/*
 * Double (64-bit) Accelerate-optimized Gaussian Jacobian computation
 *
 * Computes partial derivatives: d/di0, d/dmu, d/dsigma
 * This is the float64 version for compatibility with existing code
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

/* Use Accelerate framework on macOS/iOS (unless FORCE_SCALAR is set) */
#if defined(__APPLE__) && !defined(FORCE_SCALAR)
#define USE_ACCELERATE 1
#include <Accelerate/Accelerate.h>
#endif

#ifdef USE_ACCELERATE
/*
 * Accelerate-optimized Gaussian Jacobian computation using double (64-bit)
 * Computes all three partial derivatives in a vectorized manner
 * Output is (n, 3) array: [d_i0, d_mu, d_sigma] for each x value
 */
static void compute_jacobian_accelerate_double(const double *x, double i0, double mu, double sigma,
                                               double *result, npy_intp n)
{
  const double two_sigma_sq = 2.0 * sigma * sigma;
  const double inv_two_sigma_sq = 1.0 / two_sigma_sq;
  const double scale = -inv_two_sigma_sq;
  const double neg_mu = -mu;
  const double sigma_sq = sigma * sigma;
  const double sigma_cubed = sigma * sigma * sigma;

  /* Allocate temporary arrays for computation */
  double *diff = (double *)malloc(n * sizeof(double));
  double *exp_term = (double *)malloc(n * sizeof(double));

  if (diff == NULL || exp_term == NULL)
  {
    free(diff);
    free(exp_term);
    return;
  }

  /* Step 1: diff = x - mu */
  vDSP_vsaddD(x, 1, &neg_mu, diff, 1, n);

  /* Step 2: exp_term = diff^2 */
  vDSP_vsqD(diff, 1, exp_term, 1, n);

  /* Step 3: exp_term = exp_term * (-1 / two_sigma_sq) */
  vDSP_vsmulD(exp_term, 1, &scale, exp_term, 1, n);

  /* Step 4: exp_term = exp(exp_term) */
  int n_int = (int)n;
  vvexp(exp_term, exp_term, &n_int);

  /* Now compute the three derivatives:
   * d_i0 = exp_term
   * d_mu = i0 * exp_term * diff / sigma^2
   * d_sigma = i0 * exp_term * diff^2 / sigma^3
   */

  /* Column 0: d_i0 = exp_term (stride 3 in output) */
  for (npy_intp i = 0; i < n; i++)
  {
    result[i * 3 + 0] = exp_term[i];
  }

  /* Column 1: d_mu = i0 * exp_term * diff / sigma^2 */
  double *temp = (double *)malloc(n * sizeof(double));
  if (temp != NULL)
  {
    /* temp = i0 * exp_term */
    vDSP_vsmulD(exp_term, 1, &i0, temp, 1, n);
    /* temp = temp * diff */
    vDSP_vmulD(temp, 1, diff, 1, temp, 1, n);
    /* temp = temp / sigma^2 */
    double inv_sigma_sq = 1.0 / sigma_sq;
    vDSP_vsmulD(temp, 1, &inv_sigma_sq, temp, 1, n);

    for (npy_intp i = 0; i < n; i++)
    {
      result[i * 3 + 1] = temp[i];
    }
    free(temp);
  }

  /* Column 2: d_sigma = i0 * exp_term * diff^2 / sigma^3 */
  /* Reuse diff for diff^2 computation */
  vDSP_vsaddD(x, 1, &neg_mu, diff, 1, n);
  vDSP_vsqD(diff, 1, diff, 1, n);

  temp = (double *)malloc(n * sizeof(double));
  if (temp != NULL)
  {
    /* temp = i0 * exp_term */
    vDSP_vsmulD(exp_term, 1, &i0, temp, 1, n);
    /* temp = temp * diff^2 */
    vDSP_vmulD(temp, 1, diff, 1, temp, 1, n);
    /* temp = temp / sigma^3 */
    double inv_sigma_cubed = 1.0 / sigma_cubed;
    vDSP_vsmulD(temp, 1, &inv_sigma_cubed, temp, 1, n);

    for (npy_intp i = 0; i < n; i++)
    {
      result[i * 3 + 2] = temp[i];
    }
    free(temp);
  }

  free(diff);
  free(exp_term);
}
#endif /* USE_ACCELERATE */

/*
 * Scalar fallback implementation (double)
 * Only compile when Accelerate is not used to avoid unused-function warnings
 */
#ifndef USE_ACCELERATE
static void compute_jacobian_scalar_double(const double *x, double i0, double mu, double sigma,
                                           double *result, npy_intp n)
{
  const double two_sigma_sq = 2.0 * sigma * sigma;
  const double sigma_sq = sigma * sigma;
  const double sigma_cubed = sigma * sigma * sigma;

  for (npy_intp i = 0; i < n; i++)
  {
    double diff = x[i] - mu;
    double exp_term = exp(-(diff * diff) / two_sigma_sq);

    /* d/di0 */
    result[i * 3 + 0] = exp_term;

    /* d/dmu */
    result[i * 3 + 1] = i0 * exp_term * diff / sigma_sq;

    /* d/dsigma */
    result[i * 3 + 2] = i0 * exp_term * (diff * diff) / sigma_cubed;
  }
}
#endif /* USE_ACCELERATE */

/*
 * Main computation function - dispatches to appropriate implementation
 */
static void compute_jacobian_double(const double *x, double i0, double mu, double sigma,
                                    double *result, npy_intp n)
{
#ifdef USE_ACCELERATE
  compute_jacobian_accelerate_double(x, i0, mu, sigma, result, n);
#else
  compute_jacobian_scalar_double(x, i0, mu, sigma, result, n);
#endif
}

/*
 * Python interface: fgaussian_jacobian_f64(x, i0, mu, sigma)
 */
static PyObject *fgaussian_jacobian_f64_fgaussian_jacobian_f64(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL;
  double i0, mu, sigma;

  /* Parse arguments */
  if (!PyArg_ParseTuple(args, "O!ddd",
                        &PyArray_Type, &x_array,
                        &i0, &mu, &sigma))
  {
    return NULL;
  }

  /* Validate x is a numpy array */
  if (!PyArray_Check(x_array))
  {
    PyErr_SetString(PyExc_TypeError, "x must be a numpy array");
    return NULL;
  }

  /* Ensure x is float64 type and contiguous */
  PyArrayObject *x_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)x_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (x_contig == NULL)
  {
    return NULL;
  }

  /* Get array dimensions */
  int ndim = PyArray_NDIM(x_contig);
  npy_intp total_size = PyArray_SIZE(x_contig);

  /* Validate that x is 1D */
  if (ndim != 1)
  {
    Py_DECREF(x_contig);
    PyErr_SetString(PyExc_ValueError, "x must be a 1D array");
    return NULL;
  }

  /* Create output array (n, 3) shape for Jacobian matrix */
  npy_intp result_dims[2] = {total_size, 3};
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, result_dims, NPY_DOUBLE);
  if (result == NULL)
  {
    Py_DECREF(x_contig);
    return NULL;
  }

  /* Get pointers to data */
  const double *x_data = (const double *)PyArray_DATA(x_contig);
  double *result_data = (double *)PyArray_DATA(result);

  /* Validate sigma is positive */
  if (sigma <= 0.0)
  {
    Py_DECREF(x_contig);
    Py_DECREF(result);
    PyErr_SetString(PyExc_ValueError, "sigma must be positive");
    return NULL;
  }

  /* Compute Jacobian matrix */
  compute_jacobian_double(x_data, i0, mu, sigma, result_data, total_size);

  Py_DECREF(x_contig);

  return (PyObject *)result;
}

/* Method definition */
static PyMethodDef FGaussianJacobianF64Methods[] = {
    {"fgaussian_jacobian_f64", fgaussian_jacobian_f64_fgaussian_jacobian_f64, METH_VARARGS,
     "Accelerate-optimized Gaussian Jacobian computation\n\n"
     "Computes partial derivatives d/di0, d/dmu, d/dsigma using float64."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fgaussian_jacobian_f64_module = {
    PyModuleDef_HEAD_INIT,
    "fgaussian_jacobian_f64_ext",
    "Float64 Accelerate-optimized C extension for computing Gaussian Jacobian",
    -1,
    FGaussianJacobianF64Methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fgaussian_jacobian_f64_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fgaussian_jacobian_f64_module);
}
