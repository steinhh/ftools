/*
 * Float (32-bit) Accelerate-optimized Gaussian Jacobian computation
 *
 * Computes partial derivatives: d/di0, d/dmu, d/dsigma
 * Uses float instead of double for optimal performance
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
 * Accelerate-optimized Gaussian Jacobian computation using float (32-bit)
 * Computes all three partial derivatives in a vectorized manner
 * Output is (n, 3) array: [d_i0, d_mu, d_sigma] for each x value
 */
static void compute_jacobian_accelerate_float(const float *x, float i0, float mu, float sigma,
                                              float *result, npy_intp n)
{
  const float two_sigma_sq = 2.0f * sigma * sigma;
  const float inv_two_sigma_sq = 1.0f / two_sigma_sq;
  const float scale = -inv_two_sigma_sq;
  const float neg_mu = -mu;
  const float sigma_sq = sigma * sigma;
  const float sigma_cubed = sigma * sigma * sigma;

  /* Allocate temporary arrays for computation */
  float *diff = (float *)malloc(n * sizeof(float));
  float *exp_term = (float *)malloc(n * sizeof(float));

  if (diff == NULL || exp_term == NULL)
  {
    free(diff);
    free(exp_term);
    return;
  }

  /* Step 1: diff = x - mu */
  vDSP_vsadd(x, 1, &neg_mu, diff, 1, n);

  /* Step 2: exp_term = diff^2 */
  vDSP_vsq(diff, 1, exp_term, 1, n);

  /* Step 3: exp_term = exp_term * (-1 / two_sigma_sq) */
  vDSP_vsmul(exp_term, 1, &scale, exp_term, 1, n);

  /* Step 4: exp_term = exp(exp_term) */
  int n_int = (int)n;
  vvexpf(exp_term, exp_term, &n_int);

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
  float *temp = (float *)malloc(n * sizeof(float));
  if (temp != NULL)
  {
    /* temp = i0 * exp_term */
    vDSP_vsmul(exp_term, 1, &i0, temp, 1, n);
    /* temp = temp * diff */
    vDSP_vmul(temp, 1, diff, 1, temp, 1, n);
    /* temp = temp / sigma^2 */
    float inv_sigma_sq = 1.0f / sigma_sq;
    vDSP_vsmul(temp, 1, &inv_sigma_sq, temp, 1, n);

    for (npy_intp i = 0; i < n; i++)
    {
      result[i * 3 + 1] = temp[i];
    }
    free(temp);
  }

  /* Column 2: d_sigma = i0 * exp_term * diff^2 / sigma^3 */
  /* Reuse diff for diff^2 computation */
  vDSP_vsadd(x, 1, &neg_mu, diff, 1, n);
  vDSP_vsq(diff, 1, diff, 1, n);

  temp = (float *)malloc(n * sizeof(float));
  if (temp != NULL)
  {
    /* temp = i0 * exp_term */
    vDSP_vsmul(exp_term, 1, &i0, temp, 1, n);
    /* temp = temp * diff^2 */
    vDSP_vmul(temp, 1, diff, 1, temp, 1, n);
    /* temp = temp / sigma^3 */
    float inv_sigma_cubed = 1.0f / sigma_cubed;
    vDSP_vsmul(temp, 1, &inv_sigma_cubed, temp, 1, n);

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
 * Scalar fallback implementation (float)
 * Only compile when Accelerate is not used to avoid unused-function warnings
 */
#ifndef USE_ACCELERATE
static void compute_jacobian_scalar_float(const float *x, float i0, float mu, float sigma,
                                          float *result, npy_intp n)
{
  const float two_sigma_sq = 2.0f * sigma * sigma;
  const float sigma_sq = sigma * sigma;
  const float sigma_cubed = sigma * sigma * sigma;

  for (npy_intp i = 0; i < n; i++)
  {
    float diff = x[i] - mu;
    float exp_term = expf(-(diff * diff) / two_sigma_sq);

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
static void compute_jacobian_float(const float *x, float i0, float mu, float sigma,
                                   float *result, npy_intp n)
{
#ifdef USE_ACCELERATE
  compute_jacobian_accelerate_float(x, i0, mu, sigma, result, n);
#else
  compute_jacobian_scalar_float(x, i0, mu, sigma, result, n);
#endif
}

/*
 * Python interface: fgaussian_jacobian_f32(x, i0, mu, sigma)
 */
static PyObject *fgaussian_jacobian_f32_fgaussian_jacobian_f32(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL;
  float i0, mu, sigma;

  /* Parse arguments */
  if (!PyArg_ParseTuple(args, "O!fff",
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

  /* Ensure x is float32 type and contiguous */
  PyArrayObject *x_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)x_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
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
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, result_dims, NPY_FLOAT);
  if (result == NULL)
  {
    Py_DECREF(x_contig);
    return NULL;
  }

  /* Get pointers to data */
  const float *x_data = (const float *)PyArray_DATA(x_contig);
  float *result_data = (float *)PyArray_DATA(result);

  /* Validate sigma is positive */
  if (sigma <= 0.0f)
  {
    Py_DECREF(x_contig);
    Py_DECREF(result);
    PyErr_SetString(PyExc_ValueError, "sigma must be positive");
    return NULL;
  }

  /* Compute Jacobian matrix */
  compute_jacobian_float(x_data, i0, mu, sigma, result_data, total_size);

  Py_DECREF(x_contig);

  return (PyObject *)result;
}

/* Method definition */
static PyMethodDef FGaussianJacobianF32Methods[] = {
    {"fgaussian_jacobian_f32", fgaussian_jacobian_f32_fgaussian_jacobian_f32, METH_VARARGS,
     "Accelerate-optimized Gaussian Jacobian computation\n\n"
     "Computes partial derivatives d/di0, d/dmu, d/dsigma using float32."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fgaussian_jacobian_f32_module = {
    PyModuleDef_HEAD_INIT,
    "fgaussian_jacobian_f32_ext",
    "Float32 Accelerate-optimized C extension for computing Gaussian Jacobian",
    -1,
    FGaussianJacobianF32Methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fgaussian_jacobian_f32_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fgaussian_jacobian_f32_module);
}
