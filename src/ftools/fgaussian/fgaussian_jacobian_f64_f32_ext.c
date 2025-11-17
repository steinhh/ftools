/*
 * Mixed-precision Gaussian Jacobian: accepts float64 input, computes in float32, returns float64 output
 *
 * All internal calculations are performed in float32 for speed, but the interface is float64.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

/* Use Accelerate framework on macOS/iOS */
#if defined(__APPLE__)
#define USE_ACCELERATE 1
#include <Accelerate/Accelerate.h>
#endif


#ifdef USE_ACCELERATE
/*
 * Accelerate-optimized Gaussian Jacobian computation using float (32-bit)
 * All calculations in float32, input/output conversion handled in wrapper
 */
static void compute_jacobian_accelerate_float(const float *x, float i0, float mu, float sigma,
                                              float *result, npy_intp n)
{
  const float one = 1.0f;
  const float neg_mu = -mu;

  float *diff = (float *)malloc(sizeof(float) * (size_t)n);
  float *exp_term = (float *)malloc(sizeof(float) * (size_t)n);
  if (diff == NULL || exp_term == NULL)
  {
    free(diff);
    free(exp_term);
    return;
  }

  /* diff = x - mu */
  vDSP_vsadd((float *)x, 1, &neg_mu, diff, 1, n);

  /* exp_term = diff^2 */
  vDSP_vsq(diff, 1, exp_term, 1, n);

  /* exp_term = -(diff^2) / (2*sigma^2) */
  float scale = - (1.0f / (2.0f * sigma * sigma));
  vDSP_vsmul(exp_term, 1, &scale, exp_term, 1, n);

  /* exp_term = exp(exp_term) */
  int n_int = (int)n;
  vvexpf(exp_term, exp_term, &n_int);

  /* Column 0: d_i0 = exp_term (store with stride 3) */
  vDSP_vsmul(exp_term, 1, &one, result + 0, 3, n);

  /* Column 1: d_mu = i0 * exp_term * diff / sigma^2 */
  float *temp = (float *)malloc(sizeof(float) * (size_t)n);
  if (temp != NULL)
  {
    vDSP_vsmul(exp_term, 1, &i0, temp, 1, n); /* temp = i0 * exp_term */
    vDSP_vmul(temp, 1, diff, 1, temp, 1, n);   /* temp *= diff */
    float inv_sigma_sq = 1.0f / (sigma * sigma);
    vDSP_vsmul(temp, 1, &inv_sigma_sq, temp, 1, n);
    vDSP_vsmul(temp, 1, &one, result + 1, 3, n); /* store into column 1 */
    free(temp);
  }

  /* Column 2: d_sigma = i0 * exp_term * diff^2 / sigma^3 */
  temp = (float *)malloc(sizeof(float) * (size_t)n);
  if (temp != NULL)
  {
    vDSP_vsq(diff, 1, temp, 1, n); /* temp = diff^2 */
    vDSP_vsmul(exp_term, 1, &i0, exp_term, 1, n); /* exp_term = i0 * exp_term */
    vDSP_vmul(exp_term, 1, temp, 1, temp, 1, n); /* temp = i0*exp_term*diff^2 */
    float inv_sigma_cubed = 1.0f / (sigma * sigma * sigma);
    vDSP_vsmul(temp, 1, &inv_sigma_cubed, temp, 1, n);
    vDSP_vsmul(temp, 1, &one, result + 2, 3, n); /* store into column 2 */
    free(temp);
  }

  free(diff);
  free(exp_term);
}
#endif /* USE_ACCELERATE */

/*
 * Scalar fallback implementation (double)
 */
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
 * Python interface: fgaussian_jacobian_f64(x, i0, mu, sigma)
 */
static PyObject *fgaussian_jacobian_f64_f32_fgaussian_jacobian_f64_f32(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL;
  double i0_d, mu_d, sigma_d;

  if (!PyArg_ParseTuple(args, "O!ddd",
                        &PyArray_Type, &x_array,
                        &i0_d, &mu_d, &sigma_d))
  {
    return NULL;
  }

  if (!PyArray_Check(x_array))
  {
    PyErr_SetString(PyExc_TypeError, "x must be a numpy array");
    return NULL;
  }

  /* Ensure x is float64 type and contiguous */
  PyArrayObject *x_contig = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)x_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (x_contig == NULL)
  {
    return NULL;
  }

  int ndim = PyArray_NDIM(x_contig);
  npy_intp total_size = PyArray_SIZE(x_contig);

  if (ndim != 1)
  {
    Py_DECREF(x_contig);
    PyErr_SetString(PyExc_ValueError, "x must be a 1D array");
    return NULL;
  }

  /* Convert parameters to float32 */
  float i0 = (float)i0_d;
  float mu = (float)mu_d;
  float sigma = (float)sigma_d;

  /* Validate sigma */
  if (sigma <= 0.0f)
  {
    Py_DECREF(x_contig);
    PyErr_SetString(PyExc_ValueError, "sigma must be positive");
    return NULL;
  }

  /* Convert input to float32 buffer */
  const double *x_data_d = (const double *)PyArray_DATA(x_contig);
  float *x_data_f = (float *)malloc(sizeof(float) * (size_t)total_size);
  if (x_data_f == NULL)
  {
    Py_DECREF(x_contig);
    PyErr_NoMemory();
    return NULL;
  }
  for (npy_intp i = 0; i < total_size; i++)
    x_data_f[i] = (float)x_data_d[i];

  /* Allocate float32 result buffer */
  float *result_f = (float *)malloc(sizeof(float) * (size_t)(total_size * 3));
  if (result_f == NULL)
  {
    free(x_data_f);
    Py_DECREF(x_contig);
    PyErr_NoMemory();
    return NULL;
  }

  /* Compute in float32 */
  compute_jacobian_float(x_data_f, i0, mu, sigma, result_f, total_size);

  /* Create output double array (n,3) */
  npy_intp result_dims[2] = {total_size, 3};
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, result_dims, NPY_DOUBLE);
  if (result == NULL)
  {
    free(x_data_f);
    free(result_f);
    Py_DECREF(x_contig);
    return NULL;
  }

  /* Convert float results to double output */
  double *result_d = (double *)PyArray_DATA(result);
  for (npy_intp i = 0; i < total_size * 3; i++)
    result_d[i] = (double)result_f[i];

  free(x_data_f);
  free(result_f);
  Py_DECREF(x_contig);

  return (PyObject *)result;
}

/* Method definition */
static PyMethodDef FGaussianJacobianF64F32Methods[] = {
  {"fgaussian_jacobian_f64_f32", fgaussian_jacobian_f64_f32_fgaussian_jacobian_f64_f32, METH_VARARGS,
   "Gaussian Jacobian: accepts f64 input, computes in f32, returns f64 (n,3)."},
  {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fgaussian_jacobian_f64_f32_module = {
  PyModuleDef_HEAD_INIT,
  "fgaussian_jacobian_f64_f32_ext",
  "Mixed-precision Gaussian Jacobian: input f64 -> compute f32 -> output f64",
  -1,
  FGaussianJacobianF64F32Methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fgaussian_jacobian_f64_f32_ext(void)
{
  import_array();
  if (PyErr_Occurred())
  {
    return NULL;
  }
  return PyModule_Create(&fgaussian_jacobian_f64_f32_module);
}
