/*
 * Float (32-bit) Accelerate-optimized Gaussian computation
 *
 * Uses float instead of double to reduce memory bandwidth by 50%
 * Should provide significant speedup for memory-bound operations
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
 * Accelerate-optimized Gaussian computation using float (32-bit)
 * Uses vForce vvexpf for vectorized single-precision exponential
 * Works directly in the output array to avoid temporary allocation
 */
static void compute_gaussian_accelerate_float(const float *x, float i0, float mu, float sigma,
                                              float *result, npy_intp n)
{
  const float two_sigma_sq = 2.0f * sigma * sigma;
  const float inv_two_sigma_sq = 1.0f / two_sigma_sq;
  const float scale = -inv_two_sigma_sq;
  const float neg_mu = -mu;

  /* Step 1: result = x - mu (vectorized subtraction) */
  vDSP_vsadd(x, 1, &neg_mu, result, 1, n);

  /* Step 2: result = result * result (vectorized squaring - in place) */
  vDSP_vsq(result, 1, result, 1, n);

  /* Step 3: result = result * (-1 / two_sigma_sq) (vectorized scale - in place) */
  vDSP_vsmul(result, 1, &scale, result, 1, n);

  /* Step 4: result = exp(result) (vectorized exponential - in place) */
  int n_int = (int)n;
  vvexpf(result, result, &n_int);

  /* Step 5: result = i0 * result (vectorized scale - in place) */
  vDSP_vsmul(result, 1, &i0, result, 1, n);
}
#endif /* USE_ACCELERATE */

/*
 * Scalar fallback implementation (float)
 * Only compile this fallback when Accelerate is not used to avoid
 * unused-function warnings on platforms where the vectorized path is active.
 */
#ifndef USE_ACCELERATE
static void compute_gaussian_scalar_float(const float *x, float i0, float mu, float sigma,
                                          float *result, npy_intp n)
{
  const float two_sigma_sq = 2.0f * sigma * sigma;

  for (npy_intp i = 0; i < n; i++)
  {
    float diff = x[i] - mu;
    result[i] = i0 * expf(-(diff * diff) / two_sigma_sq);
  }
}
#endif

/*
 * Main computation function - dispatches to appropriate implementation
 */
static void compute_gaussian_float(const float *x, float i0, float mu, float sigma,
                                   float *result, npy_intp n)
{
#ifdef USE_ACCELERATE
  compute_gaussian_accelerate_float(x, i0, mu, sigma, result, n);
#else
  compute_gaussian_scalar_float(x, i0, mu, sigma, result, n);
#endif
}

/*
 * Python interface: fgaussian_f32(x, i0, mu, sigma)
 */
static PyObject *fgaussian_f32_fgaussian_f32(PyObject *self, PyObject *args)
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
  npy_intp *dims = PyArray_DIMS(x_contig);
  npy_intp total_size = PyArray_SIZE(x_contig);

  /* Create output array (float32) */
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_FLOAT);
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

  /* Compute Gaussian profile ? run pure C compute without holding the GIL
     (MP-safe: compute_gaussian_* do not call Python C-API or touch Python objects) */
  {
    Py_BEGIN_ALLOW_THREADS
        compute_gaussian_float(x_data, i0, mu, sigma, result_data, total_size);
    Py_END_ALLOW_THREADS
  }

  Py_DECREF(x_contig);

  return (PyObject *)result;
}

/* Method definition */
static PyMethodDef FGaussianF32Methods[] = {
    {"fgaussian_f32", fgaussian_f32_fgaussian_f32, METH_VARARGS,
     "Accelerate-optimized Gaussian profile computation\n\n"
     "Uses float32 for optimal performance (~5x faster than NumPy float64)."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fgaussian_f32_module = {
    PyModuleDef_HEAD_INIT,
    "fgaussian_f32_ext",
    "Float32 Accelerate-optimized C extension for computing Gaussian profiles",
    -1,
    FGaussianF32Methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fgaussian_f32_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fgaussian_f32_module);
}
