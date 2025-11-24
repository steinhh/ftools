/*
 * Mixed-precision Gaussian Jacobian (f64 input -> f32 compute -> f64 output)
 *
 * Accepts double precision `x` but computes internally in float32. Uses Apple's
 * Accelerate framework for vectorization where available, and a scalar fallback
 * otherwise.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

/* Use Accelerate framework on macOS/iOS (unless FORCE_SCALAR is set) */
#if defined(__APPLE__) && !defined(FORCE_SCALAR)
#define USE_ACCELERATE 1
#include <Accelerate/Accelerate.h>
#endif

/*
 * Scalar implementation (float32)
 * Only compile when Accelerate is not used to avoid unused-function warnings
 */
#ifndef USE_ACCELERATE
static void compute_jacobian_scalar_float(const float *x, float i0, float mu, float sigma,
                                          float *result, float *scratch, npy_intp n)
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

/* Static scratch buffers to avoid repeated malloc/free */
/* Thread-local scratch buffers */
typedef struct
{
  float *buf;     /* pointer to allocated float buffer */
  npy_intp cap_n; /* capacity per small array (n) */
} thread_scratch_t;

static pthread_key_t thread_scratch_key;
static pthread_once_t thread_scratch_once = PTHREAD_ONCE_INIT;

static void free_thread_scratch(void *p)
{
  if (p == NULL)
    return;
  thread_scratch_t *ts = (thread_scratch_t *)p;
  free(ts->buf);
  free(ts);
}

static void make_thread_scratch_key(void)
{
  pthread_key_create(&thread_scratch_key, free_thread_scratch);
}

static float *get_thread_scratch(npy_intp n)
{
  pthread_once(&thread_scratch_once, make_thread_scratch_key);
  thread_scratch_t *ts = (thread_scratch_t *)pthread_getspecific(thread_scratch_key);
  size_t needed = (size_t)8 * (size_t)n; /* 4*n scratch + n input + 3*n result */
  if (ts == NULL)
  {
    ts = (thread_scratch_t *)malloc(sizeof(thread_scratch_t));
    if (ts == NULL)
      return NULL;
    ts->buf = (float *)malloc(sizeof(float) * needed);
    if (ts->buf == NULL)
    {
      free(ts);
      return NULL;
    }
    ts->cap_n = n;
    if (pthread_setspecific(thread_scratch_key, ts) != 0)
    {
      free(ts->buf);
      free(ts);
      return NULL;
    }
    return ts->buf;
  }
  if (ts->cap_n < n)
  {
    float *newbuf = (float *)realloc(ts->buf, sizeof(float) * needed);
    if (newbuf == NULL)
      return NULL;
    ts->buf = newbuf;
    ts->cap_n = n;
  }
  return ts->buf;
}

#ifdef USE_ACCELERATE
/*
 * Accelerate vectorized implementation (float32)
 */
static void compute_jacobian_accelerate_float(const float *x_in, float i0, float mu, float sigma,
                                              float *result, float *scratch, npy_intp n)
{
  const float one = 1.0f;
  const float neg_mu = -mu;

  /* Partition the per-call scratch buffer into arrays */
  float *diff = scratch;          /* len n */
  float *exp_term = scratch + n;  /* len n */
  float *temp = scratch + 2 * n;  /* len n */
  float *temp2 = scratch + 3 * n; /* len n */

  /* diff = x - mu */
  vDSP_vsadd((float *)x_in, 1, &neg_mu, diff, 1, n);

  /* exp_term = diff^2 */
  vDSP_vsq(diff, 1, exp_term, 1, n);

  /* exp_term = -(diff^2) / (2*sigma^2) */
  float scale = -(1.0f / (2.0f * sigma * sigma));
  vDSP_vsmul(exp_term, 1, &scale, exp_term, 1, n);

  /* exp_term = exp(exp_term) */
  int n_int = (int)n;
  vvexpf(exp_term, exp_term, &n_int);

  /* Column 0: d_i0 = exp_term (store with stride 3) */
  vDSP_vsmul(exp_term, 1, &one, result + 0, 3, n);

  /* Column 1: d_mu = i0 * exp_term * diff / sigma^2 */
  vDSP_vmul(exp_term, 1, diff, 1, temp, 1, n); /* temp = exp_term * diff */
  vDSP_vsmul(temp, 1, &i0, temp, 1, n);        /* temp = i0 * exp_term * diff */
  float inv_sigma_sq = 1.0f / (sigma * sigma);
  vDSP_vsmul(temp, 1, &inv_sigma_sq, temp, 1, n); /* temp /= sigma^2 */
  vDSP_vsmul(temp, 1, &one, result + 1, 3, n);    /* store into col 1 */

  /* Column 2: d_sigma = i0 * exp_term * diff^2 / sigma^3 */
  vDSP_vsq(diff, 1, temp2, 1, n);                /* temp2 = diff^2 */
  vDSP_vmul(exp_term, 1, temp2, 1, temp2, 1, n); /* temp2 = exp_term * diff^2 */
  vDSP_vsmul(temp2, 1, &i0, temp2, 1, n);        /* temp2 = i0 * exp_term * diff^2 */
  float inv_sigma_cubed = 1.0f / (sigma * sigma * sigma);
  vDSP_vsmul(temp2, 1, &inv_sigma_cubed, temp2, 1, n);
  vDSP_vsmul(temp2, 1, &one, result + 2, 3, n); /* store into col 2 */

  /* scratch arrays are reused across invocations */
}
#endif /* USE_ACCELERATE */

/*
 * Dispatcher: choose vectorized (if available) or scalar
 */
static void compute_jacobian_float(const float *x, float i0, float mu, float sigma,
                                   float *result, float *scratch, npy_intp n)
{
#ifdef USE_ACCELERATE
  compute_jacobian_accelerate_float(x, i0, mu, sigma, result, scratch, n);
#else
  compute_jacobian_scalar_float(x, i0, mu, sigma, result, scratch, n);
#endif
}

/*
 * Python wrapper: fgaussian_jacobian_f64_f32(x, i0, mu, sigma)
 * Accepts double array x, computes in float32, returns double array
 */
static PyObject *fgaussian_jacobian_f64_f32(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL;
  double i0_d, mu_d, sigma_d;

  if (!PyArg_ParseTuple(args, "O!ddd", &PyArray_Type, &x_array, &i0_d, &mu_d, &sigma_d))
  {
    return NULL;
  }

  if (!PyArray_Check(x_array))
  {
    PyErr_SetString(PyExc_TypeError, "x must be a numpy array");
    return NULL;
  }

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

  if (sigma <= 0.0f)
  {
    Py_DECREF(x_contig);
    PyErr_SetString(PyExc_ValueError, "sigma must be positive");
    return NULL;
  }

#ifdef USE_ACCELERATE
  const double *x_data_d = (const double *)PyArray_DATA(x_contig);
#else
  const double *x_data_d = (const double *)PyArray_DATA(x_contig);
#endif

  /* Layout: scratch (4*n floats), x_data_f (n floats), result_f (3*n floats) => 8*n floats needed */
  size_t n = (size_t)total_size;
  size_t floats_needed = 8 * n;
  float *all = get_thread_scratch(total_size);
  int must_free_all = 0;
  if (all == NULL)
  {
    all = (float *)malloc(sizeof(float) * floats_needed);
    if (all == NULL)
    {
      Py_DECREF(x_contig);
      PyErr_NoMemory();
      return NULL;
    }
    must_free_all = 1;
  }
  float *scratch = all;          /* 4*n */
  float *x_data_f = all + 4 * n; /* n */
#ifdef USE_ACCELERATE
  int n_int = (int)total_size;
  vDSP_vdpsp(x_data_d, 1, x_data_f, 1, n_int);
#else
  for (npy_intp i = 0; i < total_size; i++)
    x_data_f[i] = (float)x_data_d[i];
#endif

  /* Compute in float32 using per-call/thread scratch and result buffer */
  float *result_f = all + 5 * n;
  compute_jacobian_float(x_data_f, i0, mu, sigma, result_f, scratch, total_size);

  /* Create output double array and convert results */
  npy_intp result_dims[2] = {total_size, 3};
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, result_dims, NPY_DOUBLE);
  if (result == NULL)
  {
    free(all);
    Py_DECREF(x_contig);
    return NULL;
  }

  double *result_d = (double *)PyArray_DATA(result);
#ifdef USE_ACCELERATE
  /* Convert each float3 column to double output with stride 3 using vDSP */
  int n_int2 = (int)total_size;
  vDSP_vspdp(result_f + 0, 3, result_d + 0, 3, n_int2); /* col 0 */
  vDSP_vspdp(result_f + 1, 3, result_d + 1, 3, n_int2); /* col 1 */
  vDSP_vspdp(result_f + 2, 3, result_d + 2, 3, n_int2); /* col 2 */
#else
  for (npy_intp i = 0; i < total_size * 3; i++)
    result_d[i] = (double)result_f[i];
#endif

  if (must_free_all)
    free(all);
  Py_DECREF(x_contig);

  return (PyObject *)result;
}

/* Module method definition */
static PyMethodDef FGaussianJacobianF64F32Methods[] = {
    {"fgaussian_jacobian_f64_f32", fgaussian_jacobian_f64_f32, METH_VARARGS,
     "Gaussian Jacobian: accepts f64 input, computes in f32, returns f64 (n,3)."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fgaussian_jacobian_f64_f32_module = {
    PyModuleDef_HEAD_INIT,
    "fgaussian_jacobian_f64_f32_ext",
    "Mixed-precision Gaussian Jacobian: input f64 -> compute f32 -> output f64",
    -1,
    FGaussianJacobianF64F32Methods};

PyMODINIT_FUNC PyInit_fgaussian_jacobian_f64_f32_ext(void)
{
  import_array();
  if (PyErr_Occurred())
    return NULL;
  return PyModule_Create(&fgaussian_jacobian_f64_f32_module);
}
