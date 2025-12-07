/*
 * fmpfit_f32_ext.c - Python C extension for MPFIT wrapper (float32 version)
 *
 * Wraps the MPFIT library for nonlinear least-squares fitting using float32
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cmpfit-1.5/mpfit.h"

/* Include Gaussian deviate computation */
#include "gaussian_deviate.c"

/*
 * Core MPFIT function - calls MPFIT library (float32 version)
 */
static void fmpfit_f32_c_wrap(
    const float *x, const float *y, const float *error,
    const float *p0, const float *bounds,
    int mpoints, int npar, int deviate_type,
    float xtol, float ftol, float gtol,
    int maxiter, int quiet,
    float *best_params, float *bestnorm, float *orignorm,
    int *niter, int *nfev, int *status,
    float *resid, float *xerror, float *covar)
{
  int i;
  mp_par *pars = NULL;
  mp_config config;
  mp_result result;
  struct gaussian_private_data_f32 private_data;

  /* Initialize parameter array from p0 */
  for (i = 0; i < npar; i++)
  {
    best_params[i] = p0[i];
  }

  /* Setup parameter constraints from bounds array */
  /* bounds is shape (npar, 2) with [lower, upper] for each parameter */
  pars = (mp_par *)calloc(npar, sizeof(mp_par));
  if (pars)
  {
    for (i = 0; i < npar; i++)
    {
      float lower = bounds[i * 2];
      float upper = bounds[i * 2 + 1];

      /* Check if lower bound is finite */
      if (isfinite(lower))
      {
        pars[i].limited[0] = 1;
        pars[i].limits[0] = lower;
      }
      else
      {
        pars[i].limited[0] = 0;
      }

      /* Check if upper bound is finite */
      if (isfinite(upper))
      {
        pars[i].limited[1] = 1;
        pars[i].limits[1] = upper;
      }
      else
      {
        pars[i].limited[1] = 0;
      }

      pars[i].fixed = 0;
      pars[i].side = 3; /* Use analytical derivatives (side=3) */
    }
  }

  /* Configure MPFIT */
  memset(&config, 0, sizeof(config));
  config.ftol = ftol;
  config.xtol = xtol;
  config.gtol = gtol;
  config.maxiter = maxiter;

  /* Setup result structure to receive outputs */
  memset(&result, 0, sizeof(result));
  result.resid = resid;
  result.xerror = xerror;
  result.covar = covar;

  /* Setup private data for Gaussian model */
  private_data.x = x;
  private_data.y = y;
  private_data.error = error;

  /* Call MPFIT */
  if (deviate_type == 0)
  {
    /* Gaussian model */
    *status = mpfit(myfunct_gaussian_deviates_with_derivatives_f32,
                    mpoints, npar, best_params, pars, &config,
                    (void *)&private_data, &result);
  }
  else
  {
    *status = -1; /* Unknown deviate type */
  }

  /* Extract results */
  *bestnorm = result.bestnorm;
  *orignorm = result.orignorm;
  *niter = result.niter;
  *nfev = result.nfev;

  /* Free parameter constraints */
  if (pars)
  {
    free(pars);
  }
}

/*
 * Python wrapper function
 *
 * Note: mpoints and npar are inferred from array shapes
 */
static PyObject *py_fmpfit_f32(PyObject *self, PyObject *args)
{
  PyObject *x_obj, *y_obj, *error_obj, *p0_obj, *bounds_obj;
  int deviate_type, maxiter, quiet;
  double xtol_d, ftol_d, gtol_d;

  (void)self;

  /* Parse arguments - no mpoints/npar needed */
  if (!PyArg_ParseTuple(args, "OOOOOidddii",
                        &x_obj, &y_obj, &error_obj, &p0_obj, &bounds_obj,
                        &deviate_type,
                        &xtol_d, &ftol_d, &gtol_d,
                        &maxiter, &quiet))
  {
    return NULL;
  }

  /* Convert tolerance values to float */
  float xtol = (float)xtol_d;
  float ftol = (float)ftol_d;
  float gtol = (float)gtol_d;

  /* Convert to contiguous arrays */
  PyArrayObject *x_contig = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *y_contig = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *error_contig = (PyArrayObject *)PyArray_FROM_OTF(error_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p0_contig = (PyArrayObject *)PyArray_FROM_OTF(p0_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *bounds_contig = (PyArrayObject *)PyArray_FROM_OTF(bounds_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  if (!x_contig || !y_contig || !error_contig || !p0_contig || !bounds_contig)
  {
    Py_XDECREF(x_contig);
    Py_XDECREF(y_contig);
    Py_XDECREF(error_contig);
    Py_XDECREF(p0_contig);
    Py_XDECREF(bounds_contig);
    return NULL;
  }

  /* Infer dimensions from array shapes */
  int mpoints = (int)PyArray_DIM(x_contig, 0);
  int npar = (int)PyArray_DIM(p0_contig, 0);

  /* Get data pointers */
  float *x = (float *)PyArray_DATA(x_contig);
  float *y = (float *)PyArray_DATA(y_contig);
  float *error = (float *)PyArray_DATA(error_contig);
  float *p0 = (float *)PyArray_DATA(p0_contig);
  float *bounds = (float *)PyArray_DATA(bounds_contig);

  /* Allocate output arrays */
  float *best_params = (float *)malloc(npar * sizeof(float));
  float *resid = (float *)malloc(mpoints * sizeof(float));
  float *xerror = (float *)malloc(npar * sizeof(float));
  float *xerror_scaled = (float *)malloc(npar * sizeof(float));
  float *covar = (float *)malloc(npar * npar * sizeof(float));
  float bestnorm, orignorm;
  int niter, nfev, status;

  if (!best_params || !resid || !xerror || !xerror_scaled || !covar)
  {
    free(best_params);
    free(resid);
    free(xerror);
    free(xerror_scaled);
    free(covar);
    Py_DECREF(x_contig);
    Py_DECREF(y_contig);
    Py_DECREF(error_contig);
    Py_DECREF(p0_contig);
    Py_DECREF(bounds_contig);
    return PyErr_NoMemory();
  }

  /* Release GIL while running the compute-heavy C routine so other Python
     threads may run concurrently. MPFIT and the user callbacks are pure C
     functions (they don't call into Python), so releasing the GIL here is
     safe. */
  {
    Py_BEGIN_ALLOW_THREADS
        fmpfit_f32_c_wrap(x, y, error, p0, bounds,
                          mpoints, npar, deviate_type,
                          xtol, ftol, gtol,
                          maxiter, quiet,
                          best_params, &bestnorm, &orignorm,
                          &niter, &nfev, &status,
                          resid, xerror, covar);

    /* Compute xerror_scaled = xerror * sqrt(chi2 / dof) to match curve_fit default */
    int dof = mpoints - npar;
    float scale_factor = (dof > 0) ? sqrtf(bestnorm / dof) : 1.0f;
    for (int i = 0; i < npar; i++)
    {
      xerror_scaled[i] = xerror[i] * scale_factor;
    }
    Py_END_ALLOW_THREADS
  }

  /* Create output arrays */
  npy_intp dims_params[1] = {npar};
  npy_intp dims_resid[1] = {mpoints};
  npy_intp dims_covar[2] = {npar, npar};

  PyArrayObject *best_params_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_FLOAT32);
  PyArrayObject *resid_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_resid, NPY_FLOAT32);
  PyArrayObject *xerror_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_FLOAT32);
  PyArrayObject *covar_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_covar, NPY_FLOAT32);
  PyArrayObject *xerror_scaled_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_FLOAT32);

  if (!best_params_array || !resid_array || !xerror_array || !covar_array ||
      !xerror_scaled_array)
  {
    Py_XDECREF(best_params_array);
    Py_XDECREF(resid_array);
    Py_XDECREF(xerror_array);
    Py_XDECREF(covar_array);
    Py_XDECREF(xerror_scaled_array);
    free(best_params);
    free(resid);
    free(xerror);
    free(covar);
    free(xerror_scaled);
    Py_DECREF(x_contig);
    Py_DECREF(y_contig);
    Py_DECREF(error_contig);
    Py_DECREF(p0_contig);
    Py_DECREF(bounds_contig);
    return NULL;
  }

  /* Copy data to output arrays */
  memcpy(PyArray_DATA(best_params_array), best_params, npar * sizeof(float));
  memcpy(PyArray_DATA(resid_array), resid, mpoints * sizeof(float));
  memcpy(PyArray_DATA(xerror_array), xerror, npar * sizeof(float));
  memcpy(PyArray_DATA(covar_array), covar, npar * npar * sizeof(float));
  memcpy(PyArray_DATA(xerror_scaled_array), xerror_scaled, npar * sizeof(float));

  /* Free temporary buffers */
  free(best_params);
  free(resid);
  free(xerror);
  free(covar);
  free(xerror_scaled);

  /* Release input arrays */
  Py_DECREF(x_contig);
  Py_DECREF(y_contig);
  Py_DECREF(error_contig);
  Py_DECREF(p0_contig);
  Py_DECREF(bounds_contig);

  /* Create result dictionary */
  PyObject *result = PyDict_New();
  if (!result)
  {
    Py_DECREF(best_params_array);
    Py_DECREF(resid_array);
    Py_DECREF(xerror_array);
    Py_DECREF(covar_array);
    Py_DECREF(xerror_scaled_array);
    return NULL;
  }

  /* For now, set dummy values for nfree, npegged */
  int nfree = npar; /* In real implementation, count non-fixed parameters */
  int npegged = 0;  /* In real implementation, count parameters at bounds */

  /* Populate dictionary */
  PyDict_SetItemString(result, "best_params", (PyObject *)best_params_array);
  PyDict_SetItemString(result, "bestnorm", PyFloat_FromDouble((double)bestnorm));
  PyDict_SetItemString(result, "orignorm", PyFloat_FromDouble((double)orignorm));
  PyDict_SetItemString(result, "niter", PyLong_FromLong(niter));
  PyDict_SetItemString(result, "nfev", PyLong_FromLong(nfev));
  PyDict_SetItemString(result, "status", PyLong_FromLong(status));
  PyDict_SetItemString(result, "npar", PyLong_FromLong(npar));
  PyDict_SetItemString(result, "nfree", PyLong_FromLong(nfree));
  PyDict_SetItemString(result, "npegged", PyLong_FromLong(npegged));
  PyDict_SetItemString(result, "nfunc", PyLong_FromLong(mpoints));
  PyDict_SetItemString(result, "resid", (PyObject *)resid_array);
  PyDict_SetItemString(result, "xerror", (PyObject *)xerror_array);
  PyDict_SetItemString(result, "covar", (PyObject *)covar_array);
  PyDict_SetItemString(result, "xerror_scaled", (PyObject *)xerror_scaled_array);

  /* Decrement reference counts (dict holds references) */
  Py_DECREF(best_params_array);
  Py_DECREF(resid_array);
  Py_DECREF(xerror_array);
  Py_DECREF(covar_array);
  Py_DECREF(xerror_scaled_array);

  return result;
}

/* Method definition */
static PyMethodDef FMPFitF32Methods[] = {
    {"fmpfit_f32", py_fmpfit_f32, METH_VARARGS,
     "Levenberg-Marquardt least-squares curve fitting (float32 version)\n\n"
     "Performs constrained nonlinear least squares fitting using MPFIT with float32 precision.\n\n"
     "Parameters: x, y, error, p0, bounds, deviate_type, xtol, ftol, gtol, maxiter, quiet\n"
     "Note: mpoints and npar are inferred from array shapes."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fmpfit_f32_module = {
    PyModuleDef_HEAD_INIT,
    "fmpfit_f32_ext",
    "MPFIT curve fitting C extension - float32 version (Levenberg-Marquardt with constraints)",
    -1,
    FMPFitF32Methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fmpfit_f32_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fmpfit_f32_module);
}
