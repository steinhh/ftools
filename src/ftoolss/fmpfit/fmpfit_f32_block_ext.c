/*
 * fmpfit_f32_block_ext.c - Python C extension for MPFIT block wrapper (float32)
 *
 * Wraps the MPFIT library for nonlinear least-squares fitting of multiple spectra
 * in a single call, using single precision.
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
 * Core MPFIT function for a single spectrum - reused from fmpfit_f32_ext
 *
 * xerror_scipy: if non-NULL, will be filled with scipy-style errors from full Hessian inverse
 */
static void fmpfit_f32_single(
    const float *x, const float *y, const float *error,
    const float *p0, const float *bounds,
    int mpoints, int npar, int deviate_type,
    float xtol, float ftol, float gtol,
    int maxiter, int quiet,
    float *best_params, float *bestnorm, float *orignorm,
    int *niter, int *nfev, int *status,
    int *nfree, int *npegged,
    float *resid, float *xerror, float *covar,
    float *xerror_scipy)
{
  int i;
  mp_par *pars = NULL;
  mp_config config;
  mp_result result;
  struct gaussian_private_data_f32 private_data;

  (void)deviate_type; /* Currently only Gaussian supported */
  (void)quiet;

  /* Initialize parameter array from p0 */
  for (i = 0; i < npar; i++)
  {
    best_params[i] = p0[i];
  }

  /* Setup parameter constraints from bounds array */
  /* bounds is contiguous: [min0, max0, min1, max1, ...] for this spectrum */
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

  /* Setup MPFIT configuration */
  memset(&config, 0, sizeof(config));
  config.ftol = ftol;
  config.xtol = xtol;
  config.gtol = gtol;
  config.maxiter = maxiter;
  config.nofinitecheck = 0;

  /* Setup result structure */
  memset(&result, 0, sizeof(result));
  result.resid = resid;
  result.xerror = xerror;
  result.covar = covar;
  result.xerror_scipy = xerror_scipy; /* mpfit computes scipy-style errors from full Hessian inverse */

  /* Setup private data for user function */
  private_data.x = x;
  private_data.y = y;
  private_data.error = error;

  /* Call MPFIT */
  *status = mpfit(myfunct_gaussian_deviates_with_derivatives_f32,
                  mpoints, npar, best_params, pars, &config,
                  (void *)&private_data, &result);

  /* Extract results */
  *bestnorm = result.bestnorm;
  *orignorm = result.orignorm;
  *niter = result.niter;
  *nfev = result.nfev;
  *nfree = result.nfree;
  *npegged = result.npegged;

  /* Free allocated memory */
  if (pars)
    free(pars);
}

/*
 * Python wrapper: fmpfit_f32_block(x, y, error, p0, bounds, deviate_type,
 *                                   xtol, ftol, gtol, maxiter, quiet)
 *
 * Array shapes (row-major, C order):
 *   x:      (n_spectra, mpoints)
 *   y:      (n_spectra, mpoints)
 *   error:  (n_spectra, mpoints)
 *   p0:     (n_spectra, npar)
 *   bounds: (n_spectra, npar, 2)
 *
 * Note: n_spectra, mpoints, and npar are inferred from array shapes.
 */
static PyObject *py_fmpfit_f32_block(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL, *y_array = NULL, *error_array = NULL;
  PyArrayObject *p0_array = NULL, *bounds_array = NULL;
  int deviate_type, maxiter, quiet;
  double xtol_d, ftol_d, gtol_d;

  (void)self; /* Suppress unused parameter warning */

  /* Parse arguments - dimensions inferred from arrays */
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!idddii",
                        &PyArray_Type, &x_array,
                        &PyArray_Type, &y_array,
                        &PyArray_Type, &error_array,
                        &PyArray_Type, &p0_array,
                        &PyArray_Type, &bounds_array,
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

  /* Ensure arrays are contiguous and correct type */
  PyArrayObject *x_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)x_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *y_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)y_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *error_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)error_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p0_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)p0_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *bounds_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)bounds_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);

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
  int n_spectra = (int)PyArray_DIM(x_contig, 0);
  int mpoints = (int)PyArray_DIM(x_contig, 1);
  int npar = (int)PyArray_DIM(p0_contig, 1);

  /* Get data pointers */
  const float *x = (const float *)PyArray_DATA(x_contig);
  const float *y = (const float *)PyArray_DATA(y_contig);
  const float *error = (const float *)PyArray_DATA(error_contig);
  const float *p0 = (const float *)PyArray_DATA(p0_contig);
  const float *bounds = (const float *)PyArray_DATA(bounds_contig);

  /* Create output arrays with proper dimensions */
  npy_intp dims_params[2] = {n_spectra, npar};
  npy_intp dims_scalar[1] = {n_spectra};
  npy_intp dims_resid[2] = {n_spectra, mpoints};
  npy_intp dims_covar[3] = {n_spectra, npar, npar};

  PyArrayObject *best_params_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_params, NPY_FLOAT);
  PyArrayObject *bestnorm_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_FLOAT);
  PyArrayObject *orignorm_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_FLOAT);
  PyArrayObject *niter_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *nfev_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *status_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *npar_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *nfree_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *npegged_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *nfunc_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_scalar, NPY_INT32);
  PyArrayObject *resid_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_resid, NPY_FLOAT);
  PyArrayObject *xerror_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_params, NPY_FLOAT);
  PyArrayObject *covar_array = (PyArrayObject *)PyArray_SimpleNew(3, dims_covar, NPY_FLOAT);
  PyArrayObject *xerror_scipy_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_params, NPY_FLOAT);

  if (!best_params_array || !bestnorm_array || !orignorm_array ||
      !niter_array || !nfev_array || !status_array ||
      !npar_array || !nfree_array || !npegged_array || !nfunc_array ||
      !resid_array || !xerror_array || !covar_array ||
      !xerror_scipy_array)
  {
    Py_XDECREF(best_params_array);
    Py_XDECREF(bestnorm_array);
    Py_XDECREF(orignorm_array);
    Py_XDECREF(niter_array);
    Py_XDECREF(nfev_array);
    Py_XDECREF(status_array);
    Py_XDECREF(npar_array);
    Py_XDECREF(nfree_array);
    Py_XDECREF(npegged_array);
    Py_XDECREF(nfunc_array);
    Py_XDECREF(resid_array);
    Py_XDECREF(xerror_array);
    Py_XDECREF(covar_array);
    Py_XDECREF(xerror_scipy_array);
    Py_DECREF(x_contig);
    Py_DECREF(y_contig);
    Py_DECREF(error_contig);
    Py_DECREF(p0_contig);
    Py_DECREF(bounds_contig);
    PyErr_NoMemory();
    return NULL;
  }

  /* Get output data pointers */
  float *out_best_params = (float *)PyArray_DATA(best_params_array);
  float *out_bestnorm = (float *)PyArray_DATA(bestnorm_array);
  float *out_orignorm = (float *)PyArray_DATA(orignorm_array);
  int *out_niter = (int *)PyArray_DATA(niter_array);
  int *out_nfev = (int *)PyArray_DATA(nfev_array);
  int *out_status = (int *)PyArray_DATA(status_array);
  int *out_npar = (int *)PyArray_DATA(npar_array);
  int *out_nfree = (int *)PyArray_DATA(nfree_array);
  int *out_npegged = (int *)PyArray_DATA(npegged_array);
  int *out_nfunc = (int *)PyArray_DATA(nfunc_array);
  float *out_resid = (float *)PyArray_DATA(resid_array);
  float *out_xerror = (float *)PyArray_DATA(xerror_array);
  float *out_covar = (float *)PyArray_DATA(covar_array);
  float *out_xerror_scipy = (float *)PyArray_DATA(xerror_scipy_array);

  /* Release GIL while running the compute-heavy loop */
  Py_BEGIN_ALLOW_THREADS

      /* Process each spectrum */
      for (int s = 0; s < n_spectra; s++)
  {
    /* Pointers to this spectrum's data (contiguous in last dimension) */
    const float *x_s = x + s * mpoints;
    const float *y_s = y + s * mpoints;
    const float *error_s = error + s * mpoints;
    const float *p0_s = p0 + s * npar;
    const float *bounds_s = bounds + s * npar * 2;

    /* Pointers to this spectrum's output */
    float *best_params_s = out_best_params + s * npar;
    float *resid_s = out_resid + s * mpoints;
    float *xerror_s = out_xerror + s * npar;
    float *covar_s = out_covar + s * npar * npar;

    /* Pointers to xerror outputs for this spectrum */
    float *xerror_scipy_s = out_xerror_scipy + s * npar;

    /* Fit this spectrum */
    fmpfit_f32_single(
        x_s, y_s, error_s, p0_s, bounds_s,
        mpoints, npar, deviate_type,
        xtol, ftol, gtol, maxiter, quiet,
        best_params_s,
        &out_bestnorm[s], &out_orignorm[s],
        &out_niter[s], &out_nfev[s], &out_status[s],
        &out_nfree[s], &out_npegged[s],
        resid_s, xerror_s, covar_s,
        xerror_scipy_s); /* mpfit computes scipy-style errors internally */

    /* Set constant values */
    out_npar[s] = npar;
    out_nfunc[s] = mpoints;
  }

  Py_END_ALLOW_THREADS

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
    Py_DECREF(bestnorm_array);
    Py_DECREF(orignorm_array);
    Py_DECREF(niter_array);
    Py_DECREF(nfev_array);
    Py_DECREF(status_array);
    Py_DECREF(npar_array);
    Py_DECREF(nfree_array);
    Py_DECREF(npegged_array);
    Py_DECREF(nfunc_array);
    Py_DECREF(resid_array);
    Py_DECREF(xerror_array);
    Py_DECREF(covar_array);
    Py_DECREF(xerror_scipy_array);
    return NULL;
  }

  /* Populate dictionary */
  PyDict_SetItemString(result, "best_params", (PyObject *)best_params_array);
  PyDict_SetItemString(result, "bestnorm", (PyObject *)bestnorm_array);
  PyDict_SetItemString(result, "orignorm", (PyObject *)orignorm_array);
  PyDict_SetItemString(result, "niter", (PyObject *)niter_array);
  PyDict_SetItemString(result, "nfev", (PyObject *)nfev_array);
  PyDict_SetItemString(result, "status", (PyObject *)status_array);
  PyDict_SetItemString(result, "npar", (PyObject *)npar_array);
  PyDict_SetItemString(result, "nfree", (PyObject *)nfree_array);
  PyDict_SetItemString(result, "npegged", (PyObject *)npegged_array);
  PyDict_SetItemString(result, "nfunc", (PyObject *)nfunc_array);
  PyDict_SetItemString(result, "resid", (PyObject *)resid_array);
  PyDict_SetItemString(result, "xerror", (PyObject *)xerror_array);
  PyDict_SetItemString(result, "covar", (PyObject *)covar_array);
  PyDict_SetItemString(result, "xerror_scipy", (PyObject *)xerror_scipy_array);

  /* Decrement reference counts (dict holds references) */
  Py_DECREF(best_params_array);
  Py_DECREF(bestnorm_array);
  Py_DECREF(orignorm_array);
  Py_DECREF(niter_array);
  Py_DECREF(nfev_array);
  Py_DECREF(status_array);
  Py_DECREF(npar_array);
  Py_DECREF(nfree_array);
  Py_DECREF(npegged_array);
  Py_DECREF(nfunc_array);
  Py_DECREF(resid_array);
  Py_DECREF(xerror_array);
  Py_DECREF(covar_array);
  Py_DECREF(xerror_scipy_array);

  return result;
}

/* Method definition */
static PyMethodDef FMPFitBlockF32Methods[] = {
    {"fmpfit_f32_block", py_fmpfit_f32_block, METH_VARARGS,
     "Levenberg-Marquardt least-squares curve fitting for multiple spectra (float32)\n\n"
     "Performs constrained nonlinear least squares fitting using MPFIT with single precision\n"
     "on multiple spectra in a single call.\n\n"
     "Parameters: x, y, error, p0, bounds, deviate_type, xtol, ftol, gtol, maxiter, quiet\n"
     "Note: n_spectra, mpoints, and npar are inferred from array shapes."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fmpfit_f32_block_module = {
    PyModuleDef_HEAD_INIT,
    "fmpfit_f32_block_ext",
    "MPFIT block curve fitting C extension (Levenberg-Marquardt with constraints) - float32",
    -1,
    FMPFitBlockF32Methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fmpfit_f32_block_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fmpfit_f32_block_module);
}
