/*
 * fmpfit_ext.c - Python C extension for MPFIT curve fitting
 *
 * Wraps the MPFIT Levenberg-Marquardt least-squares fitting library
 * Provides efficient parameter estimation with bounds constraints
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cmpfit-1.5/mpfit.h"

/* Private data structure for passing x, y, error to user function */
struct gaussian_private_data
{
  const double *x;     /* Independent variable */
  const double *y;     /* Measured values */
  const double *error; /* Measurement uncertainties */
};

/*
 * User function: compute Gaussian deviates (y - f(x))/error
 * For Gaussian model: f(x) = p[0] * exp(-0.5 * ((x - p[1]) / p[2])^2)
 * Parameters: p[0] = amplitude, p[1] = mean, p[2] = sigma
 */
int myfunct_gaussian_deviates_with_derivatives(int m, int n, double *p, double *deviates,
                                               double **derivs, void *private_data)
{
  struct gaussian_private_data *private = (struct gaussian_private_data *)private_data;
  int i;
  double amp, mean, sigma;

  /* Extract parameters */
  amp = p[0];
  mean = p[1];
  sigma = p[2];

  /* Compute weighted deviates: (y - model) / error */
  for (i = 0; i < m; i++)
  {
    double x_i = private->x[i];
    double y_i = private->y[i];
    double err_i = private->error[i];
    double z = (x_i - mean) / sigma;
    double model = amp * exp(-0.5 * z * z);

    deviates[i] = (y_i - model) / err_i;
  }

  /* If derivs is non-zero, compute analytical derivatives */
  /* Since deviate = (y - model)/error, d(deviate)/d(p) = -d(model)/d(p)/error */
  if (derivs)
  {
    for (i = 0; i < m; i++)
    {
      double x_i = private->x[i];
      double err_i = private->error[i];
      double z = (x_i - mean) / sigma;
      double expterm = exp(-0.5 * z * z);

      /* d(deviate)/d(amp) = -d(model)/d(amp)/error = -exp(-0.5*z^2) / error */
      if (derivs[0])
        derivs[0][i] = -expterm / err_i;

      /* d(deviate)/d(mean) = -d(model)/d(mean)/error = -amp * z * exp(-0.5*z^2) / (sigma * error) */
      if (derivs[1])
        derivs[1][i] = -amp * z * expterm / (sigma * err_i);

      /* d(deviate)/d(sigma) = -d(model)/d(sigma)/error = -amp * z^2 * exp(-0.5*z^2) / (sigma * error) */
      if (derivs[2])
        derivs[2][i] = -amp * z * z * expterm / (sigma * err_i);
    }
  }

  return 0;
}

/*
 * Core MPFIT function - calls MPFIT library
 */
static void fmpfit_c_wrap(
    const double *x, const double *y, const double *error,
    const double *p0, const double *bounds,
    int mpoints, int npar, int deviate_type,
    double xtol, double ftol, double gtol,
    int maxiter, int quiet,
    double *best_params, double *bestnorm, double *orignorm,
    int *niter, int *nfev, int *status,
    double *resid, double *xerror, double *covar)
{
  int i;
  mp_par *pars = NULL;
  mp_config config;
  mp_result result;
  struct gaussian_private_data private;

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
      double lower = bounds[i * 2];
      double upper = bounds[i * 2 + 1];

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

  /* Setup private data for user function */
  private.x = x;
  private.y = y;
  private.error = error;

  /* Call MPFIT */
  *status = mpfit(myfunct_gaussian_deviates_with_derivatives,
                  mpoints, npar, best_params, pars, &config,
                  (void *)&private, &result);

  /* Extract results */
  *bestnorm = result.bestnorm;
  *orignorm = result.orignorm;
  *niter = result.niter;
  *nfev = result.nfev;

  /* Free allocated memory */
  if (pars)
    free(pars);
}

/*
 * Python wrapper: fmpfit(x, y, error, p0, bounds, mpoints, npar, deviate_type,
 *                        xtol, ftol, gtol, maxiter, quiet)
 */
static PyObject *py_fmpfit(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL, *y_array = NULL, *error_array = NULL;
  PyArrayObject *p0_array = NULL, *bounds_array = NULL;
  int mpoints, npar, deviate_type, maxiter, quiet;
  double xtol, ftol, gtol;

  /* Parse arguments */
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiidddii",
                        &PyArray_Type, &x_array,
                        &PyArray_Type, &y_array,
                        &PyArray_Type, &error_array,
                        &PyArray_Type, &p0_array,
                        &PyArray_Type, &bounds_array,
                        &mpoints, &npar, &deviate_type,
                        &xtol, &ftol, &gtol,
                        &maxiter, &quiet))
  {
    return NULL;
  }

  /* Ensure arrays are contiguous and correct type */
  PyArrayObject *x_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)x_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *y_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)y_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *error_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)error_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p0_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)p0_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *bounds_contig = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)bounds_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (!x_contig || !y_contig || !error_contig || !p0_contig || !bounds_contig)
  {
    Py_XDECREF(x_contig);
    Py_XDECREF(y_contig);
    Py_XDECREF(error_contig);
    Py_XDECREF(p0_contig);
    Py_XDECREF(bounds_contig);
    return NULL;
  }

  /* Get data pointers */
  const double *x = (const double *)PyArray_DATA(x_contig);
  const double *y = (const double *)PyArray_DATA(y_contig);
  const double *error = (const double *)PyArray_DATA(error_contig);
  const double *p0 = (const double *)PyArray_DATA(p0_contig);
  const double *bounds = (const double *)PyArray_DATA(bounds_contig);

  /* Allocate output arrays */
  double *best_params = (double *)malloc(npar * sizeof(double));
  double *resid = (double *)malloc(mpoints * sizeof(double));
  double *xerror = (double *)malloc(npar * sizeof(double));
  double *covar = (double *)malloc(npar * npar * sizeof(double));

  if (!best_params || !resid || !xerror || !covar)
  {
    free(best_params);
    free(resid);
    free(xerror);
    free(covar);
    Py_DECREF(x_contig);
    Py_DECREF(y_contig);
    Py_DECREF(error_contig);
    Py_DECREF(p0_contig);
    Py_DECREF(bounds_contig);
    PyErr_NoMemory();
    return NULL;
  }

  /* Output scalars */
  double bestnorm, orignorm;
  int niter, nfev, status;

  /* Call core fitting function */
  fmpfit_c_wrap(x, y, error, p0, bounds,
                mpoints, npar, deviate_type,
                xtol, ftol, gtol, maxiter, quiet,
                best_params, &bestnorm, &orignorm,
                &niter, &nfev, &status,
                resid, xerror, covar);

  /* Create output arrays */
  npy_intp dims_params[1] = {npar};
  npy_intp dims_resid[1] = {mpoints};
  npy_intp dims_covar[2] = {npar, npar};

  PyArrayObject *best_params_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_DOUBLE);
  PyArrayObject *resid_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_resid, NPY_DOUBLE);
  PyArrayObject *xerror_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_DOUBLE);
  PyArrayObject *covar_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_covar, NPY_DOUBLE);

  if (!best_params_array || !resid_array || !xerror_array || !covar_array)
  {
    Py_XDECREF(best_params_array);
    Py_XDECREF(resid_array);
    Py_XDECREF(xerror_array);
    Py_XDECREF(covar_array);
    free(best_params);
    free(resid);
    free(xerror);
    free(covar);
    Py_DECREF(x_contig);
    Py_DECREF(y_contig);
    Py_DECREF(error_contig);
    Py_DECREF(p0_contig);
    Py_DECREF(bounds_contig);
    return NULL;
  }

  /* Copy data to output arrays */
  memcpy(PyArray_DATA(best_params_array), best_params, npar * sizeof(double));
  memcpy(PyArray_DATA(resid_array), resid, mpoints * sizeof(double));
  memcpy(PyArray_DATA(xerror_array), xerror, npar * sizeof(double));
  memcpy(PyArray_DATA(covar_array), covar, npar * npar * sizeof(double));

  /* Free temporary buffers */
  free(best_params);
  free(resid);
  free(xerror);
  free(covar);

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
    return NULL;
  }

  /* For now, set dummy values for nfree, npegged */
  int nfree = npar; /* In real implementation, count non-fixed parameters */
  int npegged = 0;  /* In real implementation, count parameters at bounds */

  /* Populate dictionary */
  PyDict_SetItemString(result, "best_params", (PyObject *)best_params_array);
  PyDict_SetItemString(result, "bestnorm", PyFloat_FromDouble(bestnorm));
  PyDict_SetItemString(result, "orignorm", PyFloat_FromDouble(orignorm));
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

  /* Decrement reference counts (dict holds references) */
  Py_DECREF(best_params_array);
  Py_DECREF(resid_array);
  Py_DECREF(xerror_array);
  Py_DECREF(covar_array);

  return result;
}

/* Method definition */
static PyMethodDef FMPFitMethods[] = {
    {"fmpfit", py_fmpfit, METH_VARARGS,
     "Levenberg-Marquardt least-squares curve fitting\n\n"
     "Performs constrained nonlinear least squares fitting using MPFIT."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fmpfit_module = {
    PyModuleDef_HEAD_INIT,
    "fmpfit_ext",
    "MPFIT curve fitting C extension (Levenberg-Marquardt with constraints)",
    -1,
    FMPFitMethods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fmpfit_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fmpfit_module);
}
