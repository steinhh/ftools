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
#include <cmpfit-1.5/mpfit.h>

/*
 * Core MPFIT function (stub - to be implemented)
 *
 * This function performs the actual Levenberg-Marquardt fitting.
 * It will be implemented separately using the MPFIT library.
 */
static void fmpfit_core(
    const double *x, const double *y, const double *error,
    const double *p0, const double *bounds,
    int mpoints, int npar, int deviate_type,
    double xtol, double ftol, double gtol,
    int maxiter, int quiet,
    double *best_params, double *bestnorm, double *orignorm,
    int *niter, int *nfev, int *status,
    double *resid, double *xerror, double *covar)
{
  /*
   * STUB IMPLEMENTATION
   *
   * This is a placeholder that demonstrates the interface.
   * The actual implementation will use the MPFIT library to:
   * 1. Set up the model function based on deviate_type
   * 2. Configure MPFIT parameters (tolerances, limits, etc.)
   * 3. Run the Levenberg-Marquardt optimization
   * 4. Extract results and populate output arrays
   *
   * For now, just copy initial parameters and set dummy values
   */

  int i, j;

  /* Copy initial parameters to output */
  for (i = 0; i < npar; i++)
  {
    best_params[i] = p0[i];
  }

  /* Calculate initial residuals and chi-square */
  double chisq = 0.0;
  for (i = 0; i < mpoints; i++)
  {
    /* For Gaussian model: y = A * exp(-0.5 * ((x - mu) / sigma)^2) */
    double model_val = 0.0;
    if (deviate_type == 0 && npar >= 3)
    {
      /* Gaussian: p[0]=amplitude, p[1]=mean, p[2]=sigma */
      double A = p0[0];
      double mu = p0[1];
      double sigma = p0[2];
      model_val = A * exp(-0.5 * pow((x[i] - mu) / sigma, 2));
    }

    resid[i] = (y[i] - model_val) / error[i];
    chisq += resid[i] * resid[i];
  }

  *orignorm = chisq;
  *bestnorm = chisq; /* In real implementation, this would be the final chi-square */

  /* Set dummy output values */
  *niter = 0;
  *nfev = 0;
  *status = -999; /* -999 = stub not implemented */

  /* Set dummy uncertainties */
  for (i = 0; i < npar; i++)
  {
    xerror[i] = 0.0;
  }

  /* Set identity covariance matrix (dummy) */
  for (i = 0; i < npar; i++)
  {
    for (j = 0; j < npar; j++)
    {
      covar[i * npar + j] = (i == j) ? 1.0 : 0.0;
    }
  }
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
  fmpfit_core(x, y, error, p0, bounds,
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
