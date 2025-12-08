/*
 * fmpfit_f64_ext.c - Python C extension for MPFIT wrapper (float64)
 *
 * Wraps the MPFIT library for nonlinear least-squares fitting using double precision
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
 * Core MPFIT function - calls MPFIT library (float64 version)
 */
static void fmpfit_f64_c_wrap(
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
  struct gaussian_private_data private_data;

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

  /* Setup private data for user function */
  private_data.x = x;
  private_data.y = y;
  private_data.error = error;

  /* Call MPFIT */
  *status = mpfit(myfunct_gaussian_deviates_with_derivatives,
                  mpoints, npar, best_params, pars, &config,
                  (void *)&private_data, &result);

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
 * Python wrapper: fmpfit_f64(x, y, error, p0, bounds, deviate_type,
 *                            xtol, ftol, gtol, maxiter, quiet)
 *
 * Note: mpoints and npar are inferred from array shapes
 */
static PyObject *py_fmpfit_f64(PyObject *self, PyObject *args)
{
  PyArrayObject *x_array = NULL, *y_array = NULL, *error_array = NULL;
  PyArrayObject *p0_array = NULL, *bounds_array = NULL;
  int deviate_type, maxiter, quiet;
  double xtol, ftol, gtol;

  (void)self;

  /* Parse arguments - no mpoints/npar needed */
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!idddii",
                        &PyArray_Type, &x_array,
                        &PyArray_Type, &y_array,
                        &PyArray_Type, &error_array,
                        &PyArray_Type, &p0_array,
                        &PyArray_Type, &bounds_array,
                        &deviate_type,
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

  /* Infer dimensions from array shapes */
  int mpoints = (int)PyArray_DIM(x_contig, 0);
  int npar = (int)PyArray_DIM(p0_contig, 0);

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
  double *xerror_scaled = (double *)malloc(npar * sizeof(double));
  double *xerror_scipy = (double *)malloc(npar * sizeof(double));

  if (!best_params || !resid || !xerror || !covar || !xerror_scaled || !xerror_scipy)
  {
    free(best_params);
    free(resid);
    free(xerror);
    free(covar);
    free(xerror_scaled);
    free(xerror_scipy);
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
  /* Release GIL while running the compute-heavy C routine so other Python
     threads may run concurrently. The MPFIT library and callbacks are pure
     C and do not call back into Python, so this is safe. */
  {
    Py_BEGIN_ALLOW_THREADS
        fmpfit_f64_c_wrap(x, y, error, p0, bounds,
                          mpoints, npar, deviate_type,
                          xtol, ftol, gtol, maxiter, quiet,
                          best_params, &bestnorm, &orignorm,
                          &niter, &nfev, &status,
                          resid, xerror, covar);

    /* Compute xerror_scaled = xerror * sqrt(chi2 / dof) to match curve_fit default */
    int dof = mpoints - npar;
    double scale_factor = (dof > 0) ? sqrt(bestnorm / dof) : 1.0;
    for (int i = 0; i < npar; i++)
    {
      xerror_scaled[i] = xerror[i] * scale_factor;
    }

    /* Compute xerror_scipy: errors from full Hessian inverse (scipy-style)
     * MPFIT's covar is computed from the reduced Jacobian (excluding bounded params).
     * scipy computes covar from the full Jacobian. To match scipy, we need to
     * recompute the full Hessian H = J^T W J and invert it.
     * Here we use a simpler approach: compute Jacobian, form Hessian, invert.
     */
    {
      /* Allocate workspace for Jacobian and Hessian */
      double *jac = (double *)malloc(mpoints * npar * sizeof(double));
      double *hess = (double *)malloc(npar * npar * sizeof(double));
      double *hess_inv = (double *)malloc(npar * npar * sizeof(double));

      if (jac && hess && hess_inv)
      {
        /* Compute Jacobian at best_params */
        double I_val = best_params[0];
        double v_val = best_params[1];
        double w_val = best_params[2];

        for (int k = 0; k < mpoints; k++)
        {
          double xk = x[k];
          double ek = error[k];
          double xmv = xk - v_val;
          double g = I_val * exp(-(xmv * xmv) / (2 * w_val * w_val));

          /* Jacobian: d(residual)/d(param) = (1/error) * d(model)/d(param) */
          /* For weighted least squares: J_ij = (1/sigma_i) * df/dp_j */
          jac[k * npar + 0] = (g / I_val) / ek;                               /* dg/dI */
          jac[k * npar + 1] = (g * xmv / (w_val * w_val)) / ek;               /* dg/dv */
          jac[k * npar + 2] = (g * xmv * xmv / (w_val * w_val * w_val)) / ek; /* dg/dw */
        }

        /* Compute Hessian H = J^T J (this is the normal equations matrix) */
        for (int i = 0; i < npar; i++)
        {
          for (int j = 0; j < npar; j++)
          {
            double sum = 0.0;
            for (int k = 0; k < mpoints; k++)
            {
              sum += jac[k * npar + i] * jac[k * npar + j];
            }
            hess[i * npar + j] = sum;
          }
        }

        /* Invert Hessian using Gaussian elimination with partial pivoting */
        /* First copy hess to hess_inv and make it identity */
        for (int i = 0; i < npar; i++)
        {
          for (int j = 0; j < npar; j++)
          {
            hess_inv[i * npar + j] = (i == j) ? 1.0 : 0.0;
          }
        }

        /* Gaussian elimination */
        double *aug = (double *)malloc(npar * npar * sizeof(double));
        if (aug)
        {
          memcpy(aug, hess, npar * npar * sizeof(double));

          for (int col = 0; col < npar; col++)
          {
            /* Find pivot */
            int pivot = col;
            double max_val = fabs(aug[col * npar + col]);
            for (int row = col + 1; row < npar; row++)
            {
              if (fabs(aug[row * npar + col]) > max_val)
              {
                max_val = fabs(aug[row * npar + col]);
                pivot = row;
              }
            }

            /* Swap rows if needed */
            if (pivot != col)
            {
              for (int k = 0; k < npar; k++)
              {
                double tmp = aug[col * npar + k];
                aug[col * npar + k] = aug[pivot * npar + k];
                aug[pivot * npar + k] = tmp;

                tmp = hess_inv[col * npar + k];
                hess_inv[col * npar + k] = hess_inv[pivot * npar + k];
                hess_inv[pivot * npar + k] = tmp;
              }
            }

            /* Scale pivot row */
            double pivot_val = aug[col * npar + col];
            if (fabs(pivot_val) > 1e-15)
            {
              for (int k = 0; k < npar; k++)
              {
                aug[col * npar + k] /= pivot_val;
                hess_inv[col * npar + k] /= pivot_val;
              }

              /* Eliminate column */
              for (int row = 0; row < npar; row++)
              {
                if (row != col)
                {
                  double factor = aug[row * npar + col];
                  for (int k = 0; k < npar; k++)
                  {
                    aug[row * npar + k] -= factor * aug[col * npar + k];
                    hess_inv[row * npar + k] -= factor * hess_inv[col * npar + k];
                  }
                }
              }
            }
          }
          free(aug);
        }

        /* Extract errors from diagonal of hess_inv, scaled by sqrt(chi2/dof) */
        for (int i = 0; i < npar; i++)
        {
          double var = hess_inv[i * npar + i];
          if (var > 0)
          {
            xerror_scipy[i] = sqrt(var) * scale_factor;
          }
          else
          {
            xerror_scipy[i] = 0.0;
          }
        }
      }
      else
      {
        /* Fallback if allocation failed */
        for (int i = 0; i < npar; i++)
        {
          xerror_scipy[i] = xerror_scaled[i];
        }
      }

      free(jac);
      free(hess);
      free(hess_inv);
    }
    Py_END_ALLOW_THREADS
  }

  /* Create output arrays */
  npy_intp dims_params[1] = {npar};
  npy_intp dims_resid[1] = {mpoints};
  npy_intp dims_covar[2] = {npar, npar};

  PyArrayObject *best_params_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_DOUBLE);
  PyArrayObject *resid_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_resid, NPY_DOUBLE);
  PyArrayObject *xerror_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_DOUBLE);
  PyArrayObject *covar_array = (PyArrayObject *)PyArray_SimpleNew(2, dims_covar, NPY_DOUBLE);
  PyArrayObject *xerror_scaled_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_DOUBLE);
  PyArrayObject *xerror_scipy_array = (PyArrayObject *)PyArray_SimpleNew(1, dims_params, NPY_DOUBLE);

  if (!best_params_array || !resid_array || !xerror_array || !covar_array ||
      !xerror_scaled_array || !xerror_scipy_array)
  {
    Py_XDECREF(best_params_array);
    Py_XDECREF(resid_array);
    Py_XDECREF(xerror_array);
    Py_XDECREF(covar_array);
    Py_XDECREF(xerror_scaled_array);
    Py_XDECREF(xerror_scipy_array);
    free(best_params);
    free(resid);
    free(xerror);
    free(covar);
    free(xerror_scaled);
    free(xerror_scipy);
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
  memcpy(PyArray_DATA(xerror_scaled_array), xerror_scaled, npar * sizeof(double));
  memcpy(PyArray_DATA(xerror_scipy_array), xerror_scipy, npar * sizeof(double));

  /* Free temporary buffers */
  free(best_params);
  free(resid);
  free(xerror);
  free(covar);
  free(xerror_scaled);
  free(xerror_scipy);

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
  PyDict_SetItemString(result, "xerror_scaled", (PyObject *)xerror_scaled_array);
  PyDict_SetItemString(result, "xerror_scipy", (PyObject *)xerror_scipy_array);

  /* Decrement reference counts (dict holds references) */
  Py_DECREF(best_params_array);
  Py_DECREF(resid_array);
  Py_DECREF(xerror_array);
  Py_DECREF(covar_array);
  Py_DECREF(xerror_scaled_array);
  Py_DECREF(xerror_scipy_array);

  return result;
}

/* Method definition */
static PyMethodDef FMPFitMethods[] = {
    {"fmpfit_f64", py_fmpfit_f64, METH_VARARGS,
     "Levenberg-Marquardt least-squares curve fitting (float64)\n\n"
     "Performs constrained nonlinear least squares fitting using MPFIT with double precision.\n\n"
     "Parameters: x, y, error, p0, bounds, deviate_type, xtol, ftol, gtol, maxiter, quiet\n"
     "Note: mpoints and npar are inferred from array shapes."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fmpfit_f64_module = {
    PyModuleDef_HEAD_INIT,
    "fmpfit_f64_ext",
    "MPFIT curve fitting C extension (Levenberg-Marquardt with constraints) - float64",
    -1,
    FMPFitMethods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fmpfit_f64_ext(void)
{
  import_array();

  if (PyErr_Occurred())
  {
    return NULL;
  }

  return PyModule_Create(&fmpfit_f64_module);
}
