/*
 * xerror_scipy.c - Compute scipy-style parameter errors from full Hessian inverse
 *
 * This file is included by all fmpfit extensions (f32, f64, block variants).
 * Use preprocessor macros to control the data type:
 *
 *   #define XERROR_SCIPY_FLOAT 1   // for float32
 *   // or leave undefined for float64 (default)
 *
 * Before including, define XERROR_SCIPY_FLOAT for float32 version.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Type-specific definitions based on XERROR_SCIPY_FLOAT macro */
#ifdef XERROR_SCIPY_FLOAT
#define XS_REAL float
#define XS_EXP expf
#define XS_SQRT sqrtf
#define XS_FABS fabsf
#define XS_EPSILON 1e-7f
#define XS_ZERO 0.0f
#define XS_ONE 1.0f
#define XS_TWO 2.0f
#define XS_FUNC_NAME compute_xerror_scipy_f32
#else
#define XS_REAL double
#define XS_EXP exp
#define XS_SQRT sqrt
#define XS_FABS fabs
#define XS_EPSILON 1e-15
#define XS_ZERO 0.0
#define XS_ONE 1.0
#define XS_TWO 2.0
#define XS_FUNC_NAME compute_xerror_scipy_f64
#endif

/*
 * Compute scipy-style parameter errors using full Hessian inverse.
 *
 * This matches scipy.optimize.curve_fit error computation, which uses
 * the full Hessian matrix regardless of whether parameters hit bounds.
 * MPFIT's covar uses a reduced Hessian (excluding pegged parameters).
 *
 * For a Gaussian model: g(x) = I * exp(-0.5 * ((x - v) / w)^2)
 *   Parameters: I (intensity), v (velocity/center), w (width/sigma)
 *
 * Parameters:
 *   x           - x coordinates (mpoints)
 *   error       - measurement uncertainties (mpoints)
 *   best_params - fitted parameter values [I, v, w]
 *   mpoints     - number of data points
 *   npar        - number of parameters (must be 3 for Gaussian)
 *   scale_factor - sqrt(chi2/dof) scaling factor
 *   xerror_scipy - output: scipy-style errors (npar)

 * Returns: 0 on success, -1 on allocation failure
 */
static int XS_FUNC_NAME(
    const XS_REAL *x,
    const XS_REAL *error,
    const XS_REAL *best_params,
    int mpoints,
    int npar,
    XS_REAL scale_factor,
    XS_REAL *xerror_scipy)
{
  int i, j, k, col, row, pivot;
  XS_REAL *jac = NULL;
  XS_REAL *hess = NULL;
  XS_REAL *hess_inv = NULL;
  XS_REAL *aug = NULL;
  XS_REAL I_val, v_val, w_val;
  XS_REAL xk, ek, xmv, g, w2, w3;
  XS_REAL sum, max_val, pivot_val, factor, tmp, var;
  int ret = 0;

  /* Allocate workspace */
  jac = (XS_REAL *)malloc(mpoints * npar * sizeof(XS_REAL));
  hess = (XS_REAL *)malloc(npar * npar * sizeof(XS_REAL));
  hess_inv = (XS_REAL *)malloc(npar * npar * sizeof(XS_REAL));

  if (!jac || !hess || !hess_inv)
  {
    /* Allocation failed */
    ret = -1;
    goto cleanup;
  }

  /* Extract parameters */
  I_val = best_params[0];
  v_val = best_params[1];
  w_val = best_params[2];
  w2 = w_val * w_val;
  w3 = w2 * w_val;

  /* Compute Jacobian at best_params
   * For Gaussian: g = I * exp(-0.5 * ((x - v) / w)^2)
   * Weighted Jacobian: J_ij = (1/sigma_i) * dg/dp_j
   */
  for (k = 0; k < mpoints; k++)
  {
    xk = x[k];
    ek = error[k];
    xmv = xk - v_val;
    g = I_val * XS_EXP(-(xmv * xmv) / (XS_TWO * w2));

    jac[k * npar + 0] = (g / I_val) / ek;          /* dg/dI = g/I */
    jac[k * npar + 1] = (g * xmv / w2) / ek;       /* dg/dv = g*(x-v)/w^2 */
    jac[k * npar + 2] = (g * xmv * xmv / w3) / ek; /* dg/dw = g*(x-v)^2/w^3 */
  }

  /* Compute Hessian H = J^T J (normal equations matrix) */
  for (i = 0; i < npar; i++)
  {
    for (j = 0; j < npar; j++)
    {
      sum = XS_ZERO;
      for (k = 0; k < mpoints; k++)
      {
        sum += jac[k * npar + i] * jac[k * npar + j];
      }
      hess[i * npar + j] = sum;
    }
  }

  /* Initialize hess_inv as identity matrix */
  for (i = 0; i < npar; i++)
  {
    for (j = 0; j < npar; j++)
    {
      hess_inv[i * npar + j] = (i == j) ? XS_ONE : XS_ZERO;
    }
  }

  /* Invert Hessian using Gauss-Jordan elimination with partial pivoting */
  aug = (XS_REAL *)malloc(npar * npar * sizeof(XS_REAL));
  if (!aug)
  {
    ret = -1;
    goto cleanup;
  }

  memcpy(aug, hess, npar * npar * sizeof(XS_REAL));

  for (col = 0; col < npar; col++)
  {
    /* Find pivot (row with maximum absolute value in column) */
    pivot = col;
    max_val = XS_FABS(aug[col * npar + col]);
    for (row = col + 1; row < npar; row++)
    {
      if (XS_FABS(aug[row * npar + col]) > max_val)
      {
        max_val = XS_FABS(aug[row * npar + col]);
        pivot = row;
      }
    }

    /* Swap rows if needed */
    if (pivot != col)
    {
      for (k = 0; k < npar; k++)
      {
        tmp = aug[col * npar + k];
        aug[col * npar + k] = aug[pivot * npar + k];
        aug[pivot * npar + k] = tmp;

        tmp = hess_inv[col * npar + k];
        hess_inv[col * npar + k] = hess_inv[pivot * npar + k];
        hess_inv[pivot * npar + k] = tmp;
      }
    }

    /* Scale pivot row */
    pivot_val = aug[col * npar + col];
    if (XS_FABS(pivot_val) > XS_EPSILON)
    {
      for (k = 0; k < npar; k++)
      {
        aug[col * npar + k] /= pivot_val;
        hess_inv[col * npar + k] /= pivot_val;
      }

      /* Eliminate column in other rows */
      for (row = 0; row < npar; row++)
      {
        if (row != col)
        {
          factor = aug[row * npar + col];
          for (k = 0; k < npar; k++)
          {
            aug[row * npar + k] -= factor * aug[col * npar + k];
            hess_inv[row * npar + k] -= factor * hess_inv[col * npar + k];
          }
        }
      }
    }
  }

  free(aug);
  aug = NULL;

  /* Extract errors from diagonal of hess_inv, scaled by sqrt(chi2/dof) */
  for (i = 0; i < npar; i++)
  {
    var = hess_inv[i * npar + i];
    xerror_scipy[i] = (var > XS_ZERO) ? XS_SQRT(var) * scale_factor : XS_ZERO;
  }

cleanup:
  free(jac);
  free(hess);
  free(hess_inv);

  return ret;
}

/* Clean up macros to allow re-inclusion with different settings */
#undef XS_REAL
#undef XS_EXP
#undef XS_SQRT
#undef XS_FABS
#undef XS_EPSILON
#undef XS_ZERO
#undef XS_ONE
#undef XS_TWO
#undef XS_FUNC_NAME
