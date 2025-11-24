/*
 * gaussian_deviate.c - Gaussian model deviate function for MPFIT
 *
 * Computes weighted residuals and analytical derivatives for Gaussian model
 */

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

  /* Compute weighted deviates and cache exp() values for derivatives */
  /* Allocate temporary storage for exp terms if derivatives are needed */
  double *expterms = NULL;
  double *z_values = NULL;

  if (derivs)
  {
    expterms = (double *)malloc(m * sizeof(double));
    z_values = (double *)malloc(m * sizeof(double));
  }

  for (i = 0; i < m; i++)
  {
    double x_i = private->x[i];
    double y_i = private->y[i];
    double err_i = private->error[i];
    double z = (x_i - mean) / sigma;
    double expterm = exp(-0.5 * z * z);
    double model = amp * expterm;

    deviates[i] = (y_i - model) / err_i;

    /* Cache values for derivative calculation */
    if (derivs)
    {
      expterms[i] = expterm;
      z_values[i] = z;
    }
  }

  /* If derivs is non-zero, compute analytical derivatives using cached values */
  /* Since deviate = (y - model)/error, d(deviate)/d(p) = -d(model)/d(p)/error */
  if (derivs)
  {
    for (i = 0; i < m; i++)
    {
      double err_i = private->error[i];
      double expterm = expterms[i];
      double z = z_values[i];

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

    /* Free temporary storage */
    free(expterms);
    free(z_values);
  }

  return 0;
}
