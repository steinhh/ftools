/*
 * gaussian_deviate_f32.c - Gaussian model deviate function for MPFIT (float32 version)
 *
 * Computes weighted residuals and analytical derivatives for Gaussian model
 * using a standard C implementation with cached exponential values.
 *
 * Includes analytical derivatives for amplitude, mean, and sigma parameters.
 */

/* Private data structure for passing x, y, error to user function */
struct gaussian_private_data_f32
{
  const float *x;     /* Independent variable */
  const float *y;     /* Measured values */
  const float *error; /* Measurement uncertainties */
};

/*
 * User function: compute Gaussian deviates (y - f(x))/error
 * For Gaussian model: f(x) = p[0] * exp(-0.5 * ((x - p[1]) / p[2])^2)
 * Parameters: p[0] = amplitude, p[1] = mean, p[2] = sigma
 */
int myfunct_gaussian_deviates_with_derivatives_f32(int m, int n, float *p, float *deviates,
                                                   float **derivs, void *private_data)
{
  struct gaussian_private_data_f32 *private = (struct gaussian_private_data_f32 *)private_data;
  int i;
  float amp, mean, sigma;
  (void)n; /* Suppress unused parameter warning */

  /* Extract parameters */
  amp = p[0];
  mean = p[1];
  sigma = p[2];

  /* Compute weighted deviates and cache exp() values for derivatives */
  /* Allocate temporary storage for exp terms if derivatives are needed */
  float *expterms = NULL;
  float *z_values = NULL;

  if (derivs)
  {
    expterms = (float *)malloc(m * sizeof(float));
    z_values = (float *)malloc(m * sizeof(float));
  }

  for (i = 0; i < m; i++)
  {
    float x_i = private->x[i];
    float y_i = private->y[i];
    float err_i = private->error[i];
    float z = (x_i - mean) / sigma;
    float expterm = expf(-0.5f * z * z);
    float model = amp * expterm;

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
      float err_i = private->error[i];
      float expterm = expterms[i];
      float z = z_values[i];

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
