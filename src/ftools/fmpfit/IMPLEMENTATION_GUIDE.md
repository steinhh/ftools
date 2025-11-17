## Implementing fmpfit_core() - Quick Guide

The stub in `fmpfit_ext.c` needs to be replaced with actual MPFIT integration.

### Current Stub Location

File: `src/ftools/fmpfit/fmpfit_ext.c`, lines 22-77

### Implementation Steps

1. **Include MPFIT headers**

   ```c
   #include "cmpfit-1.5/mpfit.h"
   ```

2. **Define model function**

   ```c
   /* User function for computing residuals */
   int gaussian_func(int m, int npar, double *p, double *deviates,
                     double **derivs, void *private_data)
   {
       struct fit_data *data = (struct fit_data *)private_data;
       
       for (int i = 0; i < m; i++) {
           double A = p[0];
           double mu = p[1]; 
           double sigma = p[2];
           double model = A * exp(-0.5 * pow((data->x[i] - mu) / sigma, 2));
           deviates[i] = (data->y[i] - model) / data->error[i];
       }
       return 0;
   }
   ```

3. **Define private data structure**

   ```c
   struct fit_data {
       const double *x;
       const double *y;
       const double *error;
   };
   ```

4. **Set up MPFIT parameters**

   ```c
   mp_par pars[npar];
   for (int i = 0; i < npar; i++) {
       pars[i].fixed = 0;
       pars[i].limited[0] = 1;  /* lower limit */
       pars[i].limited[1] = 1;  /* upper limit */
       pars[i].limits[0] = bounds[i * 2 + 0];
       pars[i].limits[1] = bounds[i * 2 + 1];
   }
   ```

5. **Set up MPFIT config**

   ```c
   mp_config config;
   memset(&config, 0, sizeof(config));
   config.ftol = ftol;
   config.xtol = xtol;
   config.gtol = gtol;
   config.maxiter = maxiter;
   config.nofinitecheck = 1;
   ```

6. **Prepare result structure**

   ```c
   mp_result result_info;
   memset(&result_info, 0, sizeof(result_info));
   result_info.resid = resid;
   result_info.xerror = xerror;
   result_info.covar = covar;
   ```

7. **Call MPFIT**

   ```c
   struct fit_data data = {x, y, error};
   
   mp_func func;
   func = (deviate_type == 0) ? gaussian_func : NULL;
   
   int mpfit_status = mpfit(func, mpoints, npar, best_params, 
                            pars, &config, &data, &result_info);
   ```

8. **Extract results**

   ```c
   *bestnorm = result_info.bestnorm;
   *orignorm = result_info.orignorm;
   *niter = result_info.niter;
   *nfev = result_info.nfev;
   *status = mpfit_status;
   ```

### Compilation

Update setup.py to link MPFIT library:

```python
Extension(
    "ftools.fmpfit.fmpfit_ext",
    sources=[
        os.path.join("src", "ftools", "fmpfit", "fmpfit_ext.c"),
        os.path.join("src", "ftools", "fmpfit", "cmpfit-1.5", "mpfit.c"),
    ],
    include_dirs=include_dirs + [
        os.path.join("src", "ftools", "fmpfit", "cmpfit-1.5")
    ],
    extra_compile_args=["-O3"],
)
```

### Testing

After implementation:

1. Remove status=-999 check from tests
2. Verify actual fitting works with known data
3. Check parameter uncertainties are reasonable
4. Validate covariance matrix is positive definite

### References

- MPFIT documentation: `cmpfit-1.5/README`
- Example usage: `fit_gaussian_mpfit.py`
- Interface notes: `callsign.txt`
