## fmpfit Extension - Implementation Summary

### Created Files

1. **`src/ftools/fmpfit/__init__.py`** (205 lines)
   - Python wrapper for MPFIT curve fitting
   - `mpfit()` function with signature matching specifications
   - `MPFitResult` class for returning fit results
   - Full input validation and error handling
   - Converts Python data structures to C-compatible arrays

2. **`src/ftools/fmpfit/fmpfit_ext.c`** (313 lines)
   - C extension module
   - `py_fmpfit()` - Python/C interface function
   - `fmpfit_core()` - Stub for core fitting algorithm (to be implemented)
   - Memory management and array conversion
   - Returns dictionary with all required result fields

3. **`src/ftools/fmpfit/example_fmpfit.py`** (96 lines)
   - Demonstrates usage with synthetic Gaussian data
   - Shows complete workflow from data generation to results
   - Executable example script

4. **`tests/test_fmpfit.py`** (163 lines)
   - 9 comprehensive unit tests
   - Tests import, basic calls, result shapes, error handling
   - All tests pass ?

5. **`src/ftools/fmpfit/README.md`** (130 lines)
   - Complete documentation
   - API reference
   - Usage examples
   - Implementation notes

**Function signature:**

```python
result = fmpfit.mpfit(deviate_type, parinfo=parinfo, functkw=functkw, 
                      xtol=1.0e-6, ftol=1.0e-6, gtol=1.0e-6,
                      maxiter=2000, quiet=1)
```

**Input parameters:** ? All implemented as specified

- `deviate_type`: int32 (model type)
- `functkw`: dict with x, y, error arrays
- `parinfo`: list of dicts with value and limits
- `xtol`, `ftol`, `gtol`: float64
- `maxiter`, `quiet`: int32

**Return object:** ? All attributes implemented

- `best_params`, `bestnorm`, `orignorm`
- `niter`, `nfev`, `status`
- `npar`, `nfree`, `npegged`, `nfunc`
- `resid`, `xerror`, `covar`

### C Function Interface (Stub Ready)

The `fmpfit_core()` function has the exact signature specified:

```c
void fmpfit_core(
    const double *x, const double *y, const double *error,
    const double *p0, const double *bounds,
    int mpoints, int npar, int deviate_type,
    double xtol, double ftol, double gtol,
    int maxiter, int quiet,
    double *best_params, double *bestnorm, double *orignorm,
    int *niter, int *nfev, int *status,
    double *resid, double *xerror, double *covar)
```

### Usage Example

```python
from ftools import fmpfit

parinfo = [
    {'value': 1.0, 'limits': [0.0, 10.0]},
    {'value': 0.0, 'limits': [-5.0, 5.0]},
    {'value': 1.0, 'limits': [0.1, 5.0]}
]

functkw = {'x': x_data, 'y': y_data, 'error': errors}

result = fmpfit.mpfit(0, parinfo=parinfo, functkw=functkw)

print(result.best_params)  # Access fit results
```
