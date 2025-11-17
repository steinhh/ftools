# fmpfit - Levenberg-Marquardt Least-Squares Fitting

Python wrapper for MPFIT (MINPACK-1 Least Squares Fitting Library in C) with parameter constraints.

## Status

**This is a work in progress.** The Python wrapper and C extension infrastructure are complete, but the core MPFIT fitting algorithm needs to be implemented in `fmpfit_core()`.

Current status: `-999` (stub implementation)

## Features

- Levenberg-Marquardt least-squares optimization
- Parameter bounds constraints
- Support for Gaussian model (extensible to other models)
- Efficient C implementation
- Clean Python API

## Usage

```python
import numpy as np
from ftools import fmpfit

# Generate data
x = np.linspace(-5, 5, 100)
y = 2.5 * np.exp(-0.5 * ((x - 1.0) / 0.8)**2)
error = np.ones_like(y) * 0.1

# Define parameter info
parinfo = [
    {'value': 1.0, 'limits': [0.0, 10.0]},   # amplitude
    {'value': 0.0, 'limits': [-5.0, 5.0]},   # mean
    {'value': 1.0, 'limits': [0.1, 5.0]}     # sigma
]

# Prepare data
functkw = {'x': x, 'y': y, 'error': error}

# Run fit
result = fmpfit.mpfit(
    deviate_type=0,  # 0 = Gaussian model
    parinfo=parinfo,
    functkw=functkw,
    xtol=1.0e-6,
    ftol=1.0e-6,
    gtol=1.0e-6,
    maxiter=2000,
    quiet=1
)

# Access results
print(f"Status: {result.status}")
print(f"Best-fit parameters: {result.best_params}")
print(f"Chi-square: {result.bestnorm}")
print(f"Iterations: {result.niter}")
```

## API

### mpfit()

```python
result = fmpfit.mpfit(deviate_type, parinfo=None, functkw=None, 
                      xtol=1.0e-6, ftol=1.0e-6, gtol=1.0e-6, 
                      maxiter=2000, quiet=1)
```

**Parameters:**

- `deviate_type` (int): Model type (0 = Gaussian)
- `parinfo` (list of dict): Parameter info, each dict contains:
  - `'value'`: Initial parameter value
  - `'limits'`: `[lower, upper]` bounds
- `functkw` (dict): Function keywords:
  - `'x'`: Independent variable (ndarray)
  - `'y'`: Dependent variable (ndarray)
  - `'error'`: Measurement uncertainties (ndarray)
- `xtol` (float): Relative tolerance in parameters
- `ftol` (float): Relative tolerance in chi-square
- `gtol` (float): Orthogonality tolerance
- `maxiter` (int): Maximum iterations
- `quiet` (int): 1=quiet, 0=verbose

**Returns:** `MPFitResult` object with attributes:

- `best_params`: Best-fit parameter values
- `bestnorm`: Final chi-square
- `orignorm`: Initial chi-square
- `niter`: Number of iterations
- `nfev`: Number of function evaluations
- `status`: Status code (0 = success)
- `resid`: Final residuals
- `xerror`: Parameter uncertainties (1-sigma)
- `covar`: Covariance matrix

## Implementation Notes

The C extension (`fmpfit_ext.c`) provides:

1. Python wrapper with input validation
2. Array conversion and memory management
3. Stub for `fmpfit_core()` - the actual fitting algorithm

To complete the implementation, `fmpfit_core()` needs to:

1. Set up MPFIT parameter structures
2. Define model functions (Gaussian, etc.)
3. Call MPFIT library routines
4. Extract results and populate output arrays

See `callsign.txt` for interface details and `cmpfit-1.5/` for the MPFIT library.

## Testing

```bash
# Run unit tests
pytest tests/test_fmpfit.py -v

# Run example
python src/ftools/fmpfit/example_fmpfit.py
```

## Files

- `__init__.py` - Python wrapper and API
- `fmpfit_ext.c` - C extension with stub
- `example_fmpfit.py` - Usage example
- `test_fmpfit.py` - Unit tests
- `callsign.txt` - Implementation notes
- `cmpfit-1.5/` - MPFIT library source
