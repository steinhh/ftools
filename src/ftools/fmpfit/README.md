# fmpfit - Levenberg-Marquardt Least-Squares Fitting

Python wrapper for MPFIT (cmpfit, MINPACK-1 Least Squares Fitting Library in C) with parameter constraints.

## Features

- Levenberg-Marquardt least-squares optimization
- Parameter bounds constraints
- Gaussian model with analytical Jacobian (fast, accurate)
- float32 and float64 precision variants
- Single-spectrum and block (multi-spectrum) fitting
- GIL-free C implementation for parallel execution
- Clean Python API with `MPFitResult` objects

## Usage

### Single-Spectrum Fitting

```python
import numpy as np
from ftools.fmpfit import fmpfit_pywrap  # auto dtype dispatch

# Generate data
x = np.linspace(-5, 5, 100)
y = 2.5 * np.exp(-0.5 * ((x - 1.0) / 0.8)**2) + np.random.randn(100) * 0.1
error = np.ones_like(y) * 0.1

# Define parameter info
parinfo = [
    {'value': 1.0, 'limits': [0.0, 10.0]},   # amplitude
    {'value': 0.0, 'limits': [-5.0, 5.0]},   # mean
    {'value': 1.0, 'limits': [0.1, 5.0]}     # sigma
]

# Prepare data
functkw = {'x': x, 'y': y, 'error': error}

# Run fit (auto-selects f64 based on input dtype)
result = fmpfit_pywrap(
    deviate_type=0,  # 0 = Gaussian model
    parinfo=parinfo,
    functkw=functkw
)

# Access results
print(f"Status: {result.status}")
print(f"Best-fit parameters: {result.best_params}")
print(f"Chi-square: {result.bestnorm}")
print(f"Iterations: {result.niter}")
```

### Block Fitting (Multiple Spectra)

```python
from ftools.fmpfit import fmpfit_block_pywrap
import numpy as np

n_spectra, mpoints, npar = 100, 200, 3

# 2D arrays: (n_spectra, mpoints)
x = np.tile(np.linspace(-5, 5, mpoints), (n_spectra, 1))
y = ...  # your data, shape (n_spectra, mpoints)
error = np.ones((n_spectra, mpoints)) * 0.1

# Initial params: (n_spectra, npar)
p0 = np.tile([1.0, 0.0, 1.0], (n_spectra, 1))

# Bounds: (n_spectra, npar, 2)
bounds = np.array([[[0, 10], [-5, 5], [0.1, 5]]] * n_spectra, dtype=np.float64)

# Fit all spectra (GIL released during C computation)
results = fmpfit_block_pywrap(x, y, error, p0, bounds)

# Results dict with arrays of shape (n_spectra, ...)
print(results['best_params'].shape)  # (n_spectra, npar)
print(results['status'])             # (n_spectra,)
```

## API

### fmpfit_pywrap (auto dtype dispatch)

```python
result = fmpfit_pywrap(deviate_type, parinfo=None, functkw=None,
                       dtype=None, xtol=1e-6, ftol=1e-6, gtol=1e-6,
                       maxiter=2000, quiet=1)
```

### fmpfit_f64_pywrap / fmpfit_f32_pywrap

```python
result = fmpfit_f64_pywrap(deviate_type, parinfo=None, functkw=None, 
                           xtol=1e-6, ftol=1e-6, gtol=1e-6, 
                           maxiter=2000, quiet=1)
```

**Parameters:**

- `deviate_type` (int): Model type (0 = Gaussian with analytical Jacobian)
- `parinfo` (list of dict): Parameter info, each dict contains:
  - `'value'`: Initial parameter value
  - `'limits'`: `[lower, upper]` bounds
- `functkw` (dict): Function keywords:
  - `'x'`: Independent variable (ndarray)
  - `'y'`: Dependent variable (ndarray)
  - `'error'`: Measurement uncertainties (ndarray)
- `xtol`, `ftol`, `gtol` (float): Convergence tolerances
- `maxiter` (int): Maximum iterations
- `quiet` (int): 1=quiet, 0=verbose

**Returns:** `MPFitResult` object with attributes:

- `best_params`: Best-fit parameter values
- `bestnorm`: Final chi-square
- `orignorm`: Initial chi-square
- `niter`: Number of iterations
- `nfev`: Number of function evaluations
- `status`: Status code (positive = success)
- `resid`: Final residuals
- `xerror`: Parameter uncertainties (1-sigma)
- `covar`: Covariance matrix
- `c_time`: Time spent in C extension

### Block Functions

```python
results = fmpfit_block_pywrap(x, y, error, p0, bounds, deviate_type=0,
                               dtype=None, xtol=1e-6, ftol=1e-6, gtol=1e-6,
                               maxiter=2000, quiet=1)
```

Also: `fmpfit_f64_block_pywrap`, `fmpfit_f32_block_pywrap`

**Parameters:** 2D arrays with shape `(n_spectra, mpoints)` for data and `(n_spectra, npar)` for parameters.

**Returns:** Dict with arrays: `best_params`, `bestnorm`, `status`, `niter`, `nfev`, `xerror`, `covar`.

## Multithreading Performance

All fmpfit functions release the Python GIL during C computation, enabling true parallel execution:

- **4.14× speedup** with 6 threads (69% efficiency)
- **3.37× speedup** with 4 threads (84% efficiency)
- **1.85× speedup** with 2 threads (93% efficiency)

Threading is beneficial when individual fits take >0.5ms.

### Parallel Batch Fitting

```python
from concurrent.futures import ThreadPoolExecutor
from ftools.fmpfit import fmpfit_f64_pywrap

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_fit, dataset_list))
```

## Testing

```bash
pytest tests/test_fmpfit.py -v               # Unit tests
pytest tests/test_fmpfit_concurrent.py -v    # Thread safety
pytest tests/test_fmpfit_vs_curvefit.py -v   # SciPy comparison
pytest tests/test_fmpfit_block_vs_curvefit.py -v  # Block vs SciPy
```

## Files

- `__init__.py` - Python wrapper and API
- `fmpfit_f64_ext.c` / `fmpfit_f32_ext.c` - Single-spectrum C extensions
- `fmpfit_f64_block_ext.c` / `fmpfit_f32_block_ext.c` - Block C extensions
- `gaussian_deviate_f64.c` / `gaussian_deviate_f32.c` - Gaussian model + Jacobian
- `cmpfit-1.5/` - Unified MPFIT library source (precision via compile-time define)
- `example_fmpfit_*.py` - Usage examples
- `benchmark_*.py` - Performance benchmarks
