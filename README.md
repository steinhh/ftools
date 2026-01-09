# ftoolss

High-performance C extensions for image processing and curve fitting.

## Features

| Export | Description |
|--------|-------------|
| `fmedian` | Local median filter (2D/3D auto-dispatch) |
| `fmedian2d`, `fmedian3d` | Direct 2D/3D median |
| `fsigma` | Local ? filter (2D/3D auto-dispatch) |
| `fsigma2d`, `fsigma3d` | Direct 2D/3D sigma |
| `fgaussian_f32`, `fgaussian_f64` | Gaussian profile (Accelerate-optimized) |
| `fmpfit_pywrap` | Auto dtype dispatch (single spectrum) |
| `fmpfit_f64_pywrap`, `fmpfit_f32_pywrap` | Single spectrum fitting |
| `fmpfit_block_pywrap` | Auto dtype dispatch (block fitting) |
| `fmpfit_f64_block_pywrap`, `fmpfit_f32_block_pywrap` | Block fitting |
| `MPFitResult` | Named tuple for fit results |

**Common features:** NaN handling, edge handling, optional center exclusion (fmedian/fsigma).

## Installation

```bash
pip install .                        # Standard install
pip install -e .                     # Editable (development)
python setup.py build_ext --inplace  # Build extensions only
```

**Requirements:** Python 3.8+, NumPy ?1.20, C compiler. macOS Accelerate used when available.

## Quick Start

```python
import numpy as np
from ftoolss import fmedian, fsigma, fgaussian_f32, fgaussian_f64

# 2D/3D median and sigma filters
data = np.random.randn(100, 200)
median = fmedian(data, (3, 3), exclude_center=1)
sigma = fsigma(data, (3, 3), exclude_center=1)

# 3D works the same way
data_3d = np.random.randn(50, 100, 50)
median_3d = fmedian(data_3d, (3, 3, 3))

# Gaussian profiles
x = np.linspace(-10, 10, 1000, dtype=np.float32)
profile = fgaussian_f32(x, i0=1.0, mu=0.0, sigma=1.5)
```

### Curve Fitting (fmpfit)

Single-spectrum fitting with auto dtype dispatch:

```python
from ftoolss.fmpfit import fmpfit_pywrap
import numpy as np

x = np.linspace(-5, 5, 100)
y = 2.5 * np.exp(-0.5 * ((x - 1.0) / 0.8)**2) + np.random.randn(100) * 0.1

result = fmpfit_pywrap(
    deviate_type=0,  # Gaussian model
    parinfo=[
        {'value': 1.0, 'limits': [0.0, 10.0]},  # amplitude
        {'value': 0.0, 'limits': [-5.0, 5.0]},  # mean
        {'value': 1.0, 'limits': [0.1, 5.0]}    # sigma
    ],
    functkw={'x': x, 'y': y, 'error': np.ones_like(y) * 0.1}
)
print(f"Best-fit: {result.best_params}, ?²={result.bestnorm:.2f}")
print(f"SciPy-style errors (full Hessian): {result.xerror_scipy}")
```

Block fitting (multiple spectra in one call, GIL-free):

```python
from ftoolss.fmpfit import fmpfit_block_pywrap
import numpy as np

n_spectra, mpoints, npar = 100, 200, 3
x = np.tile(np.linspace(-5, 5, mpoints), (n_spectra, 1))
y = ...  # shape (n_spectra, mpoints)
error = np.ones_like(y) * 0.1
p0 = np.tile([1.0, 0.0, 1.0], (n_spectra, 1))
bounds = np.array([[[0, 10], [-5, 5], [0.1, 5]]] * n_spectra)

results = fmpfit_block_pywrap(0, x, y, error, p0, bounds)
# results['best_params'] shape: (n_spectra, npar)
```

## API Reference

### fmedian / fsigma

```python
fmedian(data, window_size, exclude_center=0)
fsigma(data, window_size, exclude_center=0)
```

- `window_size`: `(x, y)` or `(x, y, z)` ? odd positive integers
- `exclude_center`: 1 to exclude center from calculation (useful for outlier detection)
- Returns: float64 array, same shape as input

### fgaussian_f32 / fgaussian_f64

Computes: `i0 * exp(-((x - mu)² / (2 * sigma²))`

```python
fgaussian_f32(x, i0, mu, sigma)  # float32, fastest
fgaussian_f64(x, i0, mu, sigma)  # float64
```

### fmpfit

Levenberg-Marquardt least-squares fitting. See `src/ftoolss/fmpfit/README.md` for details.

| Function | Description |
|----------|-------------|
| `fmpfit_pywrap` | Auto dtype dispatch (single spectrum) |
| `fmpfit_f64_pywrap` | float64 single spectrum |
| `fmpfit_f32_pywrap` | float32 single spectrum |
| `fmpfit_block_pywrap` | Auto dtype dispatch (block of spectra) |
| `fmpfit_f64_block_pywrap` | float64 block fitting |
| `fmpfit_f32_block_pywrap` | float32 block fitting |

## Examples

See `examples/` directory for complete working examples.

## Testing

```bash
pytest                              # Run all tests
pytest --cov=ftoolss --cov-report=html  # With coverage
```

## Performance

| Module | Speedup vs NumPy | Notes |
|--------|------------------|-------|
| fgaussian_f32 | 5-10× | Accelerate vvexpf |
| fgaussian_f64 | 2-3× | Accelerate vvexp |
| fmedian/fsigma | ? | Sorting networks for small windows |
| fmpfit | ? | GIL-free, 4× speedup with 6 threads |

## License

MIT

## Credits

- Sorting networks: [Bert Dobbelaere](https://bertdobbelaere.github.io/sorting_networks.html)
- MPFIT: Craig Markwardt (cmpfit, MINPACK-1 derivative)
