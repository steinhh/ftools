# ftools

High-performance C extensions for image processing and curve fitting.

## Features

| Module | Description |
|--------|-------------|
| `fmedian` | Local median filter (2D/3D auto-dispatch) |
| `fsigma` | Local σ filter (2D/3D auto-dispatch) |
| `fgaussian_f32/f64` | Gaussian profile (Accelerate-optimized) |
| `fmpfit_f32/f64_wrap` | Levenberg-Marquardt fitting (GIL-free) |

**Common features:** NaN handling, edge handling, optional center exclusion (fmedian/fsigma).

## Installation

```bash
pip install .                        # Standard install
pip install -e .                     # Editable (development)
python setup.py build_ext --inplace  # Build extensions only
```

**Requirements:** Python 3.8+, NumPy ≥1.20, C compiler. macOS Accelerate used when available.

## Quick Start

```python
import numpy as np
from ftools import fmedian, fsigma, fgaussian_f32, fgaussian_f64

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

```python
from ftools import fmpfit_f64_wrap
import numpy as np

x = np.linspace(-5, 5, 100)
y = 2.5 * np.exp(-0.5 * ((x - 1.0) / 0.8)**2) + np.random.randn(100) * 0.1

result = fmpfit_f64_wrap(
    deviate_type=0,  # Gaussian model
    parinfo=[
        {'value': 1.0, 'limits': [0.0, 10.0]},  # amplitude
        {'value': 0.0, 'limits': [-5.0, 5.0]},  # mean
        {'value': 1.0, 'limits': [0.1, 5.0]}    # sigma
    ],
    functkw={'x': x, 'y': y, 'error': np.ones_like(y) * 0.1}
)
print(f"Best-fit: {result.best_params}, χ²={result.bestnorm:.2f}")
```

## API Reference

### fmedian / fsigma

```python
fmedian(data, window_size, exclude_center=0)
fsigma(data, window_size, exclude_center=0)
```

- `window_size`: `(x, y)` or `(x, y, z)` – odd positive integers
- `exclude_center`: 1 to exclude center from calculation (useful for outlier detection)
- Returns: float64 array, same shape as input

### fgaussian_f32 / fgaussian_f64

Computes: `i0 * exp(-((x - mu)² / (2 * sigma²))`

```python
fgaussian_f32(x, i0, mu, sigma)  # float32, fastest
fgaussian_f64(x, i0, mu, sigma)  # float64
```

### fmpfit_f64_wrap / fmpfit_f32_wrap

Levenberg-Marquardt least-squares fitting. See `src/ftools/fmpfit/README.md` for details.

## Examples

See `examples/` directory for complete working examples.

## Testing

```bash
pytest                              # Run all tests
pytest --cov=ftools --cov-report=html  # With coverage
```

## Performance

| Module | Speedup vs NumPy | Notes |
|--------|------------------|-------|
| fgaussian_f32 | 5-10× | Accelerate vvexpf |
| fgaussian_f64 | 2-3× | Accelerate vvexp |
| fmedian/fsigma | — | Sorting networks for small windows |
| fmpfit | — | GIL-free, 4× speedup with 6 threads |

## License

MIT

## Credits

- Sorting networks: [Bert Dobbelaere](https://bertdobbelaere.github.io/sorting_networks.html)
- MPFIT: Craig Markwardt (MINPACK-1 derivative)
