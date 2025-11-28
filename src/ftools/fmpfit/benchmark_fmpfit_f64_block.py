import numpy as np
from ftools.fmpfit import fmpfit_f64_block_pywrap, fmpfit_f64_pywrap
import time
import sys

# Get N spectra from command line (default 1000)
n_spectra = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n_points = 5
n_params = 3

# Create test data
rng = np.random.default_rng(42)
x = np.tile(np.linspace(-2, 2, n_points), (n_spectra, 1))

# True parameters
true_I = rng.uniform(5, 15, n_spectra)
true_v = rng.uniform(-0.5, 0.5, n_spectra)
true_w = rng.uniform(0.8, 1.5, n_spectra)

# Generate y values
y = np.zeros((n_spectra, n_points))
for s in range(n_spectra):
    y[s] = true_I[s] * np.exp(-0.5 * ((x[s] - true_v[s]) / true_w[s])**2)

y = rng.poisson(np.maximum(y, 0.1)).astype(np.float64)
error = np.sqrt(np.maximum(y, 1.0))

# Initial guesses
p0 = np.zeros((n_spectra, n_params))
for s in range(n_spectra):
    max_idx = np.argmax(y[s])
    p0[s, 0] = y[s, max_idx]
    p0[s, 1] = x[s, max_idx]
    p0[s, 2] = 1.0

# Bounds
bounds = np.zeros((n_spectra, n_params, 2))
bounds[:, 0, :] = [0.0, 100.0]
bounds[:, 1, :] = [-2.0, 2.0]
bounds[:, 2, :] = [0.5, 3.0]

# Time block fitting
t0 = time.perf_counter()
result_block = fmpfit_f64_block_pywrap(x, y, error, p0, bounds)
t_block = time.perf_counter() - t0

# Time individual fitting (first 100 for comparison)
n_compare = 100
t0 = time.perf_counter()
for s in range(n_compare):
    parinfo = [
        {'value': p0[s, 0], 'limits': [bounds[s, 0, 0], bounds[s, 0, 1]]},
        {'value': p0[s, 1], 'limits': [bounds[s, 1, 0], bounds[s, 1, 1]]},
        {'value': p0[s, 2], 'limits': [bounds[s, 2, 0], bounds[s, 2, 1]]},
    ]
    functkw = {'x': x[s], 'y': y[s], 'error': error[s]}
    _ = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
t_single = (time.perf_counter() - t0) / n_compare * n_spectra

# Results
n_converged = np.sum((result_block['status'] >= 1) & (result_block['status'] <= 4))

print(f'Block fitting {n_spectra} spectra:')
print(f'  Total time: {t_block*1000:.1f} ms')
print(f'  Per spectrum: {t_block/n_spectra*1e6:.1f} us')
print(f'  Converged: {n_converged}/{n_spectra}')
print()
print(f'Individual fitting (extrapolated from {n_compare}):')
print(f'  Total time: {t_single*1000:.1f} ms')
print(f'  Per spectrum: {t_single/n_spectra*1e6:.1f} us')
print()
print(f'Speedup: {t_single/t_block:.1f}x')
