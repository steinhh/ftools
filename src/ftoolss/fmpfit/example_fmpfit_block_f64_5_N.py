#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of fmpfit_f64_block extension - Minimal data case

Demonstrates fitting Gaussian models with only 5 data points spanning
no more than the FWHM. Uses Poisson-distributed noise (error = sqrt(signal)).
This is a challenging case with minimal data and realistic photon noise.
Fits N spectra in a single block call using float64 precision.

Usage:
    python example_fmpfit_block_f64_5_N.py [N]
    
where N is the number of spectra to fit in one block (default: 1000)
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local module if running from fmpfit directory
try:
    from ftoolss.fmpfit import fmpfit_f64_block_pywrap
except ModuleNotFoundError:
    # Try direct import if in fmpfit directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("fmpfit_module", "__init__.py")
    fmpfit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmpfit_module)
    fmpfit_f64_block_pywrap = fmpfit_module.fmpfit_f64_block_pywrap

# Parse command line argument for number of spectra
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

print("=" * 70)
print(f"FMPFIT BLOCK FLOAT64 Minimal Data Example: 5 pixels, {N} spectra")
print("=" * 70)

# Fixed parameters for all spectra
true_params = np.array([2.5, 0.0, 0.8], dtype=np.float64)  # amplitude, mean, sigma
fwhm = 2.355 * true_params[2]
npar = 3
mpoints = 5

# 5 pixels spanning the FWHM (centered at mean=0.0)
x_1d = np.linspace(-fwhm/2, fwhm/2, mpoints, dtype=np.float64)

p0_1d = np.array([2.0, 0.0, 1.0], dtype=np.float64)  # Initial guesses
bounds_1d = np.array([[0.0, 10.0], [-2.0, 2.0], [0.1, 5.0]], dtype=np.float64)

print(f"\nTrue parameters: amplitude={true_params[0]}, mean={true_params[1]}, sigma={true_params[2]}")
print(f"FWHM: {fwhm:.3f}")
print(f"Data points per spectrum: {mpoints}")
print(f"X range: [{x_1d[0]:.3f}, {x_1d[-1]:.3f}]")
print(f"Initial guesses: {p0_1d.tolist()}")
print("Precision: float64 (64-bit)")
print(f"\nGenerating {N} spectra with different noise realizations...")

# Generate all spectra at once
np.random.seed(42)

# x array: same for all spectra, shape (N, mpoints)
x = np.tile(x_1d, (N, 1))

# True signal for each spectrum (same true params for all)
y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)

# Poisson noise: y ~ Poisson(y_true), error = sqrt(y)
y = np.random.poisson(y_true).astype(np.float64)
error = np.sqrt(np.maximum(y, 1.0)).astype(np.float64)  # Avoid division by zero

# Initial parameters and bounds: same for all spectra
p0 = np.tile(p0_1d, (N, 1))
bounds = np.tile(bounds_1d, (N, 1, 1))

print(f"Data shapes: x={x.shape}, y={y.shape}, error={error.shape}")
print(f"Parameter shapes: p0={p0.shape}, bounds={bounds.shape}")

print(f"\nFitting {N} spectra in a single block call...")

# Time the block fit
t0 = time.perf_counter()
result = fmpfit_f64_block_pywrap(
    0,  # deviate_type = Gaussian
    x, y, error, p0, bounds,
    xtol=1.0e-6,
    ftol=1.0e-6,
    gtol=1.0e-6,
    maxiter=2000,
)
elapsed = time.perf_counter() - t0

# Extract results
all_params = result['best_params']
all_errors = result['xerror']
all_chi2 = result['bestnorm']
all_niter = result['niter']
all_nfev = result['nfev']
all_status = result['status']
c_time = result.get('c_time', elapsed)

# Calculate statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\nStatus codes:")
unique_status, counts = np.unique(all_status, return_counts=True)
for status, count in zip(unique_status, counts):
    status_name = {
        1: "MP_OK_CHI (chi-square converged)",
        2: "MP_OK_PAR (parameter converged)",
        3: "MP_OK_BOTH (both converged)",
        4: "MP_OK_DIR (orthogonality)",
        5: "MP_MAXITER (max iterations)"
    }.get(status, f"Unknown ({status})")
    print(f"  Status {status:2d}: {count:6d} fits ({100*count/N:5.1f}%) - {status_name}")

# DOF = 5 data points - 3 parameters = 2
dof = mpoints - npar
print(f"\nDegrees of freedom: {dof}")
print(f"Expected chi^2 ~ {dof}")

print("\nReduced chi^2 statistics:")
reduced_chi2 = all_chi2 / dof
print(f"  Mean:   {reduced_chi2.mean():.4f}")
print(f"  Std:    {reduced_chi2.std():.4f}")
print(f"  Min:    {reduced_chi2.min():.4f}")
print(f"  Max:    {reduced_chi2.max():.4f}")
print(f"  Median: {np.median(reduced_chi2):.4f}")

print("\nIteration statistics:")
print(f"  Mean:   {all_niter.mean():.2f}")
print(f"  Std:    {all_niter.std():.2f}")
print(f"  Min:    {all_niter.min()}")
print(f"  Max:    {all_niter.max()}")

print("\nFunction evaluation statistics:")
print(f"  Mean:   {all_nfev.mean():.2f}")
print(f"  Std:    {all_nfev.std():.2f}")
print(f"  Min:    {all_nfev.min()}")
print(f"  Max:    {all_nfev.max()}")

print("\nTiming statistics:")
print(f"  Total time:         {elapsed:.4f} s")
print(f"  C extension time:   {c_time:.4f} s")
print(f"  Python overhead:    {(elapsed - c_time):.4f} s")
print(f"  Time per spectrum:  {elapsed/N*1e6:.2f} us")
print(f"  C time per spectrum: {c_time/N*1e6:.2f} us")
print(f"  Throughput:         {N/elapsed:.0f} spectra/s")

print("\nParameter recovery (fitted - true):")
param_names = ['Amplitude', 'Mean', 'Sigma']
for i, name in enumerate(param_names):
    offsets = all_params[:, i] - true_params[i]
    mean_error = all_errors[:, i].mean()
    print(f"  {name:12s}: mean offset = {offsets.mean():+.6f} +/- {offsets.std():.6f}")
    print(f"                 mean error   = {mean_error:.6f}")
    if mean_error > 0:
        print(f"                 pull = {offsets.mean()/mean_error:+.3f} +/- {offsets.std()/mean_error:.3f}")

# Count failed fits
failed_mask = ~np.isin(all_status, [1, 2, 3, 4])
n_failed = failed_mask.sum()
if n_failed > 0:
    print(f"\nWARNING: {n_failed} spectra ({100*n_failed/N:.1f}%) failed to converge properly")

print("=" * 70)
print("\nNOTE: This 5-point example reflects typical MUSE spectral fitting,")
print("where spectra are extracted in narrow windows around emission lines.")
print("With only 5 data points and 3 parameters (2 DOF), and Poisson noise")
print("(error = sqrt(signal)), fits are highly sensitive to noise and may")
print("show large scatter. Some fits may converge to local minima or")
print("parameter boundaries.")
print("\nThis example uses float64 precision block fitting, which processes")
print("all spectra in a single C call with the GIL released.")
print("Typical throughput: ~85,000 spectra/s (~12 us/spectrum) for 5-point fits.")
print("=" * 70)
