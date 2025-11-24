#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of fmpfit_f32 extension - Multiple runs

Demonstrates fitting a Gaussian model to synthetic noisy data N times,
each with different random noise realizations. Useful for testing
convergence statistics and performance using float32 precision.

Usage:
    python example_fmpfit_f32_N.py [N]
    
where N is the number of fitting runs (default: 10)
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftools import fmpfit_f32_wrap

# Parse command line argument for number of runs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

print("=" * 70)
print(f"FMPFIT FLOAT32 Multiple Runs Example: Fitting {N} datasets")
print("=" * 70)

# Fixed parameters for all runs
x = np.linspace(-5, 5, 100, dtype=np.float32)
true_params = np.array([2.5, 1.0, 0.8], dtype=np.float32)  # amplitude, mean, sigma
error = np.ones_like(x, dtype=np.float32) * 0.1
p0 = [1.5, 0.5, 1.0]  # Initial guesses
bounds = [[0.0, 10.0], [-5.0, 5.0], [0.1, 5.0]]

print(f"\nData points per fit: {len(x)}")
print(f"Precision: float32 (32-bit)")
print(f"True parameters: {true_params.tolist()}")
print(f"Initial guesses: {p0}")
print(f"\nRunning {N} fits with different noise realizations...\n")

# Storage for results
all_params = []
all_errors = []
all_chi2 = []
all_niter = []
all_nfev = []
all_status = []
all_times = []

# Run N fits
for i in range(N):
    # Generate new noise realization for each run
    np.random.seed(42 + i)
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
    noise = np.random.normal(0, 0.1, len(x)).astype(np.float32)
    y = y_true + noise
    
    # Setup for this fit
    parinfo = [{'value': p0[j], 'limits': bounds[j]} for j in range(len(p0))]
    functkw = {'x': x, 'y': y, 'error': error}
    
    # Time the fit
    t0 = time.time()
    result = fmpfit_f32_wrap(
        deviate_type=0,
        parinfo=parinfo,
        functkw=functkw,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
        maxiter=2000,
        quiet=1
    )
    elapsed = time.time() - t0
    
    # Store results
    all_params.append(result.best_params)
    all_errors.append(result.xerror)
    all_chi2.append(result.bestnorm)
    all_niter.append(result.niter)
    all_nfev.append(result.nfev)
    all_status.append(result.status)
    all_times.append(elapsed)
    
    if (i + 1) % max(1, N // 10) == 0 or i == 0:
        print(f"  Run {i+1:4d}/{N}: chi^2={result.bestnorm:6.1f}, "
              f"iter={result.niter:2d}, status={result.status}, "
              f"time={elapsed*1000:.2f}ms")

# Convert to arrays for statistics
all_params = np.array(all_params)
all_errors = np.array(all_errors)
all_chi2 = np.array(all_chi2)
all_niter = np.array(all_niter)
all_nfev = np.array(all_nfev)
all_status = np.array(all_status)
all_times = np.array(all_times)

# Calculate statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\nStatus codes:")
unique_status, counts = np.unique(all_status, return_counts=True)
for status, count in zip(unique_status, counts):
    print(f"  Status {status:2d}: {count:4d} fits ({100*count/N:5.1f}%)")

print("\nReduced chi^2 statistics:")
reduced_chi2 = all_chi2 / (len(x) - len(p0))
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
print(f"  Mean:   {all_times.mean()*1000:.2f} ms")
print(f"  Std:    {all_times.std()*1000:.2f} ms")
print(f"  Min:    {all_times.min()*1000:.2f} ms")
print(f"  Max:    {all_times.max()*1000:.2f} ms")
print(f"  Total:  {all_times.sum():.3f} s")

print("\nParameter recovery (fitted - true):")
param_names = ['Amplitude', 'Mean', 'Sigma']
for i, name in enumerate(param_names):
    offsets = all_params[:, i] - true_params[i]
    mean_error = all_errors[:, i].mean()
    print(f"  {name:12s}: mean offset = {offsets.mean():+.6f} +/- {offsets.std():.6f}")
    print(f"                 mean error   = {mean_error:.6f}")
    print(f"                 pull = {offsets.mean()/mean_error:+.3f} +/- {offsets.std()/mean_error:.3f}")

print("=" * 70)
