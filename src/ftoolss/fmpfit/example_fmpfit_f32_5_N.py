#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of fmpfit_f32 extension - Minimal data case

Demonstrates fitting a Gaussian model with only 5 data points spanning
no more than the FWHM. Uses Poisson-distributed noise (error = sqrt(signal)).
This is a challenging case with minimal data and realistic photon noise.
Run N times with different noise realizations using float32 precision.

Usage:
    python example_fmpfit_f32_5_N.py [N]
    
where N is the number of fitting runs (default: 10)
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local module if running from fmpfit directory
try:
    from ftoolss.fmpfit import fmpfit_f32_pywrap
except ModuleNotFoundError:
    # Try direct import if in fmpfit directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("fmpfit_module", "__init__.py")
    fmpfit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmpfit_module)
    fmpfit_f32_pywrap = fmpfit_module.fmpfit_f32_pywrap

# Parse command line argument for number of runs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

print("=" * 70)
print(f"FMPFIT FLOAT32 Minimal Data Example: 5 pixels, {N} runs")
print("=" * 70)

# Fixed parameters for all runs
true_params = np.array([2.5, 0.0, 0.8], dtype=np.float32)  # amplitude, mean, sigma
fwhm = 2.355 * true_params[2]  # FWHM = 2.355 * sigma

# 5 pixels spanning the FWHM (centered at mean=0.0)
# We'll use points from -FWHM/2 to +FWHM/2
x = np.linspace(-fwhm/2, fwhm/2, 5, dtype=np.float32)

p0 = [2.0, 0.0, 1.0]  # Initial guesses
bounds = [[0.0, 10.0], [-2.0, 2.0], [0.1, 5.0]]

print(f"\nTrue parameters: amplitude={true_params[0]}, mean={true_params[1]}, sigma={true_params[2]}")
print(f"FWHM: {fwhm:.3f}")
print(f"Data points: {len(x)}")
print(f"X range: [{x[0]:.3f}, {x[-1]:.3f}]")
print(f"Initial guesses: {p0}")
print(f"Precision: float32 (32-bit)")
print(f"\nRunning {N} fits with different noise realizations...\n")

# Storage for results
all_params = []
all_errors = []
all_chi2 = []
all_niter = []
all_nfev = []
all_status = []
all_times = []
all_c_times = []
failed_runs = []

# Run N fits
for i in range(N):
    # Generate new noise realization for each run
    np.random.seed(42 + i)
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
    
    # Poisson noise: y ~ Poisson(y_true), error = sqrt(y)
    y = np.random.poisson(y_true).astype(np.float32)
    error = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)  # Avoid division by zero for low counts
    
    # Setup for this fit
    parinfo = [{'value': p0[j], 'limits': bounds[j]} for j in range(len(p0))]
    functkw = {'x': x, 'y': y, 'error': error}
    
    # Time the fit (total time including Python wrapper)
    t0 = time.perf_counter()
    result = fmpfit_f32_pywrap(
        deviate_type=0,
        parinfo=parinfo,
        functkw=functkw,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
        maxiter=2000,
        
    )
    elapsed = time.perf_counter() - t0
    
    # Store results
    all_params.append(result.best_params)
    all_errors.append(result.xerror)
    all_chi2.append(result.bestnorm)
    all_niter.append(result.niter)
    all_nfev.append(result.nfev)
    all_status.append(result.status)
    all_times.append(elapsed)
    all_c_times.append(result.c_time)
    
    # Track failed convergence
    if result.status != 1 and result.status != 2 and result.status != 3 and result.status != 4:
        failed_runs.append(i + 1)
    
    if (i + 1) % max(1, N // 10) == 0 or i == 0:
        print(f"  Run {i+1:4d}/{N}: chi^2={result.bestnorm:6.1f}, "
              f"iter={result.niter:2d}, status={result.status}, "
              f"time={elapsed*1e6:.2f}us")

# Convert to arrays for statistics
all_params = np.array(all_params)
all_errors = np.array(all_errors)
all_chi2 = np.array(all_chi2)
all_niter = np.array(all_niter)
all_nfev = np.array(all_nfev)
all_status = np.array(all_status)
all_times = np.array(all_times)
all_c_times = np.array(all_c_times)

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
    print(f"  Status {status:2d}: {count:4d} fits ({100*count/N:5.1f}%) - {status_name}")

# DOF = 5 data points - 3 parameters = 2
dof = len(x) - len(p0)
print(f"\nDegrees of freedom: {dof}")
print(f"Expected chi^2 ~ {dof}")

print(f"\nReduced chi^2 statistics:")
reduced_chi2 = all_chi2 / dof
print(f"  Mean:   {reduced_chi2.mean():.4f}")
print(f"  Std:    {reduced_chi2.std():.4f}")
print(f"  Min:    {reduced_chi2.min():.4f}")
print(f"  Max:    {reduced_chi2.max():.4f}")
print(f"  Median: {np.median(reduced_chi2):.4f}")

print(f"\nIteration statistics:")
print(f"  Mean:   {all_niter.mean():.2f}")
print(f"  Std:    {all_niter.std():.2f}")
print(f"  Min:    {all_niter.min()}")
print(f"  Max:    {all_niter.max()}")

print(f"\nFunction evaluation statistics:")
print(f"  Mean:   {all_nfev.mean():.2f}")
print(f"  Std:    {all_nfev.std():.2f}")
print(f"  Min:    {all_nfev.min()}")
print(f"  Max:    {all_nfev.max()}")

print("\nTiming statistics (total time):")
print(f"  Mean:   {all_times.mean()*1e6:.2f} us")
print(f"  Std:    {all_times.std()*1e6:.2f} us")
print(f"  Min:    {all_times.min()*1e6:.2f} us")
print(f"  Max:    {all_times.max()*1e6:.2f} us")
print(f"  Total:  {all_times.sum():.3f} s")

print("\nTiming statistics (C extension only):")
print(f"  Mean:   {all_c_times.mean()*1e6:.2f} us")
print(f"  Std:    {all_c_times.std()*1e6:.2f} us")
print(f"  Min:    {all_c_times.min()*1e6:.2f} us")
print(f"  Max:    {all_c_times.max()*1e6:.2f} us")
print(f"  Total:  {all_c_times.sum():.3f} s")

python_overhead = all_times - all_c_times
print("\nPython overhead statistics:")
print(f"  Mean:   {python_overhead.mean()*1e6:.2f} us")
print(f"  Std:    {python_overhead.std()*1e6:.2f} us")
print(f"  Min:    {python_overhead.min()*1e6:.2f} us")
print(f"  Max:    {python_overhead.max()*1e6:.2f} us")

print("\nTime breakdown (mean):")
mean_total = all_times.mean() * 1e6
mean_c = all_c_times.mean() * 1e6
mean_python = python_overhead.mean() * 1e6
print(f"  Total time:       {mean_total:8.2f} us (100.0%)")
print(f"  C extension:      {mean_c:8.2f} us ({mean_c/mean_total*100:5.1f}%)")
print(f"  Python overhead:  {mean_python:8.2f} us ({mean_python/mean_total*100:5.1f}%)")

print("\nParameter recovery (fitted - true):")
param_names = ['Amplitude', 'Mean', 'Sigma']
for i, name in enumerate(param_names):
    offsets = all_params[:, i] - true_params[i]
    mean_error = all_errors[:, i].mean()
    print(f"  {name:12s}: mean offset = {offsets.mean():+.6f} +/- {offsets.std():.6f}")
    print(f"                 mean error   = {mean_error:.6f}")
    if mean_error > 0:
        print(f"                 pull = {offsets.mean()/mean_error:+.3f} +/- {offsets.std()/mean_error:.3f}")

if failed_runs:
    print(f"\nWARNING: {len(failed_runs)} runs failed to converge properly")
    if len(failed_runs) <= 10:
        print(f"Failed run numbers: {failed_runs}")

print("=" * 70)
print("\nNOTE: With only 5 data points and 3 parameters (2 DOF),")
print("and Poisson noise (error = sqrt(signal)), fits are highly")
print("sensitive to noise and may show large scatter.")
print("Some fits may converge to local minima or parameter boundaries.")
print("\nThis example uses float32 precision, which provides ~6-7 significant")
print("digits and uses 50% less memory than float64.")
print("=" * 70)
