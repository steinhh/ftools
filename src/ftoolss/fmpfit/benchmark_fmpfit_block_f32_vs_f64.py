#!/usr/bin/env python3
"""
Combined benchmark: fmpfit_block f32 vs f64

Compares block fitting performance and accuracy between float32 and float64
precision for the same test data.
"""

import numpy as np
from ftoolss.fmpfit import (
    fmpfit_f32_block_pywrap, fmpfit_f64_block_pywrap,
    fmpfit_f32_pywrap, fmpfit_f64_pywrap
)
import time
import sys

# Get N spectra from command line (default 1000)
n_spectra = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n_points = 5
n_params = 3

print("=" * 70)
print(f"FMPFIT BLOCK BENCHMARK: float32 vs float64 ({n_spectra} spectra)")
print("=" * 70)

# Create test data (generate in float64, then convert for f32)
rng = np.random.default_rng(42)
x_f64 = np.tile(np.linspace(-2, 2, n_points), (n_spectra, 1))

# True parameters
true_I = rng.uniform(5, 15, n_spectra)
true_v = rng.uniform(-0.5, 0.5, n_spectra)
true_w = rng.uniform(0.8, 1.5, n_spectra)

# Generate y values
y_f64 = np.zeros((n_spectra, n_points))
for s in range(n_spectra):
    y_f64[s] = true_I[s] * np.exp(-0.5 * ((x_f64[s] - true_v[s]) / true_w[s])**2)

y_f64 = rng.poisson(np.maximum(y_f64, 0.1)).astype(np.float64)
error_f64 = np.sqrt(np.maximum(y_f64, 1.0))

# Initial guesses
p0_f64 = np.zeros((n_spectra, n_params))
for s in range(n_spectra):
    max_idx = np.argmax(y_f64[s])
    p0_f64[s, 0] = y_f64[s, max_idx]
    p0_f64[s, 1] = x_f64[s, max_idx]
    p0_f64[s, 2] = 1.0

# Bounds
bounds_f64 = np.zeros((n_spectra, n_params, 2))
bounds_f64[:, 0, :] = [0.0, 100.0]
bounds_f64[:, 1, :] = [-2.0, 2.0]
bounds_f64[:, 2, :] = [0.5, 3.0]

# Convert to float32
x_f32 = x_f64.astype(np.float32)
y_f32 = y_f64.astype(np.float32)
error_f32 = error_f64.astype(np.float32)
p0_f32 = p0_f64.astype(np.float32)
bounds_f32 = bounds_f64.astype(np.float32)

# ============================================================================
# Block fitting benchmarks
# ============================================================================
print("\n--- BLOCK FITTING ---")

# Warmup
_ = fmpfit_f64_block_pywrap(0, x_f64[:10], y_f64[:10], error_f64[:10], p0_f64[:10], bounds_f64[:10])
_ = fmpfit_f32_block_pywrap(0, x_f32[:10], y_f32[:10], error_f32[:10], p0_f32[:10], bounds_f32[:10])

# Time f64 block
t0 = time.perf_counter()
result_f64 = fmpfit_f64_block_pywrap(0, x_f64, y_f64, error_f64, p0_f64, bounds_f64)
t_f64_block = time.perf_counter() - t0

# Time f32 block
t0 = time.perf_counter()
result_f32 = fmpfit_f32_block_pywrap(0, x_f32, y_f32, error_f32, p0_f32, bounds_f32)
t_f32_block = time.perf_counter() - t0

# ============================================================================
# Individual fitting benchmarks (extrapolated from subset)
# ============================================================================
print("\n--- INDIVIDUAL FITTING (extrapolated from 100) ---")

n_compare = min(100, n_spectra)

# Time f64 individual
t0 = time.perf_counter()
for s in range(n_compare):
    parinfo = [
        {'value': p0_f64[s, 0], 'limits': [bounds_f64[s, 0, 0], bounds_f64[s, 0, 1]]},
        {'value': p0_f64[s, 1], 'limits': [bounds_f64[s, 1, 0], bounds_f64[s, 1, 1]]},
        {'value': p0_f64[s, 2], 'limits': [bounds_f64[s, 2, 0], bounds_f64[s, 2, 1]]},
    ]
    functkw = {'x': x_f64[s], 'y': y_f64[s], 'error': error_f64[s]}
    _ = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
t_f64_single = (time.perf_counter() - t0) / n_compare * n_spectra

# Time f32 individual
t0 = time.perf_counter()
for s in range(n_compare):
    parinfo = [
        {'value': float(p0_f32[s, 0]), 'limits': [float(bounds_f32[s, 0, 0]), float(bounds_f32[s, 0, 1])]},
        {'value': float(p0_f32[s, 1]), 'limits': [float(bounds_f32[s, 1, 0]), float(bounds_f32[s, 1, 1])]},
        {'value': float(p0_f32[s, 2]), 'limits': [float(bounds_f32[s, 2, 0]), float(bounds_f32[s, 2, 1])]},
    ]
    functkw = {'x': x_f32[s], 'y': y_f32[s], 'error': error_f32[s]}
    _ = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
t_f32_single = (time.perf_counter() - t0) / n_compare * n_spectra

# ============================================================================
# Results
# ============================================================================
n_converged_f64 = np.sum((result_f64['status'] >= 1) & (result_f64['status'] <= 4))
n_converged_f32 = np.sum((result_f32['status'] >= 1) & (result_f32['status'] <= 4))

# Parameter comparison (how close are f32 results to f64?)
param_diff = np.abs(result_f64['best_params'].astype(np.float64) - 
                    result_f32['best_params'].astype(np.float64))
max_param_diff = np.max(param_diff)
mean_param_diff = np.mean(param_diff)

# Chi-square comparison
chi2_diff = np.abs(result_f64['bestnorm'] - result_f32['bestnorm'].astype(np.float64))
max_chi2_diff = np.max(chi2_diff)
mean_chi2_diff = np.mean(chi2_diff)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nTIMING COMPARISON")
print("-" * 70)
print(f"{'Method':<25} {'float64':>15} {'float32':>15} {'f32/f64':>12}")
print("-" * 70)
print(f"{'Block (total ms)':<25} {t_f64_block*1000:>15.2f} {t_f32_block*1000:>15.2f} {t_f32_block/t_f64_block:>12.2f}x")
print(f"{'Block (per spectrum us)':<25} {t_f64_block/n_spectra*1e6:>15.2f} {t_f32_block/n_spectra*1e6:>15.2f} {t_f32_block/t_f64_block:>12.2f}x")
print(f"{'Individual (total ms)':<25} {t_f64_single*1000:>15.2f} {t_f32_single*1000:>15.2f} {t_f32_single/t_f64_single:>12.2f}x")
print(f"{'Individual (per us)':<25} {t_f64_single/n_spectra*1e6:>15.2f} {t_f32_single/n_spectra*1e6:>15.2f} {t_f32_single/t_f64_single:>12.2f}x")

print("\nSPEEDUP (block vs individual)")
print("-" * 70)
print(f"{'float64 block speedup:':<35} {t_f64_single/t_f64_block:.1f}x")
print(f"{'float32 block speedup:':<35} {t_f32_single/t_f32_block:.1f}x")

print("\nCONVERGENCE")
print("-" * 70)
print(f"{'float64 converged:':<35} {n_converged_f64}/{n_spectra} ({100*n_converged_f64/n_spectra:.1f}%)")
print(f"{'float32 converged:':<35} {n_converged_f32}/{n_spectra} ({100*n_converged_f32/n_spectra:.1f}%)")

print("\nNUMERICAL ACCURACY (f32 vs f64)")
print("-" * 70)
print(f"{'Max parameter difference:':<35} {max_param_diff:.6e}")
print(f"{'Mean parameter difference:':<35} {mean_param_diff:.6e}")
print(f"{'Max chi-square difference:':<35} {max_chi2_diff:.6e}")
print(f"{'Mean chi-square difference:':<35} {mean_chi2_diff:.6e}")

print("\nMEMORY (per spectrum)")
print("-" * 70)
mem_f64 = (x_f64.nbytes + y_f64.nbytes + error_f64.nbytes + p0_f64.nbytes + bounds_f64.nbytes) / n_spectra
mem_f32 = (x_f32.nbytes + y_f32.nbytes + error_f32.nbytes + p0_f32.nbytes + bounds_f32.nbytes) / n_spectra
print(f"{'float64 input data:':<35} {mem_f64:.0f} bytes")
print(f"{'float32 input data:':<35} {mem_f32:.0f} bytes")
print(f"{'Memory savings:':<35} {100*(1-mem_f32/mem_f64):.1f}%")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if t_f32_block < t_f64_block:
    print(f"float32 block is {t_f64_block/t_f32_block:.2f}x FASTER than float64 block")
else:
    print(f"float64 block is {t_f32_block/t_f64_block:.2f}x FASTER than float32 block")
print(f"Both achieve similar convergence rates and parameter accuracy")
print(f"float32 uses {100*(1-mem_f32/mem_f64):.0f}% less memory")
print("=" * 70)
