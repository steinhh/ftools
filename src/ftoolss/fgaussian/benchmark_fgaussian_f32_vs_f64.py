#!/usr/bin/env python
"""
Benchmark comparison: fgaussian (float32) vs fgaussian_f64 (float64)

Shows the performance difference between float32 and float64 versions.
"""

import numpy as np
import time
from . import fgaussian_f32_ext, fgaussian_f64_ext

# Test parameters
i0, mu, sigma = 1.0, 0.0, 1.5

# Test different array sizes
test_sizes = [10, 100, 1000, 10000, 100000]

print("Benchmark: fgaussian_f32 (float32) vs fgaussian_f64 (float64)")
print("=" * 70)
print(f"{'N':<10} {'f32 (?s)':<15} {'f64 (?s)':<15} {'Ratio (f64/f32)':<15}")
print("-" * 70)

for n in test_sizes:
    # Prepare arrays
    x_f32 = np.linspace(-10, 10, n, dtype=np.float32)
    x_f64 = np.linspace(-10, 10, n, dtype=np.float64)
    
    # Warm up
    _ = fgaussian_f32_ext.fgaussian_f32(x_f32, i0, mu, sigma)
    _ = fgaussian_f64_ext.fgaussian_f64(x_f64, i0, mu, sigma)
    
    # Benchmark float32
    n_iter = max(1000, 100000 // n)
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = fgaussian_f32_ext.fgaussian_f32(x_f32, i0, mu, sigma)
    t_f32 = (time.perf_counter() - start) / n_iter * 1e6
    
    # Benchmark float64
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = fgaussian_f64_ext.fgaussian_f64(x_f64, i0, mu, sigma)
    t_f64 = (time.perf_counter() - start) / n_iter * 1e6
    
    ratio = t_f64 / t_f32
    print(f"{n:<10} {t_f32:<15.2f} {t_f64:<15.2f} {ratio:<15.2f}x")

print("=" * 70)
print("\nNote: Float32 version is faster due to:")
print("  - 50% less memory bandwidth")
print("  - 2x wider SIMD vectors")
print("  - Faster exponential computation")
