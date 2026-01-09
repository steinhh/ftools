#!/usr/bin/env python3
"""
Benchmark fgaussian_f32 performance across different array sizes.
Compares C extension (float32 + Accelerate) vs NumPy (float64).
"""

import numpy as np
import time
from . import fgaussian_f32_ext


def numpy_gaussian(x, i0, mu, sigma):
    """Reference NumPy implementation (float64)"""
    return i0 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def benchmark_size(n, num_iterations=1000):
    """Benchmark for a specific array size"""
    # Setup
    x_f32 = np.linspace(-10, 10, n, dtype=np.float32)  # float32 for C extension
    x_f64 = np.linspace(-10, 10, n)  # float64 for NumPy
    i0, mu, sigma = 2.5, 1.5, 3.0
    
    # Warm up
    for _ in range(10):
        _ = fgaussian_f32_ext.fgaussian_f32(x_f32, i0, mu, sigma)
        _ = numpy_gaussian(x_f64, i0, mu, sigma)
    
    # Benchmark C extension (float32)
    start = time.perf_counter()
    for _ in range(num_iterations):
        result_c = fgaussian_f32_ext.fgaussian_f32(x_f32, i0, mu, sigma)
    time_c = (time.perf_counter() - start) / num_iterations
    
    # Benchmark NumPy (float64)
    start = time.perf_counter()
    for _ in range(num_iterations):
        result_np = numpy_gaussian(x_f64, i0, mu, sigma)
    time_np = (time.perf_counter() - start) / num_iterations
    
    # Calculate speedup and accuracy
    speedup = time_np / time_c
    max_diff = np.max(np.abs(result_c - result_np))
    
    return time_c, time_np, speedup, max_diff


def main():
    """Run benchmarks for different array sizes"""
    print("=" * 80)
    print("fgaussian Benchmark: C Extension (float32) vs NumPy (float64)")
    print("=" * 80)
    print()
    
    sizes = [5, 10, 100, 1000, 10000]
    iterations = [100000, 100000, 10000, 1000, 100]  # Fewer iterations for larger arrays
    
    print(f"{'N':<10} {'C (?s)':<12} {'NumPy (?s)':<12} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 80)
    
    for n, num_iter in zip(sizes, iterations):
        time_c, time_np, speedup, max_diff = benchmark_size(n, num_iter)
        
        # Convert to microseconds for readability
        time_c_us = time_c * 1e6
        time_np_us = time_np * 1e6
        
        print(f"{n:<10} {time_c_us:<12.3f} {time_np_us:<12.3f} {speedup:<10.2f}x {max_diff:<12.2e}")
    
    print()
    print("Notes:")
    print("  - C extension uses float32 with Apple Accelerate framework")
    print("  - NumPy uses float64 with standard operations")
    print("  - Speedup = NumPy time / C time")
    print("  - Max Diff = maximum absolute difference between results")
    print()


if __name__ == "__main__":
    main()
