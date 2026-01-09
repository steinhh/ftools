#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head-to-head benchmark: fmpfit float32 vs float64

Comprehensive comparison of performance, memory usage, and accuracy
using 5-pixel Gaussian fits with the same constraints as example_fmpfit_5_N.
Tests with Poisson noise and realistic bounds.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftoolss.fmpfit import fmpfit_f64_pywrap, fmpfit_f32_pywrap


def benchmark_single_fit(dtype_name, wrap_func, use_float32, seed=42):
    """Run a single fit and return timing/results
    
    Uses 5-pixel Gaussian with Poisson noise, same as example_fmpfit_5_N.
    """
    
    # Fixed parameters (same as example_fmpfit_5_N)
    true_params = np.array([2.5, 0.0, 0.8])  # amplitude, mean, sigma
    fwhm = 2.355 * true_params[2]  # FWHM = 2.355 * sigma
    
    # 5 pixels spanning the FWHM (centered at mean=0.0)
    x = np.linspace(-fwhm/2, fwhm/2, 5)
    
    # Generate Poisson noise for this realization
    np.random.seed(seed)
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
    
    # Poisson noise: y ~ Poisson(y_true), error = sqrt(y)
    y = np.random.poisson(y_true).astype(float)  # noqa: NPY002
    error = np.sqrt(np.maximum(y, 1.0))  # Avoid division by zero for low counts
    
    # Convert to appropriate dtype
    if use_float32:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        error = error.astype(np.float32)
    
    # Initial guess and bounds (same as example_fmpfit_5_N)
    p0 = [2.0, 0.0, 1.0]
    bounds = [[0.0, 10.0], [-2.0, 2.0], [0.1, 5.0]]
    
    # Prepare parinfo and functkw
    parinfo = [{'value': p0[i], 'limits': bounds[i]} for i in range(len(p0))]
    functkw = {'x': x, 'y': y, 'error': error}
    
    # Warmup run
    _ = wrap_func(
        deviate_type=0,
        parinfo=parinfo,
        functkw=functkw,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
            maxiter=2000,
    )
    
    # Timed run
    start = time.perf_counter()
    result = wrap_func(
        deviate_type=0,
        parinfo=parinfo,
        functkw=functkw,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
            maxiter=2000,
    )
    end = time.perf_counter()
    
    elapsed = (end - start) * 1e6  # Convert to microseconds
    
    return {
        'time_us': elapsed,
        'c_time_us': result.c_time * 1e6,
        'status': result.status,
        'niter': result.niter,
        'nfev': result.nfev,
        'bestnorm': result.bestnorm,
        'params': result.best_params.copy(),
        'memory': (result.best_params.nbytes + result.resid.nbytes + 
                   result.xerror.nbytes + result.covar.nbytes),
        'true_params': true_params
    }


def benchmark_batch(nruns, dtype_name, wrap_func, use_float32):
    """Run multiple fits with different noise realizations and collect statistics"""
    
    times = []
    c_times = []
    statuses = []
    iters = []
    fev_counts = []
    all_params = []
    
    for i in range(nruns):
        # Use different seed for each run to get different noise realizations
        result = benchmark_single_fit(dtype_name, wrap_func, use_float32, seed=42+i)
        times.append(result['time_us'])
        c_times.append(result['c_time_us'])
        statuses.append(result['status'])
        iters.append(result['niter'])
        fev_counts.append(result['nfev'])
        all_params.append(result['params'])
    
    # Get final result for memory and true params
    final = benchmark_single_fit(dtype_name, wrap_func, use_float32, seed=42)
    
    # Calculate parameter statistics
    all_params = np.array(all_params)
    param_offsets = all_params - final['true_params']
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_c_time': np.mean(c_times),
        'std_c_time': np.std(c_times),
        'mean_iters': np.mean(iters),
        'mean_fev': np.mean(fev_counts),
        'params': final['params'],
        'memory': final['memory'],
        'param_mean_offset': np.mean(param_offsets, axis=0),
        'param_std_offset': np.std(param_offsets, axis=0),
        'true_params': final['true_params']
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite with 5-pixel Gaussians"""
    
    print("=" * 80)
    print("FMPFIT FLOAT32 VS FLOAT64 HEAD-TO-HEAD BENCHMARK")
    print("5-pixel Gaussians with Poisson noise (same as example_fmpfit_5_N)")
    print("=" * 80)
    print()
    
    # Test configurations - different numbers of runs to test
    configs = [
        ('Quick', 100),
        ('Standard', 1000),
        ('Extended', 5000),
    ]
    
    results_table = []
    
    for config_name, nruns in configs:
        print(f"Running {config_name} benchmark ({nruns} runs with different noise)...")
        
        # Run float64
        results_f64 = benchmark_batch(nruns, 'float64', fmpfit_f64_pywrap, False)
        
        # Run float32
        results_f32 = benchmark_batch(nruns, 'float32', fmpfit_f32_pywrap, True)
        
        # Calculate speedup
        speedup = results_f64['mean_time'] / results_f32['mean_time']
        c_speedup = results_f64['mean_c_time'] / results_f32['mean_c_time']
        memory_ratio = results_f32['memory'] / results_f64['memory']
        
        # Calculate parameter differences (compare mean offsets from true values)
        f64_offset_rms = np.sqrt(np.mean(results_f64['param_mean_offset']**2))
        f32_offset_rms = np.sqrt(np.mean(results_f32['param_mean_offset']**2))
        
        # Direct comparison of fitted parameters from same noise realization
        param_diff = np.abs(results_f64['params'].astype(np.float64) - 
                           results_f32['params'].astype(np.float64))
        max_diff = np.max(param_diff)
        
        results_table.append({
            'name': config_name,
            'nruns': nruns,
            'f64_time': results_f64['mean_time'],
            'f64_std': results_f64['std_time'],
            'f32_time': results_f32['mean_time'],
            'f32_std': results_f32['std_time'],
            'speedup': speedup,
            'f64_c_time': results_f64['mean_c_time'],
            'f32_c_time': results_f32['mean_c_time'],
            'c_speedup': c_speedup,
            'f64_mem': results_f64['memory'],
            'f32_mem': results_f32['memory'],
            'memory_ratio': memory_ratio,
            'max_param_diff': max_diff,
            'f64_offset_rms': f64_offset_rms,
            'f32_offset_rms': f32_offset_rms,
            'f64_iters': results_f64['mean_iters'],
            'f32_iters': results_f32['mean_iters'],
            'f64_fev': results_f64['mean_fev'],
            'f32_fev': results_f32['mean_fev'],
        })
    
    # Print results table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()
    
    # Timing comparison
    print("TIMING COMPARISON (microseconds)")
    print("-" * 80)
    print(f"{'Config':<12} {'Runs':<8} {'Float64':<15} {'Float32':<15} {'Speedup':<10}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['name']:<12} {r['nruns']:<8} "
              f"{r['f64_time']:>7.2f} +/- {r['f64_std']:>5.2f}  "
              f"{r['f32_time']:>7.2f} +/- {r['f32_std']:>5.2f}  "
              f"{r['speedup']:>6.3f}x")
    print()
    
    # C extension timing
    print("C EXTENSION TIMING (microseconds)")
    print("-" * 80)
    print(f"{'Config':<12} {'Runs':<8} {'Float64':<15} {'Float32':<15} {'Speedup':<10}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['name']:<12} {r['nruns']:<8} "
              f"{r['f64_c_time']:>7.2f}         "
              f"{r['f32_c_time']:>7.2f}         "
              f"{r['c_speedup']:>6.3f}x")
    print()
    
    # Memory comparison
    print("MEMORY USAGE (bytes)")
    print("-" * 80)
    print(f"{'Config':<12} {'Runs':<8} {'Float64':<12} {'Float32':<12} {'Ratio':<10} {'Savings':<10}")
    print("-" * 80)
    for r in results_table:
        savings = 100 * (1 - r['memory_ratio'])
        print(f"{r['name']:<12} {r['nruns']:<8} "
              f"{r['f64_mem']:<12} {r['f32_mem']:<12} "
              f"{r['memory_ratio']:<10.3f} {savings:>6.1f}%")
    print()
    
    # Accuracy comparison
    print("NUMERICAL ACCURACY")
    print("-" * 80)
    print(f"{'Config':<12} {'Runs':<8} {'Max Diff':<12} {'F64 RMS':<12} {'F32 RMS':<12} {'Iters (F64/F32)':<18}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['name']:<12} {r['nruns']:<8} "
              f"{r['max_param_diff']:<12.6e} "
              f"{r['f64_offset_rms']:<12.6e} "
              f"{r['f32_offset_rms']:<12.6e} "
              f"{r['f64_iters']:>5.1f} / {r['f32_iters']:<5.1f}")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    avg_speedup = np.mean([r['speedup'] for r in results_table])
    avg_c_speedup = np.mean([r['c_speedup'] for r in results_table])
    avg_memory_savings = np.mean([100 * (1 - r['memory_ratio']) for r in results_table])
    max_param_diff = np.max([r['max_param_diff'] for r in results_table])
    
    print(f"Test configuration:             5-pixel Gaussians with Poisson noise")
    print(f"Average overall speedup:        {avg_speedup:.3f}x")
    print(f"Average C extension speedup:    {avg_c_speedup:.3f}x")
    print(f"Average memory savings:         {avg_memory_savings:.1f}%")
    print(f"Maximum parameter difference:   {max_param_diff:.6e}")
    print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if avg_speedup > 1.05:
        print("? Float32 is FASTER on average")
    elif avg_speedup > 0.95:
        print("? Float32 and Float64 have SIMILAR performance")
    else:
        print("? Float64 is faster on average")
    
    print(f"? Float32 saves ~{avg_memory_savings:.0f}% memory")
    
    if max_param_diff < 1e-4:
        print("? Float32 accuracy is EXCELLENT (max diff < 1e-4)")
    elif max_param_diff < 1e-3:
        print("? Float32 accuracy is GOOD (max diff < 1e-3)")
    else:
        print("? Float32 accuracy differences may be noticeable")
    
    print()
    print("Use float32 when:")
    print("  ? Memory is constrained")
    print("  ? Processing large batches")
    print("  ? ~6-7 significant digits precision is sufficient")
    print("  ? Working with minimal (5-pixel) data")
    print()
    print("Use float64 when:")
    print("  ? Maximum precision is required")
    print("  ? Working with high-precision measurements")
    print("  ? No memory constraints")
    print()
    print("=" * 80)
    print()
    print("NOTE: This benchmark uses 5-pixel Gaussians with Poisson noise,")
    print("matching the constraints in example_fmpfit_5_N. Each run uses a")
    print("different noise realization to test robustness.")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
