#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multithreading Speedup Benchmark for fmpfit_f32

Tests the GIL-release performance of fmpfit_f32_ext by running multiple
fits concurrently and measuring wall-clock speedup vs sequential execution.

The fmpfit_f32_ext.c releases the GIL around the core C computation using
Py_BEGIN_ALLOW_THREADS/Py_END_ALLOW_THREADS, allowing true parallel execution.
"""

import numpy as np
import time
import threading
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftools.fmpfit import fmpfit_f32_wrap


def generate_test_data(seed, n_points=50):
    """Generate synthetic Gaussian data with Poisson noise"""
    np.random.seed(seed)
    
    # True parameters: amplitude, mean, sigma
    true_params = np.array([2.5, 0.0, 0.8], dtype=np.float32)
    
    # Generate x values
    x = np.linspace(-5, 5, n_points, dtype=np.float32)
    
    # True Gaussian
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
    
    # Add Poisson noise
    y = np.random.poisson(np.maximum(y_true, 1.0)).astype(np.float32)
    error = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)
    
    return x, y, error, true_params


def run_single_fit(seed, n_points=50):
    """Run a single fit - returns (elapsed_time, result)"""
    x, y, error, true_params = generate_test_data(seed, n_points)
    
    # Initial guess and bounds
    parinfo = [
        {'value': 2.0, 'limits': [0.0, 10.0]},
        {'value': 0.0, 'limits': [-5.0, 5.0]},
        {'value': 1.0, 'limits': [0.1, 5.0]}
    ]
    
    functkw = {'x': x, 'y': y, 'error': error}
    
    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    
    return elapsed, result


def benchmark_sequential(n_fits, n_points=50):
    """Run fits sequentially (baseline)"""
    times = []
    
    print(f"\n{'='*60}")
    print(f"Sequential Execution: {n_fits} fits")
    print(f"{'='*60}")
    
    total_start = time.perf_counter()
    
    for i in range(n_fits):
        elapsed, result = run_single_fit(seed=i, n_points=n_points)
        times.append(elapsed)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Fit {i+1:3d}: {elapsed*1000:6.2f} ms | status={result.status} niter={result.niter}")
    
    total_elapsed = time.perf_counter() - total_start
    
    print(f"\nTotal time: {total_elapsed:.3f} s")
    print(f"Mean time per fit: {np.mean(times)*1000:.2f} ms")
    print(f"Std time per fit: {np.std(times)*1000:.2f} ms")
    
    return total_elapsed, times


def benchmark_threaded(n_fits, n_threads, n_points=50):
    """Run fits concurrently using ThreadPoolExecutor"""
    times = []
    
    print(f"\n{'='*60}")
    print(f"Threaded Execution: {n_fits} fits on {n_threads} threads")
    print(f"{'='*60}")
    
    total_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit all tasks
        futures = {executor.submit(run_single_fit, seed=i, n_points=n_points): i 
                   for i in range(n_fits)}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            elapsed, result = future.result()
            times.append(elapsed)
            completed += 1
            
            if completed % 10 == 0 or completed == 1:
                print(f"  Completed {completed:3d}/{n_fits} | Last fit: {elapsed*1000:6.2f} ms | status={result.status}")
    
    total_elapsed = time.perf_counter() - total_start
    
    print(f"\nTotal time: {total_elapsed:.3f} s")
    print(f"Mean time per fit: {np.mean(times)*1000:.2f} ms")
    print(f"Std time per fit: {np.std(times)*1000:.2f} ms")
    
    return total_elapsed, times


def benchmark_threading_overhead(n_threads, n_points=50):
    """Measure threading overhead with minimal workload"""
    def dummy_worker():
        time.sleep(0.001)  # 1ms sleep
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(dummy_worker) for _ in range(n_threads)]
        for future in as_completed(futures):
            future.result()
    elapsed = time.perf_counter() - start
    
    return elapsed


def run_speedup_analysis(n_fits=40, n_points=50, max_threads=8):
    """
    Run comprehensive speedup analysis
    
    Tests thread counts from 1 to max_threads and compares to sequential baseline.
    """
    print("\n" + "="*70)
    print("FMPFIT_F32 MULTITHREADING SPEEDUP BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of fits: {n_fits}")
    print(f"  Data points per fit: {n_points}")
    print(f"  Max threads tested: {max_threads}")
    
    # Get CPU count
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"  Available CPUs: {cpu_count}")
    
    # Run sequential baseline
    seq_time, seq_times = benchmark_sequential(n_fits, n_points)
    
    # Test different thread counts
    results = []
    if max_threads <= 4:
        thread_counts = list(range(1, max_threads + 1))
    elif max_threads <= 8:
        thread_counts = [1, 2, 4, 6, 8]
    else:
        # For 9+ threads, test: 1, 2, 4, 6, 8, 10, 12, ...
        thread_counts = [1, 2, 4, 6, 8] + list(range(10, max_threads + 1, 2))
    thread_counts = [t for t in thread_counts if t <= max_threads]
    
    for n_threads in thread_counts:
        threaded_time, threaded_times = benchmark_threaded(n_fits, n_threads, n_points)
        speedup = seq_time / threaded_time
        efficiency = speedup / n_threads * 100
        
        results.append({
            'threads': n_threads,
            'time': threaded_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
    
    # Summary table
    print("\n" + "="*70)
    print("SPEEDUP SUMMARY")
    print("="*70)
    print(f"\nSequential baseline: {seq_time:.3f} s ({seq_time/n_fits*1000:.2f} ms/fit)")
    print(f"\n{'Threads':>8} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['threads']:8d} {r['time']:10.3f} {r['speedup']:10.2f}x {r['efficiency']:11.1f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    best_speedup = max(results, key=lambda x: x['speedup'])
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    
    print(f"\nBest speedup: {best_speedup['speedup']:.2f}x with {best_speedup['threads']} threads")
    print(f"Best efficiency: {best_efficiency['efficiency']:.1f}% with {best_efficiency['threads']} thread(s)")
    
    # Check if GIL release is working
    if results[-1]['speedup'] > 1.5:
        print(f"\n? GIL release is working effectively!")
        print(f"  Achieved {results[-1]['speedup']:.2f}x speedup with {results[-1]['threads']} threads")
    else:
        print(f"\n? Limited speedup detected ({results[-1]['speedup']:.2f}x with {results[-1]['threads']} threads)")
        print(f"  This may indicate GIL contention or I/O bottlenecks")
    
    return results


def run_verification_test():
    """Quick test to verify all threads produce correct results"""
    print("\n" + "="*70)
    print("VERIFICATION TEST: Correctness Check")
    print("="*70)
    
    n_fits = 20
    
    # Pre-generate all test data with fixed seeds
    print("\nGenerating test data...")
    test_datasets = []
    for i in range(n_fits):
        x, y, error, true_params = generate_test_data(seed=i, n_points=50)
        test_datasets.append((x.copy(), y.copy(), error.copy()))
    
    def run_fit_with_data(data_idx):
        """Run fit on pre-generated data"""
        x, y, error = test_datasets[data_idx]
        
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-5.0, 5.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]}
        ]
        
        functkw = {'x': x, 'y': y, 'error': error}
        
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
        return result
    
    # Run sequential
    print("Running sequential fits...")
    seq_results = []
    for i in range(n_fits):
        result = run_fit_with_data(i)
        seq_results.append(result.best_params.copy())
    
    # Run threaded
    print("Running threaded fits...")
    threaded_results = [None] * n_fits
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_fit_with_data, i): i for i in range(n_fits)}
        
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            threaded_results[i] = result.best_params.copy()
    
    # Compare results
    print("\nComparing results...")
    max_diff = 0.0
    all_match = True
    
    for i in range(n_fits):
        diff = np.max(np.abs(seq_results[i] - threaded_results[i]))
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            all_match = False
            print(f"  Fit {i}: diff = {diff:.2e} (MISMATCH)")
    
    if all_match:
        print(f"\n? All results match! (max diff: {max_diff:.2e})")
    else:
        print(f"\n? Some results differ (max diff: {max_diff:.2e})")
        print("  This may indicate race conditions or numerical instability")
    
    return all_match


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark multithreading speedup for fmpfit_f32')
    parser.add_argument('--fits', type=int, default=40, help='Number of fits to run (default: 40)')
    parser.add_argument('--points', type=int, default=50, help='Data points per fit (default: 50)')
    parser.add_argument('--max-threads', type=int, default=8, help='Maximum threads to test (default: 8)')
    parser.add_argument('--verify-only', action='store_true', help='Only run verification test')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Just run verification
        success = run_verification_test()
        sys.exit(0 if success else 1)
    else:
        # Run verification first
        print("Running verification test first...")
        success = run_verification_test()
        
        if not success:
            print("\n? Verification failed! Proceeding with caution...")
        
        # Run full speedup analysis
        results = run_speedup_analysis(
            n_fits=args.fits,
            n_points=args.points,
            max_threads=args.max_threads
        )
