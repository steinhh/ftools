#!/usr/bin/env python
"""
Benchmark fmpfit_wrap overhead

Measures the overhead of calling fmpfit C extension versus pure Python
computation of Gaussian model and residuals. This helps quantify the
Python/C interface cost.
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftools import fmpfit_wrap


def benchmark_single_fmpfit_call(x, y, error, p0, bounds, n_iterations):
    """Benchmark single fmpfit call (includes fitting work)"""
    
    # Prepare parinfo and functkw once
    parinfo = [{'value': p0[i], 'limits': bounds[i]} for i in range(len(p0))]
    functkw = {'x': x, 'y': y, 'error': error}
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = fmpfit_wrap(deviate_type=0, parinfo=parinfo, functkw=functkw)
    elapsed = time.perf_counter() - start
    
    return elapsed, result


def fmpfit_wrap_stub(deviate_type=0, parinfo=None, functkw=None, **kwargs):  # noqa: ARG001
    """
    Stub version of fmpfit_wrap that does all Python overhead but skips C call.
    Used to measure pure overhead of wrapper function.
    """
    import numpy as np
    
    # Validate inputs (same as real function)
    if functkw is None:
        raise ValueError("functkw must be provided")
    if parinfo is None:
        raise ValueError("parinfo must be provided")
    if 'x' not in functkw or 'y' not in functkw or 'error' not in functkw:
        raise ValueError("functkw must contain 'x', 'y', 'error'")
    
    # Extract arrays
    x = np.asarray(functkw['x'], dtype=np.float64)
    _ = np.asarray(functkw['y'], dtype=np.float64)
    _ = np.asarray(functkw['error'], dtype=np.float64)
    
    mpoints = len(x)
    npar = len(parinfo)
    
    # Extract initial parameter values and bounds
    p0 = np.zeros(npar, dtype=np.float64)
    bounds = np.zeros((npar, 2), dtype=np.float64)
    
    for i, pinfo in enumerate(parinfo):
        if 'value' not in pinfo:
            raise ValueError(f"parinfo[{i}] must contain 'value'")
        if 'limits' not in pinfo:
            raise ValueError(f"parinfo[{i}] must contain 'limits'")
        
        p0[i] = float(pinfo['value'])
        limits = pinfo['limits']
        if len(limits) != 2:
            raise ValueError(f"parinfo[{i}]['limits'] must have 2 elements")
        bounds[i, 0] = float(limits[0])
        bounds[i, 1] = float(limits[1])
        
        if bounds[i, 0] >= bounds[i, 1]:
            raise ValueError(f"parinfo[{i}]: lower bound must be < upper bound")
    
    # SKIP C extension call - return mock result
    # result_dict = fmpfit_ext.fmpfit(...)
    
    # Create mock result object with minimal values
    class MockResult:
        def __init__(self):
            self.best_params = p0
            self.bestnorm = 0.0
            self.orignorm = 0.0
            self.niter = 0
            self.nfev = 0
            self.status = 1
            self.npar = npar
            self.nfree = npar
            self.npegged = 0
            self.nfunc = mpoints
            self.resid = np.zeros(mpoints)
            self.xerror = np.zeros(npar)
            self.covar = np.zeros((npar, npar))
    
    return MockResult()


def benchmark_fmpfit_minimal_work(x, y, error, converged_params, n_iterations):
    """
    Benchmark fmpfit when starting from converged solution.
    This minimizes the fitting work to isolate overhead.
    """
    
    # Use converged parameters as initial guess - should converge in 1 iteration
    parinfo = [
        {'value': converged_params[i], 'limits': [[0.0, 10.0], [-5.0, 5.0], [0.1, 5.0]][i]} 
        for i in range(len(converged_params))
    ]
    functkw = {'x': x, 'y': y, 'error': error}
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = fmpfit_wrap(deviate_type=0, parinfo=parinfo, functkw=functkw)
    elapsed = time.perf_counter() - start
    
    return elapsed, result


def benchmark_fmpfit_stub_overhead(x, y, error, converged_params, n_iterations):
    """
    Benchmark stub version to measure pure Python overhead.
    """
    parinfo = [
        {'value': converged_params[i], 'limits': [[0.0, 10.0], [-5.0, 5.0], [0.1, 5.0]][i]} 
        for i in range(len(converged_params))
    ]
    functkw = {'x': x, 'y': y, 'error': error}
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = fmpfit_wrap_stub(deviate_type=0, parinfo=parinfo, functkw=functkw)
    elapsed = time.perf_counter() - start
    
    return elapsed, result


def main():
    # Parse command line argument
    if len(sys.argv) > 1:
        try:
            n_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid argument '{sys.argv[1]}'. Expected integer.")
            print("Usage: python benchmark_fmpfit_overhead.py [N]")
            print("  N: number of iterations (default: 1000)")
            sys.exit(1)
    else:
        n_iterations = 1000
    
    # Generate synthetic data
    np.random.seed(42)
    n_points = 100
    x = np.linspace(-5, 5, n_points)
    
    # True parameters for Gaussian: amp=2.5, mean=1.0, sigma=0.8
    true_params = [2.5, 1.0, 0.8]
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
    noise = np.random.normal(0, 0.1, len(x))  # noqa: NPY002
    y = y_true + noise
    error = np.ones_like(y) * 0.1
    
    # Initial guess (deliberately off)
    p0 = [1.5, 0.5, 1.0]
    bounds = [[0.0, 10.0], [-5.0, 5.0], [0.1, 5.0]]
    
    print("=" * 70)
    print("FMPFIT Overhead Benchmark")
    print("=" * 70)
    print(f"\nData points: {n_points}")
    print(f"Iterations: {n_iterations}")
    print()
    
    # First, do a single fit to get converged parameters
    parinfo = [{'value': p0[i], 'limits': bounds[i]} for i in range(len(p0))]
    functkw = {'x': x, 'y': y, 'error': error}
    result_initial = fmpfit_wrap(deviate_type=0, parinfo=parinfo, functkw=functkw)
    converged_params = result_initial.best_params
    
    print("Initial fit (for reference):")
    print(f"  Status: {result_initial.status}")
    print(f"  Iterations: {result_initial.niter}")
    print(f"  Function evaluations: {result_initial.nfev}")
    print(f"  Converged params: {converged_params}")
    print()
    
    # Benchmark 1: Full fmpfit call from initial guess
    print("-" * 70)
    print("Benchmark 1: Full fmpfit fit (from initial guess)")
    print("-" * 70)
    elapsed_full, result_full = benchmark_single_fmpfit_call(
        x, y, error, p0, bounds, n_iterations
    )
    time_per_fit_full = elapsed_full / n_iterations * 1e6
    print(f"Total time: {elapsed_full:.4f} s")
    print(f"Time per fit: {time_per_fit_full:.2f} us")
    print(f"Avg iterations: {result_full.niter}")
    print(f"Avg function evaluations: {result_full.nfev}")
    print()
    
    # Benchmark 2: Minimal work fmpfit call (from converged solution)
    print("-" * 70)
    print("Benchmark 2: Minimal fmpfit call (from converged solution)")
    print("-" * 70)
    elapsed_minimal, result_minimal = benchmark_fmpfit_minimal_work(
        x, y, error, converged_params, n_iterations
    )
    time_per_fit_minimal = elapsed_minimal / n_iterations * 1e6
    print(f"Total time: {elapsed_minimal:.4f} s")
    print(f"Time per fit: {time_per_fit_minimal:.2f} us")
    print(f"Avg iterations: {result_minimal.niter}")
    print(f"Avg function evaluations: {result_minimal.nfev}")
    print()
    
    # Benchmark 3: Stub overhead (no C call)
    print("-" * 70)
    print("Benchmark 3: Python wrapper overhead (stub, no C call)")
    print("-" * 70)
    elapsed_stub, _ = benchmark_fmpfit_stub_overhead(
        x, y, error, converged_params, n_iterations
    )
    time_per_stub = elapsed_stub / n_iterations * 1e6
    print(f"Total time: {elapsed_stub:.4f} s")
    print(f"Time per call: {time_per_stub:.2f} us")
    print()
    
    # Analysis
    print("=" * 70)
    print("Overhead Analysis")
    print("=" * 70)
    
    # Calculate per-evaluation time
    time_per_eval_c = time_per_fit_minimal / result_minimal.nfev
    c_extension_overhead = time_per_fit_minimal - time_per_stub
    
    print("\nOverhead breakdown:")
    print(f"  Python wrapper only (stub):     {time_per_stub:.2f} us")
    print(f"  Minimal fmpfit call:            {time_per_fit_minimal:.2f} us")
    print(f"  C extension overhead:           {c_extension_overhead:.2f} us")
    print(f"    ({result_minimal.nfev} function evaluations @ {time_per_eval_c:.2f} us/eval)")
    
    print("\nFull fit breakdown:")
    print(f"  Total time:                     {time_per_fit_full:.2f} us")
    print(f"  Python wrapper overhead:        {time_per_stub:.2f} us ({time_per_stub/time_per_fit_full*100:.1f}%)")
    print(f"  C extension work:               {time_per_fit_full - time_per_stub:.2f} us ({(time_per_fit_full - time_per_stub)/time_per_fit_full*100:.1f}%)")
    print(f"    - {result_full.nfev} function evaluations @ {time_per_eval_c:.2f} us = {result_full.nfev * time_per_eval_c:.2f} us")
    print(f"    - MPFIT algorithm overhead:     {time_per_fit_full - time_per_stub - result_full.nfev * time_per_eval_c:.2f} us")
    
    print("\nSummary:")
    print(f"  Python wrapper overhead: {time_per_stub:.2f} us per call")
    print(f"  C extension overhead: {c_extension_overhead:.2f} us per call")
    print(f"  Full fit from initial guess: {time_per_fit_full:.2f} us ({time_per_fit_full/1000:.2f} ms)")
    print(f"  with {result_full.nfev} function evaluations")
    print()


if __name__ == "__main__":
    main()
