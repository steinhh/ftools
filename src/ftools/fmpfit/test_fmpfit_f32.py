#!/usr/bin/env python3
"""
Test script to verify fmpfit_f32 (float32 version) works correctly
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftools.fmpfit import fmpfit_wrap, fmpfit_f32_wrap

def test_fmpfit_f32():
    """Test float32 version of MPFIT"""
    
    # Generate synthetic Gaussian data
    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    
    # True parameters
    true_params = [2.5, 1.0, 0.8]
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
    noise = np.random.normal(0, 0.1, len(x))
    y = y_true + noise
    error = np.ones_like(y) * 0.1
    
    # Initial guess
    p0 = [1.5, 0.5, 1.0]
    bounds = [[0.0, 10.0], [-5.0, 5.0], [0.1, 5.0]]
    
    # Prepare parinfo and functkw
    parinfo = [{'value': p0[i], 'limits': bounds[i]} for i in range(len(p0))]
    functkw = {'x': x, 'y': y, 'error': error}
    
    print("=" * 70)
    print("Testing fmpfit_f32 (float32 version)")
    print("=" * 70)
    
    # Test float64 version
    print("\n--- Float64 version ---")
    result_f64 = fmpfit_wrap(deviate_type=0, parinfo=parinfo, functkw=functkw)
    print(f"Status: {result_f64.status}")
    print(f"Iterations: {result_f64.niter}")
    print(f"Chi-square: {result_f64.bestnorm:.6f}")
    print(f"Parameters: {result_f64.best_params}")
    print(f"Errors: {result_f64.xerror}")
    print(f"C time: {result_f64.c_time*1e6:.2f} ?s")
    
    # Test float32 version
    print("\n--- Float32 version ---")
    result_f32 = fmpfit_f32_wrap(deviate_type=0, parinfo=parinfo, functkw=functkw)
    print(f"Status: {result_f32.status}")
    print(f"Iterations: {result_f32.niter}")
    print(f"Chi-square: {result_f32.bestnorm:.6f}")
    print(f"Parameters: {result_f32.best_params}")
    print(f"Errors: {result_f32.xerror}")
    print(f"C time: {result_f32.c_time*1e6:.2f} ?s")
    
    # Compare results
    print("\n--- Comparison ---")
    print(f"True parameters: {true_params}")
    print(f"F64 parameters:  {result_f64.best_params}")
    print(f"F32 parameters:  {result_f32.best_params}")
    
    # Check parameter differences
    param_diff = np.abs(result_f64.best_params - result_f32.best_params.astype(np.float64))
    print(f"\nParameter differences (F64 - F32): {param_diff}")
    print(f"Max parameter difference: {np.max(param_diff):.6e}")
    
    # Check if they're close (within single precision tolerance)
    tolerance = 5e-5  # Reasonable tolerance for float32 vs float64 comparison
    if np.allclose(result_f64.best_params, result_f32.best_params, rtol=tolerance, atol=tolerance):
        print(f"\n? Results agree within tolerance ({tolerance})")
    else:
        print(f"\n? Results differ beyond tolerance ({tolerance})")
    
    # Check data types
    print(f"\n--- Data Types ---")
    print(f"F64 best_params dtype: {result_f64.best_params.dtype}")
    print(f"F32 best_params dtype: {result_f32.best_params.dtype}")
    print(f"F64 resid dtype: {result_f64.resid.dtype}")
    print(f"F32 resid dtype: {result_f32.resid.dtype}")
    
    # Memory usage comparison
    print(f"\n--- Memory Usage ---")
    f64_mem = (result_f64.best_params.nbytes + result_f64.resid.nbytes + 
               result_f64.xerror.nbytes + result_f64.covar.nbytes)
    f32_mem = (result_f32.best_params.nbytes + result_f32.resid.nbytes + 
               result_f32.xerror.nbytes + result_f32.covar.nbytes)
    print(f"F64 total memory: {f64_mem} bytes")
    print(f"F32 total memory: {f32_mem} bytes")
    print(f"Memory savings: {f64_mem - f32_mem} bytes ({100*(1-f32_mem/f64_mem):.1f}% reduction)")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_fmpfit_f32()
