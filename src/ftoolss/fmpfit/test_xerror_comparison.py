#!/usr/bin/env python
"""
Test script to compare fmpfit xerror_scipy against scipy.optimize.curve_fit errors.

The xerror_scipy values are computed by mpfit using the user-provided deviate callback
to compute the Jacobian, then inverting the full Hessian matrix (J^T*J).
"""

import numpy as np
from scipy.optimize import curve_fit

try:
    import ftoolss
    from ftoolss.fmpfit import fmpfit_f32_pywrap
    print(f"ftools version: {ftoolss.__version__}")
except ImportError:
    print("ERROR: ftools not installed")
    raise


def gaussian(x, amp, vel, lw):
    """Gaussian model: amp * exp(-0.5 * ((x - vel) / lw)^2)"""
    return amp * np.exp(-0.5 * ((x - vel) / lw) ** 2)


def create_test_case(true_amp, true_vel, true_lw, n_points=30, noise_level=0.01, seed=42):
    """Create a synthetic test case with known parameters."""
    np.random.seed(seed)
    
    # x range centered on velocity
    x = np.linspace(true_vel - 3*true_lw, true_vel + 3*true_lw, n_points).astype(np.float32)
    
    # Generate Gaussian + noise
    y = gaussian(x, true_amp, true_vel, true_lw)
    y = (y + noise_level * np.random.randn(n_points)).astype(np.float32)
    
    # Uniform errors
    error = np.ones(n_points, dtype=np.float32) * noise_level
    
    return x, y, error


def fit_with_fmpfit(x, y, error, bounds_lower, bounds_upper, p0):
    """Fit using ftools fmpfit_f32_pywrap."""
    fa = {'x': x, 'y': y, 'error': error}
    parinfo = [
        {'value': p0[0], 'limits': [bounds_lower[0], bounds_upper[0]]},
        {'value': p0[1], 'limits': [bounds_lower[1], bounds_upper[1]]},
        {'value': p0[2], 'limits': [bounds_lower[2], bounds_upper[2]]},
    ]
    
    mp_ = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=fa,
                           xtol=1.0E-6, ftol=1.0E-6, gtol=1.0E-6,
                           maxiter=2000, quiet=1)
    return mp_


def fit_with_scipy(x, y, error, bounds_lower, bounds_upper, p0):
    """Fit using scipy.optimize.curve_fit for reference."""
    bounds = (bounds_lower, bounds_upper)
    popt, pcov = curve_fit(gaussian, x, y, p0=p0, sigma=error, 
                           bounds=bounds, absolute_sigma=False)  # Match fmpfit's scaling
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def run_test_case(name, true_params, bounds_upper_lw=200.0, n_points=30, noise=0.01):
    """Run a single test case and compare fmpfit against scipy."""
    true_amp, true_vel, true_lw = true_params
    
    print(f"\n{'='*80}")
    print(f"Test: {name}")
    print(f"  True params: amp={true_amp:.4f}, vel={true_vel:.2f}, lw={true_lw:.2f}")
    print(f"  Linewidth bound: {bounds_upper_lw}")
    print(f"{'='*80}")
    
    # Create test data
    x, y, error = create_test_case(true_amp, true_vel, true_lw, n_points, noise)
    
    # Bounds
    bounds_lower = [0.0, float(x.min()), 59.14]
    bounds_upper = [10.0, float(x.max()), bounds_upper_lw]
    
    # Initial guess
    p0 = [float(y.max()), float(x[np.argmax(y)]), 100.0]
    
    # Fit with fmpfit
    mp_ = fit_with_fmpfit(x, y, error, bounds_lower, bounds_upper, p0)
    
    print("\nfmpfit results:")
    print(f"  params:       amp={mp_.best_params[0]:.6f}, vel={mp_.best_params[1]:.4f}, lw={mp_.best_params[2]:.4f}")
    print(f"  xerror_scipy: amp={mp_.xerror_scipy[0]:.6f}, vel={mp_.xerror_scipy[1]:.4f}, lw={mp_.xerror_scipy[2]:.4f}")
    print(f"  status={mp_.status}, npegged={mp_.npegged}")
    
    # Fit with scipy for reference
    popt_scipy, perr_scipy = fit_with_scipy(x, y, error, bounds_lower, bounds_upper, p0)
    
    print("\nscipy results (absolute_sigma=False):")
    print(f"  params: amp={popt_scipy[0]:.6f}, vel={popt_scipy[1]:.4f}, lw={popt_scipy[2]:.4f}")
    print(f"  errors: amp={perr_scipy[0]:.6f}, vel={perr_scipy[1]:.4f}, lw={perr_scipy[2]:.4f}")
    
    # Check if linewidth at bound
    at_bound = mp_.best_params[2] >= (bounds_upper_lw - 0.01)
    print(f"\nLinewidth at bound: {at_bound}")
    
    # Compare to scipy
    diff_vs_scipy = mp_.xerror_scipy - perr_scipy
    rel_diff_scipy = np.abs(diff_vs_scipy) / (perr_scipy + 1e-10)
    
    print("\nComparison fmpfit vs scipy:")
    print(f"  Absolute diff: amp={diff_vs_scipy[0]:.6e}, vel={diff_vs_scipy[1]:.4e}, lw={diff_vs_scipy[2]:.4e}")
    print(f"  Relative diff: amp={rel_diff_scipy[0]:.2%}, vel={rel_diff_scipy[1]:.2%}, lw={rel_diff_scipy[2]:.2%}")
    
    if np.max(rel_diff_scipy) > 0.01:  # >1% difference
        print("\n*** NOTE: >1% difference between fmpfit and scipy (expected at bounds) ***")
    
    return {
        'name': name,
        'at_bound': at_bound,
        'fmpfit_params': mp_.best_params,
        'xerror_scipy': mp_.xerror_scipy,
        'scipy_errors': perr_scipy,
        'rel_diff_scipy': rel_diff_scipy,
    }


def main():
    print("Testing fmpfit xerror_scipy vs scipy curve_fit errors")
    print("=" * 80)
    
    results = []
    
    # Test 1: Normal fit (linewidth within bounds)
    results.append(run_test_case(
        "Normal fit (lw within bounds)",
        true_params=(0.5, 100.0, 150.0),
        bounds_upper_lw=200.0
    ))
    
    # Test 2: Linewidth at upper bound
    results.append(run_test_case(
        "Linewidth at upper bound (true_lw > bound)",
        true_params=(0.5, 100.0, 300.0),
        bounds_upper_lw=200.0
    ))
    
    # Test 3: Linewidth exactly at bound
    results.append(run_test_case(
        "Linewidth exactly at bound",
        true_params=(0.5, 100.0, 200.0),
        bounds_upper_lw=200.0
    ))
    
    # Test 4: Very wide Gaussian (extreme bound violation)
    results.append(run_test_case(
        "Very wide Gaussian (extreme)",
        true_params=(0.5, 100.0, 500.0),
        bounds_upper_lw=200.0
    ))
    
    # Test 5: Narrow Gaussian at lower bound
    results.append(run_test_case(
        "Narrow Gaussian (at lower bound)",
        true_params=(0.5, 100.0, 50.0),
        bounds_upper_lw=200.0
    ))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for r in results:
        max_rel_diff = np.max(r['rel_diff_scipy'])
        status = "NOTE" if max_rel_diff > 0.01 else "OK"
        print(f"  {r['name']:<45} | at_bound={r['at_bound']!s:<5} | max_rel_diff={max_rel_diff:.2%} | {status}")


if __name__ == "__main__":
    main()
