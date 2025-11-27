"""Module for comparing fitting results using different optimization methods.

Compares scipy.optimize.curve_fit with fmpfit_f64_wrap for Gaussian fitting.
"""

import numpy as np
from scipy.optimize import curve_fit
from ftools.fmpfit import fmpfit_f64_wrap


def gaussian(x, i0, mu, sigma):
    """Gaussian function: i0 * exp(-((x - mu)^2) / (2 * sigma^2))"""
    return i0 * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_jacobian(x, i0, mu, sigma):
    """Jacobian of the Gaussian function for curve_fit."""
    inv_sigma = 1.0 / sigma
    inv_sigma2 = inv_sigma * inv_sigma
    exp_term = gaussian(x, i0, mu, sigma)
    d_i0 = exp_term / i0
    xmu = x - mu
    d_mu = exp_term * xmu * inv_sigma2
    d_sigma = d_mu * xmu * inv_sigma
    return np.stack((d_i0, d_mu, d_sigma), axis=-1)


def compare_scipy_fmpfit(x, y, error, p0, bounds):
    """
    Compare fitting results from scipy.optimize.curve_fit and fmpfit_f64_wrap.
    
    Parameters
    ----------
    x : ndarray
        Independent variable (1D array)
    y : ndarray
        Dependent variable (1D array)
    error : ndarray
        Measurement uncertainties (1D array)
    p0 : list
        Initial parameter guesses [amplitude, mean, sigma]
    bounds : list of tuples
        Parameter bounds [(low, high), ...] for each parameter
    
    Returns
    -------
    dict
        Dictionary with results from both methods and comparison metrics
    """
    results = {}
    
    # --- scipy.optimize.curve_fit ---
    try:
        scipy_bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
        popt_scipy, pcov_scipy = curve_fit(
            gaussian, x, y, p0=p0, bounds=scipy_bounds,
            sigma=error, absolute_sigma=True, jac=gaussian_jacobian
        )
        perr_scipy = np.sqrt(np.diag(pcov_scipy))
        residuals_scipy = y - gaussian(x, *popt_scipy)
        chisq_scipy = np.sum((residuals_scipy / error) ** 2)
        dof = len(x) - len(p0)
        reduced_chisq_scipy = chisq_scipy / dof if dof > 0 else np.nan
        
        results['scipy'] = {
            'params': popt_scipy,
            'errors': perr_scipy,
            'chisq': chisq_scipy,
            'reduced_chisq': reduced_chisq_scipy,
            'success': True
        }
    except Exception as e:
        results['scipy'] = {
            'params': np.array([np.nan] * len(p0)),
            'errors': np.array([np.nan] * len(p0)),
            'chisq': np.nan,
            'reduced_chisq': np.nan,
            'success': False,
            'error': str(e)
        }
    
    # --- fmpfit_f64_wrap ---
    try:
        parinfo = [
            {'value': p0[i], 'limits': list(bounds[i])}
            for i in range(len(p0))
        ]
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_mpfit = fmpfit_f64_wrap(
            deviate_type=0,  # Gaussian
            parinfo=parinfo,
            functkw=functkw
        )
        
        results['fmpfit'] = {
            'params': result_mpfit.best_params,
            'errors': result_mpfit.xerror,
            'chisq': result_mpfit.bestnorm,
            'reduced_chisq': result_mpfit.bestnorm / (len(x) - len(p0)) if len(x) > len(p0) else np.nan,
            'niter': result_mpfit.niter,
            'nfev': result_mpfit.nfev,
            'status': result_mpfit.status,
            'c_time': result_mpfit.c_time,
            'success': result_mpfit.status >= 0
        }
    except Exception as e:
        results['fmpfit'] = {
            'params': np.array([np.nan] * len(p0)),
            'errors': np.array([np.nan] * len(p0)),
            'chisq': np.nan,
            'reduced_chisq': np.nan,
            'success': False,
            'error': str(e)
        }
    
    # --- Comparison ---
    if results['scipy']['success'] and results['fmpfit']['success']:
        param_diff = results['scipy']['params'] - results['fmpfit']['params']
        results['comparison'] = {
            'param_diff': param_diff,
            'param_diff_percent': 100 * np.abs(param_diff) / np.abs(results['scipy']['params']),
            'chisq_diff': results['scipy']['chisq'] - results['fmpfit']['chisq'],
            'match': np.allclose(results['scipy']['params'], results['fmpfit']['params'], rtol=1e-3)
        }
    
    return results


def run_comparison_example():
    """Run a comparison example with synthetic Gaussian data."""
    # Generate synthetic data with Gaussian noise
    rng = np.random.default_rng(42)
    x = np.linspace(-5, 5, 100)
    true_params = [2.5, 1.0, 0.8]  # amplitude, mean, sigma
    y_true = gaussian(x, *true_params)
    noise_level = 0.1
    y = y_true + rng.normal(0, noise_level, len(x))
    error = np.ones_like(y) * noise_level
    
    # Initial guess and bounds
    p0 = [2.0, 0.5, 1.0]
    bounds = [(0.0, 10.0), (-5.0, 5.0), (0.1, 5.0)]
    
    # Run comparison
    results = compare_scipy_fmpfit(x, y, error, p0, bounds)
    
    # Print results
    print("=" * 60)
    print("Comparison: scipy.optimize.curve_fit vs fmpfit_f64_wrap")
    print("=" * 60)
    print(f"\nTrue parameters: amplitude={true_params[0]}, mean={true_params[1]}, sigma={true_params[2]}")
    print(f"Initial guess:   amplitude={p0[0]}, mean={p0[1]}, sigma={p0[2]}")
    
    print("\n--- scipy.optimize.curve_fit ---")
    if results['scipy']['success']:
        print(f"  Parameters: {results['scipy']['params']}")
        print(f"  Errors:     {results['scipy']['errors']}")
        print(f"  Chi-square: {results['scipy']['chisq']:.6f}")
        print(f"  Reduced chi-sq: {results['scipy']['reduced_chisq']:.6f}")
    else:
        print(f"  FAILED: {results['scipy'].get('error', 'Unknown error')}")
    
    print("\n--- fmpfit_f64_wrap ---")
    if results['fmpfit']['success']:
        print(f"  Parameters: {results['fmpfit']['params']}")
        print(f"  Errors:     {results['fmpfit']['errors']}")
        print(f"  Chi-square: {results['fmpfit']['chisq']:.6f}")
        print(f"  Reduced chi-sq: {results['fmpfit']['reduced_chisq']:.6f}")
        print(f"  Iterations: {results['fmpfit']['niter']}")
        print(f"  Func evals: {results['fmpfit']['nfev']}")
        print(f"  C time:     {results['fmpfit']['c_time']*1000:.3f} ms")
    else:
        print(f"  FAILED: {results['fmpfit'].get('error', 'Unknown error')}")
    
    if 'comparison' in results:
        print("\n--- Comparison ---")
        print(f"  Parameter difference: {results['comparison']['param_diff']}")
        print(f"  Percent difference:   {results['comparison']['param_diff_percent']}%")
        print(f"  Chi-square diff:      {results['comparison']['chisq_diff']:.6e}")
        print(f"  Parameters match:     {results['comparison']['match']}")
    
    return results


if __name__ == "__main__":
    run_comparison_example()

