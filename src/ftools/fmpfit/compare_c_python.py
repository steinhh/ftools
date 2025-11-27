"""Module for comparing fitting results using different optimization methods.

Compares scipy.optimize.curve_fit with fmpfit_f64_wrap for Gaussian fitting.

Usage:
    python compare_c_python.py [N]
    
    N: number of comparison runs (default: 10)
"""

import sys
import os

# Handle imports when run directly from within the package
if __name__ == "__main__":
    # Add the src directory to path to allow direct execution
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _src_dir = os.path.dirname(os.path.dirname(os.path.dirname(_this_dir)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

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
        popt_scipy, pcov_scipy, infodict, mesg, ier = curve_fit(
            gaussian, x, y, p0=p0, bounds=scipy_bounds,
            sigma=error, absolute_sigma=True, jac=gaussian_jacobian,
            full_output=True
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
            'nfev': infodict.get('nfev', 0),
            'success': True
        }
    except Exception as e:
        results['scipy'] = {
            'params': np.array([np.nan] * len(p0)),
            'errors': np.array([np.nan] * len(p0)),
            'chisq': np.nan,
            'reduced_chisq': np.nan,
            'nfev': 0,
            'success': False,
            'error': str(e)
        }
    
    # --- fmpfit_f64_wrap ---
    # Note: fmpfit uses analytical derivatives for the Gaussian model internally.
    # The nfev count includes both residual and Jacobian evaluations.
    try:
        parinfo = [
            {'value': p0[i], 'limits': list(bounds[i])}
            for i in range(len(p0))
        ]
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_mpfit = fmpfit_f64_wrap(
            deviate_type=0,  # Gaussian (uses analytical derivatives)
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


def run_comparison_n_times(n_runs, seed=42):
    """Run comparison N times with 5-pixel data covering the FWHM."""
    rng = np.random.default_rng(seed)
    
    # True parameters
    true_params = [2.5, 0.0, 0.8]  # amplitude, mean, sigma
    noise_level = 0.1
    
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ~ 2.355 * sigma
    # Cover FWHM with 5 pixels centered on mean
    fwhm = 2.355 * true_params[2]
    half_fwhm = fwhm / 2
    x = np.linspace(true_params[1] - half_fwhm, true_params[1] + half_fwhm, 5)
    
    # Initial guess and bounds
    p0 = [2.0, 0.0, 1.0]
    bounds = [(0.0, 10.0), (-2.0, 2.0), (0.1, 5.0)]
    
    # Storage for results
    results_list = []
    
    for i in range(n_runs):
        y_true = gaussian(x, *true_params)
        y = y_true + rng.normal(0, noise_level, len(x))
        error = np.ones_like(y) * noise_level
        
        result = compare_scipy_fmpfit(x, y, error, p0, bounds)
        result['run'] = i + 1
        results_list.append(result)
    
    # Print header
    print("=" * 120)
    print(f"Comparison: scipy.optimize.curve_fit vs fmpfit_f64_wrap ({n_runs} runs, 5 pixels covering FWHM)")
    print(f"True parameters: amplitude={true_params[0]}, mean={true_params[1]}, sigma={true_params[2]}")
    print(f"FWHM = {fwhm:.4f}, x = {x}")
    print("=" * 120)
    
    # Table header
    print(f"\n{'Run':>4} | {'scipy_amp':>10} {'scipy_mu':>10} {'scipy_sig':>10} {'sc_nfev':>7} | "
          f"{'mpfit_amp':>10} {'mpfit_mu':>10} {'mpfit_sig':>10} {'mp_nfev':>7} | "
          f"{'amp_diff%':>9} {'mu_diff%':>9} {'sig_diff%':>9} | {'match':>5}")
    print("-" * 130)
    
    # Print each run
    n_matches = 0
    for r in results_list:
        if r['scipy']['success'] and r['fmpfit']['success']:
            sp = r['scipy']['params']
            mp = r['fmpfit']['params']
            pct = r['comparison']['param_diff_percent']
            match = r['comparison']['match']
            sc_nfev = r['scipy']['nfev']
            mp_nfev = r['fmpfit']['nfev']
            if match:
                n_matches += 1
            print(f"{r['run']:>4} | {sp[0]:>10.6f} {sp[1]:>10.6f} {sp[2]:>10.6f} {sc_nfev:>7} | "
                  f"{mp[0]:>10.6f} {mp[1]:>10.6f} {mp[2]:>10.6f} {mp_nfev:>7} | "
                  f"{pct[0]:>9.4f} {pct[1]:>9.4f} {pct[2]:>9.4f} | {'Yes' if match else 'No':>5}")
        else:
            scipy_err = 'OK' if r['scipy']['success'] else 'FAIL'
            mpfit_err = 'OK' if r['fmpfit']['success'] else 'FAIL'
            print(f"{r['run']:>4} | scipy: {scipy_err}, mpfit: {mpfit_err}")
    
    print("-" * 130)
    print(f"Summary: {n_matches}/{n_runs} runs matched (within 0.1% tolerance)")
    
    return results_list


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_comparison_n_times(n)

