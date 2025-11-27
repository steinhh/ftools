"""Module for comparing fitting results using different optimization methods.

Compares scipy.optimize.curve_fit with fmpfit_f64_wrap for Gaussian fitting.

Usage:
    python compare_c_python.py [N]
    
    N: number of comparison runs (default: 10)
"""
# 
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PNG output
import matplotlib.pyplot as plt


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


def compare_scipy_fmpfit(x, y, error, p0, bounds, pixel_spacing=None):
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
        Initial parameter guesses [intensity, velocity, width]
    bounds : list of tuples
        Parameter bounds [(low, high), ...] for each parameter
    pixel_spacing : float, optional
        Pixel spacing for computing mu difference in pixels
    
    Returns
    -------
    dict
        Dictionary with results from both methods and comparison metrics
    """
    results = {}
    
    # --- scipy.optimize.curve_fit ---
    try:
        scipy_bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
        
        # Wrap jacobian to count calls
        jac_count = [0]  # Use list to allow mutation in nested function
        def counted_jacobian(x, i0, mu, sigma):
            jac_count[0] += 1
            return gaussian_jacobian(x, i0, mu, sigma)
        
        popt_scipy, pcov_scipy, infodict, mesg, ier = curve_fit( # NOSONAR
            gaussian, x, y, p0=p0, bounds=scipy_bounds,
            sigma=error, absolute_sigma=True, jac=counted_jacobian,
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
            'njac': jac_count[0],
            'success': True
        }
    except Exception as e:
        results['scipy'] = {
            'params': np.array([np.nan] * len(p0)),
            'errors': np.array([np.nan] * len(p0)),
            'chisq': np.nan,
            'reduced_chisq': np.nan,
            'nfev': 0,
            'njac': 0,
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
    
    return results


def plot_fit_comparison(result, output_dir=None):
    """
    Create a PNG file showing scipy (left) and mpfit (right) fits.
    
    Parameters
    ----------
    result : dict
        Result dictionary from compare_scipy_fmpfit with added x, y, error, true_params
    output_dir : str
        Directory to save PNG files (default: figs subdir relative to this script)
    """
    if output_dir is None:
        # Use figs subdirectory relative to this script
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(_this_dir, 'figs')
        os.makedirs(output_dir, exist_ok=True)
    run_num = result['run']
    x = result['x']
    y = result['y']
    error = result['error']
    tp = result['true_params']
    
    # High-resolution x for smooth fit curves (10x resolution)
    x_fine = np.linspace(x[0], x[-1], len(x) * 10)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Common settings
    bar_width = (x[1] - x[0]) * 0.6
    
    for ax_idx, (method, label) in enumerate([('scipy', 'Scipy'), ('fmpfit', 'MPFIT')]):
        ax = axes[ax_idx]
        
        # Plot data as bar chart with error bars
        ax.bar(x, y, width=bar_width, alpha=0.6, color='steelblue', label='Data')
        ax.errorbar(x, y, yerr=error, fmt='none', ecolor='black', capsize=3)
        
        # Plot true Gaussian (grey)
        y_true_fine = gaussian(x_fine, *tp)
        ax.plot(x_fine, y_true_fine, color='grey', linestyle='--', linewidth=2, label='True')
        
        # Plot fitted Gaussian if successful (green)
        if result[method]['success']:
            params = result[method]['params']
            y_fit_fine = gaussian(x_fine, *params)
            ax.plot(x_fine, y_fit_fine, color='green', linestyle='-', linewidth=2, label=f'{label} fit')
            rchi2 = result[method]['reduced_chisq']
            ax.set_title(f'{label}: I={params[0]:.1f}, v={params[1]:.2f}, w={params[2]:.2f}, rchi2={rchi2:.2f}')
        else:
            ax.set_title(f'{label}: FAILED')
        
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('Intensity')
        ax.legend(loc='upper right')
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    
    fig.suptitle(f'Run {run_num}: True I={tp[0]:.1f}, v={tp[1]:.2f}, w={tp[2]:.2f}', fontsize=12)
    plt.tight_layout()
    
    # Save PNG
    output_path = os.path.join(output_dir, f'fit_run_{run_num:03d}.png')
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def estimate_fwhm_from_data(x, y, max_idx):
    """Estimate FWHM from noisy data by finding half-max crossings."""
    max_val = y[max_idx]
    half_max = max_val / 2
    
    # Find first pixel to the left where value drops below half-max
    left_idx = None
    for j in range(max_idx - 1, -1, -1):
        if y[j] <= half_max:
            left_idx = j
            break
    
    # Find first pixel to the right where value drops below half-max
    right_idx = None
    for j in range(max_idx + 1, len(y)):
        if y[j] <= half_max:
            right_idx = j
            break
    
    # Estimate FWHM
    if left_idx is not None and right_idx is not None:
        fwhm_pixels = right_idx - left_idx
    else:
        fwhm_pixels = 3.0  # default
    
    # Convert to sigma: FWHM = 2.355 * sigma
    pixel_spacing = x[1] - x[0]
    fwhm = fwhm_pixels * pixel_spacing
    sigma = fwhm / 2.355
    return sigma


def run_comparison_n_times(n_runs, seed=42):
    """
    Run comparison N times with randomized true parameters and Poisson noise.
    
    True parameters are randomized for each run:
    - intensity (I): uniform in [1, 20]
    - velocity (v): uniform in [x[0], x[-1]], so peak can be anywhere in the 5-pixel range
    - width (w): FWHM uniform in [2, 5] pixels, converted to sigma
    
    Noise is Poisson-distributed (variance = signal), with error = sqrt(counts).
    
    Initial guesses are derived from the noisy data:
    - p0_I: value of the maximum pixel
    - p0_v: x-coordinate of the maximum pixel
    - p0_w: estimated from half-max crossings, or default 3.0 pixels FWHM if not found
    
    Parameters
    ----------
    n_runs : int
        Number of comparison runs to perform
    seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    list
        List of result dictionaries from each run
    """
    rng = np.random.default_rng(seed)
    
    # Fixed x array: 5 pixels
    x = np.linspace(-2.0, 2.0, 5)
    pixel_spacing = x[1] - x[0]
    
    # Storage for results
    results_list = []
    
    for i in range(n_runs):
        # Randomize true parameters for each run
        true_I = rng.uniform(1.0, 20.0)
        true_v = rng.uniform(x[0], x[-1])  # peak anywhere in x range
        true_fwhm_pixels = rng.uniform(2.0, 5.0)
        true_w = (true_fwhm_pixels * pixel_spacing) / 2.355
        
        true_params = [true_I, true_v, true_w]
        
        # Generate noisy data with Poisson noise
        y_true = gaussian(x, *true_params)
        # Poisson noise: variance = signal, so we sample from Poisson(y_true)
        # Need to handle cases where y_true might be very small or negative
        y_true_positive = np.maximum(y_true, 0.0)
        y = rng.poisson(y_true_positive).astype(float)
        # Error is sqrt of counts (Poisson statistics)
        error = np.sqrt(np.maximum(y, 1.0))  # minimum error of 1 to avoid division by zero
        
        # Initial guesses from noisy data
        max_idx = np.argmax(y)
        p0_I = y[max_idx]
        p0_v = x[max_idx]
        p0_w = estimate_fwhm_from_data(x, y, max_idx)
        
        p0 = [p0_I, p0_v, p0_w]
        bounds = [(0.0, 100.0), (x[0] - 1.0, x[-1] + 1.0), (0.01, 10.0)]
        
        result = compare_scipy_fmpfit(x, y, error, p0, bounds, pixel_spacing)
        result['run'] = i + 1
        result['true_params'] = true_params
        result['x'] = x
        result['y'] = y
        result['error'] = error
        results_list.append(result)
    
    # Print header
    print("=" * 120)
    print(f"Comparison: scipy.optimize.curve_fit vs fmpfit_f64_wrap ({n_runs} runs, 5 pixels, Poisson noise)")
    print(f"True params randomized: I in [1,20], v in [{x[0]:.2f},{x[-1]:.2f}], w (FWHM) in [2,5] pixels")
    print(f"x = {x}, pixel_spacing = {pixel_spacing:.4f}")
    print("=" * 120)
    
    # Table header
    print(f"\n{'':>4} || {'True':^20} || {'Scipy':^53} || {'MPFIT':^53} ||")
    print(f"{'Run':>4} || {'I':>6} {'v':>6} {'w':>6} || {'I':>6} {'v':>6} {'w':>6} | {'I%':>5}  {'v_px':>5}  {'w%':>5}  {'rchi2':>5} {'FAIL':>4} || {'I':>6} {'v':>6} {'w':>6} | {'I%':>5}  {'v_px':>5}  {'w%':>5}  {'rchi2':>5} {'FAIL':>4} ||")
    print("-" * 142)
    
    # Print each run
    n_scipy_fail = 0
    n_mpfit_fail = 0
    
    for r in results_list:
        tp = r['true_params']
        scipy_ok = r['scipy']['success']
        mpfit_ok = r['fmpfit']['success']
        
        # True params
        true_str = f"{tp[0]:>6.1f} {tp[1]:>6.1f} {tp[2]:>6.1f}"
        
        # Scipy params and fail status
        if scipy_ok:
            sp = r['scipy']['params']
            sp_rchi2 = r['scipy']['reduced_chisq']
            sp_I_pct = 100 * abs(sp[0] - tp[0]) / tp[0] if tp[0] != 0 else 0
            sp_v_px = abs(sp[1] - tp[1]) / pixel_spacing
            sp_w_pct = 100 * abs(sp[2] - tp[2]) / tp[2] if tp[2] != 0 else 0
            # Check if scipy matched truth well
            sp_I_match = sp_I_pct <= 10.0  # 10% tolerance for I
            sp_v_match = sp_v_px <= 0.1  # 0.1 pixel tolerance for v
            sp_w_match = sp_w_pct <= 10.0  # 10% tolerance for w
            scipy_good = sp_I_match and sp_v_match and sp_w_match
            scipy_fail = '****' if not scipy_good else ''
            if not scipy_good:
                n_scipy_fail += 1
            scipy_str = f"{sp[0]:>6.1f} {sp[1]:>6.1f} {sp[2]:>6.1f} | {sp_I_pct:>5.1f}  {sp_v_px:>5.2f}  {sp_w_pct:>5.1f}  {sp_rchi2:>5.2f} {scipy_fail:>4}"
        else:
            n_scipy_fail += 1
            scipy_str = f"{'-':>6} {'-':>6} {'-':>6} | {'-':>5}  {'-':>5}  {'-':>5}  {'-':>5} {'****':>4}"
        
        # Mpfit params and fail status
        if mpfit_ok:
            mp = r['fmpfit']['params']
            mp_rchi2 = r['fmpfit']['reduced_chisq']
            mp_I_pct = 100 * abs(mp[0] - tp[0]) / tp[0] if tp[0] != 0 else 0
            mp_v_px = abs(mp[1] - tp[1]) / pixel_spacing
            mp_w_pct = 100 * abs(mp[2] - tp[2]) / tp[2] if tp[2] != 0 else 0
            # Check if mpfit matched truth well
            mp_I_match = mp_I_pct <= 10.0
            mp_v_match = mp_v_px <= 0.1
            mp_w_match = mp_w_pct <= 10.0
            mpfit_good = mp_I_match and mp_v_match and mp_w_match
            mpfit_fail = '****' if not mpfit_good else ''
            if not mpfit_good:
                n_mpfit_fail += 1
            mpfit_str = f"{mp[0]:>6.1f} {mp[1]:>6.1f} {mp[2]:>6.1f} | {mp_I_pct:>5.1f}  {mp_v_px:>5.2f}  {mp_w_pct:>5.1f}  {mp_rchi2:>5.2f} {mpfit_fail:>4}"
        else:
            n_mpfit_fail += 1
            mpfit_str = f"{'-':>6} {'-':>6} {'-':>6} | {'-':>5}  {'-':>5}  {'-':>5}  {'-':>5} {'****':>4}"
        
        print(f"{r['run']:>4} || {true_str} || {scipy_str} || {mpfit_str} ||")
    
    print("-" * 142)
    print(f"Summary: Scipy {n_scipy_fail}/{n_runs} failed, MPFIT {n_mpfit_fail}/{n_runs} failed (I/w: 10% tol, v: 0.1 px tol)")
    
    # Generate plots for the first 10 runs
    print(f"\nGenerating plots for first {min(10, n_runs)} runs...")
    for r in results_list[:10]:
        plot_fit_comparison(r)
    
    return results_list


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_comparison_n_times(n)

