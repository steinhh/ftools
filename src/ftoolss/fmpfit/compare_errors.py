"""Module for comparing fitting results using different optimization methods.

Compares scipy.optimize.curve_fit with fmpfit_f64_pywrap for Gaussian fitting.

Usage:
    python compare_c_python.py [N]
    
    N: number of comparison runs (default: 10)
    
Prints a summary table comparing true parameters, fitted parameters,
Generates PNG plots for the first 20 runs in a 'figs' subdirectory.

Results:

   Pass - for all practical purposes results are identical.

   Biases (intensity I in [ 2 ... 10 ]):

      I = +10%
      w = -12%

      As expected with so few pixels.
      Basically cancels out for total flux!

   Biases (intensity I in [20 ... 50]):

      I = +1.1%
      w = -1.8%

      Excellent!
      
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
from ftoolss.fmpfit import fmpfit_f64_pywrap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PNG output
import matplotlib.pyplot as plt

# MPFIT status codes from mpfit.h
MPFIT_STATUS = {
    0: 'General input parameter error',
    -16: 'User function produced non-finite values',
    -17: 'No user function was supplied',
    -18: 'No user data points were supplied',
    -19: 'No free parameters',
    -20: 'Memory allocation error',
    -21: 'Initial values inconsistent w constraints',
    -22: 'Initial constraints inconsistent',
    -23: 'General input parameter error',
    -24: 'Not enough degrees of freedom',
    1: 'Convergence in chi-square value',
    2: 'Convergence in parameter value',
    3: 'Convergence in both chi-square and parameter',
    4: 'Convergence in orthogonality',
    5: 'Maximum number of iterations reached',
    6: 'ftol is too small; no further improvement',
    7: 'xtol is too small; no further improvement',
    8: 'gtol is too small; no further improvement',
}


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
    Compare fitting results from scipy.optimize.curve_fit and fmpfit_f64_pywrap.
    
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
            sigma=error, jac=counted_jacobian,
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
    
    # --- fmpfit_f64_pywrap ---
    # Note: fmpfit uses analytical derivatives for the Gaussian model internally.
    # The nfev count includes both residual and Jacobian evaluations.
    try:
        parinfo = [
            {'value': p0[i], 'limits': list(bounds[i])}
            for i in range(len(p0))
        ]
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_mpfit = fmpfit_f64_pywrap(
            deviate_type=0,  # Gaussian (uses analytical derivatives)
            parinfo=parinfo,
            functkw=functkw
        )
        
        results['fmpfit'] = {
            'params': result_mpfit.best_params,
            'errors': result_mpfit.xerror,
            'errors_scipy': result_mpfit.xerror_scipy,
            'chisq': result_mpfit.bestnorm,
            'reduced_chisq': result_mpfit.bestnorm / (len(x) - len(p0)) if len(x) > len(p0) else np.nan,
            'niter': result_mpfit.niter,
            'nfev': result_mpfit.nfev,
            'status': result_mpfit.status,
            'c_time': result_mpfit.c_time,
            'covar': result_mpfit.covar,
            'success': result_mpfit.status >= 0
        }
    except Exception as e:
        results['fmpfit'] = {
            'params': np.array([np.nan] * len(p0)),
            'errors': np.array([np.nan] * len(p0)),
            'errors_scipy': np.array([np.nan] * len(p0)),
            'chisq': np.nan,
            'reduced_chisq': np.nan,
            'covar': None,
            'success': False,
            'error': str(e)
        }
    
    return results


def plot_fit_comparison(result, output_dir=None):
    """
    Create a PNG file showing both scipy and mpfit fits on a single plot,
    with a residuals panel below.
    
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
    p0 = result['p0']
    pixel_spacing = x[1] - x[0]
    
    # High-resolution x for smooth fit curves (10x resolution)
    x_fine = np.linspace(x[0], x[-1], len(x) * 10)
    
    # Create figure with main plot (3/4) and residuals panel (1/4)
    fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(8, 7), 
                                             gridspec_kw={'height_ratios': [3, 1]},
                                             sharex=True)
    
    # Common settings
    bar_width = (x[1] - x[0]) * 0.6
    
    # === MAIN PLOT ===
    # Plot data as bar chart with error bars
    ax_main.bar(x, y, width=bar_width, alpha=0.6, color='steelblue', label='Data')
    ax_main.errorbar(x, y, yerr=error, fmt='none', ecolor='black', capsize=3)
    
    # Plot true Gaussian
    y_true_fine = gaussian(x_fine, *tp)
    y_true_at_x = gaussian(x, *tp)
    ax_main.plot(x_fine, y_true_fine, color='grey', linestyle='--', linewidth=2, label='True')
    
    # Build text for display (with asterisks for out-of-tolerance values)
    text_lines = [f'True:   I={tp[0]:>6.1f}   v={tp[1]:>5.2f}   w={tp[2]:>4.2f}']
    
    # Plot scipy fit first (red) so it gets covered by mpfit if identical
    y_scipy_at_x = None
    if result['scipy']['success']:
        sp = result['scipy']['params']
        y_scipy_fine = gaussian(x_fine, *sp)
        y_scipy_at_x = gaussian(x, *sp)
        ax_main.plot(x_fine, y_scipy_fine, color='red', linestyle='-', linewidth=2, label='Scipy')
        sp_rchi2 = result['scipy']['reduced_chisq']
        # Check tolerances and add asterisks
        sp_I_ok = 100 * abs(sp[0] - tp[0]) / tp[0] <= 10.0 if tp[0] != 0 else True
        sp_v_ok = abs(sp[1] - tp[1]) / pixel_spacing <= 0.1
        sp_w_ok = 100 * abs(sp[2] - tp[2]) / tp[2] <= 10.0 if tp[2] != 0 else True
        sp_I_mark = ' ' if sp_I_ok else '*'
        sp_v_mark = ' ' if sp_v_ok else '*'
        sp_w_mark = ' ' if sp_w_ok else '*'
        text_lines.append(f'Scipy:  I={sp[0]:>6.1f}{sp_I_mark}  v={sp[1]:>5.2f}{sp_v_mark}  w={sp[2]:>4.2f}{sp_w_mark}  rchi2={sp_rchi2:.2f}')
    else:
        # Get failure reason for scipy
        if 'error' in result['scipy']:
            fail_reason = result['scipy']['error']
            if len(fail_reason) > 30:
                fail_reason = fail_reason[:27] + '...'
        else:
            fail_reason = 'unknown'
        text_lines.append(f'Scipy:  FAILED ({fail_reason})')
    
    # Plot mpfit fit on top (green)
    y_mpfit_at_x = None
    if result['fmpfit']['success']:
        mp = result['fmpfit']['params']
        y_mpfit_fine = gaussian(x_fine, *mp)
        y_mpfit_at_x = gaussian(x, *mp)
        ax_main.plot(x_fine, y_mpfit_fine, color='green', linestyle='-', linewidth=2, label='MPFIT')
        mp_rchi2 = result['fmpfit']['reduced_chisq']
        # Check tolerances and add asterisks
        mp_I_ok = 100 * abs(mp[0] - tp[0]) / tp[0] <= 10.0 if tp[0] != 0 else True
        mp_v_ok = abs(mp[1] - tp[1]) / pixel_spacing <= 0.1
        mp_w_ok = 100 * abs(mp[2] - tp[2]) / tp[2] <= 10.0 if tp[2] != 0 else True
        mp_I_mark = ' ' if mp_I_ok else '*'
        mp_v_mark = ' ' if mp_v_ok else '*'
        mp_w_mark = ' ' if mp_w_ok else '*'
        text_lines.append(f'MPFIT:  I={mp[0]:>6.1f}{mp_I_mark}  v={mp[1]:>5.2f}{mp_v_mark}  w={mp[2]:>4.2f}{mp_w_mark}  rchi2={mp_rchi2:.2f}')
    else:
        # Get failure reason for mpfit
        if 'error' in result['fmpfit']:
            fail_reason = result['fmpfit']['error']
            if len(fail_reason) > 30:
                fail_reason = fail_reason[:27] + '...'
        elif 'status' in result['fmpfit']:
            status = result['fmpfit']['status']
            if status in MPFIT_STATUS:
                fail_reason = MPFIT_STATUS[status]
            else:
                fail_reason = f'status={status}'
        else:
            fail_reason = 'unknown'
        text_lines.append(f'MPFIT:  FAILED ({fail_reason})')
    
    # Add text box
    ax_main.text(0.02, 0.98, '\n'.join(text_lines), transform=ax_main.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='left',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_main.set_title(f'Run {run_num}')
    ax_main.set_ylabel('Intensity')
    ax_main.legend(loc='upper right')
    
    # === RESIDUALS PLOT ===
    # Zero line
    ax_resid.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 1-sigma error band (shaded grey)
    ax_resid.fill_between(x, -error, error, alpha=0.3, color='grey', label='1-sigma')
    
    # Noise: data - true (grey markers)
    noise = y - y_true_at_x
    ax_resid.scatter(x, noise, color='grey', s=50, zorder=3, label='Noise', marker='o')
    
    # Scipy residuals (red markers) - offset slightly left
    if y_scipy_at_x is not None:
        resid_scipy = y - y_scipy_at_x
        ax_resid.scatter(x - 0.1, resid_scipy, color='red', s=40, zorder=4, label='Scipy', marker='s')
    
    # MPFIT residuals (green markers) - offset slightly right
    if y_mpfit_at_x is not None:
        resid_mpfit = y - y_mpfit_at_x
        ax_resid.scatter(x + 0.1, resid_mpfit, color='green', s=40, zorder=5, label='MPFIT', marker='^')
    
    ax_resid.set_xlabel('x (pixels)')
    ax_resid.set_ylabel('Residual')
    ax_resid.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax_resid.legend(loc='upper right', fontsize=8, ncol=4)
    
    plt.tight_layout()
    
    # Save PNG
    output_path = os.path.join(output_dir, f'fit_run_{run_num:03d}.png')
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def mpfit_is_better(result, chi2_threshold=0.5, param_threshold=0.1):
    """Determine if MPFIT found a significantly better fit than scipy.
    
    Compares chi-squared values and parameter distances from truth to determine
    if MPFIT's solution is meaningfully better than scipy's.
    
    Parameters
    ----------
    result : dict
        Result dictionary containing scipy, fmpfit, and true_params
    chi2_threshold : float
        Minimum chi2 improvement (mpfit chi2 < scipy chi2 - threshold) to consider better
    param_threshold : float
        Minimum relative improvement in parameter distance from truth
    
    Returns
    -------
    bool
        True if MPFIT found a significantly better solution
    str
        Reason string explaining why MPFIT is better (or empty if not)
    """
    sp = result['scipy']
    mp = result['fmpfit']
    true = result['true_params']
    
    # Compare reduced chi-squared
    sp_rchi2 = sp['reduced_chisq']
    mp_rchi2 = mp['reduced_chisq']
    
    # Chi2 improvement: MPFIT significantly lower
    chi2_better = mp_rchi2 < sp_rchi2 - chi2_threshold
    
    # Parameter distance from truth
    sp_params = sp['params']
    mp_params = mp['params']
    
    # Compute weighted distance from truth (relative for I and w, absolute for v)
    def param_distance(params):
        d_I = abs(params[0] - true[0]) / true[0] if true[0] != 0 else abs(params[0] - true[0])
        d_v = abs(params[1] - true[1])  # absolute for velocity
        d_w = abs(params[2] - true[2]) / true[2] if true[2] != 0 else abs(params[2] - true[2])
        return d_I + d_v + d_w
    
    sp_dist = param_distance(sp_params)
    mp_dist = param_distance(mp_params)
    
    # MPFIT closer to truth by significant margin
    params_better = mp_dist < sp_dist * (1 - param_threshold)
    
    reasons = []
    if chi2_better:
        reasons.append(f'chi2: {mp_rchi2:.2f} vs {sp_rchi2:.2f}')
    if params_better:
        reasons.append(f'dist: {mp_dist:.2f} vs {sp_dist:.2f}')
    
    is_better = chi2_better or params_better
    reason = ', '.join(reasons) if reasons else ''
    
    return is_better, reason


def estimate_sigma_from_data(x, y, max_idx, lower_bound=0.7):
    """Estimate sigma using second moment around the peak, with safe floor.
    
    Uses the second moment (variance) of the intensity distribution around
    the peak to estimate sigma. Includes a safety floor to ensure the initial
    guess stays well above the lower bound, avoiding MPFIT getting stuck at
    local minima near bounds.
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        Intensity values (may contain noise/negative values)
    max_idx : int
        Index of the maximum pixel
    lower_bound : float
        Lower bound on sigma parameter (default: 0.7)
    
    Returns
    -------
    float
        Estimated sigma, guaranteed to be at least lower_bound + 0.5
    """
    # Use only positive values for moment calculation
    y_pos = np.maximum(y, 0)
    x0 = x[max_idx]
    total = np.sum(y_pos)
    
    if total <= 0:
        # Fallback if no positive signal
        return lower_bound + 0.57  # ~1.27, middle of typical range
    
    # Second moment: variance around the peak
    var_x = np.sum((x - x0)**2 * y_pos) / total
    est = np.sqrt(var_x) if var_x > 0 else lower_bound + 0.57
    
    # Safety floor: stay at least 0.5 above lower bound to avoid local minima
    safe_min = lower_bound + 0.5
    return max(est, safe_min)


def run_comparison_n_times(n_runs, seed=41):
    """
    Run comparison N times with randomized true parameters and Poisson noise.
    
    Simulates realistic data extraction:
    1. Generate Gaussian on 10-pixel wide array
    2. True mean (v) is within +/-0.5 pixels of center
    3. Add Poisson noise
    4. Find maximum pixel and extract 5 pixels centered on it
    
    True parameters are randomized for each run:
    - intensity (I): uniform in [2, 10]
    - velocity (v): uniform in [-0.5, 0.5] from center pixel
    - width (w): FWHM uniform in [2, 5] pixels, converted to sigma
    
    Noise is Poisson-distributed (variance = signal), with error = sqrt(counts).
    
    Initial guesses are derived from the noisy data:
    - p0_I: value of the maximum pixel
    - p0_v: x-coordinate of the maximum pixel (always 0 in extracted window)
    - p0_w: estimated from half-max crossings, or default 3.0 pixels FWHM if not found
    
    Parameters
    ----------
    n_runs : int
        Number of comparison runs to perform
    seed : int, optional
        Random seed for reproducibility (default: 41)
    
    Returns
    -------
    list
        List of result dictionaries from each run
    """
    rng = np.random.default_rng(seed)
    
    # Wide array: 10 pixels for initial data generation
    n_wide = 10
    x_wide = np.arange(n_wide) - (n_wide - 1) / 2.0  # Centered: [-4.5, -3.5, ..., 4.5]
    pixel_spacing = 1.0
    
    # Storage for results
    results_list = []

    true_I_range = (5.0, 10.0)
    true_v_range = (-0.5, 0.5)  # Within +/-0.5 pixels of center (x=0)
    true_fwhm_range = (2, 5.0)  # FWHM from 2 to 5 pixels

    for i in range(n_runs):
        # Randomize true parameters for each run
        true_I = rng.uniform(*true_I_range)
        # True mean within +/-0.5 pixels of center
        true_v = rng.uniform(*true_v_range)
        
        true_fwhm_pixels = rng.uniform(*true_fwhm_range)
        true_w = (true_fwhm_pixels * pixel_spacing) / 2.355
        
        true_params_wide = [true_I, true_v, true_w]
        
        # Generate noisy data on wide array with Poisson noise
        y_true_wide = gaussian(x_wide, *true_params_wide)
        y_true_wide_positive = np.maximum(y_true_wide, 0.0)
        y_wide = rng.poisson(y_true_wide_positive).astype(float)
        
        # Find maximum pixel
        max_idx_wide = np.argmax(y_wide)
        
        # Extract 5 pixels centered on maximum
        half_window = 2
        start_idx = max_idx_wide - half_window
        end_idx = max_idx_wide + half_window + 1
        
        # Handle edge cases (should be rare with v near center)
        if start_idx < 0:
            start_idx = 0
            end_idx = 5
        if end_idx > n_wide:
            end_idx = n_wide
            start_idx = n_wide - 5
        
        # Extract 5-pixel window
        x = x_wide[start_idx:end_idx]
        y = y_wide[start_idx:end_idx]
        
        # Shift x so the maximum pixel is at x=0
        x = x - x_wide[max_idx_wide]
        
        # True params in extracted window coordinates
        true_v_extracted = true_v - x_wide[max_idx_wide]
        true_params = [true_I, true_v_extracted, true_w]
        
        # Error is sqrt of counts (Poisson statistics)
        error = np.sqrt(np.maximum(y, 1.0))  # minimum error of 1 to avoid division by zero
        
        # Initial guesses from extracted data
        max_idx = np.argmax(y)  # Should be center (index 2) in extracted window
        p0_I = y[max_idx]
        p0_v = x[max_idx]
        p0_w = estimate_sigma_from_data(x, y, max_idx)
        
        p0 = [p0_I, p0_v, p0_w]
        # Amplitude upper bound: 5 sigma (Poisson) above max measured value
        I_upper = y[max_idx] + 5.0 * error[max_idx]
        bounds = [(0.0, I_upper), (x[0], x[-1]), (0.7, 5.0)]
        
        result = compare_scipy_fmpfit(x, y, error, p0, bounds, pixel_spacing)
        result['run'] = i + 1
        result['true_params'] = true_params
        result['p0'] = p0
        result['bounds'] = bounds
        result['x'] = x
        result['y'] = y
        result['error'] = error
        results_list.append(result)
    
    # Filter to successful runs only
    valid_results = [r for r in results_list if r['scipy']['success'] and r['fmpfit']['success']]
    
    print("=" * 120)
    print(f"ERROR ANALYSIS: scipy.optimize.curve_fit vs fmpfit_f64_pywrap ({n_runs} runs, {len(valid_results)} valid)")
    print(f"True params randomized: I in {true_I_range}, v in {true_v_range} (from center), w (FWHM) in {true_fwhm_range} pixels")
    print("=" * 120)
    
    # ==================== SCATTER PLOTS ====================
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(_this_dir, 'figs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for scatter plots
    scipy_I = np.array([r['scipy']['params'][0] for r in valid_results])
    scipy_v = np.array([r['scipy']['params'][1] for r in valid_results])
    scipy_w = np.array([r['scipy']['params'][2] for r in valid_results])
    scipy_rchi2 = np.array([r['scipy']['reduced_chisq'] for r in valid_results])
    
    mpfit_I = np.array([r['fmpfit']['params'][0] for r in valid_results])
    mpfit_v = np.array([r['fmpfit']['params'][1] for r in valid_results])
    mpfit_w = np.array([r['fmpfit']['params'][2] for r in valid_results])
    mpfit_rchi2 = np.array([r['fmpfit']['reduced_chisq'] for r in valid_results])
    
    scipy_err_I = np.array([r['scipy']['errors'][0] for r in valid_results])
    scipy_err_v = np.array([r['scipy']['errors'][1] for r in valid_results])
    scipy_err_w = np.array([r['scipy']['errors'][2] for r in valid_results])
    
    mpfit_err_I = np.array([r['fmpfit']['errors_scipy'][0] for r in valid_results])
    mpfit_err_v = np.array([r['fmpfit']['errors_scipy'][1] for r in valid_results])
    mpfit_err_w = np.array([r['fmpfit']['errors_scipy'][2] for r in valid_results])
    
    # Figure 1: Parameter comparison (4 panels: I, v, w, rchi2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, sp_data, mp_data, label in [
        (axes1[0, 0], scipy_I, mpfit_I, 'I (intensity)'),
        (axes1[0, 1], scipy_v, mpfit_v, 'v (velocity)'),
        (axes1[1, 0], scipy_w, mpfit_w, 'w (width)'),
        (axes1[1, 1], scipy_rchi2, mpfit_rchi2, 'reduced chi2'),
    ]:
        ax.scatter(sp_data, mp_data, alpha=0.5, s=20)
        # Add y=x line
        lims = [min(sp_data.min(), mp_data.min()), max(sp_data.max(), mp_data.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
        ax.set_xlabel(f'scipy {label}')
        ax.set_ylabel(f'mpfit {label}')
        ax.set_title(f'{label}: scipy vs mpfit')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'params_scatter.png')
    plt.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    print(f"\nSaved parameter scatter plots: {fig1_path}")
    
    # Figure 2: Error comparison (3 panels: err_I, err_v, err_w)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, sp_err, mp_err, label in [
        (axes2[0], scipy_err_I, mpfit_err_I, 'err_I'),
        (axes2[1], scipy_err_v, mpfit_err_v, 'err_v'),
        (axes2[2], scipy_err_w, mpfit_err_w, 'err_w'),
    ]:
        ax.scatter(sp_err, mp_err, alpha=0.5, s=20)
        # Add y=x line
        lims = [0, max(sp_err.max(), mp_err.max()) * 1.1]
        ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
        ax.set_xlabel(f'scipy {label}')
        ax.set_ylabel(f'mpfit {label} (scipy-style)')
        ax.set_title(f'{label}: scipy vs mpfit_scipy')
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'errors_scatter.png')
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"Saved error scatter plots: {fig2_path}")
    
    # ==================== FILTER RUNS WITH ERROR RATIO OUTLIERS ====================
    # Only consider runs where mpfit_scipy/scipy ratio is >1.05 or <0.95
    # But ignore cases where MPFIT found a significantly better fit
    outlier_runs = []
    mpfit_better_runs = []
    
    for r in valid_results:
        sp_err = r['scipy']['errors']
        mp_err_scipy = r['fmpfit']['errors_scipy']
        
        # Compute ratios
        ratios = []
        for i in range(3):
            if sp_err[i] > 0:
                ratios.append(mp_err_scipy[i] / sp_err[i])
            else:
                ratios.append(np.nan)
        
        # Check if any ratio is outside [0.95, 1.05]
        is_outlier = any(not np.isnan(rat) and (rat > 1.05 or rat < 0.95) for rat in ratios)
        if is_outlier:
            r['error_ratios'] = ratios
            
            # Check if MPFIT found a better solution - if so, ignore this outlier
            is_better, reason = mpfit_is_better(r)
            if is_better:
                r['mpfit_better_reason'] = reason
                mpfit_better_runs.append(r)
            else:
                outlier_runs.append(r)
    
    print(f"\n{'=' * 120}")
    print(f"OUTLIER ANALYSIS: {len(outlier_runs)} runs with error ratio outside [0.95, 1.05]")
    print(f"  (Ignored {len(mpfit_better_runs)} runs where MPFIT found a better fit)")
    print(f"{'=' * 120}")
    
    if mpfit_better_runs:
        print(f"\nIgnored runs (MPFIT better):")
        for r in mpfit_better_runs[:10]:  # Show first 10
            print(f"  Run {r['run']:>4}: {r['mpfit_better_reason']}")
        if len(mpfit_better_runs) > 10:
            print(f"  ... and {len(mpfit_better_runs) - 10} more")
    
    if not outlier_runs:
        print("No outliers found! All error ratios are within [0.99, 1.01].")
        return results_list
    
    # ==================== TABLE 1: Parameters for outlier runs ====================
    print(f"\n{'=' * 100}")
    print("TABLE 1: Parameter comparison for outlier runs (verify differences are negligible)")
    print(f"{'=' * 100}")
    print(f"{'Run':>4} || {'True I':>8} {'True v':>8} {'True w':>8} || {'Scipy I':>8} {'Scipy v':>8} {'Scipy w':>8} || {'dI':>10} {'dv':>10} {'dw':>10} ||")
    print("-" * 100)
    
    for r in outlier_runs:
        tp = r['true_params']
        sp = r['scipy']['params']
        mp = r['fmpfit']['params']
        
        # Absolute differences between scipy and mpfit
        diff_I = abs(sp[0] - mp[0])
        diff_v = abs(sp[1] - mp[1])
        diff_w = abs(sp[2] - mp[2])
        
        print(f"{r['run']:>4} || {tp[0]:>8.2f} {tp[1]:>8.4f} {tp[2]:>8.4f} || {sp[0]:>8.2f} {sp[1]:>8.4f} {sp[2]:>8.4f} || {diff_I:>10.6f} {diff_v:>10.6f} {diff_w:>10.6f} ||")
    
    print("-" * 100)
    
    # ==================== TABLE 2: Errors for outlier runs with analysis ====================
    print(f"\n{'=' * 140}")
    print("TABLE 2: Error analysis for outlier runs")
    print(f"{'=' * 140}")
    print(f"{'Run':>4} || {'scipy_err_I':>10} {'scipy_err_v':>10} {'scipy_err_w':>10} || {'mpfit_err_I':>10} {'mpfit_err_v':>10} {'mpfit_err_w':>10} || {'rat_I':>7} {'rat_v':>7} {'rat_w':>7} || {'rchi2':>5} {'Notes':>20} ||")
    print("-" * 140)
    
    # Analyze patterns
    patterns = {
        'I_at_bound': 0,
        'v_at_bound': 0,
        'w_at_bound': 0,
        'zero_error': 0,
        'low_rchi2': 0,
    }
    
    for r in outlier_runs:
        sp_err = r['scipy']['errors']
        mp_err_scipy = r['fmpfit']['errors_scipy']
        mp_err_unscaled = r['fmpfit']['errors']
        ratios = r['error_ratios']
        rchi2 = r['fmpfit']['reduced_chisq']
        bounds = r['bounds']
        params = r['fmpfit']['params']
        
        # Check for patterns
        notes = []
        
        # Check if parameters hit bounds
        tol = 1e-6
        if abs(params[0] - bounds[0][0]) < tol or abs(params[0] - bounds[0][1]) < tol:
            notes.append('I@bound')
            patterns['I_at_bound'] += 1
        if abs(params[1] - bounds[1][0]) < tol or abs(params[1] - bounds[1][1]) < tol:
            notes.append('v@bound')
            patterns['v_at_bound'] += 1
        if abs(params[2] - bounds[2][0]) < tol or abs(params[2] - bounds[2][1]) < tol:
            notes.append('w@bound')
            patterns['w_at_bound'] += 1
        
        # Check for zero errors (unscaled)
        if any(mp_err_unscaled[i] == 0 for i in range(3)):
            notes.append('zero_xerror')
            patterns['zero_error'] += 1
        
        # Check for very low rchi2
        if rchi2 < 0.1:
            notes.append('low_rchi2')
            patterns['low_rchi2'] += 1
        
        notes_str = ','.join(notes) if notes else ''
        
        # Format ratios with highlighting
        def fmt_ratio(rat):
            if np.isnan(rat):
                return '  nan  '
            elif rat > 1.01 or rat < 0.99:
                return f'{rat:>6.3f}*'
            else:
                return f'{rat:>6.3f} '
        
        rat_str = f"{fmt_ratio(ratios[0])} {fmt_ratio(ratios[1])} {fmt_ratio(ratios[2])}"
        
        print(f"{r['run']:>4} || {sp_err[0]:>10.4f} {sp_err[1]:>10.4f} {sp_err[2]:>10.4f} || {mp_err_scipy[0]:>10.4f} {mp_err_scipy[1]:>10.4f} {mp_err_scipy[2]:>10.4f} || {rat_str} || {rchi2:>5.2f} {notes_str:>20} ||")
    
    print("-" * 140)
    
    # ==================== PATTERN SUMMARY ====================
    print(f"\n{'=' * 80}")
    print("PATTERN SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total outlier runs: {len(outlier_runs)}")
    print(f"  - Parameter I at bound: {patterns['I_at_bound']}")
    print(f"  - Parameter v at bound: {patterns['v_at_bound']}")
    print(f"  - Parameter w at bound: {patterns['w_at_bound']}")
    print(f"  - Zero unscaled error:  {patterns['zero_error']}")
    print(f"  - Low reduced chi2:     {patterns['low_rchi2']}")
    
    # Additional analysis: check covariance matrices
    print(f"\n{'=' * 80}")
    print("COVARIANCE MATRIX ANALYSIS FOR OUTLIERS")
    print(f"{'=' * 80}")
    
    for r in outlier_runs[:5]:  # Show first 5
        print(f"\nRun {r['run']}:")
        print(f"  Fitted params: I={r['fmpfit']['params'][0]:.4f}, v={r['fmpfit']['params'][1]:.4f}, w={r['fmpfit']['params'][2]:.4f}")
        print(f"  Bounds: I=[{r['bounds'][0][0]:.2f}, {r['bounds'][0][1]:.2f}], v=[{r['bounds'][1][0]:.2f}, {r['bounds'][1][1]:.2f}], w=[{r['bounds'][2][0]:.2f}, {r['bounds'][2][1]:.2f}]")
        covar = r['fmpfit']['covar']
        if covar is not None:
            print(f"  Covariance diagonal: [{covar[0,0]:.6f}, {covar[1,1]:.6f}, {covar[2,2]:.6f}]")
            print(f"  xerror (unscaled): {r['fmpfit']['errors']}")
            print(f"  xerror_scipy:      {r['fmpfit'].get('errors_scipy', [np.nan, np.nan, np.nan])}")
            print(f"  scipy errors:      {r['scipy']['errors']}")
        print(f"  Error ratios: {r['error_ratios']}")
    
    print("=" * 120)
    
    # ==================== GENERATE FIT PLOTS FOR OUTLIERS ====================
    # Find parameter outliers (where scipy and mpfit params differ significantly)
    param_outliers = []
    for r in valid_results:
        sp = r['scipy']['params']
        mp = r['fmpfit']['params']
        
        # Check for significant differences
        diff_I = abs(sp[0] - mp[0])
        diff_v = abs(sp[1] - mp[1])
        diff_w = abs(sp[2] - mp[2])
        
        # Thresholds: 1% for I, 0.01 pixels for v, 1% for w
        is_outlier = False
        if sp[0] > 0 and diff_I / sp[0] > 0.01:
            is_outlier = True
        if diff_v > 0.01:
            is_outlier = True
        if sp[2] > 0 and diff_w / sp[2] > 0.01:
            is_outlier = True
        
        if is_outlier:
            param_outliers.append(r)
    
    if param_outliers:
        print(f"\n{'=' * 80}")
        print(f"GENERATING FIT PLOTS FOR {len(param_outliers)} PARAMETER OUTLIERS")
        print(f"{'=' * 80}")
        for r in param_outliers:
            plot_fit_comparison(r, output_dir)

    return results_list


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_comparison_n_times(n)

