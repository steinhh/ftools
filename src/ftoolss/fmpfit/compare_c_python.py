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
        p0_w = estimate_fwhm_from_data(x, y, max_idx)
        
        p0 = [p0_I, p0_v, p0_w]
        # Amplitude upper bound: 5 sigma (Poisson) above max measured value
        I_upper = y[max_idx] + 5.0 * error[max_idx]
        bounds = [(0.0, I_upper), (x[0], x[-1]), (0.7, 5.0)]
        
        result = compare_scipy_fmpfit(x, y, error, p0, bounds, pixel_spacing)
        result['run'] = i + 1
        result['true_params'] = true_params
        result['p0'] = p0
        result['x'] = x
        result['y'] = y
        result['error'] = error
        results_list.append(result)
    
    # Print header
    print("=" * 120)
    print(f"Comparison: scipy.optimize.curve_fit vs fmpfit_f64_pywrap ({n_runs} runs, 5 pixels extracted from 10, Poisson noise)")
    print(f"True params randomized: I in {true_I_range}, v in {true_v_range} (from center), w (FWHM) in {true_fwhm_range} pixels")
    print(f"x = {x}, pixel_spacing = {pixel_spacing:.4f}")
    print("=" * 120)
    
    # Table header
    print(f"\n{'':>4} || {'True':^20} || {'Scipy':^53} || {'MPFIT':^53} || {'':>7} ||")
    print(f"{'Run':>4} || {'I':>6} {'v':>6} {'w':>6} || {'I':>6} {'v':>6} {'w':>6} | {'I%':>6} {'v_px':>6} {'w%':>6} {'rchi2':>5} {'FAIL':>4} || {'I':>6} {'v':>6} {'w':>6} | {'I%':>6} {'v_px':>6} {'w%':>6} {'rchi2':>5} {'FAIL':>4} || {'MISMTCH':>7} ||")
    print("-" * 158)
    
    # Print each run
    n_scipy_fail = 0
    n_mpfit_fail = 0
    n_mismatch = 0
    
    # Accumulators for signed differences (for averaging)
    sp_I_pct_list, sp_v_px_list, sp_w_pct_list = [], [], []
    mp_I_pct_list, mp_v_px_list, mp_w_pct_list = [], [], []
    
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
            # Signed differences
            sp_I_pct = 100 * (sp[0] - tp[0]) / tp[0] if tp[0] != 0 else 0
            sp_v_px = (sp[1] - tp[1]) / pixel_spacing
            sp_w_pct = 100 * (sp[2] - tp[2]) / tp[2] if tp[2] != 0 else 0
            sp_I_pct_list.append(sp_I_pct)
            sp_v_px_list.append(sp_v_px)
            sp_w_pct_list.append(sp_w_pct)
            # Check if scipy matched truth well (use absolute for tolerance check)
            sp_I_match = abs(sp_I_pct) <= 10.0  # 10% tolerance for I
            sp_v_match = abs(sp_v_px) <= 0.1  # 0.1 pixel tolerance for v
            sp_w_match = abs(sp_w_pct) <= 10.0  # 10% tolerance for w
            scipy_good = sp_I_match and sp_v_match and sp_w_match
            scipy_fail = '****' if not scipy_good else ''
            if not scipy_good:
                n_scipy_fail += 1
            # Add asterisk after out-of-bounds values, space otherwise
            sp_I_mark = '*' if not sp_I_match else ' '
            sp_v_mark = '*' if not sp_v_match else ' '
            sp_w_mark = '*' if not sp_w_match else ' '
            scipy_str = f"{sp[0]:>6.1f} {sp[1]:>6.1f} {sp[2]:>6.1f} | {sp_I_pct:>+6.1f}{sp_I_mark}{sp_v_px:>+6.2f}{sp_v_mark}{sp_w_pct:>+6.1f}{sp_w_mark}{sp_rchi2:>5.2f} {scipy_fail:>4}"
        else:
            n_scipy_fail += 1
            scipy_str = f"{'-':>6} {'-':>6} {'-':>6} | {'-':>6} {'-':>6} {'-':>6} {'-':>5} {'****':>4}"
        
        # Mpfit params and fail status
        if mpfit_ok:
            mp = r['fmpfit']['params']
            mp_rchi2 = r['fmpfit']['reduced_chisq']
            # Signed differences
            mp_I_pct = 100 * (mp[0] - tp[0]) / tp[0] if tp[0] != 0 else 0
            mp_v_px = (mp[1] - tp[1]) / pixel_spacing
            mp_w_pct = 100 * (mp[2] - tp[2]) / tp[2] if tp[2] != 0 else 0
            mp_I_pct_list.append(mp_I_pct)
            mp_v_px_list.append(mp_v_px)
            mp_w_pct_list.append(mp_w_pct)
            # Check if mpfit matched truth well (use absolute for tolerance check)
            mp_I_match = abs(mp_I_pct) <= 10.0
            mp_v_match = abs(mp_v_px) <= 0.1
            mp_w_match = abs(mp_w_pct) <= 10.0
            mpfit_good = mp_I_match and mp_v_match and mp_w_match
            mpfit_fail = '****' if not mpfit_good else ''
            if not mpfit_good:
                n_mpfit_fail += 1
            # Add asterisk after out-of-bounds values, space otherwise
            mp_I_mark = '*' if not mp_I_match else ' '
            mp_v_mark = '*' if not mp_v_match else ' '
            mp_w_mark = '*' if not mp_w_match else ' '
            mpfit_str = f"{mp[0]:>6.1f} {mp[1]:>6.1f} {mp[2]:>6.1f} | {mp_I_pct:>+6.1f}{mp_I_mark}{mp_v_px:>+6.2f}{mp_v_mark}{mp_w_pct:>+6.1f}{mp_w_mark}{mp_rchi2:>5.2f} {mpfit_fail:>4}"
        else:
            n_mpfit_fail += 1
            mpfit_str = f"{'-':>6} {'-':>6} {'-':>6} | {'-':>6} {'-':>6} {'-':>6} {'-':>5} {'****':>4}"
        
        # Check for mismatch between scipy and mpfit (flag if I, v, or w differ significantly)
        mismatch_flags = ''
        if scipy_ok and mpfit_ok:
            sp = r['scipy']['params']
            mp = r['fmpfit']['params']
            # I mismatch: >0.1% relative difference
            I_mismatch = abs(sp[0] - mp[0]) / max(abs(sp[0]), abs(mp[0]), 1e-10) > 0.01
            # v mismatch: >0.05 pixels
            v_mismatch = abs(sp[1] - mp[1]) > 0.05
            # w mismatch: >1% relative difference
            w_mismatch = abs(sp[2] - mp[2]) / max(abs(sp[2]), abs(mp[2]), 1e-10) > 0.01
            if I_mismatch:
                mismatch_flags += 'I'
            if v_mismatch:
                mismatch_flags += 'v'
            if w_mismatch:
                mismatch_flags += 'w'
            if mismatch_flags:
                n_mismatch += 1
        elif scipy_ok != mpfit_ok:
            mismatch_flags = 'FAIL'
            n_mismatch += 1
        
        print(f"{r['run']:>4} || {true_str} || {scipy_str} || {mpfit_str} || {mismatch_flags:>7} ||")
    
    print("-" * 158)
    
    # Compute and print averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')
    
    sp_I_avg, sp_v_avg, sp_w_avg = avg(sp_I_pct_list), avg(sp_v_px_list), avg(sp_w_pct_list)
    mp_I_avg, mp_v_avg, mp_w_avg = avg(mp_I_pct_list), avg(mp_v_px_list), avg(mp_w_pct_list)
    
    # Average line (aligned with difference columns)
    avg_scipy = f"{'':>6} {'':>6} {'':>6} | {sp_I_avg:>+6.1f} {sp_v_avg:>+6.2f} {sp_w_avg:>+6.1f} {'':>5} {'':>4}"
    avg_mpfit = f"{'':>6} {'':>6} {'':>6} | {mp_I_avg:>+6.1f} {mp_v_avg:>+6.2f} {mp_w_avg:>+6.1f} {'':>5} {'':>4}"
    print(f"{'Avg':>4} || {'':>20} || {avg_scipy} || {avg_mpfit} || {'':>7} ||")
    
    print("-" * 158)
    print(f"Summary: Scipy {n_scipy_fail}/{n_runs} failed, MPFIT {n_mpfit_fail}/{n_runs} failed (I/w: 10% tol, v: 0.1 px tol)")
    print(f"         Scipy vs MPFIT mismatch: {n_mismatch}/{n_runs} (I/w: 5% tol, v: 0.05 px tol)")
    
    # Generate plots for the first num_plot runs
    num_plot = 20
    print(f"\nGenerating plots for first {min(num_plot, n_runs)} runs...")
    for r in results_list[:num_plot]:
        plot_fit_comparison(r)
    
    # ==================== ERROR ESTIMATE COMPARISON ====================
    print("\n" + "=" * 120)
    print("ERROR ESTIMATE COMPARISON")
    print("=" * 120)
    print("""
Error estimation methods:
Error estimation methods:
    - scipy errors:    sqrt(diag(pcov)) with absolute_sigma=False (default)
                        (scaled so reduced_chi2 = 1)
    - mpfit xerror:    sqrt(diag(covar)) from MPFIT, unscaled (assumes input errors are correct)
    - mpfit xerror_scipy: SciPy-compatible errors computed in MPFIT core (full-Hessian diagonal)
""")
    
    # Table header for error comparison
    print(f"\n{'':>4} || {'Scipy errors':^36} || {'MPFIT errors_scipy':^36} || {'Ratio (mpfit_scipy/scipy)':^20} ||")
    print(f"{'Run':>4} || {'err_I':>9} {'err_v':>9} {'err_w':>9} {'rchi2':>5} || {'err_I':>9} {'err_v':>9} {'err_w':>9} {'rchi2':>5} || {'I':>6} {'v':>6} {'w':>6} ||")
    print("-" * 124)
    
    # Collect ratio statistics
    ratios_I, ratios_v, ratios_w = [], [], []
    
    for r in results_list:
        scipy_ok = r['scipy']['success']
        mpfit_ok = r['fmpfit']['success']
        
        if scipy_ok and mpfit_ok:
            sp_err = r['scipy']['errors']
            sp_rchi2 = r['scipy']['reduced_chisq']
            # Use SciPy-compatible errors exported by MPFIT core (xerror_scipy)
            mp_err_scipy = r['fmpfit'].get('errors_scipy')
            mp_rchi2 = r['fmpfit']['reduced_chisq']
            if mp_err_scipy is None:
                mp_err_scipy = np.array([np.nan, np.nan, np.nan])

            # Compute ratios (mpfit_scipy/scipy)
            ratio_I = mp_err_scipy[0] / sp_err[0] if sp_err[0] > 0 else np.nan
            ratio_v = mp_err_scipy[1] / sp_err[1] if sp_err[1] > 0 else np.nan
            ratio_w = mp_err_scipy[2] / sp_err[2] if sp_err[2] > 0 else np.nan
            
            ratios_I.append(ratio_I)
            ratios_v.append(ratio_v)
            ratios_w.append(ratio_w)
            
            scipy_str = f"{sp_err[0]:>9.4f} {sp_err[1]:>9.4f} {sp_err[2]:>9.4f} {sp_rchi2:>5.2f}"
            mpfit_str = f"{mp_err_scipy[0]:>9.4f} {mp_err_scipy[1]:>9.4f} {mp_err_scipy[2]:>9.4f} {mp_rchi2:>5.2f}"
            ratio_str = f"{ratio_I:>6.3f} {ratio_v:>6.3f} {ratio_w:>6.3f}"
        else:
            scipy_str = f"{'-':>9} {'-':>9} {'-':>9} {'-':>5}"
            mpfit_str = f"{'-':>9} {'-':>9} {'-':>9} {'-':>5}"
            ratio_str = f"{'-':>6} {'-':>6} {'-':>6}"
        
        print(f"{r['run']:>4} || {scipy_str} || {mpfit_str} || {ratio_str} ||")
    
    print("-" * 124)
    
    # Summary statistics
    if ratios_I:
        print("\nRatio statistics (mpfit_scipy / scipy_errors):")
        print(f"  I:  mean={np.mean(ratios_I):.4f}, std={np.std(ratios_I):.4f}, min={np.min(ratios_I):.4f}, max={np.max(ratios_I):.4f}")
        print(f"  v:  mean={np.mean(ratios_v):.4f}, std={np.std(ratios_v):.4f}, min={np.min(ratios_v):.4f}, max={np.max(ratios_v):.4f}")
        print(f"  w:  mean={np.mean(ratios_w):.4f}, std={np.std(ratios_w):.4f}, min={np.min(ratios_w):.4f}, max={np.max(ratios_w):.4f}")
    
    print("=" * 120)
    
    return results_list


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_comparison_n_times(n)

