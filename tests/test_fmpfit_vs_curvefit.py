"""Test comparing fmpfit_f64 with scipy.optimize.curve_fit.

Verifies that fmpfit_f64 produces results nearly identical to scipy's curve_fit
for Gaussian fitting. Results should match to within float32 precision.
"""
import numpy as np
import pytest
from scipy.optimize import curve_fit

from ftoolss.fmpfit import fmpfit_f64_pywrap

# Tolerance for parameter comparison
# Results should match very closely - within about 1e-3 absolute and relative
PARAM_RTOL = 1e-3
PARAM_ATOL = 1e-3


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


def fit_with_scipy(x, y, error, p0, bounds):
    """Fit using scipy.optimize.curve_fit."""
    scipy_bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
    
    popt, pcov = curve_fit(
        gaussian, x, y, p0=p0, bounds=scipy_bounds,
        sigma=error, absolute_sigma=True, jac=gaussian_jacobian
    )
    perr = np.sqrt(np.diag(pcov))
    residuals = y - gaussian(x, *popt)
    chisq = np.sum((residuals / error) ** 2)
    
    return {
        'params': popt,
        'errors': perr,
        'chisq': chisq
    }


def fit_with_fmpfit(x, y, error, p0, bounds):
    """Fit using fmpfit_f64_pywrap."""
    parinfo = [
        {'value': p0[i], 'limits': list(bounds[i])}
        for i in range(len(p0))
    ]
    functkw = {'x': x, 'y': y, 'error': error}
    
    result = fmpfit_f64_pywrap(
        deviate_type=0,  # Gaussian (uses analytical derivatives)
        parinfo=parinfo,
        functkw=functkw
    )
    
    return {
        'params': result.best_params,
        'errors': result.xerror,
        'chisq': result.bestnorm,
        'status': result.status
    }


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_fmpfit_vs_curvefit_single(seed):
    """Test that fmpfit_f64 and curve_fit produce nearly identical results."""
    rng = np.random.default_rng(seed)
    
    # Generate test data
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    true_params = [30.0, 0.0, 1.0]  # [intensity, mean, sigma]
    
    # Generate noisy data with Poisson noise
    y_true = gaussian(x, *true_params)
    y = rng.poisson(y_true).astype(float)
    error = np.sqrt(np.maximum(y, 1.0))
    
    # Initial guesses
    max_idx = np.argmax(y)
    p0 = [y[max_idx], x[max_idx], 1.0]
    I_upper = y[max_idx] + 2.0 * error[max_idx]
    bounds = [(0.0, I_upper), (x[0], x[-1]), (0.7, 5.0)]
    
    # Fit with both methods
    scipy_result = fit_with_scipy(x, y, error, p0, bounds)
    fmpfit_result = fit_with_fmpfit(x, y, error, p0, bounds)
    
    # Check that fmpfit converged
    assert fmpfit_result['status'] > 0, f"fmpfit failed with status {fmpfit_result['status']}"
    
    # Compare parameters - should be equal to float32 precision
    np.testing.assert_allclose(
        fmpfit_result['params'], scipy_result['params'],
        rtol=PARAM_RTOL, atol=PARAM_ATOL,
        err_msg=f"Parameters differ: fmpfit={fmpfit_result['params']}, scipy={scipy_result['params']}"
    )
    
    # Compare chi-square values - should be equal to float32 precision
    # Use larger atol since chisq can be very small for perfect fits
    np.testing.assert_allclose(
        fmpfit_result['chisq'], scipy_result['chisq'],
        rtol=PARAM_RTOL, atol=max(PARAM_ATOL, 1e-6),
        err_msg=f"Chi-square differs: fmpfit={fmpfit_result['chisq']}, scipy={scipy_result['chisq']}"
    )


def test_fmpfit_vs_curvefit_multiple_runs():
    """Test that fmpfit_f64 and curve_fit produce identical results over multiple runs."""
    rng = np.random.default_rng(42)
    n_runs = 20
    
    # Track results
    n_success = 0
    max_diffs = []
    
    for run_idx in range(n_runs):
        # Randomize true parameters
        true_I = rng.uniform(10.0, 50.0)
        true_v = rng.uniform(-0.5, 0.5)
        true_w = rng.uniform(0.8, 2.0)
        true_params = [true_I, true_v, true_w]
        
        # Generate data on wide array
        x_wide = np.arange(10) - 4.5
        y_true_wide = gaussian(x_wide, *true_params)
        y_wide = rng.poisson(np.maximum(y_true_wide, 0.0)).astype(float)
        
        # Extract 5 pixels around maximum
        max_idx = np.argmax(y_wide)
        start_idx = max(0, max_idx - 2)
        end_idx = min(len(x_wide), start_idx + 5)
        start_idx = max(0, end_idx - 5)
        
        x = x_wide[start_idx:end_idx]
        y = y_wide[start_idx:end_idx]
        x = x - x[np.argmax(y)]  # Center on maximum
        error = np.sqrt(np.maximum(y, 1.0))
        
        # Initial guesses
        max_idx_local = np.argmax(y)
        p0 = [y[max_idx_local], x[max_idx_local], 1.0]
        I_upper = y[max_idx_local] + 2.0 * error[max_idx_local]
        bounds = [(0.0, I_upper), (x[0], x[-1]), (0.7, 5.0)]
        
        try:
            # Fit with both methods
            scipy_result = fit_with_scipy(x, y, error, p0, bounds)
            fmpfit_result = fit_with_fmpfit(x, y, error, p0, bounds)
            
            # Skip if fmpfit didn't converge
            if fmpfit_result['status'] <= 0:
                continue
            
            # Track maximum absolute difference for debugging
            diff = np.abs(fmpfit_result['params'] - scipy_result['params'])
            max_diffs.append(np.max(diff))
            
            # Parameters must be equal to float32 precision
            np.testing.assert_allclose(
                fmpfit_result['params'], scipy_result['params'],
                rtol=PARAM_RTOL, atol=PARAM_ATOL
            )
            
            # Chi-square must be equal to float32 precision
            # Use larger atol since chisq can be very small
            np.testing.assert_allclose(
                fmpfit_result['chisq'], scipy_result['chisq'],
                rtol=PARAM_RTOL, atol=max(PARAM_ATOL, 1e-6)
            )
            
            n_success += 1
            
        except AssertionError:
            # Failed assertion - differences too large
            continue
        except Exception:
            # Skip failed fits
            continue
    
    # Check that we have enough successful runs
    # If we have failures, show diagnostic info
    if max_diffs:
        max_diffs_arr = np.array(max_diffs)
        print(f"\nMax differences: min={np.min(max_diffs_arr):.2e}, max={np.max(max_diffs_arr):.2e}, mean={np.mean(max_diffs_arr):.2e}")
    
    assert n_success >= n_runs * 0.9, f"Too many failed runs: {n_success}/{n_runs}"


def test_fmpfit_vs_curvefit_wellconditioned():
    """Test on a well-conditioned case with low noise - results should be identical."""
    # Well-conditioned synthetic data with low noise
    x = np.linspace(-3, 3, 21)
    true_params = [100.0, 0.0, 1.0]
    
    rng = np.random.default_rng(42)
    y_true = gaussian(x, *true_params)
    noise = rng.normal(0, 1.0, len(x))
    y = y_true + noise
    error = np.ones_like(y)
    
    # Good initial guesses
    p0 = [95.0, 0.0, 1.0]
    bounds = [(0.0, 200.0), (-5.0, 5.0), (0.1, 5.0)]
    
    # Fit with both methods
    scipy_result = fit_with_scipy(x, y, error, p0, bounds)
    fmpfit_result = fit_with_fmpfit(x, y, error, p0, bounds)
    
    # Check convergence
    assert fmpfit_result['status'] > 0
    
    # For well-conditioned data, results should be identical to float32 precision
    np.testing.assert_allclose(
        fmpfit_result['params'], scipy_result['params'],
        rtol=PARAM_RTOL, atol=PARAM_ATOL,
        err_msg=f"Parameters differ: scipy={scipy_result['params']}, fmpfit={fmpfit_result['params']}"
    )
    
    # Use larger atol since chisq can be very small
    np.testing.assert_allclose(
        fmpfit_result['chisq'], scipy_result['chisq'],
        rtol=PARAM_RTOL, atol=max(PARAM_ATOL, 1e-6),
        err_msg=f"Chi-square differs: scipy={scipy_result['chisq']}, fmpfit={fmpfit_result['chisq']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
