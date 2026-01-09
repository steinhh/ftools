"""Test comparing fmpfit_block with scipy.optimize.curve_fit.

Verifies that fmpfit_f64_block_pywrap produces results nearly identical to
scipy's curve_fit when fitting multiple spectra.
"""
import numpy as np
import pytest
from scipy.optimize import curve_fit

from ftoolss.fmpfit import fmpfit_f64_block_pywrap, fmpfit_f32_block_pywrap

# Tolerance for parameter comparison
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


def fit_spectrum_with_scipy(x, y, error, p0, bounds):
    """Fit a single spectrum using scipy.optimize.curve_fit."""
    scipy_bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
    
    popt, pcov = curve_fit(
        gaussian, x, y, p0=p0, bounds=scipy_bounds,
        sigma=error, absolute_sigma=True, jac=gaussian_jacobian
    )
    residuals = y - gaussian(x, *popt)
    chisq = np.sum((residuals / error) ** 2)
    
    return {
        'params': popt,
        'chisq': chisq
    }


def generate_test_spectra(n_spectra, n_points, rng):
    """Generate synthetic Gaussian spectra with Poisson noise."""
    x = np.tile(np.linspace(-2, 2, n_points), (n_spectra, 1))
    
    # Random true parameters for each spectrum
    true_I = rng.uniform(20.0, 50.0, n_spectra)
    true_mu = rng.uniform(-0.5, 0.5, n_spectra)
    true_sigma = rng.uniform(0.8, 1.5, n_spectra)
    
    # Generate noisy data
    y = np.zeros((n_spectra, n_points))
    for i in range(n_spectra):
        y_true = gaussian(x[i], true_I[i], true_mu[i], true_sigma[i])
        y[i] = rng.poisson(np.maximum(y_true, 0.1)).astype(float)
    
    error = np.sqrt(np.maximum(y, 1.0))
    
    # Initial guesses
    p0 = np.zeros((n_spectra, 3))
    bounds = np.zeros((n_spectra, 3, 2))
    
    for i in range(n_spectra):
        max_idx = np.argmax(y[i])
        p0[i] = [y[i, max_idx], x[i, max_idx], 1.0]
        I_upper = y[i, max_idx] + 2.0 * error[i, max_idx]
        bounds[i] = [[0.0, I_upper], [x[i, 0], x[i, -1]], [0.5, 3.0]]
    
    return x, y, error, p0, bounds


@pytest.mark.parametrize("n_spectra", [5, 10, 20])
def test_fmpfit_block_vs_curvefit(n_spectra):
    """Test that fmpfit_block and curve_fit produce identical results."""
    rng = np.random.default_rng(42)
    n_points = 5
    
    # Generate test data
    x, y, error, p0, bounds = generate_test_spectra(n_spectra, n_points, rng)
    
    # Fit all spectra with fmpfit_block
    block_result = fmpfit_f64_block_pywrap(0, x, y, error, p0, bounds)
    
    # Fit each spectrum with scipy and compare
    for i in range(n_spectra):
        # Skip if fmpfit didn't converge for this spectrum
        if block_result['status'][i] <= 0:
            continue
        
        # Fit with scipy
        bounds_i = [(bounds[i, j, 0], bounds[i, j, 1]) for j in range(3)]
        scipy_result = fit_spectrum_with_scipy(
            x[i], y[i], error[i], p0[i], bounds_i
        )
        
        # Compare parameters
        np.testing.assert_allclose(
            block_result['best_params'][i], scipy_result['params'],
            rtol=PARAM_RTOL, atol=PARAM_ATOL,
            err_msg=f"Spectrum {i}: params differ - "
                    f"fmpfit={block_result['best_params'][i]}, scipy={scipy_result['params']}"
        )
        
        # Compare chi-square - use larger atol since bestnorm can be very small
        np.testing.assert_allclose(
            block_result['bestnorm'][i], scipy_result['chisq'],
            rtol=PARAM_RTOL, atol=max(PARAM_ATOL, 1e-6),
            err_msg=f"Spectrum {i}: chisq differs - "
                    f"fmpfit={block_result['bestnorm'][i]}, scipy={scipy_result['chisq']}"
        )


def test_fmpfit_block_vs_curvefit_large():
    """Test fmpfit_block vs curve_fit on a larger batch - all must match strictly."""
    rng = np.random.default_rng(123)
    n_spectra = 50
    n_points = 7
    
    # Generate test data
    x, y, error, p0, bounds = generate_test_spectra(n_spectra, n_points, rng)
    
    # Fit all spectra with fmpfit_block
    block_result = fmpfit_f64_block_pywrap(0, x, y, error, p0, bounds)
    
    # Fit each spectrum with scipy and compare strictly
    n_converged = 0
    for i in range(n_spectra):
        if block_result['status'][i] <= 0:
            continue
        n_converged += 1
        
        # Fit with scipy
        bounds_i = [(bounds[i, j, 0], bounds[i, j, 1]) for j in range(3)]
        scipy_result = fit_spectrum_with_scipy(
            x[i], y[i], error[i], p0[i], bounds_i
        )
        
        # Strict comparison - parameters must match
        np.testing.assert_allclose(
            block_result['best_params'][i], scipy_result['params'],
            rtol=PARAM_RTOL, atol=PARAM_ATOL,
            err_msg=f"Spectrum {i}: params differ"
        )
        
        # Strict comparison - chi-square must match
        # Use larger atol since bestnorm can be very small
        np.testing.assert_allclose(
            block_result['bestnorm'][i], scipy_result['chisq'],
            rtol=PARAM_RTOL, atol=max(PARAM_ATOL, 1e-6),
            err_msg=f"Spectrum {i}: chisq differs"
        )
    
    # At least 90% should converge
    assert n_converged >= n_spectra * 0.9, f"Too few converged fits: {n_converged}/{n_spectra}"


def test_fmpfit_block_f32_vs_curvefit():
    """Test that fmpfit_f32_block also produces results close to curve_fit."""
    rng = np.random.default_rng(456)
    n_spectra = 10
    n_points = 5
    
    # Generate test data
    x, y, error, p0, bounds = generate_test_spectra(n_spectra, n_points, rng)
    
    # Convert to float32
    x_f32 = x.astype(np.float32)
    y_f32 = y.astype(np.float32)
    error_f32 = error.astype(np.float32)
    p0_f32 = p0.astype(np.float32)
    bounds_f32 = bounds.astype(np.float32)
    
    # Fit with fmpfit_f32_block
    block_result = fmpfit_f32_block_pywrap(
        0, x_f32, y_f32, error_f32, p0_f32, bounds_f32
    )
    
    # Compare against scipy (using float64 for reference)
    for i in range(n_spectra):
        if block_result['status'][i] <= 0:
            continue
        
        # Fit with scipy (float64)
        bounds_i = [(bounds[i, j, 0], bounds[i, j, 1]) for j in range(3)]
        scipy_result = fit_spectrum_with_scipy(
            x[i], y[i], error[i], p0[i], bounds_i
        )
        
        # Same strict tolerance as f64
        np.testing.assert_allclose(
            block_result['best_params'][i], scipy_result['params'],
            rtol=PARAM_RTOL, atol=PARAM_ATOL,
            err_msg=f"Spectrum {i}: f32 params differ from scipy"
        )


def test_fmpfit_block_consistency_f32_f64():
    """Test that f32 and f64 block results are close to each other."""
    rng = np.random.default_rng(789)
    n_spectra = 10
    n_points = 5
    
    # Generate test data
    x, y, error, p0, bounds = generate_test_spectra(n_spectra, n_points, rng)
    
    # Fit with f64
    result_f64 = fmpfit_f64_block_pywrap(0, x, y, error, p0, bounds)
    
    # Convert to float32 and fit
    x_f32 = x.astype(np.float32)
    y_f32 = y.astype(np.float32)
    error_f32 = error.astype(np.float32)
    p0_f32 = p0.astype(np.float32)
    bounds_f32 = bounds.astype(np.float32)
    
    result_f32 = fmpfit_f32_block_pywrap(
        0, x_f32, y_f32, error_f32, p0_f32, bounds_f32
    )
    
    # Compare results for converged spectra
    for i in range(n_spectra):
        if result_f64['status'][i] <= 0 or result_f32['status'][i] <= 0:
            continue
        
        # Parameters should be close - same tolerance
        np.testing.assert_allclose(
            result_f32['best_params'][i], result_f64['best_params'][i],
            rtol=PARAM_RTOL, atol=PARAM_ATOL,
            err_msg=f"Spectrum {i}: f32 and f64 results differ"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
