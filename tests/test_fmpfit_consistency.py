"""Test fmpfit consistency between f32/f64 and single/block variants.

Also includes regression tests to verify outputs don't change unexpectedly.

Also includes tests against scipy's curve_fit

Verifies that:
1. fmpfit_f32_ext matches fmpfit_f64_ext outputs (within f32 tolerance)
2. fmpfit_f32_block_ext matches fmpfit_f64_block_ext outputs
3. fmpfit_f32_ext matches fmpfit_f32_block_ext for single spectrum (exactly)
4. fmpfit_f64_ext matches fmpfit_f64_block_ext for single spectrum (exactly)
5. f32 single outputs remain stable (regression tests)

Note: f32 vs f64 comparisons allow for precision-related differences in:
- bestnorm/orignorm (f32 can't achieve same residuals as f64)
- niter/nfev (convergence may differ due to precision)
- status (may reach different termination conditions)
- xerror_scipy (depends on bestnorm which differs)
"""
import numpy as np
import pytest
from ftoolss.fmpfit import (
    fmpfit_f32_pywrap, fmpfit_f64_pywrap,
    fmpfit_f32_block_pywrap, fmpfit_f64_block_pywrap,
)

# Tolerance for f32 vs f64 comparisons (~7 significant digits for f32)
RTOL = 1e-4
ATOL = 1e-6

# Looser tolerance for parameters (may converge slightly differently)
PARAM_RTOL = 1e-3
PARAM_ATOL = 1e-5

# Random seed for reproducibility
SEED = 12345


def gaussian(x, i0, mu, sigma):
    """Gaussian function: i0 * exp(-((x - mu)^2) / (2 * sigma^2))"""
    return i0 * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def generate_clean_gaussian(n_points, i0, mu, sigma, error_level=0.1):
    """Generate clean Gaussian data with uniform errors."""
    x = np.linspace(-3, 3, n_points)
    y = gaussian(x, i0, mu, sigma)
    error = np.full(n_points, error_level)
    return x, y, error


def generate_noisy_gaussian(n_points, i0, mu, sigma, rng, noise_frac=0.05):
    """Generate Gaussian data with Gaussian noise."""
    x = np.linspace(-3, 3, n_points)
    y_true = gaussian(x, i0, mu, sigma)
    noise = rng.normal(0, noise_frac * i0, n_points)
    y = y_true + noise
    error = np.full(n_points, noise_frac * i0)
    return x, y, error


def generate_poisson_gaussian(n_points, i0, mu, sigma, rng):
    """Generate Gaussian data with Poisson noise (realistic photon counting)."""
    x = np.linspace(-2, 2, n_points)
    y_true = gaussian(x, i0, mu, sigma)
    y = rng.poisson(np.maximum(y_true, 0.1)).astype(float)
    error = np.sqrt(np.maximum(y, 1.0))
    return x, y, error


def make_parinfo(p0, bounds):
    """Create parinfo list from initial parameters and bounds."""
    return [
        {'value': p0[0], 'limits': [bounds[0, 0], bounds[0, 1]]},  # i0
        {'value': p0[1], 'limits': [bounds[1, 0], bounds[1, 1]]},  # mu
        {'value': p0[2], 'limits': [bounds[2, 0], bounds[2, 1]]},  # sigma
    ]


def compare_single_results(result_a, result_b, name_a, name_b, rtol=RTOL, atol=ATOL,
                           same_precision=True):
    """Compare two MPFitResult objects and assert they match.
    
    Parameters
    ----------
    same_precision : bool
        If True, expect exact integer matches (same precision comparison).
        If False, allow for precision-related differences in convergence.
    """
    # Compare array outputs - use PARAM_RTOL for best_params since convergence may differ
    param_rtol = rtol if same_precision else PARAM_RTOL
    param_atol = atol if same_precision else PARAM_ATOL
    
    np.testing.assert_allclose(
        result_a.best_params, result_b.best_params,
        rtol=param_rtol, atol=param_atol,
        err_msg=f"best_params mismatch between {name_a} and {name_b}"
    )
    np.testing.assert_allclose(
        result_a.xerror, result_b.xerror,
        rtol=param_rtol, atol=param_atol,
        err_msg=f"xerror mismatch between {name_a} and {name_b}"
    )
    
    # xerror_scipy comparison - only when same precision, as it depends on bestnorm
    # Use atol since xerror_scipy can be very small or zero
    if same_precision:
        np.testing.assert_allclose(
            result_a.xerror_scipy, result_b.xerror_scipy,
            rtol=rtol, atol=max(atol, 1e-9),
            err_msg=f"xerror_scipy mismatch between {name_a} and {name_b}"
        )
    else:
        # For f32 vs f64, just check relative shape of errors
        # Guard against division by zero when xerror_scipy is all zeros
        # (happens when bestnorm=0, i.e., perfect fit with chi2=0)
        if result_a.xerror_scipy[0] > 0 and result_b.xerror_scipy[0] > 0:
            np.testing.assert_allclose(
                result_a.xerror_scipy / result_a.xerror_scipy[0],
                result_b.xerror_scipy / result_b.xerror_scipy[0],
                rtol=param_rtol, atol=param_atol,
                err_msg=f"xerror_scipy relative shape mismatch between {name_a} and {name_b}"
            )
        else:
            # Both should have zero in same positions (both are zero when chi2=0)
            np.testing.assert_array_equal(
                result_a.xerror_scipy == 0,
                result_b.xerror_scipy == 0,
                err_msg=f"xerror_scipy zero positions mismatch between {name_a} and {name_b}"
            )
    
    # Use atol for residuals since they can be very close to zero for good fits
    np.testing.assert_allclose(
        result_a.resid, result_b.resid,
        rtol=param_rtol, atol=max(param_atol, 1e-6),
        err_msg=f"resid mismatch between {name_a} and {name_b}"
    )
    
    # Covariance - only when same precision
    # Use atol since covariance elements can be small
    if same_precision:
        np.testing.assert_allclose(
            result_a.covar, result_b.covar,
            rtol=rtol, atol=max(atol, 1e-9),
            err_msg=f"covar mismatch between {name_a} and {name_b}"
        )
    
    # bestnorm/orignorm - only when same precision
    # Use atol since bestnorm can be very small for perfect fits
    if same_precision:
        np.testing.assert_allclose(
            result_a.bestnorm, result_b.bestnorm,
            rtol=rtol, atol=max(atol, 1e-9),
            err_msg=f"bestnorm mismatch between {name_a} and {name_b}"
        )
        np.testing.assert_allclose(
            result_a.orignorm, result_b.orignorm,
            rtol=rtol, atol=max(atol, 1e-6),
            err_msg=f"orignorm mismatch between {name_a} and {name_b}"
        )
    
    # Compare integer outputs - only exact match when same precision
    if same_precision:
        assert result_a.niter == result_b.niter, \
            f"niter mismatch: {name_a}={result_a.niter}, {name_b}={result_b.niter}"
        assert result_a.nfev == result_b.nfev, \
            f"nfev mismatch: {name_a}={result_a.nfev}, {name_b}={result_b.nfev}"
        assert result_a.status == result_b.status, \
            f"status mismatch: {name_a}={result_a.status}, {name_b}={result_b.status}"
    else:
        # For f32 vs f64, just check both converged successfully (status > 0)
        assert result_a.status > 0, f"{name_a} did not converge (status={result_a.status})"
        assert result_b.status > 0, f"{name_b} did not converge (status={result_b.status})"
    
    # These should always match
    assert result_a.npar == result_b.npar, \
        f"npar mismatch: {name_a}={result_a.npar}, {name_b}={result_b.npar}"
    assert result_a.nfree == result_b.nfree, \
        f"nfree mismatch: {name_a}={result_a.nfree}, {name_b}={result_b.nfree}"
    assert result_a.npegged == result_b.npegged, \
        f"npegged mismatch: {name_a}={result_a.npegged}, {name_b}={result_b.npegged}"
    assert result_a.nfunc == result_b.nfunc, \
        f"nfunc mismatch: {name_a}={result_a.nfunc}, {name_b}={result_b.nfunc}"


def compare_block_results(result_a, result_b, name_a, name_b, rtol=RTOL, atol=ATOL,
                          same_precision=True):
    """Compare two block result dictionaries and assert they match.
    
    Parameters
    ----------
    same_precision : bool
        If True, expect exact integer matches (same precision comparison).
        If False, allow for precision-related differences in convergence.
    """
    param_rtol = rtol if same_precision else PARAM_RTOL
    param_atol = atol if same_precision else PARAM_ATOL
    
    # Compare array outputs
    np.testing.assert_allclose(
        result_a['best_params'], result_b['best_params'],
        rtol=param_rtol, atol=param_atol,
        err_msg=f"best_params mismatch between {name_a} and {name_b}"
    )
    np.testing.assert_allclose(
        result_a['xerror'], result_b['xerror'],
        rtol=param_rtol, atol=param_atol,
        err_msg=f"xerror mismatch between {name_a} and {name_b}"
    )
    
    # Use atol since xerror_scipy can be very small or zero
    if same_precision:
        np.testing.assert_allclose(
            result_a['xerror_scipy'], result_b['xerror_scipy'],
            rtol=rtol, atol=max(atol, 1e-9),
            err_msg=f"xerror_scipy mismatch between {name_a} and {name_b}"
        )
    else:
        # For f32 vs f64, check relative shape of errors per spectrum
        for i in range(result_a['xerror_scipy'].shape[0]):
            if result_a['xerror_scipy'][i, 0] != 0 and result_b['xerror_scipy'][i, 0] != 0:
                np.testing.assert_allclose(
                    result_a['xerror_scipy'][i] / result_a['xerror_scipy'][i, 0],
                    result_b['xerror_scipy'][i] / result_b['xerror_scipy'][i, 0],
                    rtol=param_rtol, atol=param_atol,
                    err_msg=f"xerror_scipy relative shape mismatch at spectrum {i}"
                )
    
    # Use atol for residuals since they can be very close to zero for good fits
    np.testing.assert_allclose(
        result_a['resid'], result_b['resid'],
        rtol=param_rtol, atol=max(param_atol, 1e-6),
        err_msg=f"resid mismatch between {name_a} and {name_b}"
    )
    
    # Use atol since these values can be small
    if same_precision:
        np.testing.assert_allclose(
            result_a['covar'], result_b['covar'],
            rtol=rtol, atol=max(atol, 1e-9),
            err_msg=f"covar mismatch between {name_a} and {name_b}"
        )
        np.testing.assert_allclose(
            result_a['bestnorm'], result_b['bestnorm'],
            rtol=rtol, atol=max(atol, 1e-9),
            err_msg=f"bestnorm mismatch between {name_a} and {name_b}"
        )
        np.testing.assert_allclose(
            result_a['orignorm'], result_b['orignorm'],
            rtol=rtol, atol=max(atol, 1e-6),
            err_msg=f"orignorm mismatch between {name_a} and {name_b}"
        )
    
    # Compare integer arrays
    if same_precision:
        np.testing.assert_array_equal(
            result_a['niter'], result_b['niter'],
            err_msg=f"niter mismatch between {name_a} and {name_b}"
        )
        np.testing.assert_array_equal(
            result_a['nfev'], result_b['nfev'],
            err_msg=f"nfev mismatch between {name_a} and {name_b}"
        )
        np.testing.assert_array_equal(
            result_a['status'], result_b['status'],
            err_msg=f"status mismatch between {name_a} and {name_b}"
        )
    else:
        # For f32 vs f64, just check both converged successfully
        assert np.all(result_a['status'] > 0), f"{name_a} has non-converged spectra"
        assert np.all(result_b['status'] > 0), f"{name_b} has non-converged spectra"
    
    # These should always match
    np.testing.assert_array_equal(
        result_a['npar'], result_b['npar'],
        err_msg=f"npar mismatch between {name_a} and {name_b}"
    )
    np.testing.assert_array_equal(
        result_a['nfree'], result_b['nfree'],
        err_msg=f"nfree mismatch between {name_a} and {name_b}"
    )
    np.testing.assert_array_equal(
        result_a['npegged'], result_b['npegged'],
        err_msg=f"npegged mismatch between {name_a} and {name_b}"
    )
    np.testing.assert_array_equal(
        result_a['nfunc'], result_b['nfunc'],
        err_msg=f"nfunc mismatch between {name_a} and {name_b}"
    )


# =============================================================================
# Test Case 1: Clean Gaussian data
# =============================================================================

class TestCleanGaussian:
    """Tests with clean (noiseless) Gaussian data."""
    
    @pytest.fixture
    def clean_data(self):
        """Generate clean Gaussian test data."""
        x, y, error = generate_clean_gaussian(
            n_points=7, i0=10.0, mu=0.5, sigma=1.2, error_level=0.1
        )
        p0 = np.array([8.0, 0.3, 1.0])
        bounds = np.array([[0.0, 50.0], [-2.0, 2.0], [0.3, 3.0]])
        return x, y, error, p0, bounds
    
    def test_f32_vs_f64_single(self, clean_data):
        """Test f32 vs f64 single-spectrum fitting on clean data."""
        x, y, error, p0, bounds = clean_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_f32_vs_f64_block(self, clean_data):
        """Test f32 vs f64 block fitting on clean data."""
        x, y, error, p0, bounds = clean_data
        # Reshape for block interface (1 spectrum)
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        
        result_f32 = fmpfit_f32_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        result_f64 = fmpfit_f64_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        compare_block_results(result_f32, result_f64, "f32_block", "f64_block", same_precision=False)
    
    def test_single_vs_block_f32(self, clean_data):
        """Test f32 single vs f32 block produce identical results."""
        x, y, error, p0, bounds = clean_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_single = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Block interface
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        result_block = fmpfit_f32_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        # Compare single result with first (only) spectrum of block result
        # These should be EXACTLY identical since same code path
        np.testing.assert_array_equal(
            result_single.best_params, result_block['best_params'][0],
            err_msg="best_params mismatch single vs block (f32)"
        )
        np.testing.assert_array_equal(
            result_single.xerror_scipy, result_block['xerror_scipy'][0],
            err_msg="xerror_scipy mismatch single vs block (f32)"
        )
        np.testing.assert_array_equal(
            result_single.resid, result_block['resid'][0],
            err_msg="resid mismatch single vs block (f32)"
        )
        np.testing.assert_array_equal(
            result_single.covar, result_block['covar'][0],
            err_msg="covar mismatch single vs block (f32)"
        )
        assert result_single.status == result_block['status'][0], \
            f"status mismatch: single={result_single.status}, block={result_block['status'][0]}"
        assert result_single.niter == result_block['niter'][0], \
            f"niter mismatch: single={result_single.niter}, block={result_block['niter'][0]}"
        assert result_single.nfev == result_block['nfev'][0], \
            f"nfev mismatch: single={result_single.nfev}, block={result_block['nfev'][0]}"
        assert result_single.bestnorm == result_block['bestnorm'][0], \
            f"bestnorm mismatch: single={result_single.bestnorm}, block={result_block['bestnorm'][0]}"
    
    def test_single_vs_block_f64(self, clean_data):
        """Test f64 single vs f64 block produce same results."""
        x, y, error, p0, bounds = clean_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_single = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Block interface
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        result_block = fmpfit_f64_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        # Compare single result with first (only) spectrum of block result
        np.testing.assert_allclose(
            result_single.best_params, result_block['best_params'][0],
            rtol=RTOL, atol=ATOL, err_msg="best_params mismatch single vs block (f64)"
        )
        np.testing.assert_allclose(
            result_single.xerror_scipy, result_block['xerror_scipy'][0],
            rtol=RTOL, atol=ATOL, err_msg="xerror_scipy mismatch single vs block (f64)"
        )
        np.testing.assert_allclose(
            result_single.resid, result_block['resid'][0],
            rtol=RTOL, atol=ATOL, err_msg="resid mismatch single vs block (f64)"
        )
        np.testing.assert_allclose(
            result_single.covar, result_block['covar'][0],
            rtol=RTOL, atol=ATOL, err_msg="covar mismatch single vs block (f64)"
        )
        assert result_single.status == result_block['status'][0]
        assert result_single.niter == result_block['niter'][0]


# =============================================================================
# Test Case 2: Noisy Gaussian data
# =============================================================================

class TestNoisyGaussian:
    """Tests with Gaussian noise added to data."""
    
    @pytest.fixture
    def noisy_data(self):
        """Generate noisy Gaussian test data."""
        rng = np.random.default_rng(SEED)
        x, y, error = generate_noisy_gaussian(
            n_points=9, i0=25.0, mu=-0.3, sigma=0.9, rng=rng, noise_frac=0.03
        )
        p0 = np.array([20.0, 0.0, 1.0])
        bounds = np.array([[0.0, 100.0], [-2.0, 2.0], [0.2, 3.0]])
        return x, y, error, p0, bounds
    
    def test_f32_vs_f64_single(self, noisy_data):
        """Test f32 vs f64 single-spectrum fitting on noisy data."""
        x, y, error, p0, bounds = noisy_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_f32_vs_f64_block(self, noisy_data):
        """Test f32 vs f64 block fitting on noisy data."""
        x, y, error, p0, bounds = noisy_data
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        
        result_f32 = fmpfit_f32_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        result_f64 = fmpfit_f64_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        compare_block_results(result_f32, result_f64, "f32_block", "f64_block", same_precision=False)
    
    def test_single_vs_block_f32(self, noisy_data):
        """Test f32 single vs f32 block produce identical results on noisy data."""
        x, y, error, p0, bounds = noisy_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_single = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        result_block = fmpfit_f32_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        # Should be EXACTLY identical
        np.testing.assert_array_equal(
            result_single.best_params, result_block['best_params'][0]
        )
        assert result_single.status == result_block['status'][0]
        assert result_single.niter == result_block['niter'][0]
        assert result_single.bestnorm == result_block['bestnorm'][0]


# =============================================================================
# Test Case 3: Poisson noise (realistic photon counting)
# =============================================================================

class TestPoissonGaussian:
    """Tests with Poisson noise (typical for photon counting detectors)."""
    
    @pytest.fixture
    def poisson_data(self):
        """Generate Gaussian with Poisson noise."""
        rng = np.random.default_rng(SEED + 1)
        x, y, error = generate_poisson_gaussian(
            n_points=5, i0=30.0, mu=0.2, sigma=0.7, rng=rng
        )
        p0 = np.array([25.0, 0.0, 0.8])
        bounds = np.array([[0.0, 100.0], [-2.0, 2.0], [0.3, 2.0]])
        return x, y, error, p0, bounds
    
    def test_f32_vs_f64_single(self, poisson_data):
        """Test f32 vs f64 single-spectrum fitting on Poisson data."""
        x, y, error, p0, bounds = poisson_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_f32_vs_f64_block(self, poisson_data):
        """Test f32 vs f64 block fitting on Poisson data."""
        x, y, error, p0, bounds = poisson_data
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        
        result_f32 = fmpfit_f32_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        result_f64 = fmpfit_f64_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        compare_block_results(result_f32, result_f64, "f32_block", "f64_block", same_precision=False)
    
    def test_single_vs_block_f32(self, poisson_data):
        """Test f32 single vs f32 block produce identical results on Poisson data."""
        x, y, error, p0, bounds = poisson_data
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_single = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        x_block = x.reshape(1, -1)
        y_block = y.reshape(1, -1)
        error_block = error.reshape(1, -1)
        p0_block = p0.reshape(1, -1)
        bounds_block = bounds.reshape(1, 3, 2)
        result_block = fmpfit_f32_block_pywrap(0, x_block, y_block, error_block, p0_block, bounds_block)
        
        # Should be EXACTLY identical
        np.testing.assert_array_equal(
            result_single.best_params, result_block['best_params'][0]
        )
        assert result_single.status == result_block['status'][0]
        assert result_single.niter == result_block['niter'][0]
        assert result_single.bestnorm == result_block['bestnorm'][0]


# =============================================================================
# Test Case 4: Multiple spectra block fitting
# =============================================================================

class TestMultipleSpectraBlock:
    """Tests with multiple spectra in block fitting."""
    
    @pytest.fixture
    def multi_spectra_data(self):
        """Generate multiple test spectra."""
        rng = np.random.default_rng(SEED + 2)
        n_spectra = 10
        n_points = 5
        
        x = np.tile(np.linspace(-2, 2, n_points), (n_spectra, 1))
        
        # Generate different spectra with varying parameters
        true_i0 = rng.uniform(15.0, 40.0, n_spectra)
        true_mu = rng.uniform(-0.5, 0.5, n_spectra)
        true_sigma = rng.uniform(0.6, 1.2, n_spectra)
        
        y = np.zeros((n_spectra, n_points))
        for i in range(n_spectra):
            y_true = gaussian(x[i], true_i0[i], true_mu[i], true_sigma[i])
            y[i] = rng.poisson(np.maximum(y_true, 0.1)).astype(float)
        
        error = np.sqrt(np.maximum(y, 1.0))
        
        # Initial guesses
        p0 = np.zeros((n_spectra, 3))
        bounds = np.zeros((n_spectra, 3, 2))
        for i in range(n_spectra):
            max_idx = np.argmax(y[i])
            p0[i] = [y[i, max_idx], x[i, max_idx], 0.8]
            bounds[i] = [[0.0, 100.0], [-2.0, 2.0], [0.3, 3.0]]
        
        return x, y, error, p0, bounds
    
    def test_f32_vs_f64_block_multi(self, multi_spectra_data):
        """Test f32 vs f64 block fitting with multiple spectra."""
        x, y, error, p0, bounds = multi_spectra_data
        
        result_f32 = fmpfit_f32_block_pywrap(0, x, y, error, p0, bounds)
        result_f64 = fmpfit_f64_block_pywrap(0, x, y, error, p0, bounds)
        
        compare_block_results(result_f32, result_f64, "f32_block", "f64_block", same_precision=False)
    
    def test_block_matches_individual_f64(self, multi_spectra_data):
        """Test that block result matches individual fits (f64)."""
        x, y, error, p0, bounds = multi_spectra_data
        n_spectra = x.shape[0]
        
        # Block fit
        result_block = fmpfit_f64_block_pywrap(0, x, y, error, p0, bounds)
        
        # Individual fits
        for i in range(n_spectra):
            parinfo = make_parinfo(p0[i], bounds[i])
            functkw = {'x': x[i], 'y': y[i], 'error': error[i]}
            result_single = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
            
            np.testing.assert_allclose(
                result_single.best_params, result_block['best_params'][i],
                rtol=RTOL, atol=ATOL,
                err_msg=f"best_params mismatch at spectrum {i}"
            )
            np.testing.assert_allclose(
                result_single.xerror_scipy, result_block['xerror_scipy'][i],
                rtol=RTOL, atol=ATOL,
                err_msg=f"xerror_scipy mismatch at spectrum {i}"
            )
            assert result_single.status == result_block['status'][i], \
                f"status mismatch at spectrum {i}"
    
    def test_block_matches_individual_f32(self, multi_spectra_data):
        """Test that block result matches individual fits exactly (f32)."""
        x, y, error, p0, bounds = multi_spectra_data
        n_spectra = x.shape[0]
        
        # Block fit
        result_block = fmpfit_f32_block_pywrap(0, x, y, error, p0, bounds)
        
        # Individual fits - should be EXACTLY identical
        for i in range(n_spectra):
            parinfo = make_parinfo(p0[i], bounds[i])
            functkw = {'x': x[i], 'y': y[i], 'error': error[i]}
            result_single = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
            
            np.testing.assert_array_equal(
                result_single.best_params, result_block['best_params'][i],
                err_msg=f"best_params mismatch at spectrum {i}"
            )
            np.testing.assert_array_equal(
                result_single.xerror_scipy, result_block['xerror_scipy'][i],
                err_msg=f"xerror_scipy mismatch at spectrum {i}"
            )
            assert result_single.status == result_block['status'][i], \
                f"status mismatch at spectrum {i}"
            assert result_single.niter == result_block['niter'][i], \
                f"niter mismatch at spectrum {i}"
            assert result_single.bestnorm == result_block['bestnorm'][i], \
                f"bestnorm mismatch at spectrum {i}"


# =============================================================================
# Test Case 5: Edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_narrow_gaussian(self):
        """Test fitting a narrow Gaussian (sigma near lower bound)."""
        x = np.linspace(-1, 1, 5)
        y = gaussian(x, 50.0, 0.0, 0.3)
        error = np.full(5, 1.0)
        p0 = np.array([40.0, 0.0, 0.5])
        bounds = np.array([[0.0, 100.0], [-1.0, 1.0], [0.1, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_wide_gaussian(self):
        """Test fitting a wide Gaussian (sigma near upper bound)."""
        x = np.linspace(-5, 5, 7)
        y = gaussian(x, 20.0, 0.0, 2.5)
        error = np.full(7, 0.5)
        p0 = np.array([15.0, 0.0, 2.0])
        bounds = np.array([[0.0, 50.0], [-3.0, 3.0], [0.5, 5.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_off_center_gaussian(self):
        """Test fitting a Gaussian off center."""
        x = np.linspace(-3, 3, 5)
        y = gaussian(x, 30.0, 1.5, 0.8)
        error = np.full(5, 1.0)
        p0 = np.array([25.0, 1.0, 1.0])
        bounds = np.array([[0.0, 100.0], [-3.0, 3.0], [0.3, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_large_amplitude(self):
        """Test fitting a Gaussian with large amplitude."""
        x = np.linspace(-2, 2, 5)
        y = gaussian(x, 1000.0, 0.0, 0.5)
        error = np.full(5, 10.0)
        p0 = np.array([800.0, 0.0, 0.6])
        bounds = np.array([[0.0, 2000.0], [-2.0, 2.0], [0.1, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)
    
    def test_small_amplitude(self):
        """Test fitting a Gaussian with small amplitude."""
        x = np.linspace(-2, 2, 5)
        y = gaussian(x, 1.0, 0.0, 1.0)
        error = np.full(5, 0.05)
        p0 = np.array([0.8, 0.0, 1.2])
        bounds = np.array([[0.0, 5.0], [-2.0, 2.0], [0.3, 3.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        result_f64 = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        compare_single_results(result_f32, result_f64, "f32", "f64", same_precision=False)


# =============================================================================
# Regression tests: Verify f32 single outputs remain stable
# =============================================================================

class TestRegressionF32:
    """Regression tests to ensure f32 outputs don't change unexpectedly.
    
    These tests lock in the current output values to detect any changes
    in the fitting algorithm or implementation.
    """
    
    def test_clean_gaussian_regression(self):
        """Regression test for clean Gaussian fitting (f32)."""
        # Exact same setup as TestCleanGaussian
        x = np.linspace(-3, 3, 7)
        y = gaussian(x, 10.0, 0.5, 1.2)
        error = np.full(7, 0.1)
        p0 = np.array([8.0, 0.3, 1.0])
        bounds = np.array([[0.0, 50.0], [-2.0, 2.0], [0.3, 3.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 2, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 1.0551560319926878e-10, rtol=1e-6, atol=5e-7,
            err_msg="bestnorm changed"
        )
        np.testing.assert_allclose(
            result.best_params, [9.999999, 0.5, 1.2000002], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        np.testing.assert_allclose(
            result.xerror, [0.08412183, 0.011645826, 0.01171242], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_noisy_gaussian_regression(self):
        """Regression test for noisy Gaussian fitting (f32)."""
        rng = np.random.default_rng(SEED)
        n_points = 9
        i0, mu, sigma = 25.0, -0.3, 0.9
        x = np.linspace(-3, 3, n_points)
        y_true = gaussian(x, i0, mu, sigma)
        noise = rng.normal(0, 0.03 * y_true.max(), n_points)
        y = y_true + noise
        error = np.full(n_points, 0.03 * y_true.max())
        p0 = np.array([20.0, 0.0, 1.0])
        bounds = np.array([[0.0, 100.0], [-2.0, 2.0], [0.2, 3.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 1, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 5.970493316650391, rtol=1e-6, atol=5e-7,
            err_msg="bestnorm changed"
        )
        np.testing.assert_allclose(
            result.best_params, [24.93502, -0.30934057, 0.87861663], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        np.testing.assert_allclose(
            result.xerror, [0.60309494, 0.024534091, 0.024543945], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_poisson_gaussian_regression(self):
        """Regression test for Poisson-noise Gaussian fitting (f32)."""
        rng = np.random.default_rng(SEED + 1)
        n_points = 5
        i0, mu, sigma = 30.0, 0.2, 0.7
        x = np.linspace(-2, 2, n_points)
        y_true = gaussian(x, i0, mu, sigma)
        y = rng.poisson(np.maximum(y_true, 0.1)).astype(float)
        error = np.sqrt(np.maximum(y, 1.0))
        p0 = np.array([25.0, 0.0, 0.8])
        bounds = np.array([[0.0, 100.0], [-2.0, 2.0], [0.3, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 3, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 0.019765639677643776, rtol=1e-6, atol=1e-9,
            err_msg="bestnorm changed"
        )
        np.testing.assert_allclose(
            result.best_params, [32.63615, 0.3731333, 0.6239251], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        np.testing.assert_allclose(
            result.xerror, [5.700121, 0.08752606, 0.06496122], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_sigma_hits_lower_bound_regression(self):
        """Regression test: sigma hits lower bound (f32).
        
        True sigma=0.2 but lower bound=0.3, so fit pegs at bound.
        """
        x = np.linspace(-1, 1, 5)
        y = gaussian(x, 50.0, 0.0, 0.2)  # True sigma < lower bound
        error = np.full(5, 1.0)
        p0 = np.array([40.0, 0.0, 0.5])
        bounds = np.array([[0.0, 100.0], [-1.0, 1.0], [0.3, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 4, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 187.7047882080078, rtol=1e-6, atol=5e-7,
            err_msg="bestnorm changed"
        )
        # sigma should be at lower bound (0.3)
        np.testing.assert_allclose(
            result.best_params, [45.44322, 0.0, 0.3], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        # xerror[2] (sigma) should be 0 because it hit bound
        np.testing.assert_allclose(
            result.xerror, [0.9430677, 0.011227073, 0.0], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_amplitude_hits_upper_bound_regression(self):
        """Regression test: amplitude hits upper bound (f32).
        
        True amplitude=200 but upper bound=100, so fit pegs at bound.
        """
        x = np.linspace(-2, 2, 7)
        y = gaussian(x, 200.0, 0.0, 1.0)  # True amplitude > upper bound
        error = np.full(7, 5.0)
        p0 = np.array([80.0, 0.0, 1.0])
        bounds = np.array([[0.0, 100.0], [-2.0, 2.0], [0.3, 3.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 1, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 812.83447265625, rtol=1e-6, atol=5e-7,
            err_msg="bestnorm changed"
        )
        # amplitude should be at upper bound (100), sigma compensates by being wider
        np.testing.assert_allclose(
            result.best_params, [100.0, 0.0, 1.6541132], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        # xerror[0] (amplitude) should be 0 because it hit bound
        np.testing.assert_allclose(
            result.xerror, [0.0, 0.064807326, 0.06819587], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_mean_hits_lower_bound_regression(self):
        """Regression test: mean hits lower bound (f32).
        
        True mean=-2.5 but lower bound=-2.0, so fit pegs at bound.
        """
        x = np.linspace(-3, 3, 7)
        y = gaussian(x, 30.0, -2.5, 0.8)  # True mean < lower bound
        error = np.full(7, 1.0)
        p0 = np.array([25.0, -1.0, 1.0])
        bounds = np.array([[0.0, 100.0], [-2.0, 2.0], [0.3, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 1, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 196.8758087158203, rtol=1e-6, atol=5e-7,
            err_msg="bestnorm changed"
        )
        # mean should be at lower bound (-2.0)
        np.testing.assert_allclose(
            result.best_params, [25.3346, -2.0, 0.9260391], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        # xerror[1] (mean) should be 0 because it hit bound
        np.testing.assert_allclose(
            result.xerror, [0.96681243, 0.0, 0.044000473], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_two_params_hit_bounds_regression(self):
        """Regression test: two parameters hit bounds (f32).
        
        True amplitude=150 > upper=100 and true sigma=0.2 < lower=0.3.
        """
        x = np.linspace(-1, 1, 5)
        y = gaussian(x, 150.0, 0.0, 0.2)
        error = np.full(5, 2.0)
        p0 = np.array([80.0, 0.0, 0.5])
        bounds = np.array([[0.0, 100.0], [-1.0, 1.0], [0.3, 2.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 4, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 793.3380737304688, rtol=1e-6, atol=5e-7,
            err_msg="bestnorm changed"
        )
        # amplitude at upper bound (100), sigma at lower bound (0.3)
        np.testing.assert_allclose(
            result.best_params, [100.0, 0.0, 0.3], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        # xerror[0] and xerror[2] should be 0 because they hit bounds
        np.testing.assert_allclose(
            result.xerror, [0.0, 0.010203887, 0.0], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )
    
    def test_very_small_signal_regression(self):
        """Regression test: very small signal near noise level (f32)."""
        x = np.linspace(-2, 2, 5)
        y = gaussian(x, 0.5, 0.0, 1.0)  # Very small amplitude
        error = np.full(5, 0.1)
        p0 = np.array([0.3, 0.0, 1.0])
        bounds = np.array([[0.0, 10.0], [-2.0, 2.0], [0.3, 3.0]])
        
        parinfo = make_parinfo(p0, bounds)
        functkw = {'x': x, 'y': y, 'error': error}
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # Expected values (recorded 2026-01-06)
        assert result.status == 4, f"status changed: {result.status}"
        np.testing.assert_allclose(
            result.bestnorm, 0.0, atol=1e-10,
            err_msg="bestnorm changed"
        )
        np.testing.assert_allclose(
            result.best_params, [0.5, 0.0, 1.0], rtol=1e-5, atol=5e-7,
            err_msg="best_params changed"
        )
        np.testing.assert_allclose(
            result.xerror, [0.091921106, 0.21292457, 0.21287873], rtol=1e-5, atol=5e-7,
            err_msg="xerror changed"
        )


# =============================================================================
# Comparison tests: f32 single vs scipy curve_fit
# =============================================================================

class TestF32VsScipy:
    """Tests comparing fmpfit f32 single-fit against scipy curve_fit.
    
    Uses xerror_scipy from fmpfit which should match scipy's error estimates
    with absolute_sigma=False (scipy's default), which scales errors by
    sqrt(chi2/dof).
    
    Note: When bestnorm (chi-square) is very close to zero (near-perfect fit),
    the error estimates can differ because the scaling factor becomes unstable.
    These tests use data with realistic noise levels.
    """
    
    # Tolerance for scipy comparison
    SCIPY_PARAM_RTOL = 1e-3   # Parameters should match within 0.1%
    SCIPY_ERROR_RTOL = 1e-4   # Errors should match very closely (same scaling)
    SCIPY_ATOL = 1e-4         # Absolute tolerance for scipy comparisons
    
    @staticmethod
    def scipy_gaussian(x, i0, mu, sigma):
        """Gaussian function for scipy curve_fit."""
        return i0 * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def test_noisy_gaussian_vs_scipy(self):
        """Compare fmpfit f32 vs scipy on noisy Gaussian data."""
        from scipy.optimize import curve_fit
        
        rng = np.random.default_rng(SEED)
        n_points = 9
        x = np.linspace(-3, 3, n_points)
        y_true = gaussian(x, 25.0, -0.3, 0.9)
        noise = rng.normal(0, 0.75, n_points)
        y = y_true + noise
        error = np.full(n_points, 0.75)
        p0 = np.array([20.0, 0.0, 1.0])
        bounds_arr = np.array([[0.0, 100.0], [-2.0, 2.0], [0.2, 3.0]])
        
        # fmpfit f32
        parinfo = make_parinfo(p0, bounds_arr)
        functkw = {'x': x, 'y': y, 'error': error}
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # scipy (absolute_sigma=False is default, matches xerror_scipy scaling)
        bounds_scipy = ([b[0] for b in bounds_arr], [b[1] for b in bounds_arr])
        popt, pcov = curve_fit(
            self.scipy_gaussian, x, y, p0=p0, sigma=error, 
            absolute_sigma=False, bounds=bounds_scipy
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Parameters should match
        np.testing.assert_allclose(
            result_f32.best_params, popt, rtol=self.SCIPY_PARAM_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit f32 params don't match scipy"
        )
        # xerror_scipy should match scipy errors
        np.testing.assert_allclose(
            result_f32.xerror_scipy, perr, rtol=self.SCIPY_ERROR_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit xerror_scipy doesn't match scipy perr"
        )
    
    def test_poisson_gaussian_vs_scipy(self):
        """Compare fmpfit f32 vs scipy on Poisson-noise Gaussian data."""
        from scipy.optimize import curve_fit
        
        rng = np.random.default_rng(SEED + 100)
        n_points = 7
        x = np.linspace(-2, 2, n_points)
        y_true = gaussian(x, 50.0, 0.1, 0.8)
        y = rng.poisson(np.maximum(y_true, 1.0)).astype(float)
        error = np.sqrt(np.maximum(y, 1.0))
        p0 = np.array([40.0, 0.0, 1.0])
        bounds_arr = np.array([[0.0, 200.0], [-2.0, 2.0], [0.3, 3.0]])
        
        # fmpfit f32
        parinfo = make_parinfo(p0, bounds_arr)
        functkw = {'x': x, 'y': y, 'error': error}
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # scipy (absolute_sigma=False is default, matches xerror_scipy scaling)
        bounds_scipy = ([b[0] for b in bounds_arr], [b[1] for b in bounds_arr])
        popt, pcov = curve_fit(
            self.scipy_gaussian, x, y, p0=p0, sigma=error, 
            absolute_sigma=False, bounds=bounds_scipy
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Parameters should match
        np.testing.assert_allclose(
            result_f32.best_params, popt, rtol=self.SCIPY_PARAM_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit f32 params don't match scipy"
        )
        # xerror_scipy should match scipy errors
        np.testing.assert_allclose(
            result_f32.xerror_scipy, perr, rtol=self.SCIPY_ERROR_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit xerror_scipy doesn't match scipy perr"
        )
    
    def test_wide_gaussian_vs_scipy(self):
        """Compare fmpfit f32 vs scipy on wide Gaussian with more points."""
        from scipy.optimize import curve_fit
        
        rng = np.random.default_rng(SEED + 200)
        n_points = 15
        x = np.linspace(-5, 5, n_points)
        y_true = gaussian(x, 20.0, 0.5, 2.0)
        noise = rng.normal(0, 1.0, n_points)
        y = y_true + noise
        error = np.full(n_points, 1.0)
        p0 = np.array([15.0, 0.0, 1.5])
        bounds_arr = np.array([[0.0, 50.0], [-3.0, 3.0], [0.5, 5.0]])
        
        # fmpfit f32
        parinfo = make_parinfo(p0, bounds_arr)
        functkw = {'x': x, 'y': y, 'error': error}
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # scipy (absolute_sigma=False is default, matches xerror_scipy scaling)
        bounds_scipy = ([b[0] for b in bounds_arr], [b[1] for b in bounds_arr])
        popt, pcov = curve_fit(
            self.scipy_gaussian, x, y, p0=p0, sigma=error, 
            absolute_sigma=False, bounds=bounds_scipy
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Parameters should match
        np.testing.assert_allclose(
            result_f32.best_params, popt, rtol=self.SCIPY_PARAM_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit f32 params don't match scipy"
        )
        # xerror_scipy should match scipy errors
        np.testing.assert_allclose(
            result_f32.xerror_scipy, perr, rtol=self.SCIPY_ERROR_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit xerror_scipy doesn't match scipy perr"
        )
    
    def test_off_center_gaussian_vs_scipy(self):
        """Compare fmpfit f32 vs scipy on off-center Gaussian."""
        from scipy.optimize import curve_fit
        
        rng = np.random.default_rng(SEED + 300)
        n_points = 9
        x = np.linspace(-3, 3, n_points)
        y_true = gaussian(x, 30.0, 1.2, 0.7)
        noise = rng.normal(0, 2.0, n_points)
        y = y_true + noise
        error = np.full(n_points, 2.0)
        p0 = np.array([25.0, 1.0, 1.0])
        bounds_arr = np.array([[0.0, 100.0], [-3.0, 3.0], [0.3, 2.0]])
        
        # fmpfit f32
        parinfo = make_parinfo(p0, bounds_arr)
        functkw = {'x': x, 'y': y, 'error': error}
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # scipy (absolute_sigma=False is default, matches xerror_scipy scaling)
        bounds_scipy = ([b[0] for b in bounds_arr], [b[1] for b in bounds_arr])
        popt, pcov = curve_fit(
            self.scipy_gaussian, x, y, p0=p0, sigma=error, 
            absolute_sigma=False, bounds=bounds_scipy
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Parameters should match
        np.testing.assert_allclose(
            result_f32.best_params, popt, rtol=self.SCIPY_PARAM_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit f32 params don't match scipy"
        )
        # xerror_scipy should match scipy errors
        np.testing.assert_allclose(
            result_f32.xerror_scipy, perr, rtol=self.SCIPY_ERROR_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit xerror_scipy doesn't match scipy perr"
        )
    
    def test_high_snr_gaussian_vs_scipy(self):
        """Compare fmpfit f32 vs scipy on high-SNR Gaussian."""
        from scipy.optimize import curve_fit
        
        rng = np.random.default_rng(SEED + 400)
        n_points = 9
        x = np.linspace(-3, 3, n_points)
        y_true = gaussian(x, 50.0, 0.2, 1.1)  # Strong signal
        noise = rng.normal(0, 1.5, n_points)  # Moderate noise
        y = y_true + noise
        error = np.full(n_points, 1.5)
        p0 = np.array([45.0, 0.0, 1.0])
        bounds_arr = np.array([[0.0, 100.0], [-2.0, 2.0], [0.3, 3.0]])
        
        # fmpfit f32
        parinfo = make_parinfo(p0, bounds_arr)
        functkw = {'x': x, 'y': y, 'error': error}
        result_f32 = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
        
        # scipy (absolute_sigma=False is default, matches xerror_scipy scaling)
        bounds_scipy = ([b[0] for b in bounds_arr], [b[1] for b in bounds_arr])
        popt, pcov = curve_fit(
            self.scipy_gaussian, x, y, p0=p0, sigma=error, 
            absolute_sigma=False, bounds=bounds_scipy
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Parameters should match
        np.testing.assert_allclose(
            result_f32.best_params, popt, rtol=self.SCIPY_PARAM_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit f32 params don't match scipy"
        )
        # xerror_scipy should match scipy errors
        np.testing.assert_allclose(
            result_f32.xerror_scipy, perr, rtol=self.SCIPY_ERROR_RTOL, atol=self.SCIPY_ATOL,
            err_msg="fmpfit xerror_scipy doesn't match scipy perr"
        )
    

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
