#!/usr/bin/env python
"""
Golden-value (regression) tests for fmpfit extensions.

Each test uses fixed input data and exact expected output, so any change
to the C implementation that alters numerical results will be caught.
Tolerances are tight (rtol=1e-6 for f64, rtol=1e-5 for f32) to allow
only negligible floating-point variation across platforms.

Test coverage:
  - fmpfit_f64_pywrap: noise-free exact recovery, seeded Poisson noise
  - fmpfit_f32_pywrap: seeded Poisson noise, parameter-at-bound (bugcase)
  - fmpfit_f64_block_pywrap: block fit, 3 spectra, seeded Poisson noise
  - fmpfit_f32_block_pywrap: block fit, 3 spectra, seeded Poisson noise
"""

import numpy as np
import pytest

from ftoolss import fmpfit_f64_pywrap, fmpfit_f32_pywrap
from ftoolss.fmpfit import fmpfit_f64_block_pywrap, fmpfit_f32_block_pywrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_5pt_f32(seed):
    """Return (x, y, error) for a 5-point f32 Gaussian with Poisson noise."""
    true_params = np.array([2.5, 0.0, 0.8], dtype=np.float32)
    fwhm = float(2.355 * true_params[2])
    x = np.linspace(-fwhm / 2, fwhm / 2, 5, dtype=np.float32)
    rng = np.random.default_rng(seed)
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2]) ** 2)
    y = rng.poisson(y_true).astype(np.float32)
    err = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)
    return x, y, err


def _gaussian_5pt_f64(seed):
    """Return (x, y, error) for a 5-point f64 Gaussian with Poisson noise."""
    true_params = np.array([2.5, 0.0, 0.8], dtype=np.float64)
    fwhm = float(2.355 * true_params[2])
    x = np.linspace(-fwhm / 2, fwhm / 2, 5, dtype=np.float64)
    rng = np.random.default_rng(seed)
    y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2]) ** 2)
    y = rng.poisson(y_true).astype(np.float64)
    err = np.sqrt(np.maximum(y, 1.0)).astype(np.float64)
    return x, y, err


# ---------------------------------------------------------------------------
# f64 single-spectrum tests
# ---------------------------------------------------------------------------

class TestFmpfitF64Golden:
    """Regression tests for fmpfit_f64_pywrap."""

    def test_noisefree_exact_recovery(self):
        """Noise-free Gaussian: fitter must recover exact parameters to high precision."""
        x = np.linspace(-3.0, 3.0, 31, dtype=np.float64)
        y = 3.0 * np.exp(-0.5 * (x / 1.2) ** 2)
        err = np.ones_like(x) * 0.01
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-3.0, 3.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        # Parameters must be recovered to < 1e-10 absolute error
        np.testing.assert_allclose(result.best_params[0], 3.0, atol=1e-10,
                                   err_msg="amplitude")
        np.testing.assert_allclose(result.best_params[1], 0.0, atol=1e-10,
                                   err_msg="mean")
        np.testing.assert_allclose(result.best_params[2], 1.2, atol=1e-10,
                                   err_msg="sigma")

        # Residuals are all near zero so chi-square must be essentially 0
        assert result.bestnorm < 1e-20, f"bestnorm too large: {result.bestnorm}"
        assert result.status in (1, 2, 3, 4), f"unexpected status {result.status}"

    def test_noisefree_metadata(self):
        """Noise-free fit: result metadata must match known values exactly."""
        x = np.linspace(-3.0, 3.0, 31, dtype=np.float64)
        y = 3.0 * np.exp(-0.5 * (x / 1.2) ** 2)
        err = np.ones_like(x) * 0.01
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-3.0, 3.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        assert result.nfunc == 31
        assert result.npar == 3
        assert result.nfree == 3
        assert result.npegged == 0

    def test_noisefree_xerror_golden(self):
        """Noise-free fit: parameter uncertainties must match golden values."""
        x = np.linspace(-3.0, 3.0, 31, dtype=np.float64)
        y = 3.0 * np.exp(-0.5 * (x / 1.2) ** 2)
        err = np.ones_like(x) * 0.01
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-3.0, 3.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        expected_xerror = np.array([0.00376818, 0.001738, 0.00175788])
        np.testing.assert_allclose(result.xerror, expected_xerror, rtol=1e-4,
                                   err_msg="xerror golden values mismatch")

    def test_5pt_poisson_seed42_params(self):
        """f64, 5-point Poisson noise (seed=42): fixate best_params golden values."""
        # Known input (seed=42 Poisson draws from amp=2.5, mean=0, sigma=0.8)
        x = np.array([-0.942, -0.471,  0.,  0.471,  0.942])
        y = np.array([2., 1., 1., 3., 2.])
        err = np.sqrt(np.maximum(y, 1.0))
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-2.0, 2.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        expected = np.array([2.10086825, 2.0, 2.57348856])
        np.testing.assert_allclose(result.best_params, expected, rtol=1e-6,
                                   err_msg="best_params golden mismatch (f64 5pt seed42)")
        assert result.status == 1

    def test_5pt_poisson_seed42_bestnorm(self):
        """f64, 5-point Poisson noise (seed=42): fixate bestnorm golden value."""
        x = np.array([-0.942, -0.471,  0.,  0.471,  0.942])
        y = np.array([2., 1., 1., 3., 2.])
        err = np.sqrt(np.maximum(y, 1.0))
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-2.0, 2.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        np.testing.assert_allclose(result.bestnorm, 1.3372162756721824, rtol=1e-6,
                                   err_msg="bestnorm golden mismatch (f64 5pt seed42)")

    def test_5pt_poisson_seed42_xerror_scipy(self):
        """f64, 5-point (seed=42): xerror_scipy golden values (vel is at bound, so
        xerror[1] == 0 while xerror_scipy[1] is large but finite)."""
        x = np.array([-0.942, -0.471,  0.,  0.471,  0.942])
        y = np.array([2., 1., 1., 3., 2.])
        err = np.sqrt(np.maximum(y, 1.0))
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-2.0, 2.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        # mean is pegged at upper bound (2.0), so xerror[1] is 0
        assert result.xerror[1] == pytest.approx(0.0, abs=1e-10), "xerror[1] should be 0 when pegged at bound"
        # xerror_scipy uses Hessian inverse - should be large but finite
        expected_scipy = np.array([6.14382867, 21.86389779, 14.82509948])
        np.testing.assert_allclose(result.xerror_scipy, expected_scipy, rtol=1e-5,
                                   err_msg="xerror_scipy golden mismatch (f64 5pt seed42)")


# ---------------------------------------------------------------------------
# f32 single-spectrum tests
# ---------------------------------------------------------------------------

class TestFmpfitF32Golden:
    """Regression tests for fmpfit_f32_pywrap."""

    def test_5pt_poisson_seed42_params(self):
        """f32, 5-point Poisson noise (seed=42): fixate best_params golden values."""
        x = np.array([-0.94200003, -0.47100002, 0., 0.47100002, 0.94200003], dtype=np.float32)
        y = np.array([2., 1., 1., 3., 2.], dtype=np.float32)
        err = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-2.0, 2.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        expected = np.array([2.1008682, 2., 2.5734887], dtype=np.float32)
        np.testing.assert_allclose(result.best_params, expected, rtol=1e-5,
                                   err_msg="best_params golden mismatch (f32 5pt seed42)")
        assert result.status == 1

    def test_5pt_poisson_seed42_bestnorm(self):
        """f32, 5-point Poisson noise (seed=42): fixate bestnorm golden value."""
        x = np.array([-0.94200003, -0.47100002, 0., 0.47100002, 0.94200003], dtype=np.float32)
        y = np.array([2., 1., 1., 3., 2.], dtype=np.float32)
        err = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-2.0, 2.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        np.testing.assert_allclose(result.bestnorm, 1.3372162580490112, rtol=1e-5,
                                   err_msg="bestnorm golden mismatch (f32 5pt seed42)")

    def test_5pt_poisson_seed42_xerror_scipy(self):
        """f32, 5-point (seed=42): xerror_scipy golden values."""
        x = np.array([-0.94200003, -0.47100002, 0., 0.47100002, 0.94200003], dtype=np.float32)
        y = np.array([2., 1., 1., 3., 2.], dtype=np.float32)
        err = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)
        parinfo = [
            {'value': 2.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-2.0, 2.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]},
        ]
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw={'x': x, 'y': y, 'error': err})

        # mean is pegged at upper bound ? xerror[1] == 0
        assert result.xerror[1] == pytest.approx(0.0, abs=1e-10), "xerror[1] should be 0 when pegged at bound"
        expected_scipy = np.array([6.143815, 21.863874, 14.825083], dtype=np.float32)
        np.testing.assert_allclose(result.xerror_scipy, expected_scipy, rtol=1e-4,
                                   err_msg="xerror_scipy golden mismatch (f32 5pt seed42)")

    def test_lw_at_upper_bound_bugcase(self):
        """
        Regression test from real MUSE data: linewidth converges to upper bound (200).
        Fixates best_params, bestnorm, and xerror_scipy for this edge case.
        """
        xdata = np.array([-81.21540537932414, -40.60770268966206, 0.0,
                           40.60770268966209, 81.21540537932415], dtype=np.float32)
        ydata = np.array([0.8990826606750488, 0.9491528868675232, 0.9999998211860657,
                           0.9884452223777771, 0.9523726105690002], dtype=np.float32)
        y_sigma = np.ones(5, dtype=np.float32)
        parinfo = [
            {'value': 1.0,              'limits': [0.9, 1.1]},
            {'value': 0.0,              'limits': [-81.21540537932412, 81.21540537932412]},
            {'value': 69.1378943769105, 'limits': [59.137894376910495, 200.0]},
        ]
        result = fmpfit_f32_pywrap(
            0, parinfo=parinfo,
            functkw={'x': xdata, 'y': ydata, 'error': y_sigma},
            xtol=1.0e-6, ftol=1.0e-6, gtol=1.0e-6, maxiter=2000, quiet=1,
        )

        # Linewidth must be at the upper bound
        np.testing.assert_allclose(result.best_params[2], 200.0, rtol=1e-5,
                                   err_msg="linewidth should be at upper bound 200")
        # Amplitude golden value
        np.testing.assert_allclose(result.best_params[0], 1.0001495, rtol=1e-4,
                                   err_msg="amplitude golden mismatch (bugcase)")
        # Velocity golden value
        np.testing.assert_allclose(result.best_params[1], 15.571712, rtol=1e-4,
                                   err_msg="velocity golden mismatch (bugcase)")
        # bestnorm golden value
        np.testing.assert_allclose(result.bestnorm, 0.0002860676031559706, rtol=1e-4,
                                   err_msg="bestnorm golden mismatch (bugcase)")
        # npegged: linewidth is AT bound but via convergence (npegged counts forced pegs)
        assert result.npegged == 0
        # xerror_scipy must be finite and match golden values
        expected_scipy = np.array([8.2430467e-03, 4.6672206e+00, 1.6366814e+01],
                                   dtype=np.float32)
        np.testing.assert_allclose(result.xerror_scipy, expected_scipy, rtol=1e-3,
                                   err_msg="xerror_scipy golden mismatch (bugcase)")


# ---------------------------------------------------------------------------
# f64 block tests
# ---------------------------------------------------------------------------

class TestFmpfitF64BlockGolden:
    """Regression tests for fmpfit_f64_block_pywrap."""

    def _make_block_data(self, n_spectra=3):
        """Return (x, y, err, p0, bnd) for n_spectra=3 spectra.

        y values are hardcoded Poisson draws (legacy seed=42) so golden values
        remain stable regardless of the NumPy random-generator version.
        """
        tp = np.array([2.5, 0.0, 0.8], dtype=np.float64)
        fwhm = float(2.355 * tp[2])
        x1d = np.linspace(-fwhm / 2, fwhm / 2, 5, dtype=np.float64)
        p0_1d = np.array([2.0, 0.0, 1.0], dtype=np.float64)
        bnd_1d = np.array([[0.0, 10.0], [-2.0, 2.0], [0.1, 5.0]], dtype=np.float64)
        x = np.tile(x1d, (n_spectra, 1))
        y = np.array([
            [2., 1., 1., 3., 2.],
            [0., 1., 2., 1., 1.],
            [2., 2., 2., 4., 2.],
        ], dtype=np.float64)[:n_spectra]
        err = np.sqrt(np.maximum(y, 1.0)).astype(np.float64)
        p0 = np.tile(p0_1d, (n_spectra, 1))
        bnd = np.tile(bnd_1d, (n_spectra, 1, 1))
        return x, y, err, p0, bnd

    def test_block_f64_n3_seed42_params(self):
        """Block f64, N=3, seed=42: fixate best_params for all 3 spectra."""
        x, y, err, p0, bnd = self._make_block_data()
        result = fmpfit_f64_block_pywrap(0, x, y, err, p0, bnd,
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        expected = np.array([
            [2.10086825, 2.0,        2.57348856],
            [1.58751874, 0.15956157, 0.60724679],
            [2.5078758,  0.33855646, 1.54905626],
        ])
        np.testing.assert_allclose(result['best_params'], expected, rtol=1e-6,
                                   err_msg="block f64 best_params golden mismatch")

    def test_block_f64_n3_seed42_bestnorm(self):
        """Block f64, N=3, seed=42: fixate bestnorm for all 3 spectra."""
        x, y, err, p0, bnd = self._make_block_data()
        result = fmpfit_f64_block_pywrap(0, x, y, err, p0, bnd,
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        expected = np.array([1.33721628, 0.45638885, 0.75819141])
        np.testing.assert_allclose(result['bestnorm'], expected, rtol=1e-6,
                                   err_msg="block f64 bestnorm golden mismatch")

    def test_block_f64_n3_seed42_status(self):
        """Block f64, N=3, seed=42: all spectra converged (status 1)."""
        x, y, err, p0, bnd = self._make_block_data()
        result = fmpfit_f64_block_pywrap(0, x, y, err, p0, bnd,
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        np.testing.assert_array_equal(result['status'], np.array([1, 1, 1], dtype=np.int32))

    def test_block_f64_n3_matches_single(self):
        """Block f64 results must match per-spectrum single fits."""
        x, y, err, p0, bnd = self._make_block_data()
        result_block = fmpfit_f64_block_pywrap(0, x, y, err, p0, bnd,
                                               xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        bnd_1d = bnd[0]  # same for all
        for i in range(3):
            parinfo_i = [{'value': float(p0[i, j]), 'limits': [float(bnd_1d[j, 0]), float(bnd_1d[j, 1])]}
                         for j in range(3)]
            r_single = fmpfit_f64_pywrap(0, parinfo=parinfo_i,
                                         functkw={'x': x[i], 'y': y[i], 'error': err[i]},
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)
            np.testing.assert_allclose(result_block['best_params'][i], r_single.best_params,
                                       rtol=1e-10,
                                       err_msg=f"block != single for spectrum {i} (f64)")


# ---------------------------------------------------------------------------
# f32 block tests
# ---------------------------------------------------------------------------

class TestFmpfitF32BlockGolden:
    """Regression tests for fmpfit_f32_block_pywrap."""

    def _make_block_data(self, n_spectra=3):
        """Return (x, y, err, p0, bnd) for n_spectra=3 f32 spectra.

        y values are hardcoded Poisson draws (legacy seed=42) so golden values
        remain stable regardless of the NumPy random-generator version.
        """
        tp = np.array([2.5, 0.0, 0.8], dtype=np.float32)
        fwhm = float(2.355 * tp[2])
        x1d = np.linspace(-fwhm / 2, fwhm / 2, 5, dtype=np.float32)
        p0_1d = np.array([2.0, 0.0, 1.0], dtype=np.float32)
        bnd_1d = np.array([[0.0, 10.0], [-2.0, 2.0], [0.1, 5.0]], dtype=np.float32)
        x = np.tile(x1d, (n_spectra, 1))
        y = np.array([
            [2., 1., 1., 3., 2.],
            [0., 1., 2., 1., 1.],
            [2., 2., 2., 4., 2.],
        ], dtype=np.float32)[:n_spectra]
        err = np.sqrt(np.maximum(y, 1.0)).astype(np.float32)
        p0 = np.tile(p0_1d, (n_spectra, 1))
        bnd = np.tile(bnd_1d, (n_spectra, 1, 1))
        return x, y, err, p0, bnd

    def test_block_f32_n3_seed42_params(self):
        """Block f32, N=3, seed=42: fixate best_params for all 3 spectra."""
        x, y, err, p0, bnd = self._make_block_data()
        result = fmpfit_f32_block_pywrap(0, x, y, err, p0, bnd,
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        expected = np.array([
            [2.1008682,  2.,         2.5734887 ],
            [1.5875187,  0.15956157, 0.6072468 ],
            [2.5077677,  0.33879173, 1.5495867 ],
        ], dtype=np.float32)
        np.testing.assert_allclose(result['best_params'], expected, rtol=1e-5,
                                   err_msg="block f32 best_params golden mismatch")

    def test_block_f32_n3_seed42_bestnorm(self):
        """Block f32, N=3, seed=42: fixate bestnorm for all 3 spectra."""
        x, y, err, p0, bnd = self._make_block_data()
        result = fmpfit_f32_block_pywrap(0, x, y, err, p0, bnd,
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        expected = np.array([1.3372163, 0.4563889, 0.75819147], dtype=np.float32)
        np.testing.assert_allclose(result['bestnorm'], expected, rtol=1e-5,
                                   err_msg="block f32 bestnorm golden mismatch")

    def test_block_f32_n3_seed42_status(self):
        """Block f32, N=3, seed=42: all spectra converged (status 1)."""
        x, y, err, p0, bnd = self._make_block_data()
        result = fmpfit_f32_block_pywrap(0, x, y, err, p0, bnd,
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        np.testing.assert_array_equal(result['status'], np.array([1, 1, 1], dtype=np.int32))

    def test_block_f32_n3_matches_single(self):
        """Block f32 results must match per-spectrum single fits (within f32 tolerance)."""
        x, y, err, p0, bnd = self._make_block_data()
        result_block = fmpfit_f32_block_pywrap(0, x, y, err, p0, bnd,
                                               xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)

        bnd_1d = bnd[0]
        for i in range(3):
            parinfo_i = [{'value': float(p0[i, j]), 'limits': [float(bnd_1d[j, 0]), float(bnd_1d[j, 1])]}
                         for j in range(3)]
            r_single = fmpfit_f32_pywrap(0, parinfo=parinfo_i,
                                         functkw={'x': x[i], 'y': y[i], 'error': err[i]},
                                         xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000)
            np.testing.assert_allclose(result_block['best_params'][i], r_single.best_params,
                                       rtol=1e-5,
                                       err_msg=f"block != single for spectrum {i} (f32)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
