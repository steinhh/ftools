"""Unit tests for fgaussian_f32 and fgaussian_f64

Each test function tests both float32 and float64 versions using appropriate input arrays.
"""

import numpy as np
import pytest
from ftoolss.fgaussian import fgaussian_f32_ext, fgaussian_f64_ext

# Access the C extension functions directly
fgaussian_f32 = fgaussian_f32_ext.fgaussian_f32
fgaussian_f64 = fgaussian_f64_ext.fgaussian_f64


class TestGaussianBasic:
    """Basic functionality tests"""
    
    def test_basic_computation(self):
        """Test basic Gaussian computation for both f32 and f64"""
        # Test f32
        x_f32 = np.linspace(-5, 5, 100, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 1.0)
        assert result_f32.dtype == np.float32
        assert result_f32.shape == x_f32.shape
        assert 0.9 < result_f32.max() < 1.1
        
        # Test f64
        x_f64 = np.linspace(-5, 5, 100, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 1.0)
        assert result_f64.dtype == np.float64
        assert result_f64.shape == x_f64.shape
        assert 0.9 < result_f64.max() < 1.1
    
    def test_peak_location(self):
        """Test that peak is at mu for both versions"""
        mu = 3.0
        
        # Test f32
        x_f32 = np.linspace(-10, 10, 1000, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 2.5, mu, 1.5)
        peak_idx_f32 = np.argmax(result_f32)
        assert x_f32[peak_idx_f32] == pytest.approx(mu, abs=0.1)
        assert result_f32[peak_idx_f32] == pytest.approx(2.5, rel=0.01)
        
        # Test f64
        x_f64 = np.linspace(-10, 10, 1000, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 2.5, mu, 1.5)
        peak_idx_f64 = np.argmax(result_f64)
        assert x_f64[peak_idx_f64] == pytest.approx(mu, abs=0.1)
        assert result_f64[peak_idx_f64] == pytest.approx(2.5, rel=0.01)
    
    def test_symmetry(self):
        """Test Gaussian is symmetric around mu for both versions"""
        # Test f32
        x_f32 = np.linspace(-5, 5, 1001, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 1.0)
        mid = 500
        left_f32 = result_f32[:mid]
        right_f32 = result_f32[mid+1:][::-1]
        np.testing.assert_allclose(left_f32, right_f32, rtol=1e-6, atol=1e-7)
        
        # Test f64
        x_f64 = np.linspace(-5, 5, 1001, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 1.0)
        left_f64 = result_f64[:mid]
        right_f64 = result_f64[mid+1:][::-1]
        np.testing.assert_allclose(left_f64, right_f64, rtol=1e-14, atol=1e-14)
    
    def test_fwhm(self):
        """Test Full Width at Half Maximum for both versions"""
        sigma = 2.0
        expected_fwhm = 2.355 * sigma
        
        # Test f32
        x_f32 = np.linspace(-10, 10, 10000, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, sigma)
        above_half_f32 = x_f32[result_f32 >= 0.5]
        measured_fwhm_f32 = above_half_f32[-1] - above_half_f32[0]
        assert measured_fwhm_f32 == pytest.approx(expected_fwhm, rel=0.01)
        
        # Test f64
        x_f64 = np.linspace(-10, 10, 10000, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, sigma)
        above_half_f64 = x_f64[result_f64 >= 0.5]
        measured_fwhm_f64 = above_half_f64[-1] - above_half_f64[0]
        assert measured_fwhm_f64 == pytest.approx(expected_fwhm, rel=0.01)


class TestGaussianParameters:
    """Test different parameter values"""
    
    def test_different_amplitudes(self):
        """Test i0 parameter for both versions"""
        # Test f32
        x_f32 = np.array([0.0], dtype=np.float32)
        assert fgaussian_f32(x_f32, 1.0, 0.0, 1.0)[0] == pytest.approx(1.0)
        assert fgaussian_f32(x_f32, 2.0, 0.0, 1.0)[0] == pytest.approx(2.0)
        assert fgaussian_f32(x_f32, 0.5, 0.0, 1.0)[0] == pytest.approx(0.5)
        
        # Test f64
        x_f64 = np.array([0.0], dtype=np.float64)
        assert fgaussian_f64(x_f64, 1.0, 0.0, 1.0)[0] == pytest.approx(1.0)
        assert fgaussian_f64(x_f64, 2.0, 0.0, 1.0)[0] == pytest.approx(2.0)
        assert fgaussian_f64(x_f64, 0.5, 0.0, 1.0)[0] == pytest.approx(0.5)
    
    def test_different_widths(self):
        """Test sigma parameter for both versions"""
        # Test f32
        x_f32 = np.array([1.0], dtype=np.float32)
        result1_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 1.0)
        result2_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 2.0)
        assert result2_f32 > result1_f32
        assert result1_f32[0] == pytest.approx(np.exp(-0.5), rel=1e-6)
        
        # Test f64
        x_f64 = np.array([1.0], dtype=np.float64)
        result1_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 1.0)
        result2_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 2.0)
        assert result2_f64 > result1_f64
        assert result1_f64[0] == pytest.approx(np.exp(-0.5), rel=1e-14)


class TestGaussianValidation:
    """Test input validation"""
    
    def test_zero_sigma_raises(self):
        """Sigma = 0 should raise ValueError for both versions"""
        x_f32 = np.array([0.0], dtype=np.float32)
        x_f64 = np.array([0.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            fgaussian_f32(x_f32, 1.0, 0.0, 0.0)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            fgaussian_f64(x_f64, 1.0, 0.0, 0.0)
    
    def test_negative_sigma_raises(self):
        """Negative sigma should raise ValueError for both versions"""
        x_f32 = np.array([0.0], dtype=np.float32)
        x_f64 = np.array([0.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            fgaussian_f32(x_f32, 1.0, 0.0, -1.0)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            fgaussian_f64(x_f64, 1.0, 0.0, -1.0)


class TestGaussianNumerical:
    """Numerical accuracy tests"""
    
    def test_matches_numpy(self):
        """Compare with NumPy implementation for both versions"""
        i0, mu, sigma = 2.5, 1.5, 3.0
        
        # Test f32
        x_f32 = np.linspace(-10, 10, 1000, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, i0, mu, sigma)
        expected_f32 = i0 * np.exp(-((x_f32 - mu) ** 2) / (2 * sigma ** 2))
        np.testing.assert_allclose(result_f32, expected_f32, rtol=1e-6, atol=1e-7)
        
        # Test f64
        x_f64 = np.linspace(-10, 10, 1000, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, i0, mu, sigma)
        expected_f64 = i0 * np.exp(-((x_f64 - mu) ** 2) / (2 * sigma ** 2))
        np.testing.assert_allclose(result_f64, expected_f64, rtol=1e-14, atol=1e-14)
    
    def test_extreme_values(self):
        """Test with extreme x values for both versions"""
        # Test f32
        x_f32 = np.array([-1e6, -100, 0, 100, 1e6], dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 1.0)
        assert result_f32[2] == pytest.approx(1.0, rel=1e-6)  # Peak at center
        assert result_f32[0] < 1e-30  # Essentially zero far away
        assert result_f32[4] < 1e-30
        
        # Test f64
        x_f64 = np.array([-1e6, -100, 0, 100, 1e6], dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 1.0)
        assert result_f64[2] == pytest.approx(1.0, rel=1e-14)
        assert result_f64[0] < 1e-30
        assert result_f64[4] < 1e-30
    
    def test_large_sigma(self):
        """Test with large sigma (nearly flat) for both versions"""
        # Test f32
        x_f32 = np.linspace(-10, 10, 100, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 1000.0)
        assert np.all(result_f32 > 0.999)  # Nearly flat at 1.0
        
        # Test f64
        x_f64 = np.linspace(-10, 10, 100, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 1000.0)
        assert np.all(result_f64 > 0.999)
    
    def test_small_sigma(self):
        """Test with small sigma (narrow peak) for both versions"""
        # Test f32
        x_f32 = np.linspace(-1, 1, 10000, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 0.01)
        peak_idx_f32 = np.argmax(result_f32)
        assert abs(x_f32[peak_idx_f32]) < 0.001
        assert result_f32[peak_idx_f32] == pytest.approx(1.0, rel=0.01)  # Relaxed tolerance
        
        # Test f64
        x_f64 = np.linspace(-1, 1, 10000, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 0.01)
        peak_idx_f64 = np.argmax(result_f64)
        assert abs(x_f64[peak_idx_f64]) < 0.001
        assert result_f64[peak_idx_f64] == pytest.approx(1.0, rel=0.01)  # Relaxed tolerance


class TestGaussianTypes:
    """Test with different input types"""
    
    def test_float32_input(self):
        """Test explicit float32 input"""
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = fgaussian_f32(x, 1.0, 0.0, 1.0)
        assert result.dtype == np.float32
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0, rel=1e-6)
    
    def test_float64_input(self):
        """Test explicit float64 input"""
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        result = fgaussian_f64(x, 1.0, 0.0, 1.0)
        assert result.dtype == np.float64
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0, rel=1e-14)


class TestGaussianEdgeCases:
    """Edge case tests"""
    
    def test_empty_array(self):
        """Test with empty array for both versions"""
        x_f32 = np.array([], dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 1.0)
        assert result_f32.shape == (0,)
        
        x_f64 = np.array([], dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 1.0)
        assert result_f64.shape == (0,)
    
    def test_single_element(self):
        """Test with single element array for both versions"""
        x_f32 = np.array([5.0], dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 2.0, 5.0, 1.5)
        assert result_f32.shape == (1,)
        assert result_f32[0] == pytest.approx(2.0, rel=1e-6)
        
        x_f64 = np.array([5.0], dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 2.0, 5.0, 1.5)
        assert result_f64.shape == (1,)
        assert result_f64[0] == pytest.approx(2.0, rel=1e-14)
    
    def test_large_array(self):
        """Test with large array for both versions"""
        n = 100_000
        
        x_f32 = np.linspace(-100, 100, n, dtype=np.float32)
        result_f32 = fgaussian_f32(x_f32, 1.0, 0.0, 10.0)
        assert result_f32.shape == (n,)
        assert 0 < result_f32.max() <= 1.0
        
        x_f64 = np.linspace(-100, 100, n, dtype=np.float64)
        result_f64 = fgaussian_f64(x_f64, 1.0, 0.0, 10.0)
        assert result_f64.shape == (n,)
        assert 0 < result_f64.max() <= 1.0


class TestF32VsF64Comparison:
    """Test that f32 and f64 produce similar results"""
    
    def test_agreement(self):
        """Test that f32 and f64 agree to within f32 precision"""
        x_f32 = np.linspace(-10, 10, 1000, dtype=np.float32)
        x_f64 = np.linspace(-10, 10, 1000, dtype=np.float64)
        
        result_f32 = fgaussian_f32(x_f32, 2.5, 1.5, 3.0)
        result_f64 = fgaussian_f64(x_f64, 2.5, 1.5, 3.0)
        
        # They should agree to within float32 precision
        np.testing.assert_allclose(result_f32, result_f64.astype(np.float32), 
                                    rtol=1e-6, atol=1e-7)
