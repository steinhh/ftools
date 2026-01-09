"""Comprehensive tests for fsigma3 (3D sigma/standard deviation filter).

Organized into test classes:
- TestFsigma3Core: Basic functionality and correctness
- TestFsigma3EdgeCases: NaN handling, boundaries, special values
- TestFsigma3Validation: Parameter validation and error handling
"""
import numpy as np
import pytest

from ftoolss import fsigma3d as fsigma3
from ftoolss.fsigma3 import fsigma3 as fsigma3_direct


class TestFsigma3Core:
    """Test core fsigma3 functionality and correctness."""
    
    def test_basic_functionality(self):
        """Test basic functionality with a simple 3D array."""
        input_arr = np.arange(1, 28, dtype=np.float64).reshape((3, 3, 3))
        
        # Apply filter with 3x3x3 window
        out = fsigma3(input_arr, 3, 3, 3, 1)
        
        assert out.shape == input_arr.shape and out.dtype == np.float64
    
    def test_data_types(self):
        """Test that data type checking works correctly."""
        input_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        out = fsigma3(input_arr, 3, 3, 3, 1)
        assert out.dtype == np.float64
    
    def test_uniform_array(self):
        """Test with uniform array (sigma should be zero)."""
        input_arr = np.ones((5, 5, 5), dtype=np.float64)
        
        # Apply filter - should get zero sigma everywhere
        out = fsigma3(input_arr, 3, 3, 3, 1)
        
        # All values should be zero (no variation)
        assert np.allclose(out, np.zeros_like(input_arr))
    
    def test_high_variance_detection(self):
        """Test that high variance areas are detected."""
        # Create array with outlier to create high variance
        input_arr = np.ones((5, 5, 5), dtype=np.float64)
        input_arr[2, 2, 2] = 100.0  # Create high variance at center
        
        # Filter with exclude_center=0 (include outlier in calculation)
        out_with_center = fsigma3(input_arr, 3, 3, 3, 0)
        
        # Filter with exclude_center=1 (exclude outlier from calculation)
        out_without_center = fsigma3(input_arr, 3, 3, 3, 1)
        
        # The center position should have higher sigma when outlier is included
        center_sigma_with = out_with_center[2, 2, 2]
        center_sigma_without = out_without_center[2, 2, 2]
        
        assert center_sigma_with > center_sigma_without
        assert center_sigma_with > 10.0
    
    def test_window_sizes(self):
        """Test various window sizes."""
        rng = np.random.default_rng(42)
        input_arr = rng.random((7, 7, 7)).astype(np.float64)
        
        # Test 1x1x1 window (should be zero everywhere when center excluded)
        out = fsigma3(input_arr, 1, 1, 1, 1)
        assert np.allclose(out, np.zeros_like(input_arr))
        
        # Test 1x1x1 window with center included (should be zero - only one value)
        out = fsigma3(input_arr, 1, 1, 1, 0)
        assert np.allclose(out, np.zeros_like(input_arr))
        
        # Test larger windows
        out3 = fsigma3(input_arr, 3, 3, 3, 1)
        out5 = fsigma3(input_arr, 5, 5, 5, 1)
        
        assert out3.shape == out5.shape == input_arr.shape
    
    def test_fsigma3_int_input_coercion(self):
        """Test fsigma3 coerces integer input to float64."""
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
        out = fsigma3(a, 3, 3, 3, 1)
        
        assert out.dtype == np.float64
        assert out.shape == a.shape
    
    def test_fsigma3_sigma_always_nonnegative(self):
        """Test fsigma3 returns non-negative values."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 5, 5)).astype(np.float64)
        out = fsigma3(a, 3, 3, 3, 1)
        
        assert np.all(out >= 0.0)


class TestFsigma3EdgeCases:
    """Test fsigma3 with edge cases, boundaries, and special values."""
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        # Create array with some NaN values
        input_arr = np.ones((3, 3, 3), dtype=np.float64)
        input_arr[0, 0, 0] = np.nan
        input_arr[1, 1, 1] = 2.0
        
        # Filter should ignore NaN values
        out = fsigma3(input_arr, 3, 3, 3, 1)
        
        # Result should not contain unexpected NaN values in most positions
        assert not np.isnan(out[2, 2, 2])
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small array
        input_arr = np.array([[[1.0, 2.0]]], dtype=np.float64)
        out = fsigma3(input_arr, 1, 1, 1, 0)
        assert np.allclose(out, np.zeros_like(input_arr))
        
        # All NaN array
        input_arr = np.full((2, 2, 2), np.nan, dtype=np.float64)
        out = fsigma3(input_arr, 3, 3, 3, 1)
        assert np.allclose(out, np.zeros_like(input_arr), equal_nan=True)
    
    def test_fsigma3_large_window(self):
        """Test fsigma3 with window larger than the array."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        out = fsigma3(a, 11, 11, 11, 1)
        assert out.shape == a.shape
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)
    
    def test_fsigma3_preserves_input(self):
        """Verify fsigma3 does not modify the input array."""
        a = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        a_copy = a.copy()
        
        out = fsigma3(a, 3, 3, 3, 1)
        
        assert np.array_equal(a, a_copy), "Input array was modified"
        assert out is not a, "Output should be a new array"
    
    def test_fsigma3_corner_voxels(self):
        """Test fsigma3 handles corner voxels (truncated windows) correctly."""
        a = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        out = fsigma3(a, 3, 3, 3, 1)
        
        # All corners should have finite, non-negative values
        assert np.isfinite(out[0, 0, 0]) and out[0, 0, 0] >= 0.0
        assert np.isfinite(out[0, 0, 2]) and out[0, 0, 2] >= 0.0
        assert np.isfinite(out[2, 2, 2]) and out[2, 2, 2] >= 0.0
    
    def test_fsigma3_negative_values(self):
        """Test fsigma3 handles negative values correctly (sigma always positive)."""
        a = np.array([
            [[-5.0, -3.0], [-1.0, 0.0]],
            [[-4.0, -2.0], [1.0, 2.0]]
        ], dtype=np.float64)
        out = fsigma3(a, 3, 3, 3, 1)
        
        # Sigma must be non-negative
        assert np.all(out >= 0.0)
        assert np.all(np.isfinite(out))


class TestFsigma3Validation:
    """Test parameter validation and error handling for fsigma3."""
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        input_arr = np.ones((3, 3, 3), dtype=np.float64)
        
        # Test that window sizes must be odd
        with pytest.raises(ValueError, match="must be an odd number"):
            fsigma3(input_arr, 2, 3, 3, 1)
        
        with pytest.raises(ValueError, match="must be an odd number"):
            fsigma3(input_arr, 3, 2, 3, 1)
        
        with pytest.raises(ValueError, match="must be an odd number"):
            fsigma3(input_arr, 3, 3, 2, 1)
        
        # Test that window sizes must be positive
        with pytest.raises(ValueError, match="must be positive"):
            fsigma3(input_arr, 0, 3, 3, 1)
        
        with pytest.raises(ValueError, match="must be positive"):
            fsigma3(input_arr, -1, 3, 3, 1)
        
        # Test that input must be 3D
        input_2d = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="must be 3-dimensional"):
            fsigma3(input_2d, 3, 3, 3, 1)
    
    def test_fsigma3_requires_sizes_and_dimension(self):
        """Test fsigma3 dimension and size requirement checks."""
        a2 = np.ones((3, 3), dtype=np.float64)
        a3 = np.ones((3, 3, 3), dtype=np.float64)
        
        with pytest.raises(TypeError):
            fsigma3_direct(a3, None, None, None)
        
        with pytest.raises(ValueError, match="3-dimensional"):
            fsigma3_direct(a2, 3, 3, 3, 0)
        
        with pytest.raises(ValueError, match="zsize must be an odd number"):
            fsigma3_direct(a3, 3, 3, 2, 0)
    
    def test_fsigma3_negative_ysize_and_zsize_rejected(self):
        """Test fsigma3 rejects negative ysize and zsize."""
        a3 = np.ones((3, 3, 3), dtype=np.float64)
        
        # Negative ysize should trigger positive-size validation
        with pytest.raises(ValueError, match="ysize must be positive"):
            fsigma3_direct(a3, 3, -1, 3, 0)
        
        # Negative zsize should trigger positive-size validation
        with pytest.raises(ValueError, match="zsize must be positive"):
            fsigma3_direct(a3, 3, 3, -1, 0)
    
    def test_fsigma3_accepts_non_int_but_coerces(self):
        """Test fsigma3 accepts non-int but coerces."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        out = fsigma3_direct(a, 3.0, 3.0, 3.0, 0)
        assert out.shape == a.shape
    
    def test_fsigma3_empty_array(self):
        """Test fsigma3 with empty array."""
        a = np.array([], dtype=np.float64).reshape(0, 0, 0)
        try:
            out = fsigma3(a, 1, 1, 1, 1)
            assert out.shape == (0, 0, 0)
        except (ValueError, RuntimeError):
            pass
