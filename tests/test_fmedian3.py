"""Comprehensive tests for fmedian3 (3D median filter).

Organized into test classes:
- TestFmedian3Core: Basic functionality and correctness
- TestFmedian3EdgeCases: NaN handling, boundaries, special values
- TestFmedian3Validation: Parameter validation and error handling
"""
import numpy as np
import pytest

from ftoolss import fmedian3d as fmedian3
from ftoolss.fmedian3 import fmedian3 as fmedian3_direct


class TestFmedian3Core:
    """Test core fmedian3 functionality and correctness."""
    
    def test_basic_functionality(self):
        """Test basic functionality with a simple 3D array."""
        input_arr = np.arange(1, 28, dtype=np.float64).reshape((3, 3, 3))
        
        # Apply filter with 3x3x3 window (includes immediate neighbors)
        out = fmedian3(input_arr, 3, 3, 3, 1)
        
        assert out.shape == input_arr.shape and out.dtype == np.float64
    
    def test_data_types(self):
        """Test that data type checking works correctly."""
        # Test with correct types
        input_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        out = fmedian3(input_arr, 3, 3, 3, 1)
        assert out.dtype == np.float64
    
    def test_window_sizes(self):
        """Test various window sizes."""
        input_arr = np.ones((5, 5, 5), dtype=np.float64)
        
        # Test 1x1x1 window (should fall back to original values when center excluded)
        out = fmedian3(input_arr, 1, 1, 1, 1)
        assert np.allclose(out, input_arr)
        
        # Test 1x1x1 window with center included
        out = fmedian3(input_arr, 1, 1, 1, 0)
        assert np.allclose(out, input_arr)
        
        # Test 3x3x3 window
        out = fmedian3(input_arr, 3, 3, 3, 1)
        assert np.allclose(out, np.ones_like(input_arr))
        
        # Test 5x5x5 window
        out = fmedian3(input_arr, 5, 5, 5, 1)
        assert np.allclose(out, np.ones_like(input_arr))
    
    def test_outlier_removal(self):
        """Test outlier detection/removal capability."""
        # Create array with outlier
        input_arr = np.ones((5, 5, 5), dtype=np.float64)
        input_arr[2, 2, 2] = 100.0  # Center outlier
        
        # Filter with exclude_center=1 (should remove outlier influence from neighbors)
        out = fmedian3(input_arr, 3, 3, 3, 1)
        
        # The outlier position should get the median of its neighbors (which are all 1)
        assert np.isclose(out[2, 2, 2], 1.0)
        
        # Other positions should also be 1 (neighbors are all 1)
        assert np.allclose(out, np.ones_like(input_arr))
    
    def test_int_input_coercion(self):
        """Test fmedian3 coerces integer input to float64."""
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
        out = fmedian3(a, 3, 3, 3, 1)
        
        assert out.dtype == np.float64
        assert out.shape == a.shape


class TestFmedian3EdgeCases:
    """Test fmedian3 with edge cases, boundaries, and special values."""
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        # Create array with NaN
        input_arr = np.ones((3, 3, 3), dtype=np.float64)
        input_arr[0, 0, 0] = np.nan
        input_arr[1, 1, 1] = np.nan  # center
        
        # Filter should ignore NaN values
        out = fmedian3(input_arr, 3, 3, 3, 1)
        
        # Result should not contain NaN in most positions (except where all neighbors are NaN)
        assert not np.isnan(out[2, 2, 2])
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small array
        input_arr = np.array([[[1.0]]], dtype=np.float64)
        out = fmedian3(input_arr, 1, 1, 1, 0)
        assert np.isclose(out[0, 0, 0], 1.0)
        
        # All NaN array
        input_arr = np.full((2, 2, 2), np.nan, dtype=np.float64)
        out = fmedian3(input_arr, 3, 3, 3, 1)
        assert np.all(np.isnan(out))
    
    def test_fmedian3_large_window(self):
        """Test fmedian3 with window larger than the array."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        out = fmedian3(a, 11, 11, 11, 1)
        assert out.shape == a.shape
        assert np.all(np.isfinite(out))
    
    def test_fmedian3_preserves_input(self):
        """Verify fmedian3 does not modify the input array."""
        a = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        a_copy = a.copy()
        
        out = fmedian3(a, 3, 3, 3, 1)
        
        assert np.array_equal(a, a_copy), "Input array was modified"
        assert out is not a, "Output should be a new array"
    
    def test_fmedian3_corner_voxels(self):
        """Test fmedian3 handles corner voxels (truncated windows) correctly."""
        a = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        out = fmedian3(a, 3, 3, 3, 1)
        
        # All corners should have finite values
        assert np.isfinite(out[0, 0, 0])
        assert np.isfinite(out[0, 0, 2])
        assert np.isfinite(out[0, 2, 0])
        assert np.isfinite(out[2, 0, 0])
        assert np.isfinite(out[2, 2, 2])
    
    def test_fmedian3_uniform_array(self):
        """Test fmedian3 with uniform values."""
        a = np.full((5, 5, 5), 42.0, dtype=np.float64)
        out = fmedian3(a, 3, 3, 3, 1)
        assert np.allclose(out, 42.0)


class TestFmedian3Validation:
    """Test parameter validation and error handling for fmedian3."""
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        input_arr = np.ones((3, 3, 3), dtype=np.float64)
        
        # Test that window sizes must be odd
        with pytest.raises(ValueError, match="must be an odd number"):
            fmedian3(input_arr, 2, 3, 3, 1)
        
        with pytest.raises(ValueError, match="must be an odd number"):
            fmedian3(input_arr, 3, 2, 3, 1)
        
        with pytest.raises(ValueError, match="must be an odd number"):
            fmedian3(input_arr, 3, 3, 2, 1)
        
        # Test that window sizes must be positive
        with pytest.raises(ValueError, match="must be positive"):
            fmedian3(input_arr, 0, 3, 3, 1)
        
        with pytest.raises(ValueError, match="must be positive"):
            fmedian3(input_arr, -1, 3, 3, 1)
        
        # Test that input must be 3D
        input_2d = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="must be 3-dimensional"):
            fmedian3(input_2d, 3, 3, 3, 1)
    
    def test_fmedian3_requires_sizes_not_none(self):
        """Test fmedian3 requires non-None sizes."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        with pytest.raises(TypeError):
            fmedian3_direct(a, None, None, None)
    
    def test_fmedian3_even_size_checks_and_dimension(self):
        """Test fmedian3 dimension and oddness checks."""
        a2 = np.ones((3, 3), dtype=np.float64)
        a3 = np.ones((3, 3, 3), dtype=np.float64)
        
        # wrong dimensionality (expects 3D input)
        with pytest.raises(ValueError, match="3-dimensional"):
            fmedian3_direct(a2, 3, 3, 3, 0)
        
        # even sizes trigger odd-number ValueError
        with pytest.raises(ValueError, match="xsize must be an odd number"):
            fmedian3_direct(a3, 2, 3, 3, 0)
    
    def test_fmedian3_negative_ysize_and_zsize_rejected(self):
        """Test fmedian3 rejects negative ysize and zsize."""
        a3 = np.ones((3, 3, 3), dtype=np.float64)
        
        # Negative ysize should trigger positive-size validation
        with pytest.raises(ValueError, match="ysize must be positive"):
            fmedian3_direct(a3, 3, -1, 3, 0)
        
        # Negative zsize should trigger positive-size validation
        with pytest.raises(ValueError, match="zsize must be positive"):
            fmedian3_direct(a3, 3, 3, -1, 0)
    
    def test_fmedian3_accepts_non_int_but_coerces(self):
        """Test fmedian3 accepts non-int but coerces."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        out = fmedian3_direct(a, 3.0, 3.0, 3.0, 0)
        assert out.shape == a.shape
    
    def test_fmedian3_empty_array(self):
        """Test fmedian3 with empty array."""
        a = np.array([], dtype=np.float64).reshape(0, 0, 0)
        try:
            out = fmedian3(a, 1, 1, 1, 1)
            assert out.shape == (0, 0, 0)
        except (ValueError, RuntimeError):
            pass
