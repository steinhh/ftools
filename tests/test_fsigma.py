"""Comprehensive tests for fsigma (2D sigma/standard deviation filter).

Organized into test classes:
- TestFsigmaCore: Basic functionality and correctness
- TestFsigmaEdgeCases: NaN handling, boundaries, special values
- TestFsigmaValidation: Parameter validation and error handling
"""
import numpy as np
import pytest

from ftoolss import fsigma
from ftoolss.fsigma import fsigma as fsigma_direct


class TestFsigmaCore:
    """Test core fsigma functionality and correctness."""
    
    def test_sigma_zero_on_constant_window(self):
        """Sigma of a constant-valued neighborhood should be 0.0."""
        a = np.full((5, 5), 7.0, dtype=np.float64)
        # 3x3 window including center
        out = fsigma(a, (3, 3), 0)
        assert np.allclose(out, 0.0)

        # 3x3 window excluding center
        out = fsigma(a, (3, 3), 1)
        assert np.allclose(out, 0.0)

    def test_center_exclusion_reduces_sigma_with_outlier(self):
        """Excluding the center outlier should reduce local sigma at the center pixel."""
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 999.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        # With center included
        out = fsigma(a, (3, 3), 0)
        sigma_with = out[1, 1]

        # With center excluded
        out = fsigma(a, (3, 3), 1)
        sigma_without = out[1, 1]

        assert sigma_with > sigma_without

    def test_nan_values_are_ignored(self):
        """NaN values should be ignored in sigma calculation (center or neighbors)."""
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],  # center NaN
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        # Center included: NaN should be ignored; sigma at center should equal
        # population std of [1,2,3,4,6,7,8,9] which is sqrt(7.5)
        out = fsigma(a, (3, 3), 0)
        assert np.isclose(out[1, 1], np.sqrt(7.5))

        # Make a neighbor NaN (not center) and exclude center; sigma should be finite
        b = np.array([
            [np.nan, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        out = fsigma(b, (3, 3), 1)
        assert np.isfinite(out[1, 1])

    def test_1x1_excluding_center_yields_zero(self):
        """With a 1x1 window and center excluded (no neighbors), sigma is defined as 0.0."""
        a = np.array([[42.0]], dtype=np.float64)
        out = fsigma(a, (1, 1), 1)
        assert np.isclose(out[0, 0], 0.0)

    def test_dtype_enforced_float64(self):
        """fsigma requires float64 input and output arrays."""
        good = np.ones((2, 2), dtype=np.float64)

        # Works with float64
        out = fsigma(good, (3, 3), 1)
        assert out.dtype == np.float64

        # New API coerces input to float64; float32 input is accepted and coerced
        bad_in = good.astype(np.float32)
        out2 = fsigma(bad_in, (3, 3), 1)
        assert out2.dtype == np.float64

    def test_dimension_checks(self):
        """Non-2D arrays or mismatched shapes should raise errors."""
        a = np.ones((2, 3), dtype=np.float64)
        # Happy path
        out = fsigma(a, (3, 3), 1)
        assert out.shape == a.shape and out.dtype == np.float64

        # 1D array should fail
        with pytest.raises(ValueError):
            a1 = np.ones(3, dtype=np.float64)
            fsigma(a1, (3, 3), 1)

    def test_fsigma_preserves_input(self):
        """Verify fsigma does not modify the input array."""
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        a_copy = a.copy()
        
        out = fsigma(a, (3, 3), 1)
        
        assert np.array_equal(a, a_copy), "Input array was modified"
        assert out is not a, "Output should be a new array"

    def test_fsigma_int_input_coercion(self):
        """Test fsigma coerces integer input to float64."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        out = fsigma(a, (3, 3), 1)
        
        assert out.dtype == np.float64
        assert out.shape == a.shape

    def test_fsigma_5x5_single_outlier(self):
        """Test fsigma with 5x5 dataset containing a single non-zero value.
        
        This tests a realistic cosmic ray detection scenario where a single
        pixel has a significantly different value from its uniform background.
        """
        # 5x5 array with a single 1.0 at center, rest zeros
        data = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)
        
        # Test with 5x5 filter, center included
        result_with_center = fsigma(data, (5, 5), exclude_center=0)
        
        # For center pixel (2,2): 25 values (24 zeros, 1 one)
        # Mean = 1/25 = 0.04
        # Variance = (24*(0-0.04)^2 + 1*(1-0.04)^2) / 25 = 0.0384
        # Sigma = sqrt(0.0384) ? 0.196
        expected_sigma_center = np.sqrt(0.0384)
        assert np.isclose(result_with_center[2, 2], expected_sigma_center)
        
        # All other pixels have non-zero sigma due to the outlier in their windows
        assert np.all(result_with_center > 0.0)
        
        # Test with 5x5 filter, center excluded
        result_without_center = fsigma(data, (5, 5), exclude_center=1)
        
        # For center pixel (2,2): 24 neighbors all zero
        # Mean = 0, Variance = 0, Sigma = 0
        assert np.isclose(result_without_center[2, 2], 0.0)
        
        # Center pixel excluded mode should have lower sigma at center
        assert result_without_center[2, 2] < result_with_center[2, 2]
        
        # Corner pixels should have highest sigma (furthest from outlier)
        assert result_with_center[0, 0] > result_with_center[2, 2]
        
        # Explicit corner value checks with center included:
        # Corner (0,0): 9 pixels in window, one is 1.0, rest are 0.0
        # Mean = 1/9, Variance = (8*(0-1/9)^2 + 1*(1-1/9)^2)/9 ? 0.0988
        # Sigma ? 0.3143
        expected_corner_with = np.sqrt((8 * (0 - 1/9)**2 + 1 * (1 - 1/9)**2) / 9)
        assert np.isclose(result_with_center[0, 0], expected_corner_with)
        assert np.isclose(result_with_center[0, 4], expected_corner_with)  # top-right
        assert np.isclose(result_with_center[4, 0], expected_corner_with)  # bottom-left
        assert np.isclose(result_with_center[4, 4], expected_corner_with)  # bottom-right
        
        # Explicit corner value checks with center excluded:
        # Corner (0,0): 8 neighbors, one is 1.0, rest are 0.0
        # Mean = 1/8, Variance = (7*(0-1/8)^2 + 1*(1-1/8)^2)/8 ? 0.1094
        # Sigma ? 0.3307
        expected_corner_without = np.sqrt((7 * (0 - 1/8)**2 + 1 * (1 - 1/8)**2) / 8)
        assert np.isclose(result_without_center[0, 0], expected_corner_without)
        assert np.isclose(result_without_center[0, 4], expected_corner_without)
        assert np.isclose(result_without_center[4, 0], expected_corner_without)
        assert np.isclose(result_without_center[4, 4], expected_corner_without)


class TestFsigmaEdgeCases:
    """Test fsigma with edge cases, boundaries, and special values."""
    
    def test_fsigma_large_window(self):
        """Test fsigma with window larger than the array."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        # Window extends beyond array bounds (must be odd)
        out = fsigma(a, (21, 21), 1)
        assert out.shape == a.shape
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_fsigma_asymmetric_windows(self):
        """Test fsigma with asymmetric window sizes (xsize != ysize)."""
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        
        # Wide horizontal window
        out1 = fsigma(a, (5, 1), 1)
        assert out1.shape == a.shape
        
        # Tall vertical window
        out2 = fsigma(a, (1, 5), 1)
        assert out2.shape == a.shape
        
        # Results should differ
        assert not np.allclose(out1, out2)

    def test_fsigma_all_nan_input(self):
        """Test fsigma with an array of all NaNs."""
        a = np.full((3, 3), np.nan, dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        assert out.shape == a.shape

    def test_fsigma_mixed_nan_and_values(self):
        """Test fsigma with mixed NaN and finite values."""
        a = np.array([
            [np.nan, 1.0, np.nan],
            [2.0, 3.0, 4.0],
            [np.nan, 5.0, np.nan]
        ], dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        
        # Center sigma computed from finite neighbors only
        assert np.isfinite(out[1, 1])
        assert out[1, 1] >= 0.0

    def test_fsigma_negative_values(self):
        """Test fsigma handles negative values correctly (sigma always positive)."""
        a = np.array([
            [-5.0, -3.0, -1.0],
            [-4.0, -2.0, 0.0],
            [-3.0, -1.0, 1.0]
        ], dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        
        # Sigma must be non-negative
        assert np.all(out >= 0.0)
        assert np.all(np.isfinite(out))

    def test_fsigma_very_large_values(self):
        """Test fsigma with very large floating point values."""
        a = np.array([
            [1e100, 1e100, 1e100],
            [1e100, 1e100, 1e100],
            [1e100, 1e100, 1e100]
        ], dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        # All same values -> sigma should be 0
        assert np.allclose(out, 0.0)

    def test_fsigma_zero_window(self):
        """Test fsigma with zero window size (only center pixel)."""
        a = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = fsigma(a, (1, 1), 1)
        # With 1x1 window and exclude_center=1, no values -> sigma = 0
        assert np.allclose(out, 0.0)

    def test_fsigma_include_vs_exclude_center(self):
        """Verify include/exclude center produces different results with outlier."""
        a = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 100.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float64)
        
        out_excl = fsigma(a, (3, 3), 1)
        out_incl = fsigma(a, (3, 3), 0)
        
        # Excluding center: sigma of [1,1,1,1,1,1,1,1] = 0.0
        assert np.isclose(out_excl[1, 1], 0.0)
        
        # Including center: sigma of [1,1,1,1,100,1,1,1,1] > 0
        assert out_incl[1, 1] > 0.0

    def test_fsigma_rectangular_array(self):
        """Test fsigma with non-square arrays."""
        a = np.arange(20, dtype=np.float64).reshape(4, 5)
        out = fsigma(a, (3, 3), 1)
        assert out.shape == (4, 5)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_fsigma_single_row(self):
        """Test fsigma with a single row (height=1)."""
        a = np.arange(10, dtype=np.float64).reshape(1, 10)
        out = fsigma(a, (3, 3), 1)
        assert out.shape == (1, 10)
        assert np.all(np.isfinite(out))

    def test_fsigma_single_column(self):
        """Test fsigma with a single column (width=1)."""
        a = np.arange(10, dtype=np.float64).reshape(10, 1)
        out = fsigma(a, (3, 3), 1)
        assert out.shape == (10, 1)
        assert np.all(np.isfinite(out))

    def test_fsigma_corner_pixels(self):
        """Test fsigma handles corner pixels (truncated windows) correctly."""
        a = np.arange(9, dtype=np.float64).reshape(3, 3)
        out = fsigma(a, (3, 3), 1)
        
        # All corners should have finite, non-negative values
        assert np.isfinite(out[0, 0]) and out[0, 0] >= 0.0
        assert np.isfinite(out[0, 2]) and out[0, 2] >= 0.0
        assert np.isfinite(out[2, 0]) and out[2, 0] >= 0.0
        assert np.isfinite(out[2, 2]) and out[2, 2] >= 0.0

    def test_fsigma_edge_pixels(self):
        """Test fsigma handles edge pixels (partial windows) correctly."""
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        out = fsigma(a, (3, 3), 1)
        
        # Check all edge pixels are finite and non-negative
        assert np.all(np.isfinite(out[0, :])) and np.all(out[0, :] >= 0.0)
        assert np.all(np.isfinite(out[-1, :])) and np.all(out[-1, :] >= 0.0)
        assert np.all(np.isfinite(out[:, 0])) and np.all(out[:, 0] >= 0.0)
        assert np.all(np.isfinite(out[:, -1])) and np.all(out[:, -1] >= 0.0)

    def test_fsigma_uniform_array(self):
        """Test fsigma with uniform values returns zero sigma."""
        a = np.full((5, 5), 42.0, dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        assert np.allclose(out, 0.0)

    def test_fsigma_two_values(self):
        """Test fsigma with exactly two distinct values."""
        a = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        
        # Center should have non-zero sigma
        assert out[1, 1] > 0.0

    def test_fsigma_numerical_precision(self):
        """Test fsigma maintains numerical precision with small differences."""
        a = np.array([
            [1.0, 1.0001, 1.0],
            [1.0001, 1.0, 1.0001],
            [1.0, 1.0001, 1.0]
        ], dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        
        # Should detect small variance
        assert out[1, 1] > 0.0
        assert out[1, 1] < 0.001

    def test_fsigma_inf_values(self):
        """Test fsigma with infinity values."""
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.inf, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float64)
        out = fsigma(a, (3, 3), 1)
        
        # At minimum, output should have same shape
        assert out.shape == a.shape


class TestFsigmaValidation:
    """Test parameter validation and error handling for fsigma."""
    
    def test_fsigma_requires_x_y_sizes_not_none(self):
        """Test fsigma requires non-None sizes."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(TypeError):
            fsigma_direct(a, None, None)

    def test_fsigma_even_and_positive_checks(self):
        """Test fsigma rejects even or non-positive sizes."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="xsize must be an odd number"):
            fsigma_direct(a, 2, 3, 0)
        with pytest.raises(ValueError, match="xsize must be positive"):
            fsigma_direct(a, -1, 3, 0)
        with pytest.raises(ValueError, match="ysize must be an odd number"):
            fsigma_direct(a, 3, 2, 0)
    
    def test_fsigma_negative_xsize(self):
        """Test fsigma rejects negative xsize."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError, MemoryError)):
            fsigma(a, (-1, 1))
    
    def test_fsigma_negative_ysize(self):
        """Test fsigma rejects negative ysize."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError, MemoryError)):
            fsigma(a, (1, -1))
    
    def test_fsigma_invalid_exclude_center(self):
        """Test fsigma with invalid exclude_center values."""
        a = np.ones((3, 3), dtype=np.float64)
        # Non-0/1 values for exclude_center should work (treated as boolean)
        out = fsigma(a, (1, 1), exclude_center=2)
        assert out.shape == (3, 3)
    
    def test_fsigma_none_parameters(self):
        """Test fsigma rejects None parameters."""
        a = np.ones((3, 3), dtype=np.float64)
        
        # None for window_size should raise TypeError
        with pytest.raises(TypeError):
            fsigma(a, None)
        
        # Tuple with None elements should raise an error
        with pytest.raises((TypeError, ValueError)):
            fsigma(a, (None, 1))

    def test_fsigma_non_integer_parameters(self):
        """Test fsigma handles float parameters (should convert to int)."""
        a = np.ones((3, 3), dtype=np.float64)
        # Float values in tuple should work (will be converted to int by underlying functions)
        result = fsigma(a, (1.0, 1.0))
        assert result.shape == (3, 3)

    def test_fsigma_empty_array(self):
        """Test fsigma with empty array."""
        a = np.array([], dtype=np.float64).reshape(0, 0)
        # Should either work or raise a clear error
        try:
            out = fsigma(a, (1, 1), 1)
            assert out.shape == (0, 0)
        except (ValueError, RuntimeError):
            pass
    
    def test_fsigma_3d_array(self):
        """Test fsigma accepts 3D arrays with 3-tuple window_size."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        # Should work with 3-tuple
        result = fsigma(a, (1, 1, 1))
        assert result.shape == (3, 3, 3)
        
        # Should fail with 2-tuple for 3D array (dispatches to 2D function)
        with pytest.raises(ValueError, match="Arrays must be 2-dimensional"):
            fsigma(a, (1, 1))
    
    def test_fsigma_no_input(self):
        """Test fsigma raises TypeError when called with no arguments."""
        with pytest.raises(TypeError):
            fsigma()
    
    def test_fsigma_missing_parameters(self):
        """Test fsigma raises TypeError when called with too few arguments."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(TypeError):
            fsigma()
        
        with pytest.raises(TypeError):
            fsigma(a)
        
        with pytest.raises(TypeError):
            fsigma(a, 1)
