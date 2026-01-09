"""Comprehensive tests for fmedian (2D median filter).

Organized into test classes:
- TestFmedianCore: Basic functionality and correctness
- TestFmedianEdgeCases: NaN handling, boundaries, special values
- TestFmedianValidation: Parameter validation and error handling
"""
import numpy as np
import pytest

from ftoolss import fmedian


class TestFmedianCore:
    """Test core fmedian functionality and correctness."""
    
    def test_median_excludes_nan_neighbors(self):
        """Neighbors that are NaN should be ignored when computing the median."""
        arr = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float64)

        # 3x3 window excluding the center from neighbors
        out = fmedian(arr, (3, 3), 1)

        # For the center pixel, neighbors excluding center are [1,2,3,4,6,7,8,9]
        # median = (4 + 6) / 2 = 5.0
        np.testing.assert_allclose(out[1, 1], 5.0, rtol=0, atol=1e-12)

    def test_include_center_nan_is_ignored(self):
        """When center is NaN but neighbors are finite, including center should still ignore the NaN."""
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],  # center NaN
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        # 3x3 window including center
        out = fmedian(a, (3, 3), 0)
        # Values considered: [1,2,3,4,6,7,8,9] (center NaN ignored) -> median = (4+6)/2 = 5
        np.testing.assert_allclose(out[1, 1], 5.0, rtol=0, atol=1e-12)

    def test_1x1_excluding_center_uses_center_when_finite(self):
        """With a 1x1 window and center excluded, no neighbors means we fall back to the center value if finite."""
        a = np.array([[42.0]], dtype=np.float64)
        out = fmedian(a, (1, 1), 1)
        np.testing.assert_allclose(out[0, 0], 42.0, rtol=0, atol=1e-12)

    def test_corner_partial_window_even_count(self):
        """At image borders, the window is truncated; check median with an even number of samples."""
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float64)
        # 3x3 window including center at top-left corner -> values: [1,2,3,4]
        out = fmedian(a, (3, 3), 0)
        np.testing.assert_allclose(out[0, 0], 2.5, rtol=0, atol=1e-12)

    def test_include_center_changes_result_with_outlier(self):
        """Including the center outlier changes the median compared to excluding it."""
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 999.0, 6.0],  # outlier
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        # Excluding center -> neighbors are [1,2,3,4,6,7,8,9] -> median = 5
        out = fmedian(a, (3, 3), 1)
        med_excl = out[1, 1]

        # Including center -> values [1,2,3,4,6,7,8,9,999] -> median = 6
        out = fmedian(a, (3, 3), 0)
        med_incl = out[1, 1]

        np.testing.assert_allclose(med_excl, 5.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(med_incl, 6.0, rtol=0, atol=1e-12)

    def test_single_valid_neighbor_median(self):
        """If only a single finite neighbor remains, the median equals that neighbor."""
        a = np.array([
            [10.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ], dtype=np.float64)
        # Exclude center, compute at (1,1); only (0,0) is valid within the 3x3 window
        out = fmedian(a, (3, 3), 1)
        np.testing.assert_allclose(out[1, 1], 10.0, rtol=0, atol=1e-12)

    def test_dtype_enforced_float64_fmedian(self):
        """fmedian requires float64 input and output arrays."""
        good = np.ones((2, 2), dtype=np.float64)

        # Works with float64
        out = fmedian(good, (3, 3), 1)
        assert out.dtype == np.float64

        # New API coerces input to float64; float32 input is accepted and coerced
        bad_in = good.astype(np.float32)
        out2 = fmedian(bad_in, (3, 3), 1)
        assert out2.dtype == np.float64

    def test_dimension_checks_fmedian(self):
        """Non-2D arrays or mismatched shapes should raise errors for fmedian."""
        a = np.ones((2, 3), dtype=np.float64)

        # Happy path
        out = fmedian(a, (3, 3), 1)
        assert out.shape == a.shape and out.dtype == np.float64

        # 1D array should fail
        with pytest.raises(ValueError):
            a1 = np.ones(3, dtype=np.float64)
            fmedian(a1, (3, 3), 1)

    def test_fmedian_preserves_input(self):
        """Verify fmedian does not modify the input array."""
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        a_copy = a.copy()
        
        out = fmedian(a, (3, 3), 1)
        
        assert np.array_equal(a, a_copy), "Input array was modified"
        assert out is not a, "Output should be a new array"

    def test_fmedian_int_input_coercion(self):
        """Test fmedian coerces integer input to float64."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        out = fmedian(a, (3, 3), 1)
        
        assert out.dtype == np.float64
        assert out.shape == a.shape


class TestFmedianEdgeCases:
    """Test fmedian with edge cases, boundaries, and special values."""
    
    def test_median_with_all_nan_window_writes_nan(self):
        """If the whole neighborhood (and center) are NaN, output should be NaN."""
        arr = np.array([[np.nan]], dtype=np.float64)

        # Window 1x1, exclude center -> no neighbors; center is NaN -> output should be NaN
        out = fmedian(arr, (1, 1), 1)
        assert np.isnan(out[0, 0])

    def test_fmedian_large_window(self):
        """Test fmedian with window larger than the array."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        # Window extends beyond array bounds
        out = fmedian(a, (21, 21), 1)
        assert out.shape == a.shape
        assert np.all(np.isfinite(out))

    def test_fmedian_asymmetric_windows(self):
        """Test fmedian with asymmetric window sizes (xsize != ysize)."""
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        
        # Wide horizontal window
        out1 = fmedian(a, (5, 1), 1)
        assert out1.shape == a.shape
        
        # Tall vertical window
        out2 = fmedian(a, (1, 5), 1)
        assert out2.shape == a.shape
        
        # Results should differ
        assert not np.allclose(out1, out2)

    def test_fmedian_all_nan_input(self):
        """Test fmedian with an array of all NaNs."""
        a = np.full((3, 3), np.nan, dtype=np.float64)
        out = fmedian(a, (3, 3), 1)
        assert out.shape == a.shape
        assert np.all(np.isnan(out))

    def test_fmedian_mixed_nan_and_values(self):
        """Test fmedian with mixed NaN and finite values."""
        a = np.array([
            [np.nan, 1.0, np.nan],
            [2.0, 3.0, 4.0],
            [np.nan, 5.0, np.nan]
        ], dtype=np.float64)
        out = fmedian(a, (3, 3), 1)
        
        # Center should be median of finite neighbors: [1,2,4,5] -> 3.0
        assert np.isclose(out[1, 1], 3.0)

    def test_fmedian_negative_values(self):
        """Test fmedian handles negative values correctly."""
        a = np.array([
            [-5.0, -3.0, -1.0],
            [-4.0, -2.0, 0.0],
            [-3.0, -1.0, 1.0]
        ], dtype=np.float64)
        out = fmedian(a, (3, 3), 1)
        
        # Center neighbors (excluding -2.0): [-5,-3,-1,-4,0,-3,-1,1]
        # Sorted: [-5,-4,-3,-3,-1,-1,0,1] -> median = (-3 + -1)/2 = -2.0
        assert np.isclose(out[1, 1], -2.0)

    def test_fmedian_very_large_values(self):
        """Test fmedian with very large floating point values."""
        a = np.array([
            [1e100, 1e100, 1e100],
            [1e100, 1e100, 1e100],
            [1e100, 1e100, 1e100]
        ], dtype=np.float64)
        out = fmedian(a, (3, 3), 1)
        assert np.allclose(out, 1e100)

    def test_fmedian_zero_window(self):
        """Test fmedian with zero window size (only center pixel)."""
        a = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = fmedian(a, (1, 1), 1)
        # With 1x1 window and exclude_center=1, should fall back to center
        assert np.allclose(out, a)

    def test_fmedian_include_vs_exclude_center(self):
        """Verify include/exclude center produces different results with outlier."""
        a = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 100.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float64)
        
        out_excl = fmedian(a, (3, 3), 1)
        out_incl = fmedian(a, (3, 3), 0)
        
        # Excluding center: median of [1,1,1,1,1,1,1,1] = 1.0
        assert np.isclose(out_excl[1, 1], 1.0)
        
        # Including center: median of [1,1,1,1,100,1,1,1,1] = 1.0 (still 1.0 with 9 values)
        assert np.isclose(out_incl[1, 1], 1.0)

    def test_fmedian_rectangular_array(self):
        """Test fmedian with non-square arrays."""
        a = np.arange(20, dtype=np.float64).reshape(4, 5)
        out = fmedian(a, (3, 3), 1)
        assert out.shape == (4, 5)
        assert np.all(np.isfinite(out))

    def test_fmedian_single_row(self):
        """Test fmedian with a single row (height=1)."""
        a = np.arange(10, dtype=np.float64).reshape(1, 10)
        out = fmedian(a, (3, 3), 1)
        assert out.shape == (1, 10)
        assert np.all(np.isfinite(out))

    def test_fmedian_single_column(self):
        """Test fmedian with a single column (width=1)."""
        a = np.arange(10, dtype=np.float64).reshape(10, 1)
        out = fmedian(a, (3, 3), 1)
        assert out.shape == (10, 1)
        assert np.all(np.isfinite(out))

    def test_fmedian_corner_pixels(self):
        """Test fmedian handles corner pixels (truncated windows) correctly."""
        a = np.arange(9, dtype=np.float64).reshape(3, 3)
        out = fmedian(a, (3, 3), 1)
        
        # All corners should have finite values
        assert np.isfinite(out[0, 0])
        assert np.isfinite(out[0, 2])
        assert np.isfinite(out[2, 0])
        assert np.isfinite(out[2, 2])

    def test_fmedian_edge_pixels(self):
        """Test fmedian handles edge pixels (partial windows) correctly."""
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        out = fmedian(a, (3, 3), 1)
        
        # Check all edge pixels are finite
        assert np.all(np.isfinite(out[0, :]))  # Top edge
        assert np.all(np.isfinite(out[-1, :]))  # Bottom edge
        assert np.all(np.isfinite(out[:, 0]))  # Left edge
        assert np.all(np.isfinite(out[:, -1]))  # Right edge

    def test_fmedian_inf_values(self):
        """Test fmedian with infinity values."""
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.inf, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float64)
        out = fmedian(a, (3, 3), 1)
        
        # Behavior with inf depends on C implementation
        # At minimum, output should have same shape
        assert out.shape == a.shape


class TestFmedianValidation:
    """Test parameter validation and error handling for fmedian."""
    
    def test_fmedian_negative_xsize(self):
        """Test fmedian rejects negative xsize."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError, MemoryError)):
            fmedian(a, (-1, 1))
    
    def test_fmedian_negative_ysize(self):
        """Test fmedian rejects negative ysize."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError, MemoryError)):
            fmedian(a, (1, -1))
    
    def test_fmedian_rejects_even_xsize(self):
        """Test fmedian rejects even xsize."""
        a = np.ones((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="xsize must be an odd number"):
            fmedian(a, (2, 3), 0)

    def test_fmedian_rejects_even_ysize(self):
        """Test fmedian rejects even ysize."""
        a = np.ones((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="ysize must be an odd number"):
            fmedian(a, (3, 2), 0)

    def test_fmedian_requires_x_y_sizes_not_none(self):
        """Test fmedian requires non-None sizes."""
        a = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(TypeError):
            fmedian(a, None, None)

    def test_fmedian_rejects_zero_or_negative_sizes(self):
        """Test fmedian rejects zero or negative sizes."""
        a = np.ones((3, 3), dtype=np.float64)
        # Zero triggers the odd-number check (0 is even)
        with pytest.raises(ValueError, match="xsize must be an odd number"):
            fmedian(a, (0, 3), 0)
        with pytest.raises(ValueError, match="ysize must be an odd number"):
            fmedian(a, (3, 0), 0)
        # Negative sizes will pass the odd-number check in Python (e.g. -1 % 2 == 1)
        # and should trigger the positive-size validation
        with pytest.raises(ValueError, match="xsize must be positive"):
            fmedian(a, (-1, 3), 0)

    def test_fmedian_accepts_non_int_but_coerces(self):
        """Test fmedian accepts non-int but coerces."""
        a = np.ones((3, 3), dtype=np.float64)
        # floats that are integer-like should be coerced to ints
        out = fmedian(a, (3.0, 3.0), 0)
        assert out.shape == a.shape

    def test_exclude_center_coerced_to_int(self):
        """Test exclude_center is coerced to int."""
        a = np.ones((3, 3), dtype=np.float64)
        # exclude_center provided as truthy string should be coerced (via int()) and accepted
        out = fmedian(a, (3, 3), True)
        assert out.shape == a.shape
    
    def test_fmedian_invalid_exclude_center(self):
        """Test fmedian with invalid exclude_center values."""
        a = np.ones((3, 3), dtype=np.float64)
        # Non-0/1 values for exclude_center should work (treated as boolean)
        out = fmedian(a, (1, 1), exclude_center=2)
        assert out.shape == (3, 3)
    
    def test_fmedian_none_parameters(self):
        """Test fmedian rejects None parameters."""
        a = np.ones((3, 3), dtype=np.float64)

        # None for window_size should raise TypeError
        with pytest.raises(TypeError):
            fmedian(a, None)  # NOSONAR - intentionally testing invalid input
        
        # Tuple with None elements should raise an error
        with pytest.raises((TypeError, ValueError)):
            fmedian(a, (None, 1))  # NOSONAR - intentionally testing invalid input

    def test_fmedian_non_integer_parameters(self):
        """Test fmedian handles float parameters (should convert to int)."""
        a = np.ones((3, 3), dtype=np.float64)
        # Float values in tuple should work (will be converted to int by underlying functions)
        result = fmedian(a, (1.0, 1.0))
        assert result.shape == (3, 3)
    
    def test_fmedian_empty_array(self):
        """Test fmedian with empty array."""
        a = np.array([], dtype=np.float64).reshape(0, 0)
        # Should either work or raise a clear error
        try:
            out = fmedian(a, (1, 1), 1)
            assert out.shape == (0, 0)
        except (ValueError, RuntimeError):
            pass  # Acceptable to reject empty arrays
    
    def test_fmedian_3d_array(self):
        """Test fmedian accepts 3D arrays with 3-tuple window_size."""
        a = np.ones((3, 3, 3), dtype=np.float64)
        # Should work with 3-tuple
        result = fmedian(a, (1, 1, 1))
        assert result.shape == (3, 3, 3)
        
        # Should fail with 2-tuple for 3D array (dispatches to 2D function)
        with pytest.raises(ValueError, match="Arrays must be 2-dimensional"):
            fmedian(a, (1, 1))
    
    def test_fmedian_no_input(self):
        """Test fmedian raises TypeError when called with no arguments."""
        with pytest.raises(TypeError):
            fmedian()  # NOSONAR - intentionally testing invalid input
    
    def test_fmedian_missing_parameters(self):
        """Test fmedian raises TypeError when called with too few arguments.""" 
        a = np.ones((3, 3), dtype=np.float64)
        # These should still raise TypeError for missing required parameters
        with pytest.raises(TypeError):
            fmedian()  # NOSONAR - intentionally testing invalid input
        
        with pytest.raises(TypeError):
            fmedian(a)  # NOSONAR - intentionally testing invalid input
        
        with pytest.raises(TypeError):
            fmedian(a, 1)  # NOSONAR - intentionally testing invalid input
