# Bugfix: mp_xerror_scipy Hessian Inversion Failure

## Summary

Fixed a bug in `mp_xerror_scipy()` in `cmpfit-1.5/mpfit.c` where Gauss-Jordan elimination with partial pivoting failed to correctly invert ill-conditioned 3x3 Hessian matrices, resulting in parameter errors being ~1000x too small.

## Bug Description

### Symptom

When fitting Gaussians with parameters hitting bounds (e.g., linewidth at upper bound), the computed parameter uncertainties (`xerror_scipy`) were drastically underestimated:

- Expected error (from scipy): 329.88
- Actual error (from buggy mp_xerror_scipy): 0.30
- Ratio: ~1000x too small

### Root Cause

The Hessian matrix H = J^T * J for these cases has:

- **Condition number**: ~9×10?
- **Diagonal values spanning 6 orders of magnitude**: [4.6, 9.4e-6, 1.5e-6]

The original Gauss-Jordan elimination with partial pivoting failed because:

1. First column (largest diagonal ~4.6) processes correctly
2. After elimination, remaining 2×2 submatrix has tiny pivots (~1e-6)
3. These pass the singularity check (`> MACHEP*100`) but produce incorrect results
4. Result: H^-1 diagonal = [0.22, 1, 1] instead of correct [0.48, 152292, 1872780]

### Affected Code

- `mp_xerror_scipy()` in `cmpfit-1.5/mpfit.c`
- Used when fitting with analytical derivatives and `result->xerror_scipy` is non-NULL

## Fix Applied

Added a new helper function `mp_invert_3x3_symmetric()` that uses **cofactor expansion (Cramer's rule)** for 3×3 symmetric matrix inversion:

```c
static mp_real mp_invert_3x3_symmetric(mp_real *A, mp_real *Ainv)
{
  // For symmetric 3x3: A = | a b c |
  //                        | b d e |
  //                        | c e f |
  
  // Compute cofactors
  mp_real C00 = d*f - e*e;
  mp_real C01 = -(b*f - e*c);
  mp_real C02 = b*e - d*c;
  mp_real C11 = a*f - c*c;
  mp_real C12 = -(a*e - b*c);
  mp_real C22 = a*d - b*b;
  
  // Determinant
  mp_real det = a*C00 + b*C01 + c*C02;
  
  // Inverse = adjugate / det
  ...
}
```

This approach:

- Is numerically stable for ill-conditioned 3×3 matrices
- Uses closed-form formulas (no iterative elimination)
- Falls back to scaled Gauss-Jordan for matrices larger than 3×3

## Test Cases

### test_xerror_bugcase.py

Extracted from real MUSE data processing. Tests a specific case where:

- Linewidth hits upper bound (200 km/s)
- Only 5 data points, 2 degrees of freedom
- Very small chi-squared (0.000286)

```python
xdata = np.array([-81.2, -40.6, 0.0, 40.6, 81.2], dtype=np.float32)
ydata = np.array([0.899, 0.949, 1.0, 0.988, 0.952], dtype=np.float32)
```

Expected result: errors match scipy curve_fit within 0.01%.

### test_xerror_comparison.py

Comprehensive test comparing `xerror_scipy` (external formula) vs `xerror_scipy_mp` (internal callback-based):

1. Normal fit (linewidth within bounds)
2. Linewidth at upper bound
3. Linewidth exactly at bound
4. Very wide Gaussian (extreme bound violation)
5. Narrow Gaussian (at lower bound)

## Verification

After fix:

```
Point (y=13, x=355):
  scipy error_velocity: 329.8776
  minimal error_velocity: 329.8782

Max abs diff in error_velocity: 5.49e-04  (was 330)
```

Improvement: **600,000x** reduction in error difference.

## Files Modified

- `cmpfit-1.5/mpfit.c`: Added `mp_invert_3x3_symmetric()`, modified `mp_xerror_scipy()`

## Related Changes

- Added `xerror_scipy_mp` output to `fmpfit_f32_ext.c` to expose internal mp_xerror_scipy alongside external xerror_scipy.c computation (for debugging/comparison)
- Updated `__init__.py` to include `xerror_scipy_mp` in `MPFitResult` class

## Notes

- The external `xerror_scipy.c` (using explicit Gaussian derivative formulas) was always correct
- The bug only affected `mp_xerror_scipy()` which uses the user-provided deviate callback
- Both methods now produce identical results matching scipy curve_fit
