# FMPFIT Float32 Implementation

## Overview

Float32 (single-precision) versions of all FMPFIT files have been created with `_f32` suffix in the file/function names. These provide memory-efficient and faster fitting for applications where full float64 precision is not required.

## Files Created

### 1. C Extension Files

**gaussian_deviate_f32.c**

- Float32 version of Gaussian deviate computation
- Uses `expf()` for single-precision exponentials
- Data structure: `gaussian_private_data_f32`
- Function: `myfunct_gaussian_deviates_with_derivatives_f32()`

**fmpfit_f32_ext.c**

- Python C extension wrapper for float32 MPFIT
- Links against `cmpfit-1.5_f32/mpfit.c` library
- Core function: `fmpfit_f32_c_wrap()`
- Python wrapper: `py_fmpfit_f32()`
- Module: `fmpfit_f32_ext` with function `fmpfit_f32()`
- Uses NPY_FLOAT32 for NumPy array types

### 2. CMPFIT Library (cmpfit-1.5_f32/)

Complete float32 version of the MPFIT library:

- **mpfit.h**: Header with float types and float32 constants
- **mpfit.c**: Core MPFIT implementation (all double ? float)
- **testmpfit.c**: Test program

All math functions converted to float versions:

- `sqrt()` ? `sqrtf()`
- `fabs()` ? `fabsf()`
- `exp()` ? `expf()`

Numeric constants updated for float32 precision:

- MP_MACHEP0: 1.19209290e-07f (machine epsilon)
- MP_DWARF: 1.17549435e-38f (smallest normalized float)
- MP_GIANT: 3.40282347e+38f (largest float)

### 3. Python Wrapper

****init**.py updates:**

- Added `fmpfit_f32_wrap()` function
- Mirrors the API of `fmpfit_wrap()` but uses float32 internally
- Accepts the same parameters
- Returns `MPFitResult` with float32 arrays

**Usage:**

```python
from ftools.fmpfit import fmpfit_f32_wrap

result = fmpfit_f32_wrap(
    deviate_type=0,
    parinfo=parinfo,
    functkw={'x': x, 'y': y, 'error': error}
)
```

### 4. Test File

**test_fmpfit_f32.py**

- Comprehensive test comparing float64 vs float32 versions
- Validates numerical accuracy
- Shows memory savings (50% reduction)
- Demonstrates dtype verification

## Performance Characteristics

### Memory Usage

- **50% reduction** in memory for all arrays
- Example (100 data points, 3 parameters):
  - Float64: 920 bytes
  - Float32: 460 bytes

### Numerical Accuracy

- Results typically agree within 5e-5 relative tolerance
- Max parameter difference: ~3-4e-5 (well within float32 precision)
- Both versions converge to nearly identical solutions

### Speed

- Similar or slightly faster than float64 version
- Float32: ~27-50 ?s per fit
- Float64: ~43-57 ?s per fit
- Speed varies by problem complexity and hardware

## API Compatibility

The float32 version maintains full API compatibility:

- Same function signature
- Same parameter structure
- Same return type (MPFitResult)
- Only difference: internal precision and output array dtypes

## When to Use Float32

**Use float32 when:**

- Memory is constrained
- Processing large batches of fits
- Precision requirements are moderate (~6-7 significant digits)
- Working with naturally noisy data

**Use float64 when:**

- Maximum precision is required
- Working with high-precision measurements
- Accumulated errors from many operations matter
- No memory constraints

## Integration with Setup.py

Added to `setup.py`:

```python
Extension(
    "ftools.fmpfit.fmpfit_f32_ext",
    sources=[
        "src/ftools/fmpfit/fmpfit_f32_ext.c",
        "src/ftools/fmpfit/cmpfit-1.5_f32/mpfit.c",
    ],
    include_dirs=include_dirs + ["src/ftools/fmpfit"],
    extra_compile_args=["-O3"],
    extra_link_args=fgaussian_extra_link_args,
)
```

## Exports

From `ftools.fmpfit`:

- `fmpfit_wrap` - float64 version (original)
- `fmpfit_f32_wrap` - float32 version (new)
- `MPFitResult` - result class (works with both)

All functions are exported in `__all__`.
