# ftools Copilot Instructions

## Project Overview

`ftools` provides high-performance C-extension local image filters (`fmedian`, `fsigma`) for cosmic ray detection. The architecture separates 2D and 3D implementations into parallel modules that share a unified Python API.

## Architecture Pattern: Parallel 2D/3D Modules

Four parallel C extension modules with identical structure:

- `fmedian/` (2D median) and `fmedian3/` (3D median)
- `fsigma/` (2D sigma) and `fsigma3/` (3D sigma)

**Each module contains:**

- `*_ext.c` - Python/C API extension using NumPy C API
- `__init__.py` - Module loader and Python wrapper
- `example_*.py` - Standalone usage examples
- `test_*.py` - Module-specific tests

**Unified API dispatch in `src/ftools/__init__.py`:**

```python
def fmedian(input_array, window_size: tuple, exclude_center: int = 0):
    if len(window_size) == 2:
        return fmedian2d(arr, xsize, ysize, exclude_center)
    elif len(window_size) == 3:
        return fmedian3(arr, xsize, ysize, zsize, exclude_center)
```

Users call `fmedian(arr, (3,3))` or `fmedian(arr, (3,3,3))` - dispatch is automatic.

## Critical Implementation Details

### Sorting Networks (Internal Optimization)

**Location:** `src/ftools/sorting/`

The median/sigma filters include sorting.c directly. This provides optimized sorting networks for small arrays (3-27 elements) used in window neighborhoods.

**Key files:**

- `sorting_networks_generated.c` - Auto-generated optimal sorting networks (sort3-sort24)
- `generate_sorting_networks.py` - Python script to generate networks from specifications
- `sorting.c` - Dispatcher that selects sorting network by size or falls back to insertion sort/qsort
- `test_all_sorts.c` - Standalone C test suite comparing networks against qsort

**Critical:** Networks must use indices 0 to N-1 for N elements. The sort11 bug (accessing d[11] for 11 elements) was fixed by replacing with a correct 35-comparator network. Always verify with `test_all_sorts.c`.

**To regenerate networks:**

```bash
cd src/ftools/sorting
python3 generate_sorting_networks.py > sorting_networks_generated_new.c
gcc -O2 -o test_all_sorts.exe test_all_sorts.c -lm && ./test_all_sorts.exe
```

### NaN Handling Pattern

All C extensions treat NaN values specially:

1. NaN neighbors are **excluded** from calculations (not counted)
2. If all values in window are NaN ? output NaN
3. `exclude_center=1` excludes center pixel from calculation (for outlier detection)
4. If `exclude_center=1` and no neighbors exist ? use center value if finite

See `tests/test_fmedian_unit.py::test_median_excludes_nan_neighbors` for canonical behavior.

### C Extension Pattern

All `*_ext.c` files follow this structure:

1. Include sorting: `include "../sorting/sorting.c"`
2. Define `compute_median()` or `compute_sigma()` using `sort_doubles(values, count)`
3. Define `check_inputs()` for array validation (dimensions, types, sizes)
4. Main function loops over array, builds window buffer, calls compute function
5. Module init with `PyInit_<module>` and method table

**Data flow:** NumPy array ? `PyArray_DATA()` pointer ? double\* iteration ? window buffer ? sort ? compute ? write result

## Development Workflows

### Building Extensions

```bash
# Activate the Python 3.13 environment with NumPy
conda activate py313

# In-place build (for development/testing)
python setup.py build_ext --inplace

# Install in editable mode
pip install -e .
```

**Important:** Always activate `py313` environment first, then run `build_ext --inplace` after modifying C code before running tests.

### Testing

```bash
# Activate environment first
conda activate py313

# All tests (requires built extensions)
pytest

# Specific test module
pytest tests/test_fmedian_unit.py

# With coverage
pytest --cov=ftools --cov-report=html
```

**Test organization:**

- `tests/test_*_unit.py` - Core functionality tests
- `tests/test_*_edge_cases.py` - NaN handling, boundaries, corner cases
- `tests/test_parameter_validation.py` - Input validation
- `tests/test_smoke_integration.py` - End-to-end integration
- Module-level `test_*.py` - Module-specific tests (also run by pytest)

### C Sorting Network Tests

```bash
cd src/ftools/sorting
gcc -O2 -o test_all_sorts.exe test_all_sorts.c -lm
./test_all_sorts.exe  # Compares all networks against qsort with 10,000 random tests each
```

## Project Conventions

### Window Size Convention

- Python API accepts full window sizes: `(3, 3)` means 3×3 window
- C implementation converts to half-sizes internally: `xsize_half = xsize / 2`
- All window sizes must be **odd positive integers**

### Include Pattern for Sorting

All `*_ext.c` files include sorting via relative path:

```c
include "../sorting/sorting.c"
```

Never link as object file - sorting.c is header-included.

### Array Indexing

- 2D: `data[y * width + x]` (row-major)
- 3D: `data[z * height * width + y * width + x]` (z-y-x order)

### Error Handling

Use `PyErr_SetString()` for validation errors, return NULL to propagate to Python.

## Key Files Reference

- `setup.py` - Defines 4 Extension modules, requires numpy>=1.20
- `src/ftools/__init__.py` - Unified API dispatch logic
- `src/ftools/sorting/sorting.c` - Lines 40-130: switch statement for network selection
- `tests/test_parameter_validation.py` - Canonical validation tests
- `README.md` - User-facing documentation (keep updated with API changes)

## Shell Environment

This project uses `tcsh` shell. Use tcsh-compatible syntax:

- Output redirection: `command >& file` not `command &> file`
- Pipe stderr: `command |& filter` not `command 2>&1 | filter`

## Developer-added instructions

When creating executable files from C test programs, use file extension `.exe` to avoid confusion with source files. For example:

```bash
gcc -O2 -o test_all_sorts.exe test_all_sorts.c -lm
./test_all_sorts.exe  # Compares all networks against qsort with 10,000 random tests each
```

Use a terse style for the README file, try to avoid duplication of information.

Use redirect to top-level file dev_null.txt instead of redirecting to /dev/null

Warn me when I'm introducing new features that warrant an increment of the major version number.
