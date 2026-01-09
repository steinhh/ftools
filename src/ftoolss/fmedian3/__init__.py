"""ftoolss.fmedian3 package loader.

This module locates the pre-built extension shared object in the repository
(`../fmedian3/fmedian3_ext*.so`) and loads it as a private module, then
exposes the `fmedian3` function at package level.

We avoid moving any compiled artifacts; the loader simply finds the first
matching shared object in the repository layout and imports it.
"""
from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os

# Try to import a bundled extension if available (e.g. when package is
# installed with aligned module names).
try:
    from . import _fmedian3_ext as _ext  # type: ignore
except Exception:
    # First, look for a compiled extension in the package directory (useful when
    # built in-place or installed). If not found, fall back to the repository
    # top-level location (legacy layout).
    _HERE = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

    candidates = glob.glob(os.path.join(_HERE, "fmedian3_ext*.so"))
    if not candidates:
        # Fallback to legacy location at the repository root
        candidates = glob.glob(os.path.join(repo_root, "fmedian3", "fmedian3_ext*.so"))
        if not candidates:
            # macOS sometimes adds extra suffixes
            candidates = glob.glob(os.path.join(repo_root, "fmedian3", "fmedian3_ext*.*.so"))

    if not candidates:
        raise ImportError(
            "Could not locate the compiled fmedian3 extension (expected src/ftoolss/fmedian3/fmedian3_ext*.so or fmedian3/fmedian3_ext*.so). "
            "Build it first or install the package so the extension is available."
        )

    so_path = candidates[0]
    # Ensure loader name matches the compiled module name (so the PyInit symbol matches)
    base = os.path.basename(so_path)
    base_mod = os.path.splitext(base)[0].split(".")[0]
    loader = importlib.machinery.ExtensionFileLoader(base_mod, so_path)
    spec = importlib.util.spec_from_loader(base_mod, loader)
    if spec is None:
        raise ImportError(f"Could not create module spec for {base_mod}")
    _ext = importlib.util.module_from_spec(spec)
    loader.exec_module(_ext)  # type: ignore[arg-type]

try:
    _c_fmedian3 = _ext.fmedian3  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - defensive
    raise ImportError("Loaded fmedian3 extension but could not find 'fmedian3' symbol") from exc


def fmedian3(input_array, xsize: int, ysize: int, zsize: int, exclude_center: int = 0):
    """Compute filtered median and return the output array.

    Signature: fmedian3(input_array, xsize, ysize, zsize, exclude_center=0) -> numpy.ndarray

    Parameters:
    - xsize, ysize, zsize: Full window sizes (must be odd numbers)
    - exclude_center: Whether to exclude the center voxel from the calculation (default: 0)

    The input will be coerced to float64; the returned array is float64.
    """
    import numpy as _np

    if xsize is None or ysize is None or zsize is None:
        raise TypeError("fmedian3 requires xsize, ysize, and zsize parameters")

    # Convert to integers and validate
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    
    # Check that sizes are positive
    if xsize <= 0:
        raise ValueError(f"xsize must be positive, got {xsize}")
    if ysize <= 0:
        raise ValueError(f"ysize must be positive, got {ysize}")
    if zsize <= 0:
        raise ValueError(f"zsize must be positive, got {zsize}")
    
    # Check that xsize, ysize and zsize are odd numbers
    if xsize % 2 == 0:
        raise ValueError(f"xsize must be an odd number, got {xsize}")
    if ysize % 2 == 0:
        raise ValueError(f"ysize must be an odd number, got {ysize}")
    if zsize % 2 == 0:
        raise ValueError(f"zsize must be an odd number, got {zsize}")
    
    arr = _np.asarray(input_array, dtype=_np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Input array must be 3-dimensional, got {arr.ndim}D")
    
    out = _np.empty_like(arr, dtype=_np.float64)
    _c_fmedian3(arr, out, xsize, ysize, zsize, int(exclude_center))
    return out


__all__ = ["fmedian3"]