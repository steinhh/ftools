"""ftoolss.fmedian package loader.

This module locates the pre-built extension shared object in the repository
(`../fmedian/fmedian_ext*.so`) and loads it as a private module, then
exposes the `fmedian` function at package level.

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
    from . import _fmedian_ext as _ext  # type: ignore
except Exception:
    # First, look for a compiled extension in the package directory (useful when
    # built in-place or installed). If not found, fall back to the repository
    # top-level location (legacy layout).
    _HERE = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

    candidates = glob.glob(os.path.join(_HERE, "fmedian_ext*.so"))
    if not candidates:
        # Fallback to legacy location at the repository root
        candidates = glob.glob(os.path.join(repo_root, "fmedian", "fmedian_ext*.so"))
        if not candidates:
            # macOS sometimes adds extra suffixes
            candidates = glob.glob(os.path.join(repo_root, "fmedian", "fmedian_ext*.*.so"))

    if not candidates:
        raise ImportError(
            "Could not locate the compiled fmedian extension (expected src/ftoolss/fmedian/fmedian_ext*.so or fmedian/fmedian_ext*.so). "
            "Build it first or install the package so the extension is available."
        )

    so_path = candidates[0]
    # Ensure loader name matches the compiled module name (so the PyInit symbol matches)
    base = os.path.basename(so_path)
    base_mod = os.path.splitext(base)[0].split(".")[0]
    loader = importlib.machinery.ExtensionFileLoader(base_mod, so_path)
    spec = importlib.util.spec_from_loader(base_mod, loader)
    _ext = importlib.util.module_from_spec(spec)
    loader.exec_module(_ext)  # type: ignore[arg-type]

try:
    _c_fmedian = _ext.fmedian  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - defensive
    raise ImportError("Loaded fmedian extension but could not find 'fmedian' symbol") from exc


def fmedian(input_array, xsize: int, ysize: int, exclude_center: int = 0):
    """Compute filtered median and return the output array.

    Signature: fmedian(input_array, xsize, ysize, exclude_center=0) -> numpy.ndarray

    Parameters:
    - xsize, ysize: Full window sizes (must be odd numbers)
    - exclude_center: Whether to exclude the center pixel from the calculation (default: 0)

    The input will be coerced to float64; the returned array is float64.
    """
    import numpy as _np

    if xsize is None or ysize is None:
        raise TypeError("fmedian requires xsize and ysize parameters")

    # Convert to integers and validate
    xsize = int(xsize)
    ysize = int(ysize)
    
    # Check that xsize and ysize are odd numbers
    if xsize % 2 == 0:
        raise ValueError(f"xsize must be an odd number, got {xsize}")
    if ysize % 2 == 0:
        raise ValueError(f"ysize must be an odd number, got {ysize}")
    
    # Check that sizes are positive
    if xsize <= 0:
        raise ValueError(f"xsize must be positive, got {xsize}")
    if ysize <= 0:
        raise ValueError(f"ysize must be positive, got {ysize}")
    
    arr = _np.asarray(input_array, dtype=_np.float64)
    out = _np.empty_like(arr, dtype=_np.float64)
    _c_fmedian(arr, out, xsize, ysize, int(exclude_center))
    return out


__all__ = ["fmedian"]
