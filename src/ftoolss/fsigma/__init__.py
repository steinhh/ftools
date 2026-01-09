"""ftoolss.fsigma package loader.

Locate the pre-built `fsigma` extension in the repository and load it.
Expose `fsigma` at package level for `from ftoolss import fsigma` imports.
"""
from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os

try:
    from . import _fsigma_ext as _ext  # type: ignore
except Exception:
    _HERE = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

    candidates = glob.glob(os.path.join(_HERE, "fsigma_ext*.so"))
    if not candidates:
        candidates = glob.glob(os.path.join(repo_root, "fsigma", "fsigma_ext*.so"))
        if not candidates:
            candidates = glob.glob(os.path.join(repo_root, "fsigma", "fsigma_ext*.*.so"))

    if not candidates:
        raise ImportError(
            "Could not locate the compiled fsigma extension (expected src/ftoolss/fsigma/fsigma_ext*.so or fsigma/fsigma_ext*.so). "
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
    _c_fsigma = _ext.fsigma  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - defensive
    raise ImportError("Loaded fsigma extension but could not find 'fsigma' symbol") from exc


def fsigma(input_array, xsize: int, ysize: int, exclude_center: int = 0):
    """Compute local population sigma and return the output array.

    Signature: fsigma(input_array, xsize, ysize, exclude_center=0) -> numpy.ndarray

    Parameters:
    - xsize, ysize: Full window sizes (must be odd numbers)
    - exclude_center: Whether to exclude the center pixel from the calculation (default: 0)

    The input will be coerced to float64; the returned array is float64.
    """
    import numpy as _np

    if xsize is None or ysize is None:
        raise TypeError("fsigma requires xsize and ysize parameters")

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
    _c_fsigma(arr, out, xsize, ysize, int(exclude_center))
    return out


__all__ = ["fsigma"]
