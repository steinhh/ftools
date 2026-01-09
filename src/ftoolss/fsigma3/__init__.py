"""ftoolss.fsigma3 package loader.

Locate the pre-built `fsigma3` extension in the repository and load it.
Expose `fsigma3` at package level for `from ftoolss import fsigma3` imports.
"""
from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os

try:
    from . import _fsigma3_ext as _ext  # type: ignore
except Exception:
    _HERE = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

    candidates = glob.glob(os.path.join(_HERE, "fsigma3_ext*.so"))
    if not candidates:
        candidates = glob.glob(os.path.join(repo_root, "fsigma3", "fsigma3_ext*.so"))
        if not candidates:
            candidates = glob.glob(os.path.join(repo_root, "fsigma3", "fsigma3_ext*.*.so"))

    if not candidates:
        raise ImportError(
            "Could not locate the compiled fsigma3 extension (expected src/ftoolss/fsigma3/fsigma3_ext*.so or fsigma3/fsigma3_ext*.so). "
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
    _c_fsigma3 = _ext.fsigma3  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - defensive
    raise ImportError("Loaded fsigma3 extension but could not find 'fsigma3' symbol") from exc


def fsigma3(input_array, xsize: int, ysize: int, zsize: int, exclude_center: int = 0):
    """Compute local population sigma and return the output array.

    Signature: fsigma3(input_array, xsize, ysize, zsize, exclude_center=0) -> numpy.ndarray

    Parameters:
    - xsize, ysize, zsize: Full window sizes (must be odd numbers)
    - exclude_center: Whether to exclude the center voxel from the calculation (default: 0)

    The input will be coerced to float64; the returned array is float64.
    """
    import numpy as _np

    if xsize is None or ysize is None or zsize is None:
        raise TypeError("fsigma3 requires xsize, ysize, and zsize parameters")

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
    _c_fsigma3(arr, out, xsize, ysize, zsize, int(exclude_center))
    return out


__all__ = ["fsigma3"]