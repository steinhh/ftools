"""Top level package for ftools.

Expose unified function names at package level so users can call
`fmedian` and `fsigma` with either 2D or 3D arrays.
"""
from .fmedian import fmedian as _fmedian2d
from .fsigma import fsigma as _fsigma2d
from .fmedian3 import fmedian3 as _fmedian3d
from .fsigma3 import fsigma3 as _fsigma3d
from .fgaussian import fgaussian_f32, fgaussian_f64
from .fmpfit import fmpfit_wrap, fmpfit_f64_wrap, fmpfit_f32_wrap


def fmedian(input_array, window_size: tuple, exclude_center: int = 0):
    """Compute filtered median for 2D or 3D arrays.

    Automatically dispatches to the appropriate implementation based on window_size length.

    Parameters:
    - input_array: 2D or 3D numpy array
    - window_size: Tuple of window dimensions (xsize, ysize) for 2D or (xsize, ysize, zsize) for 3D
                   All values must be odd positive integers
    - exclude_center: Whether to exclude center pixel/voxel (default: 0)

    Returns:
    - Filtered array of same shape as input

    Examples:
    >>> # 2D array
    >>> result = fmedian(array_2d, (3, 3))
    >>> # 3D array
    >>> result = fmedian(array_3d, (3, 3, 3))
    """
    import numpy as np

    arr = np.asarray(input_array)
    
    if not isinstance(window_size, (tuple, list)):
        raise TypeError("window_size must be a tuple or list")
    
    if len(window_size) == 2:
        xsize, ysize = window_size
        from .fmedian import fmedian as fmedian2d
        return fmedian2d(arr, xsize, ysize, exclude_center)
    
    elif len(window_size) == 3:
        xsize, ysize, zsize = window_size
        from .fmedian3 import fmedian3
        return fmedian3(arr, xsize, ysize, zsize, exclude_center)
    
    else:
        raise ValueError(f"window_size must be a 2-tuple or 3-tuple, got {len(window_size)} elements")


def fsigma(input_array, window_size: tuple, exclude_center: int = 0):
    """Compute local population standard deviation for 2D or 3D arrays.

    Automatically dispatches to the appropriate implementation based on window_size length.

    Parameters:
    - input_array: 2D or 3D numpy array
    - window_size: Tuple of window dimensions (xsize, ysize) for 2D or (xsize, ysize, zsize) for 3D
                   All values must be odd positive integers
    - exclude_center: Whether to exclude center pixel/voxel (default: 0)

    Returns:
    - Array of local standard deviations, same shape as input

    Examples:
    >>> # 2D array
    >>> result = fsigma(array_2d, (3, 3))
    >>> # 3D array
    >>> result = fsigma(array_3d, (3, 3, 3))
    """
    import numpy as np

    arr = np.asarray(input_array)
    
    if not isinstance(window_size, (tuple, list)):
        raise TypeError("window_size must be a tuple or list")
    
    if len(window_size) == 2:
        xsize, ysize = window_size
        from .fsigma import fsigma as fsigma2d
        return fsigma2d(arr, xsize, ysize, exclude_center)
    
    elif len(window_size) == 3:
        xsize, ysize, zsize = window_size
        from .fsigma3 import fsigma3
        return fsigma3(arr, xsize, ysize, zsize, exclude_center)
    
    else:
        raise ValueError(f"window_size must be a 2-tuple or 3-tuple, got {len(window_size)} elements")


# Keep the specific implementations available for direct access if needed
fmedian2d = _fmedian2d
fsigma2d = _fsigma2d
fmedian3d = _fmedian3d
fsigma3d = _fsigma3d

__version__ = "3.0.0"
__all__ = ["fmedian", "fsigma", "fgaussian_f32", "fgaussian_f64", "fmedian2d", "fsigma2d", "fmedian3d", "fsigma3d", "fmpfit_wrap"]
