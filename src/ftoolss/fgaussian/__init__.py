"""
Gaussian profile computation module

Provides high-performance computation of Gaussian profiles using C extensions.
Uses float32 for optimal performance (~5x faster than NumPy float64).
"""

from . import fgaussian_f32_ext
from . import fgaussian_f64_ext


def fgaussian_f32(x, i0, mu, sigma):
    """
    Compute Gaussian profile: i0 * exp(-((x - mu)^2) / (2 * sigma^2))
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array (Doppler or wavelength values), dtype=float32.
    i0 : float
        Peak intensity. Must be scalar.
    mu : float
        Center position (Doppler shift). Must be scalar.
    sigma : float
        Width parameter. Must be scalar and positive.
    
    Returns
    -------
    numpy.ndarray
        Gaussian profile with same shape as x, dtype=float32.
    
    Notes
    -----
    Uses Apple Accelerate framework for vectorized computation.
    No validation or type conversion is performed.
    Assumes x is already float32 numpy array.
    
    Performance: ~5x faster than NumPy with float64.
    Accuracy: <1e-7 difference vs float64 for typical values.
    
    Examples
    --------
    >>> import numpy as np
    >>> from ftoolss import fgaussian_f32
    >>> x = np.linspace(-5, 5, 100, dtype=np.float32)
    >>> profile = fgaussian_f32(x, i0=1.0, mu=0.0, sigma=1.0)
    """
    return fgaussian_f32_ext.fgaussian_f32(x, i0, mu, sigma)


def fgaussian_f64(x, i0, mu, sigma):
    """
    Compute Gaussian profile: i0 * exp(-((x - mu)^2) / (2 * sigma^2))
    
    Float64 version for compatibility with existing code.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array (Doppler or wavelength values), dtype=float64.
    i0 : float
        Peak intensity. Must be scalar.
    mu : float
        Center position (Doppler shift). Must be scalar.
    sigma : float
        Width parameter. Must be scalar and positive.
    
    Returns
    -------
    numpy.ndarray
        Gaussian profile with same shape as x, dtype=float64.
    
    Notes
    -----
    Uses Apple Accelerate framework for vectorized computation.
    Less performant than float32 version due to memory bandwidth.
    
    Examples
    --------
    >>> import numpy as np
    >>> from ftoolss import fgaussian_f64
    >>> x = np.linspace(-5, 5, 100, dtype=np.float64)
    >>> profile = fgaussian_f64(x, i0=1.0, mu=0.0, sigma=1.0)
    """
    return fgaussian_f64_ext.fgaussian_f64(x, i0, mu, sigma)


__all__ = ['fgaussian_f32', 'fgaussian_f64']
