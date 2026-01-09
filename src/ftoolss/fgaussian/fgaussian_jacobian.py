import numpy as np


def gaussian_jacobian(x, i0, mu, sigma):
    """
    Jacobian of the Gaussian function.

    Parameters
    ----------
    x : `numpy.ndarray`
        Doppler or wavelength values.
    i0 : `float`
        Peak intensity.
    mu : `float`
        Doppler shift in same units as ``x``.
    sigma : `float`
        Width in same units as ``x``.

    Returns
    -------
    `numpy.ndarray`
        Jacobian matrix of the Gaussian function.
    """
    exp_term = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    d_i0 = exp_term
    d_mu = i0 * exp_term * (x - mu) / (sigma**2)
    d_sigma = i0 * exp_term * ((x - mu) ** 2) / (sigma**3)
    return np.vstack((d_i0, d_mu, d_sigma)).T
