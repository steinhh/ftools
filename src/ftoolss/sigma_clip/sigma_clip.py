import numpy as np
from ftools import fmedian, fsigma


def sigma_clip(
    data,
    size,
    sigma=3,
    sigma_lower=None,
    sigma_upper=None,
    maxiters=5,
    exclude_center=False,
    fill=False,
):
    """
    Performs sigma-clipping of the input array.

    Parameters
     ----------
    data: numpy.ndarray
        Input array
    size: int or tuple[int]
        Size of the kernel used to compute the running median (or mean) and standard deviation
    sigma: float
        The number of standard deviations to use for both the lower and upper clipping limit.
        This is overriden by `sigma_lower` and `sigma_upper`
    sigma_lower: float
        Low threshold, in units of the standard deviation of the local intensity distribution
    sigma_upper: float
        High threshold, in units of the standard deviation of the local intensity distribution
    maxiters: int
        Maximum number of iterations to perform
    exclude_center: bool
        Whether to exclude the center pixel when computing the local median and standard deviation.
         This is useful when the center pixel is expected to be an outlier (e.g. a cosmic ray) and
         would bias the local statistics.

    Returns
    -------
    numpy.ndarray
        Filtered array, with clipped pixels replaced by the estimated value of the center of the
        local intensity distribution (either median or mean).
    """
    output = np.copy(data)
    if isinstance(size, np.ndarray):
        size = tuple(size.tolist())
    if type(size) is not tuple:
        size = (size,) * data.ndim
    sigma_lower = sigma_lower or sigma
    sigma_upper = sigma_upper or sigma
    maxiters = maxiters or np.inf
    nchanged = 1
    iteration = 0
    while nchanged != 0 and (iteration < maxiters):
        iteration += 1
        center = fmedian(output, size, exclude_center=exclude_center)
        stddev = fsigma(output, size, exclude_center=exclude_center)
        diff = output - center
        new_mask = (diff > sigma_upper * stddev) | (diff < -sigma_lower * stddev)
        output[new_mask] = np.nan
        nchanged = np.count_nonzero(new_mask)
    if fill:
        nan = np.isnan(output)
        output[nan] = center[nan]  # Last value for center used for filling
        # There is a chance that the fmedian() center value was also NaN, we need to
        # iteratively fill those as well
        while np.any(np.isnan(output)):
            center = fmedian(output, size, exclude_center=exclude_center)
            output[np.isnan(output)] = center[np.isnan(output)]

    return output
