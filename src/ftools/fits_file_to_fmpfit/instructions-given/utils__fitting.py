import multiprocessing as mp
from tqdm import tqdm

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.units import cds
from scipy.optimize import curve_fit

from muse import logger
import torch
from muse.transforms.transforms import reshape_rs2x, reshape_x2rs
from muse.utils.utils import add_history, generate_outliers, reduced_chisquare
from muse.variables import CENTROID_UNCERT_PROMISED, NPHOT_THRESHOLD
from muse.utils.mpfit import mpfit
from muse.variables import CENTROID_UNCERT_PROMISED, NPHOT_THRESHOLD, NDN_THRESHOLD
from muse.synthesis.synthesis import ph2dn

from ftools.fgaussian.fgaussian_ext import fgaussian
#from ftools.fgaussian.fgaussian_f64_ext import fgaussian_f64 as fgaussian

__all__ = [
    "calculate_moments_gaussian",
    "main_gaussian_fitter",
    "add_wavelength_information",
    "get_minimum_width",
    "fit_gaussian",
    "get_max_logt",
    "compute_thermal_width",
    "gaussian",
    "gaussian_fitting_analysis",
    "masker",
    "get_outliers",
]


def add_wavelength_information(spectrum):
    """
    This function adds new data variables to the passed spectrum.

    These variables are:

        1. Rest wavelength indices in SG_xpixel for each slit and line.
        2. Pixel size in wavelength scale for each line.
        3. Pixel size in Doppler sale for each line.
        4. Doppler axis for each slit

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        spectrum.

    Returns
    -------
    `xarray.Dataset`
        spectrum with updated variables.
    `numpy.ndarray`
        Array of spectrograph pixel coordinates for the rest wavelength.
    """
    spectrum = spectrum.copy()
    tmp_ar = np.abs(spectrum.SG_wvl - spectrum.line_wvl).argmin("SG_xpixel")
    # All this transpose is to ensure numpy adds excess dimensions automatically.
    spectrum["rest_wvl_SG_xpix_slit"] = tmp_ar
    # ------ Velocity axis
    spectrum.coords["SG_dopp"] = (spectrum.SG_wvl / spectrum.line_wvl - 1.0) * (cds.c.to("km/s"))
    spectrum.coords["SG_dopp"].attrs["units"] = str(u.km / u.s)
    add_history(spectrum, locals(), add_wavelength_information)
    return spectrum, tmp_ar


def _ensure_slit_dim(da, nslit):
    """If 'slit' dim missing, add it and broadcast."""
    if 'slit' in da.dims:
        return da
    # Put 'slit' first for consistency; remaining dims keep order
    return da.expand_dims({'slit': nslit})

def _split_spec_slitwise(dst_spec, src_spec, velocity_range, velocity_array, SG_dopp, nslit, flag_pix_slac=0):
    """
    Split a slit-summed spectrum slitwise. Also perform vetting of local
    maximum by pix_slac method if flag is passed.

    Parameters
    ----------
    dst_spec : `xarray.Dataset`
        Destination spectrum on which to save the spec.
    src_spec : `xarray.Dataset`
        Spectrum to be manipulated.
    velocity_range : `float`
        The velocity about which to search for the maximum
    velocity_array : `xarray.DataArray`
        A velocity array about which to consider the selection.
    SG_dopp : `xarray.DataArray`
        Doppler axis.
    flag_pix_slac: `int`
        A flag for performing pix_slac vetting. Defaults to 0, made non-zero if needed.

    Returns
    -------
    `xarray.Dataset`
        Return a spectrum split across slits.
    """
    # Ensure 'slit' exists everywhere and align dims
    v_arr = _ensure_slit_dim(velocity_array, nslit)
    dopp  = _ensure_slit_dim(SG_dopp, nslit)
    sig_ds   = _ensure_slit_dim(src_spec[0], nslit)
    noise_ds = _ensure_slit_dim(src_spec[1], nslit)

    # Build the base mask for the requested velocity_range: shape (slit, SG_xpixel)
    base_mask = xr.where(
        np.abs(dopp - v_arr) <= velocity_range,
        1,
        0,
    )

    # Apply to signal
    spec_vrange = sig_ds['flux'] * base_mask

    if flag_pix_slac:
        # dv per-slit from median pixel spacing of SG_dopp
        # (median of finite differences along spectral pixel axis)
        # Result dims: ('slit',)
        spacing = dopp.diff('SG_xpixel')
        dv = spacing.median('SG_xpixel') * flag_pix_slac

        # Enlarge velocity_range by dv per slit
        # Need to line up dims to (slit, SG_xpixel)
        larger_mask = xr.where(
            np.abs(dopp - v_arr) <= (velocity_range + dv).broadcast_like(dopp),
            1,
            0,
        )
        spec_largervrange = sig_ds['flux'] * larger_mask

        # Vetting: keep only points where the local maxima index agrees
        max_loc_vrange       = spec_vrange.argmax(dim='SG_xpixel')        # (slit,)
        max_loc_largervrange = spec_largervrange.argmax(dim='SG_xpixel')  # (slit,)
        agree = (np.abs(max_loc_vrange - max_loc_largervrange) == 0)       # (slit,)

        # Broadcast the slit-wise agreement to (slit, SG_xpixel) and zero out where it fails
        spec_vrange = xr.where(agree.broadcast_like(spec_vrange), spec_vrange, 0)

    # Apply the same base mask to noise
    spec_vrange_noise = noise_ds['flux'] * base_mask

    # Write results into destination (expects dst_spec[0] and dst_spec[1] have data var 'flux' with 'slit')
    dst_spec[0]['flux'].loc[{'slit': slice(None)}] = spec_vrange
    dst_spec[1]['flux'].loc[{'slit': slice(None)}] = spec_vrange_noise
    return dst_spec


def _preprocess_spec_for_fitting(input_spectrum, input_noise, difference_npix, npix, max_value, max_noise, max_indices, SG_dopp, nslit, npix_left=None, npix_right=None):
    """
    Preprocess the input spectrum and ready it for fitting.

    We first select the part of spectrum only around the local maximum.
    Then we rescale it by the max value.
    Finally, we move our spectrum to center it around the local maximum.

    Parameters
    ----------
    input_spectrum : `xarray.Dataset`
        Spectrum to preprocess.
    difference_npix : `float`
        Difference in pixel values.
    npix : `int`
        Number of pixels to consider.
    max_value : `float`
        Maximum value of the spectrum.
    max_indices : `xarray.Dataset`
        Indices of the maximum value.
    SG_dopp : `xarray.Dataset`
        SG Doppler dataset.
    nslit : `int`
        Number of slits.
    npix_left : `int`, optional
        Number of pixels to the left of the local maximum.
        Defaults to None.
    npix_right : `int`, optional
        Number of pixels to the right of the local maximum.
        Defaults to None.

    Returns
    -------
    `xarray.Dataset`
        Preprocessed spectrum.
    """
    abs_diff = np.abs(difference_npix)
    # All pixels inside mask
    all_inside_core = xr.where(abs_diff <= npix, 1.0, np.nan)
    # Check boundaries if there are less than 2*npix + 1
    need_expand = all_inside_core.sum("SG_xpixel", skipna=True) < (2 * npix + 1)
    # Generate a bigger npix mask
    all_inside_np1 = xr.where(abs_diff <= (npix + 1), 1.0, np.nan)
    # Where expansion is needed, consider extra pixels. Else consider 2*npix + 1
    all_inside = xr.where(need_expand, all_inside_np1, all_inside_core)

    output_spectrum = input_spectrum.copy(deep=True)
    output_noise = input_noise.copy(deep=True)
    if npix_left is not None and npix_right is not None:
        mask = xr.where((difference_npix >= -npix_left) & (difference_npix <= npix_right), 1, np.nan)
    else:
        mask = xr.where(max_value == 0, np.nan, 1)
    mask = mask * all_inside
    # Normalize by max value
    eps = 1e-8
    output_spectrum["flux"] = (output_spectrum["flux"] / (max_value + eps)) * mask
    output_noise["flux"] = (output_noise["flux"] / max_noise) * mask
    # Move the spectrum axis and center at the peak intensity
    tmp_arr = SG_dopp.isel(SG_xpixel=max_indices)
    output_spectrum["base_v0"] = tmp_arr
    output_spectrum["new_wvl"] = SG_dopp - tmp_arr
    output_noise["base_v0"] = tmp_arr
    output_noise["new_wvl"] = SG_dopp - tmp_arr
    return output_spectrum, output_noise


def get_max_logt(response, line=None):
    """
    Get the maximum log temperature from response function for a given line.

    Parameters
    ----------
    response : `xarray.Dataset`
        Response function.
    line : `str`, optional
        Line for which formation temperature needs to be calculated, needed if multiple lines present in response.
        Defaults to None.

    Returns
    -------
    float
        Formation temperature of the line.

    Raises
    ------
    ValueError
        If multiple lines present in response function and line keyword not provided.
    """
    if line is not None and "line" in response.SG_resp.dims:
        t_response = response.sum(["SG_xpixel", "vdop", "slit"]).sel(line=line)
    elif "line" in response.SG_resp.dims:
        msg = (
            "Line not provided, but present in response function. Subselect for one line and provide response function."
        )
        raise ValueError(
            msg,
        )
    else:
        t_response = response.sum(["SG_xpixel", "vdop", "slit"])
    return t_response.logT[np.argmax(t_response.SG_resp.values)].values.item()


def get_minimum_width(response, npix=2):
    """
    Get minimum width, including thermal and instrumental broadening for a
    response function. Response function must have SG_dopp variable.ß.

    Parameters
    ----------
    response: `xarray.Dataset`
        response function array containing only ONE spectral line.

    Returns
    -------
    `float`
        A floating point number corresponding to the width of response function line.
    """
    subresp = response.sel(vdop=0, method="nearest")  # 0 velocity
    coordinate_list = list(set(subresp.SG_resp.dims) - {"logT"})
    max_logT = subresp.SG_resp.sum(coordinate_list).argmax("logT").values.item()
    subresp = subresp.isel(logT=max_logT)  # Select max temp bin
    tmp_x = subresp.SG_dopp.values
    tmp_y = subresp.SG_resp.squeeze().transpose(*subresp.SG_dopp.dims).values / np.max(subresp.SG_resp.values)
    dv = np.median(np.gradient(subresp.SG_dopp, axis=-1))  # Find dv
    wvl = (-dv * npix, dv * npix)
    # --- Initial and boundary conditions.
    ic = [1, 0, 10]  # i0, v0, sigma0
    bc = np.asarray([(0.9, 1.1), (wvl[0], wvl[1]), (0, 200)]).T
    list_fit = [tmp_x, tmp_y, ic, bc]
    popt = fit_gaussian(list_fit)
    return popt[2]


def compute_thermal_width(logT, rest_velocity):
    """
    Compute the thermal width.

    Parameters
    ----------
    logT : `numpy.float`
        Peak formation temperature.
    rest_velocity : `xarray.Dataset`
        Rest velocity.

    Returns
    -------
    `numpy.ndarray` or scalar
        Thermal width of same shape as ``rest_velocity``.
    """
    # Find the peak logT
    # Now obtain the thermal width.
    mass_of_iron = (u.misc.u * 55.845).to("g")
    # Thermal velocity v_t = sqrt(2kT/M)
    v_t = (np.sqrt(2 * cds.k.si * (10.0**logT * u.K) / mass_of_iron)).to("km/s")
    return (v_t * rest_velocity / cds.c.to("km/s")).value


def gaussian(x, i0, mu, sigma):
    """
    Gaussian definition.

    Parameters
    ----------
    x : `xarray.Dataset` or `numpy.ndarray`
        Doppler or wavelength values.
    i0 : `xarray.Dataset` or `numpy.ndarray` or `float` :
        Peak intensity.
    mu : `xarray.Dataset` or `numpy.ndarray` or `float`:
        Doppler shift in same units as ``x``.
    sigma : `xarray.Dataset` or `numpy.ndarray` or `float`
        Width in same units as ``x``.

    Returns
    -------
    `xarray.Dataset` or `numpy.ndarray`
        Gaussian profile over ``x``.
    """
    return i0 * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def gaussian_torch(x, i0, mu, sigma):
    """
    Gaussian definition.

    Parameters
    ----------
    x : `xarray.Dataset` or `numpy.ndarray`
        Doppler or wavelength values.
    i0 : `xarray.Dataset` or `numpy.ndarray` or `float` :
        Peak intensity.
    mu : `xarray.Dataset` or `numpy.ndarray` or `float`:
        Doppler shift in same units as ``x``.
    sigma : `xarray.Dataset` or `numpy.ndarray` or `float`
        Width in same units as ``x``.

    Returns
    -------
    `xarray.Dataset` or `numpy.ndarray`
        Gaussian profile over ``x``.
    """
    return i0 * torch.exp(-((x - mu) ** 2) / (2 * sigma**2))

def gaussian_jacobian(x, i0, mu, sigma):
    """
    Jacobian of the Gaussian function.

    Parameters
    ----------
    x : `xarray.Dataset` or `numpy.ndarray`
        Doppler or wavelength values.
    i0 : `xarray.Dataset` or `numpy.ndarray` or `float`
        Peak intensity.
    mu : `xarray.Dataset` or `numpy.ndarray` or `float`
        Doppler shift in same units as ``x``.
    sigma : `xarray.Dataset` or `numpy.ndarray` or `float`
        Width in same units as ``x``.

    Returns
    -------
    `numpy.ndarray`
        Jacobian matrix of the Gaussian function.
    """
    inv_sigma = 1.0 / sigma
    inv_sigma2 = inv_sigma * inv_sigma

    exp_term = gaussian(x, i0, mu, sigma)
    d_i0 = exp_term / i0
    xmu = x - mu
    d_mu = exp_term * xmu * inv_sigma2
    d_sigma = d_mu * xmu * inv_sigma
    return np.stack((d_i0, d_mu, d_sigma), axis = -1)


def local_max(array):
    r"""
    Computes local maximum condition for a given array.

    Computes $AA = \\sgn[\nabla A]$ as the sign of gradient of array A.
    Then, compute $S = \\Sum[AA]$ as a measure of local maximum.

    Parameters
    ----------
    array : numpy.ndarray
        A 1-D numpy array of spectrum.

    Returns
    -------
    int
        A number that indicates:

        -1: Local minimum found
        0: Even number of local maxima or no local maxima.
        1: Odd number of local maxima
    """
    grad_sign = np.sign(array[1:] - array[:-1])
    ifmax = np.sum(grad_sign[1:] - grad_sign[:-1])
    return np.sign(ifmax)


def fit_gaussian_remove_badspec(in_args):
    """
    Fit a Gaussian given initial, boundary conditions and fitting variables for
    ONE spectrum.

    Return Nan if a local maximum is not found in the spectrum.

    It will return `nan` if the fit does not converge, and ignores nan values in the inputs.
    This will be called in parallel in :meth:`calculate_moments_gaussian`.

    Parameters
    ----------
    in_args[0] : `numpy.ndarray`
        Array of wavelength or doppler axis.
    in_args[1] : `numpy.ndarray`
        Spectrum intensities.
    in_args[2] : `list`
        Initial guess in the form ``[intensity, doppler shift, line width]``.
    in_args[3] : `list`
        Boundary conditions in the form ``[[lower bounds of all parameters],[upper bounds of all parameters]]``
    Returns
    -------
    `numpy.ndarray`
        Optimal parameter array from fitting.
    """
    xdata = in_args[0]
    ydata = in_args[1]
    p0 = in_args[2]
    bounds = in_args[3]
    y_sigma = in_args[4]
    neg, pos = in_args[5], in_args[6]
    mask = np.isnan(ydata)
    ydata = ydata[~mask]
    xdata = xdata[~mask]
    y_sigma = y_sigma[~mask]
    inds = np.where(np.logical_and(xdata <= pos, xdata >= neg))
    signs = local_max(ydata[inds])
    if signs >= 0:
        logger.debug("No local maximum, so not fitting")
        popt = np.array([np.nan] * (len(p0)*2 + 2))
        popt[-1] = 1
    else:
        try:
            #popt, pcov = curve_fit(gaussian, xdata, ydata, p0=p0, bounds=bounds, check_finite=False, sigma=y_sigma)
            popt, pcov = curve_fit(fgaussian, xdata, ydata, p0=p0, bounds=bounds, check_finite=False, sigma=y_sigma)
            popt, pcov = curve_fit(gaussian, xdata, ydata, p0=p0, bounds=bounds, check_finite=False, sigma=y_sigma,  jac = gaussian_jacobian)
            pcov = np.sqrt(np.diag(pcov))
            dof = len(xdata) - len(p0)  # No. of observations  - no. of fitted params.
            #chisquare = reduced_chisquare(ydata, gaussian(xdata, *popt), y_sigma, dof)
            chisquare = reduced_chisquare(ydata, fgaussian(xdata, *popt), y_sigma, dof)
            popt = np.append(popt,pcov)
            popt = np.append(popt, chisquare)
            popt = np.append(popt, 0)
        except Exception as e:  # NOQA: BLE001
            logger.debug(f"Fit did not converge: {e}")
            popt = np.array([np.nan] * (len(p0)*2 + 2))
            popt[-1] = 2
    return popt

def deviates(popt, x, y, fjac = None, error = None):
    if error is None:
        error = 1
    #return [0, (y-gaussian(x, *popt))/error]
    return [0, (y-fgaussian(x, *popt))/error]

def fit_gaussian_mpfit(in_args):
    """
    Fit a Gaussian given initial, boundary conditions and fitting variables for
    ONE spectrum.

    It will return `nan` if the fit does not converge, and ignores nan values in the inputs.
    This will be called in parallel in :meth:`calculate_moments_gaussian`.

    Parameters
    ----------
    in_args[0] : `numpy.ndarray`
        Array of wavelength or doppler axis.
    in_args[1] : `numpy.ndarray`
        Spectrum intensities.
    in_args[2] : `list`
        Initial guess in the form ``[intensity, doppler shift, line width]``.
    in_args[3] : `list`
        Boundary conditions in the form ``[[lower bounds of all parameters],[upper bounds of all parameters]]``

    Reference:
        https://github.com/segasai/astrolibpy/blob/master/mpfit/mpfit.py
        https://eispac.readthedocs.io/en/stable/guide/07-mpfit_docs.html
        https://github.com/USNavalResearchLaboratory/eispac/blob/dfa97b130bd9d286a70783e33a0f368ce6a4c3bb/eispac/core/fit_spectra.py#L66

    Returns
    -------
    `numpy.ndarray`
        Optimal parameter array from fitting.
    """
    xdata = in_args[0]
    ydata = in_args[1]
    p0 = in_args[2]
    bounds = in_args[3]
    y_sigma = np.ones_like(ydata) if len(in_args) <= 4 else in_args[4]
    mask = np.isnan(ydata)
    ydata = ydata[~mask]
    xdata = xdata[~mask]
    y_sigma = y_sigma[~mask]
    if np.sum(np.abs(ydata)) == 0 or len(ydata) == 0:
        logger.debug("No local maximum, so not fitting")
        popt = np.array([np.nan] * (len(p0) + 2))
        popt[-1] = 1
    else:
        fa = {'x': xdata, 'y': ydata, 'error': y_sigma}
        parinfo = [{'value':p0[i], 'fixed':0, 'limited':[1,1],
            'limits':[bounds[0][i],bounds[1][i]]} for i in range(len(p0))]

        mp_ = mpfit(deviates,parinfo=parinfo, functkw=fa, xtol=1.0E-6, ftol=1.0E-6, gtol=1.0E-6,
                        maxiter=2000, quiet=1)
        popt = mp_.params
        dof = len(xdata) - len(p0)  # No. of observations  - no. of fitted params.
        #chisquare = reduced_chisquare(ydata, gaussian(xdata, *popt), y_sigma, dof)
        chisquare = reduced_chisquare(ydata, fgaussian(xdata, *popt), y_sigma, dof)
        popt = np.append(popt, chisquare)
        popt = np.append(popt, 0)
    return popt


def fit_gaussian(in_args):
    """
    Fit a Gaussian given initial, boundary conditions and fitting variables for
    ONE spectrum.

    It will return `nan` if the fit does not converge, and ignores nan values in the inputs.
    This will be called in parallel in :meth:`calculate_moments_gaussian`.

    Parameters
    ----------
    in_args[0] : `numpy.ndarray`
        Array of wavelength or doppler axis.
    in_args[1] : `numpy.ndarray`
        Spectrum intensities.
    in_args[2] : `list`
        Initial guess in the form ``[intensity, doppler shift, line width]``.
    in_args[3] : `list`
        Boundary conditions in the form ``[[lower bounds of all parameters],[upper bounds of all parameters]]``

    Returns
    -------
    `numpy.ndarray`
        Optimal parameter array from fitting.
    """
    xdata = np.squeeze(in_args[0])
    ydata = np.squeeze(in_args[1])
    p0 = in_args[2]
    bounds = in_args[3]
    y_sigma = np.ones_like(ydata) if len(in_args) <= 4 else in_args[4]
    mask = np.isnan(ydata)
    ydata = ydata[~mask]
    xdata = xdata[~mask]
    y_sigma = y_sigma[~mask]
    if np.sum(np.abs(ydata)) == 0 or len(ydata) == 0:
        logger.debug("No local maximum, so not fitting")
        popt = np.array([np.nan] * (len(p0)*2 + 2))
        popt[-1] = 1
    else:
        try:
            #popt, pcov = curve_fit(gaussian, xdata, ydata, p0=p0, bounds=bounds, check_finite=False, sigma=y_sigma)
            popt, pcov = curve_fit(fgaussian, xdata, ydata, p0=p0, bounds=bounds, check_finite=False, sigma=y_sigma)
            popt, pcov = curve_fit(gaussian, xdata, ydata, p0=p0, bounds=bounds, check_finite=False, sigma=y_sigma, jac = gaussian_jacobian)
            pcov = np.sqrt(np.diag(pcov))
            dof = len(xdata) - len(p0)  # No. of observations  - no. of fitted params.
            #chisquare = reduced_chisquare(ydata, gaussian(xdata.astype(float), *popt), y_sigma, dof)
            chisquare = reduced_chisquare(ydata, fgaussian(xdata, *popt), y_sigma, dof)
            popt = np.append(popt,pcov)
            popt = np.append(popt, chisquare)
            popt = np.append(popt, 0)

        except Exception as e:  # NOQA: BLE001
            logger.debug(f"Fit did not converge: {e}")
            popt = np.array([np.nan] * (len(p0)*2 + 2))
            popt[-1] = 2
    return popt

def fit_gaussian_caruana(in_args):
    xdata = in_args[0]
    ydata = in_args[1]
    p0 = in_args[2]
    bounds = in_args[3]
    ydata[ydata<0] = 0
    ydata = ydata + 1e-5
    mask = np.isnan(ydata)
    y_sigma = np.ones_like(ydata) if len(in_args) <= 4 else ydata/in_args[4]
    y_log_transformed = np.log(ydata)
    ydata = ydata[~mask]
    y_log_transformed = y_log_transformed[~mask]
    xdata = xdata[~mask]
    y_sigma = y_sigma[~mask]
    if np.sum(np.abs(ydata)) == 0 or len(y_log_transformed) == 0:
        logger.debug("No local maximum, so not fitting")
        popt = np.array([np.nan] * (len(p0)*2 + 2))
        popt[-1] = 1
    else:
        try:
            coeffs, pcov = np.polyfit(xdata, y_log_transformed, 2, w = y_sigma, cov = True)
            # Extract Gaussian parameters from quadratic fit
            A = coeffs[0]
            B = coeffs[1]
            C = coeffs[2]
            sigma_est = np.sqrt(-1 / (2 * A))
            mu_est = B / (-2 * A)
            amp_est = np.exp(C + (mu_est ** 2) / (2 * sigma_est ** 2))
            popt = np.array([amp_est,mu_est,sigma_est])
            # Estimate the fit uncertainties by propagating them.
            delta_sigma = (sigma_est / (2 * A)) * np.sqrt(pcov[0,0])
            # Uncertainty in mu: var(mu) = var(A)*(dmu/dA)**2  + var(B)*(dmu/dB)**2 + 2*covar(A,B)*(dmu/dA)*(dmu/dB).
            delta_mu = np.sqrt((B / (2 * A**2))**2 * pcov[0,0] + (-1 / (2 * A))**2 * pcov[1,1] + 2 * (B / (2 * A**2)) * (-1 / (2 * A)) * pcov[0,1])
            # Uncertainty in amplitude
            term1 = pcov[2,2]
            term2 = (mu_est / sigma_est**2)**2 * delta_mu**2
            term3 = (-mu_est**2 / sigma_est**3)**2 * delta_sigma**2
            delta_amplitude = amp_est * np.sqrt(term1 + term2 + term3)
            # Append them to the array.
            dof = len(xdata) - len(p0)  # No. of observations  - no. of fitted params.
            #chisquare = reduced_chisquare(ydata, gaussian(xdata, *popt), y_sigma, dof)
            chisquare = reduced_chisquare(ydata, fgaussian(xdata, *popt), y_sigma, dof)
            popt = np.append(popt, np.asarray([delta_amplitude, delta_mu, delta_sigma]))
            popt = np.append(popt, chisquare)
            popt = np.append(popt, 0)
        except Exception as e:  # NOQA: BLE001
            logger.debug(f"Fit did not converge: {e}")
            popt = np.array([np.nan] * (len(p0)*2 + 2))
            popt[-1] = 2
    return popt


def main_gaussian_fitter(
    spectrum_observable,
    SG_dopp,
    *,
    velocity_range=None,
    npix=2,
    npix_left = None,
    npix_right = None,
    keep_only_local_maxima=None,
    velocity_array=None,
    width_min=0,
    cal_moment=False,
    return_spec=False,
    clip_at_vrange=False,
    spec_noise=None,
    npix_localmax_slac=1,
    ic_user=None,
    bc_user=None,
    boolean_spec_mask = True,
    method = "scipy",
    serial=False,
):
    r"""
    Accept an observable and ground truth spectrum to perform single Gaussian
    fitting on the observable spectrum.

    Parameters
    ----------
    spectrum_observable :`xarray.Dataset`
        The spectrum on which fitting is done.
        Spectrum should be of the form ``{...,SG_xpixel}``, where ``...`` can be "y" or "step".
    SG_dopp : `xarray.DataArray`
        The wavelength array corresponding to each slit on the CCD plane.
        Can be computed with the response function and :meth:`add_wavelength_information`.
    velocity_range : `float`, optional
        Velocity over which to find the peak intensity.
        Defaults to None.
    npix : `int`, optional
        Number of pixels for fitting.
        Defaults to 2.
    keep_only_local_maxima : str, optional.
        Implements logic to not fit bad spectra by asserting the existence of a local maximum.
            "gradient_sum": Use the old gradient-based method for aggressively removing bad spectra using
                            :meth:`fit_gaussian_remove_badspec`.
                            Vetting is done using :meth:local_max.
            "pix_slac": Check if the maximum is retained if we consider a slack of 1 pixel.
                        Vetting is done with :meth:`_vet_new_local_max`.
                        Fitting is done with :meth:`fit_gaussian`.
    velocity_array : `xarray.Dataset`, optional
        velocity from the inverted main line to guide the Gaussian fitting, by default None
    instrumental_width : 'float', optional
        Instrumental width to set the lower limit of line width of fitting.
    width_min : 'float', optional
        Minimum width from instrumental and thermal broadening
    cal_moment : 'bool', optional
        If True, calculate 0th, 1st, and 2nd moments instead of the single Gaussian fitting.
        Moments are defined as:
            M0 = $\Sigma_{SG\_wvl} [I] $
            M1 = $\Sigma_{SG\_wvl} [I*SG\_Dopp]/M0 $
            M2 = $\sqrt{\Sigma_{SG\_wvl} [I*(SG\_Dopp - M1)^2]/M0}$
        Defaults to False
    return_spec : 'bool', optional
        If True, return the spectrum on which fitting is done.
        Defaults to False
    clip_at_vrange : 'bool', optional
        If True, make spectrum outside of velocity_range as 0. Else consider points outside if peak finder selects external points.
        Defaults to False
    spec_noise : `xarray.Dataset`, optional
        Noise array associated with the spectrum, must be of the same shape as the spectrum.
        Defaults to None.
    npix_localmax_slac : `int`, optional
        Number of pixels given for slack in pix_slac key of keep_only_local_maxima.
        Defaults to 1.
    ic_user : `xarray.Dataset`, optional
        Initial condition for Gaussian fitting. Must be Npixel * Nslits * Nparams, and is used for normalized spectrum.
        The parameter dimension - along i0, v0, $\\sigma_0$ should be called "Params".
        Defaults to None.
    bc_user : `xarray.Dataset`, optional
        Boundary condition for Gaussian fitting. Must be Npixel * Nslits * Nparams * 2, and is used for normalized spectrum.
        The parameter dimension - along i0, v0, $\\sigma_0$ should be called "Params".
        The dimensions along the two limits should be called "Bounds", of size 2, corresponding to low and high limits respectively..
        Defaults to None.
    boolean_spec_mask: 'bool', optional
        Return a Boolean array where 1s denote the region with spectrum fit and 0s as the regions discarded.
    method : `string`, optional
        The fitting method to be used for making the Gaussian fits. The options are: `scipy`, `mpfit`, `caruana_np`, `caruana_pytorch`.
        scipy: Peforms calls scipy.curve_fit to fit a Gaussian to spectra in parallel.
        mpfit: Based on eispac mpfit, which is based on lmfit of IDL to fit Gaussian to spectra in parallel.
        caruna: Transform spectra to log() space, and fit a 2nd order polynomial to estimate the parameters.
                This algorithm is very fast, but can be sensitive to noise. Rather than a regular least square, we solve
                |(y-Rx)/sigma|, where the sigma = \frac{\partial log (spec)}{\partial spec} \noise_spec. In essence it is a
                weighted least square, and is more estimate, provides better estimates. This is based on Guo's modification from Guo, H.,
                IEEE Signal Processing Magazine vol. 28, no. 5, IEEE, pp. 134–137, 2011. doi:10.1109/MSP.2011.941846.
                We have two flavors for this algorithm:
                    caruana_np: Use np.polyfit() to estimate coeffcients, along with multiprocesing.
                    caruana_pytorch: Use torch.linalg.leastsq() to perform the fit in batch.

        Defaults to scipy.

    Returns
    -------
    `xarray.Dataset`
        Moments of spectrum_observable from Gaussian fit.
        Contains amplitude, velocity, linewidth, net_flux, rmse, and FLAG
        FLAG is:
            -1: For the moment calculation
            0: For a fitted profile
            1: If profile discarded due to lack of local maximum
            2: If the fitting algorithm did not converge.
    `xarray.Dataset`, optional
        The fitted spectrum. Returned if return_spec == True.
    """
    print()
    print('In main_gaussian_fitter')
    print('spectrum_observable')
    print(spectrum_observable)
    print('SG_dopp')
    print(SG_dopp)
    print('velocity_range')
    print(velocity_range)
    print('npix')
    print(npix)
    print('npix_left')
    print(npix_left)
    print('npix_right')
    print(npix_right)
    print('keep_only_local_maxima')
    print(keep_only_local_maxima)
    print('velocity_array')
    print(velocity_array)
    print('width_min')
    print(width_min)
    print('cal_moment')
    print(cal_moment)
    print('return_spec')
    print(return_spec)
    print('clip_at_vrange')
    print(clip_at_vrange)
    print('spec_noise')
    print(spec_noise)
    print('npix_localmax_slac')
    print(npix_localmax_slac)
    print('ic_user')
    print(ic_user)
    print('bc_user')
    print(bc_user)
    print('boolean_spec_mask')
    print(boolean_spec_mask)
    print('method')
    print(method)
    print('serial')
    print(serial)
    print()
    print('spectrum_observable.flux')
    print(spectrum_observable.flux)
    print('spectrum_observable.dims')
    print(spectrum_observable.dims)
    print('spectrum_observable.coords')
    print(spectrum_observable.coords)
    # Step 1: Make the spectrum into 35 slits using velocity_range
    dummyspec = spectrum_observable.copy(deep=True)
    flag_pix_slac = npix_localmax_slac if keep_only_local_maxima == "pix_slac" else 0
    # If we do not have a noise estimate, we just assume it is 1. This is similar to how scipy curve_fit does things.
    input_noise_spec = xr.ones_like(dummyspec) if spec_noise is None else spec_noise.copy(deep=True)
    dummy_spec_noise = input_noise_spec.copy(deep=True)

    #If there is no slit dimension, add an extra dimension.
    if "slit" not in SG_dopp.coords:
        SG_dopp = SG_dopp.expand_dims({"slit": range(1)}).copy(deep=True)
    nslit = len(SG_dopp.slit.values)

    if "slit" not in spectrum_observable.flux.dims:
        dummyspec["flux"] = spectrum_observable.flux.expand_dims({"slit": range(nslit)}).copy(deep=True)
        dummy_spec_noise["flux"] = input_noise_spec.flux.expand_dims({"slit": range(nslit)}).copy(deep=True)
        if velocity_array is None:
            # If velocity array is not given, we have no guiding!
            velocity_array = xr.zeros_like(dummyspec["flux"].isel(SG_xpixel=0))
        # For cases with velocity_array from SDC, we may get Nans. So we use this.
        velocity_array = velocity_array.fillna(0)
        if "x" in velocity_array.coords.dims and "slit" in list(spectrum_observable.coords):
            velocity_array = reshape_x2rs(velocity_array, nraster=len(dummy_spec_noise.step.values))
            # Assign the same step values (dummy_spec_noise.step.values):
            velocity_array = velocity_array.assign_coords({"step": dummy_spec_noise.step.values})
        # Step 2: Copy a subset of the spectrum into the new array for every slit
        if velocity_range is not None:
            ret = _split_spec_slitwise(
                [dummyspec,dummy_spec_noise], [spectrum_observable,input_noise_spec], velocity_range, velocity_array, SG_dopp, nslit, flag_pix_slac=flag_pix_slac
            )
            dummy_spec_noise = ret[1]
            dummyspec = ret[0]
    else:
        if velocity_array is None:
            velocity_array = xr.zeros_like(dummyspec["flux"].isel(SG_xpixel=0))
        velocity_array = velocity_array.fillna(0)
        if "x" in velocity_array.coords.dims:
            velocity_array = reshape_x2rs(velocity_array, nraster=len(dummy_spec_noise.step.values))
            # Assign the same step values (dummy_spec_noise.step.values):
            velocity_array = velocity_array.assign_coords({"step": dummy_spec_noise.step.values})
        if velocity_range is not None:
            ret = _split_spec_slitwise(
                [dummyspec,dummy_spec_noise], [spectrum_observable,input_noise_spec], velocity_range, velocity_array, SG_dopp, nslit, flag_pix_slac=flag_pix_slac
            )
            dummy_spec_noise = ret[1]
            dummyspec = ret[0]
    print()
    print('dummyspec')
    print(dummyspec)
    # We now know the max index location below
    max_indices = dummyspec["flux"].argmax(dim="SG_xpixel")  # This indexing always from 0.
    max_value = dummyspec["flux"].max(dim="SG_xpixel")
    dummdummyyspec = dummyspec.copy(deep=True)
    dummdummyyspec["SG_xpixel_array"] = dummyspec.SG_xpixel
    print()
    print('dummdummyyspec')
    print(dummdummyyspec)
    dummy_spec_noise["SG_xpixel_array"] = dummyspec.SG_xpixel
    """
    Let your SG_xpixel array be from [A,N].

    The pixel location A will be treated with index 0. The peak location
    finding subtract SG_xpixel and the max index value. However, max
    index starts from 0, and is the index corresponding to SG_xpixel
    location where we have the peak. Max indices has index starting from
    0 -> N-a, while SG_xpixel_array goes from A->N. We will need to
    subtract A from SG_xpixel_array to get the correct index.
    """
    SG_xpixel_start = dummdummyyspec["SG_xpixel_array"][0].values.item()
    # Generate N pixel mask
    diff = dummdummyyspec["SG_xpixel_array"] - SG_xpixel_start - max_indices
    if not clip_at_vrange and "slit" not in spectrum_observable.flux.dims:
        # If we do not clip outside of vrange, we need to take in the original spectrum.
        # This assignment fails if we have a GT with slit. For such a GT, we anyway consider
        # the full spectrum.
        for islit in range(nslit):
            # Original flux along all axis * mask (1 where SG_dopp<200, and 0 elsewhere).
            dummdummyyspec["flux"].loc[{"slit": islit}] = spectrum_observable.flux
            dummy_spec_noise["flux"].loc[{"slit": islit}] = input_noise_spec.flux

    max_noise = xr.ones_like(max_value) if spec_noise is None else max_value
    # Since we are rescaling the input spectrum, we will need to rescale the noise accordingly.
    # NOTE: If we do not have a noise estimate, we should NOT rescale the noise. It should remain an array of 1s.
    dummdummyyspec, dummy_spec_noise = _preprocess_spec_for_fitting(dummdummyyspec, dummy_spec_noise, diff, npix, max_value, max_noise, max_indices, SG_dopp, nslit, npix_right=npix_right, npix_left=npix_left)
    # IC and BC
    i0 = 1.0
    sigma_0 = width_min
    dv = np.median(np.gradient(SG_dopp, axis=-1))
    wvl = (-dv * npix, dv * npix)
    # # --- Make code general enough for any steps or y. We just need to transpose SG_xpixel to last coordinate
    coordinate_list = list(set(dummdummyyspec.flux.dims) - {"SG_xpixel"})
    complete_coords = [*coordinate_list, "SG_xpixel"]
    print()
    print('coordinate_list')
    print(coordinate_list)
    print()
    # --- for moment calculation
    if cal_moment:
        moments = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        vel_arr = SG_dopp.expand_dims(dim={"y": 1, "step": 1}).transpose(*complete_coords).values
        spec_arr = dummdummyyspec.flux.transpose(*complete_coords).values

        mom_peak = np.nanmax(spec_arr, axis=3)
        mom0 = np.nansum(spec_arr, axis=3)
        mom1 = np.nansum(spec_arr * vel_arr, axis=3) / mom0
        mom2 = np.sqrt(np.nansum(spec_arr * (vel_arr - mom1[..., np.newaxis]) ** 2, axis=3) / mom0)

        moments["amplitude"] = (coordinate_list, mom_peak)
        moments["velocity"] = (coordinate_list, mom1)
        moments["linewidth"] = (coordinate_list, mom2)
        moments["net_flux"] = (coordinate_list, mom0)
        moments['amplitude'] = moments.amplitude * max_value
        moments['net_flux'] = moments.net_flux * max_value
        moments["rmse"] = -1
        moments["flags"] = -1
        moments["error_amplitude"] = 0
        moments["error_velocity"] = 0
        moments["error_linewidth"] = 0
        moments["error_net_flux"] = 0
    else:
        # --- for Gaussian fitting
        N_returns = 8  # Returns 8 parameters - Amplitude, velocity, line width, Error_Amplitude, Error_velocity, Error_line width , rmse, FLAG.
        N_PIXELS = len(dummdummyyspec.SG_xpixel.values)
        xval = dummdummyyspec.new_wvl.transpose(*complete_coords).values.reshape(
            [-1, N_PIXELS],
        )
        yval = dummdummyyspec.flux.transpose(*complete_coords).values.reshape(
            [-1, N_PIXELS],
        )
        y_sigma = dummy_spec_noise.flux.transpose(*complete_coords).values.reshape(
            [-1, N_PIXELS],
        )
        if method=='caruana_pytorch':
            mask = np.isnan(yval)
            inds_not_nans = np.where(np.asarray([len(yval[i][~mask[i]]) for i in range(yval.shape[0])])!=0)[0]
            lst = [yval[i][~mask[i]] for i in inds_not_nans]
            y_val_ = torch.tensor(lst)
            y_val_[y_val_<0] = 0.0
            y_val_ = y_val_+1e-5
            x_val_ = torch.tensor([xval[i][~mask[i]] for i in inds_not_nans])
            y_sigma_ = torch.tensor([y_sigma[i][~mask[i]] for i in inds_not_nans])+1e-5
            y_log_transformed = torch.log(y_val_)
            weight_marix = y_val_/y_sigma_
            x_squared = x_val_ ** 2
            X = torch.stack([x_squared, x_val_, torch.ones_like(x_val_)], dim=2)
            fit = torch.linalg.lstsq(X*weight_marix[...,None], (y_log_transformed*weight_marix)[...,None])
            coeffs = fit.solution.squeeze(-1).T
            residual = torch.sum(((torch.einsum("pb,bip->bi", coeffs, X)-y_log_transformed)*weight_marix)**2, dim = -1)

            A = coeffs[0]
            B = coeffs[1]
            C = coeffs[2]

            # Calculate Gaussian parameters from fitted coefficients
            sigma_est = torch.sqrt(-1 / (2 * A))
            mu_est = B / (-2 * A)
            amp_est = torch.exp(C + (mu_est ** 2) / (2 * sigma_est ** 2))
            dof = len(x_val_[0]) - 3
            chisq = reduced_chisquare(y_val_.numpy(), gaussian_torch(x_val_, amp_est[...,None], mu_est[...,None],
                                                                     sigma_est[...,None]).numpy(), y_sigma_.numpy(),
                                                                     dof, axis = -1)
            flags = torch.zeros_like(amp_est)
            # Estimate uncertainties:
            # Compute Jacobian for the transformation from (A, B, C) to (sigma, mu, amplitude)
            # Partial derivatives for sigma, mu, and amplitude
            d_sigma_dA = sigma_est / (2 * A)
            d_mu_dA = B / (2 * A**2)
            d_mu_dB = -1 / (2 * A)
            d_amp_dA = amp_est * (mu_est**2 / sigma_est**3) * d_sigma_dA
            d_amp_dB = amp_est * (mu_est / sigma_est**2) * d_mu_dB
            d_amp_dC = amp_est

           # Build Jacobian matrix -  (batch_size, 3, 3) for [sigma, mu, amplitude] w.r.t. [A, B, C]
            J = torch.zeros((A.shape[0], 3, 3), dtype=A.dtype, device=A.device)
            J[:, 0, 0] = d_sigma_dA  # d(sigma)/d(A)
            J[:, 1, 0] = d_mu_dA     # d(mu)/d(A)
            J[:, 1, 1] = d_mu_dB     # d(mu)/d(B)
            J[:, 2, 0] = d_amp_dA    # d(amplitude)/d(A)
            J[:, 2, 1] = d_amp_dB    # d(amplitude)/d(B)
            J[:, 2, 2] = d_amp_dC    # d(amplitude)/d(C)

            # Covariance matrix for [A, B, C] (assume batch size covariance matrices)
            X_T = X.transpose(1, 2)  # Transpose to shape (batch_size, num_features, num_points)
            XTX = torch.bmm(X_T, X)  # Shape: (batch_size, 3, 3)
            # Compute residual variance for each spectrum
            residual_variance = residual / (X.size(1) - 3)  # Shape: (batch_size,)
            # Compute covariance matrix
            cov_matrix_ABC = torch.linalg.inv(XTX) * residual_variance.view(-1, 1, 1)  # Shape: (batch_size, 3, 3)

            # Propagate covariance to [sigma, mu, amplitude] for each spectrum
            cov_matrix_gaussian = torch.einsum('bij,bjk,bkl->bil', J, cov_matrix_ABC, J)

            # Extract uncertainties (standard deviations) for each parameter
            delta_sigma = torch.sqrt(cov_matrix_gaussian[:, 0, 0])
            delta_mu = torch.sqrt(cov_matrix_gaussian[:, 1, 1])
            delta_amplitude = torch.sqrt(cov_matrix_gaussian[:, 2, 2])

            # Save into array
            gfit_pool = np.ones([yval.shape[0],N_returns])*np.nan
            gfit_pool[:, -1] = 1
            gfit_pool[inds_not_nans,0] = amp_est.numpy()
            gfit_pool[inds_not_nans,1] = mu_est.numpy()
            gfit_pool[inds_not_nans,2] = sigma_est.numpy()
            gfit_pool[inds_not_nans,3] = delta_amplitude.numpy()
            gfit_pool[inds_not_nans,4] = delta_mu.numpy()
            gfit_pool[inds_not_nans,5] = delta_sigma.numpy()
            gfit_pool[inds_not_nans,6] = chisq
            gfit_pool[inds_not_nans,7] = flags.numpy()
        else:
            v0 = 0.0
            # --- Initial and boundary conditions.
            ic = [[i0, v0, sigma_0 + 10]] * len(yval)
            if ic_user is not None:
                ic_list = [*coordinate_list, "Params"]
                ic_vals = ic_user.transpose(*ic_list)
                ic = list(ic_vals.values.reshape([-1, len(ic_vals.Params)]))

            bc = [np.asarray([(0.9, 1.1), (wvl[0], wvl[1]), (sigma_0, 200)]).T] * len(yval)
            if bc_user is not None:
                bc_user = [*coordinate_list, "Bounds", "Params"]
                bc_vals = bc_user.transpose(*ic_list)
                bc = bc_vals.values.reshape([-1, 2, len(ic_vals.Params)])

            # This number will be used for local maximum search. Ignored otherwise.
            # If velocity_range is not given, its GT, so we put a large number here.
            bound_max_search = velocity_range if velocity_range is not None else 1000
            max_int_velocity_value = (
                dummdummyyspec["base_v0"]
                .transpose(*coordinate_list)
                .values.reshape(
                    [-1],
                )
            )
            neg_vel_bound_chng = -bound_max_search - max_int_velocity_value
            pos_vel_bound_chng = bound_max_search - max_int_velocity_value

            list_fit = list(zip(xval, yval, ic, bc, y_sigma, neg_vel_bound_chng, pos_vel_bound_chng, strict=True))
            N_returns = len(ic[0])*2+2  # Returns 8 parameters - Amplitude, velocity, line width, and the errors on them, rmse, FLAG.

            # ------------- Fit Gaussian
            if serial:
                # --- Serial fitting
                gfit_pool = np.zeros([*list(dummdummyyspec.new_wvl.isel(SG_xpixel=0).transpose(*coordinate_list).values.shape), N_returns])
                for i, args in tqdm(enumerate(list_fit)):
                    if method == 'mpfit':
                        gfit_pool[np.unravel_index(i, gfit_pool.shape[:-1])] = fit_gaussian_mpfit(args)
                    elif method == 'caruana_np':
                        gfit_pool[np.unravel_index(i, gfit_pool.shape[:-1])] = fit_gaussian_caruana(args)
                    else:
                        if keep_only_local_maxima != "gradient_sum":
                            gfit_pool[np.unravel_index(i, gfit_pool.shape[:-1])] = fit_gaussian(args)
                        else:
                            gfit_pool[np.unravel_index(i, gfit_pool.shape[:-1])] = fit_gaussian_remove_badspec(args)
            else:
            # --- Parallel fitting
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    if method=='mpfit':
                        gfit_pool = np.asarray(pool.map(fit_gaussian_mpfit, list_fit))
                    elif method=='caruana_np':
                        gfit_pool = np.asarray(pool.map(fit_gaussian_caruana, list_fit))
                    else:
                        if keep_only_local_maxima != "gradient_sum":
                                gfit_pool = np.asarray(pool.map(fit_gaussian, list_fit))
                        else:
                            gfit_pool = np.asarray(pool.map(fit_gaussian_remove_badspec, list_fit))
        # Reshape the results to match the expected output shape
        gfit_pool = gfit_pool.reshape(
            [*list(dummdummyyspec.new_wvl.isel(SG_xpixel=0).transpose(*coordinate_list).values.shape), N_returns],
        )

        # ------------ Save i0,mu and sigma arrays
        i0 = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        mu = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        sigma = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        chisq = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        flags = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        i0["flux"] = gfit_pool[..., 0] * max_value.transpose(*coordinate_list)
        mu["flux"] = gfit_pool[..., 1] + dummdummyyspec.base_v0.transpose(*coordinate_list)
        sigma["flux"] = gfit_pool[..., 2] * xr.ones_like(max_value.transpose(*coordinate_list))
        err_i0 = gfit_pool[..., 3] * max_value.transpose(*coordinate_list)
        err_mu = gfit_pool[..., 4] * xr.ones_like(max_value.transpose(*coordinate_list))
        err_sigma = gfit_pool[..., 5] * xr.ones_like(max_value.transpose(*coordinate_list))
        chisq["flux"] = gfit_pool[..., 6] * xr.ones_like(max_value.transpose(*coordinate_list))
        flags["flux"] = gfit_pool[..., 7] * xr.ones_like(max_value.transpose(*coordinate_list))
        # spectrum_profile = gaussian(SG_dopp, i0.flux, mu.flux, sigma.flux)
        # net_flux = spectrum_profile.sum(dim="SG_xpixel")
        # del spectrum_profile
        net_flux = i0.flux * sigma.flux * np.sqrt(2*np.pi) / dv #We do Amplitude (in DN) * sigma (in km/s) * sqrt(2*pi) / dv (in km/s)

        moments = xr.zeros_like(dummdummyyspec.isel(SG_xpixel=1).transpose(*coordinate_list))
        moments["amplitude"] = i0["flux"]
        moments["velocity"] = mu["flux"]
        moments["linewidth"] = sigma["flux"]
        moments["net_flux"] = net_flux
        moments["error_amplitude"] = err_i0
        moments["error_velocity"] = err_mu
        moments["error_linewidth"] = err_sigma
        moments["error_net_flux"] = np.sqrt(2 * np.pi) * np.sqrt((sigma["flux"] * err_i0) ** 2 + (i0["flux"] * err_sigma) ** 2)
        moments["RedChisq"] = chisq["flux"]
        moments["FLAGS"] = flags["flux"]
    moments.attrs = spectrum_observable.attrs

    # Adding units
    moments["amplitude"].attrs["units"], moments["net_flux"].attrs["units"] = (
        spectrum_observable.flux.attrs["units"],
        spectrum_observable.flux.attrs["units"],
    )
    if "exp_time" in spectrum_observable.flux.attrs:
        moments["amplitude"].attrs["exp_time"], moments["net_flux"].attrs["exp_time"] = (
            spectrum_observable.flux.attrs["exp_time"],
            spectrum_observable.flux.attrs["exp_time"],
        )
        moments["error_amplitude"].attrs["exp_time"], moments["error_net_flux"].attrs["exp_time"] = (
            spectrum_observable.flux.attrs["exp_time"],
            spectrum_observable.flux.attrs["exp_time"],
        )
    moments["velocity"].attrs["units"], moments["linewidth"].attrs["units"] = (
        SG_dopp.attrs["units"],
        SG_dopp.attrs["units"],
    )
    if "slit" in list(dummdummyyspec.flux.dims) and "step" in list(dummdummyyspec.flux.dims) and nslit>1:
        moments_xy = reshape_rs2x(moments,nraster=len(moments.step))
    else:
        moments_xy = moments
    # ----- Return moments and drop unnecessary variables
    list_vars = ["amplitude", "velocity", "linewidth", "net_flux", "error_amplitude", "error_velocity", "error_linewidth", "error_net_flux", "RedChisq", "FLAGS", "RMSE"]
    dropvars = [v for v in list(moments_xy.data_vars) if v not in list_vars]
    tmp = moments_xy.drop_vars(dropvars).copy(deep=True)
    if return_spec or boolean_spec_mask:
        dummdummyyspec.coords["SG_dopp"] = SG_dopp
        dummdummyyspec["flux"] = dummdummyyspec.flux * max_value.transpose(*coordinate_list)
        dummdummyyspec["flux_noise"] = dummy_spec_noise["flux"] * max_noise.transpose(*coordinate_list)
    if boolean_spec_mask:
        bool_mask = xr.where(dummdummyyspec.flux.isnull(), 0, 1)
        if "slit" in list(dummdummyyspec.flux.dims) and "step" in list(dummdummyyspec.flux.dims) and nslit>1:
            bool_mask = reshape_rs2x(bool_mask, nraster=len(dummdummyyspec.step))
            bool_mask["x"] = tmp.x.values
            if 'y' in dummdummyyspec.flux.dims:
                bool_mask["y"] = tmp.y.values
        else:
            if 'slit' in dummdummyyspec.flux.dims:
                bool_mask['slit'] =  tmp.slit.values
            if 'step' in dummdummyyspec.flux.dims:
                bool_mask['step'] = tmp.step.values

        tmp["GFIT_mask"] = bool_mask

    if return_spec:
        return tmp, dummdummyyspec

    return tmp


def calculate_moments_gaussian(spectrum, response, *, line=None, **kwargs):
    """
    Preprocess spectrum and perform a single Gaussian fit to a spectrum. Uses
    multiprocessing to fit the spectrum in parallel. Will use all available
    cores.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        The spectrum to calculate moments.
        The following cases hold for spectrum's shape:
        ``{...,SG_xpixel,slit,line}`` : Pass in the line parameter, which will be used to select from both the spectrum and response function.
        ``{...,SG_xpixel}`` : The spectra are either summed across the slit or taken only for one slit.
                              The response function should **not** contain the line.
    response: `xarray.Dataset`
        The response function corresponding to the spectrum.
    line : `str`, optional
        Either one of "Fe IX" or "Fe XV" or "Fe XIX", for fitting the spectrum.
        Need not be given if response function and spectra contain only one line.
        Defaults to None.
    **kwargs : dict, additional arguments
        These keywords are additional variables to be provided to :meth:`main_gaussian_fitter`

    Returns
    -------
    `xarray.Dataset`
        Moments of the observable from the Gaussian fitting.
        Contains the 'amplitude', 'velocity', 'linewidth', and 'net_flux'
    `xarray.Dataset`, optional
        The fitted spectrum. Returned if kwargs['return_spec'] == True.
    """
    # ----- Add Doppler information and compute logT from response function
    response = response.copy(deep=True)
    response, _ = add_wavelength_information(response.copy(deep=True))
    if "slit" in response.dims and "SG_xpixel" in response.dims:
        response = response.transpose("slit","SG_xpixel",...)

    if line is not None and "line" in spectrum.flux.dims:
        spectrum_observable = spectrum.copy(deep=True).sel(line=line)
        if "line" in response.SG_resp.dims:
            SG_dopp = response.sel(line=line).SG_dopp
            subresp = response.sel(line=line, slit=0)
        else:
            SG_dopp = response.SG_dopp
            subresp = response.sel(slit=0)
    else:
        spectrum_observable = spectrum.copy()
        SG_dopp = response.SG_dopp
        subresp = response
    kwargs["width_min"] = get_minimum_width(subresp, npix=kwargs["npix"])
    velocity_array = kwargs.pop("velocity_array", None)
    if isinstance(velocity_array, xr.DataArray) and (line is not None and "line" in velocity_array.coords.dims):
        velocity_array = velocity_array.sel(line=line).copy(deep=True).drop("line")
    if "units" not in spectrum.flux.attrs:
        unit = str(u.ph / u.s)
        logger.debug("Units for spectrum not found: assuming photons/s.")
    else:
        unit = spectrum.flux.attrs["units"]
    if not spectrum_observable.attrs:
        spectrum_observable.attrs = response.attrs
        spectrum_observable.attrs["step_size"] = spectrum.attrs['step_size']
        spectrum_observable.attrs["step_size units"] = spectrum.attrs['step_szie units']
    # ---- perform moment computation.
    tmp = main_gaussian_fitter(spectrum_observable, SG_dopp, velocity_array=velocity_array, **kwargs)
    if isinstance(tmp, tuple):
        add_history(tmp[0], locals(), calculate_moments_gaussian)
    return tmp



def gaussian_fitting_analysis(
    spectrum_observable,
    response,
    line_channel_key,
    *,
    kwargs_obs: dict | None = None,
):
    """
    Fit a single Gaussian to spectral lines. Employs
    :meth:`calculate_moments_gaussian` to perform the fitting. To calculate the
    0th, 1st, and 2nd moments under the identical condition, add
    {"cal_moment":True} in kwargs_obs.

    Parameters
    ----------
    spectrum_observable: `xarray.Dataset`
        The observable spectrum - a spectrum containing for example the main line + contaminants.
    response:  `xarray.Dataset`
        The response function cube.
        If the input spectra have already been selected with line, the response function must also be selected with the line.
    line_channel_key: list
        A list of lines or channels to perform the fitting.
        If you are giving a spectrum with single line, provide a string stating the line.
    kwargs_obs : dict, additional arguments
        These keywords are additional variables to be provided to :meth:`calculate_moments_gaussian`.

    Returns
    -------
    `xarray.Dataset`
        Moments of the spectrum from the Gaussian fitting. Contains the 'amplitude', 'velocity', 'linewidth', and 'net_flux'.
        If given a line list, select as ``moments_observable.sel(line=line)``
    `xarray.Dataset`, optional
        The fitted spectrum. Returned if kwargs_obs['return_spec'] == True.
    """
    if kwargs_obs is None:
        kwargs_obs = {}
    if "velocity_range" not in kwargs_obs:
        kwargs_obs["velocity_range"] = None
    if "npix" not in kwargs_obs:
        kwargs_obs["npix"] = 2
    if "velocity_array" not in kwargs_obs:
        kwargs_obs["velocity_array"] = None
    #spectrum_observable = spectrum_observable[["flux"]]
    # Check for units. Ensure units are in DN.'
    if "ph" in spectrum_observable.flux.attrs["units"]:
        tmp = ph2dn(spectrum_observable.copy(deep=True), response=response)
        if "channel" in spectrum_observable.flux.dims:
            spectrum_observable = tmp.swap_dims({"line": "channel"})
        else:
            spectrum_observable = tmp
    if "spec_noise" not in kwargs_obs:
        spec_noise = None
    else:
        spec_noise = kwargs_obs.pop("spec_noise")
        # Check for units. Ensure units are in DN.
        if "ph" in spec_noise.flux.attrs["units"]:
            tmp = ph2dn(spec_noise.copy(deep=True), response=response)
            if "channel" in spec_noise.flux.dims:
               spec_noise = tmp.swap_dims({"line": "channel"})
            else:
               spec_noise = tmp
    # --- Fit the observable line
    fitted_spec = []
    moments_observable = []
    if "channel" in spectrum_observable.flux.dims or "line" in spectrum_observable.flux.dims:
        if "channel" in spectrum_observable.flux.dims:
            spec_channels = response.channel.sel(line=line_channel_key)
            for l1,l2 in zip(spec_channels.data,line_channel_key):
                s = spec_noise.sel(channel = l1) if spec_noise is not None else None
                tmp = calculate_moments_gaussian(
                    spectrum_observable.sel(channel = l1),
                    response.sel(line=l2),
                    line=l2,
                    spec_noise=s,
                    **kwargs_obs,
                )
                if isinstance(tmp, tuple):
                    mom, spec = tmp[0].assign_coords({"line": l2}), tmp[1].assign_coords({"line": l2})
                    moments_observable.append(mom)
                    fitted_spec.append(spec)
                else:
                    mom = tmp
                    moments_observable.append(mom)
        else:
            spec_lines = line_channel_key
            for l1,l2 in zip(spec_lines,line_channel_key):
                s = spec_noise.sel(line = l1) if spec_noise is not None else None
                tmp = calculate_moments_gaussian(
                    spectrum_observable.sel(line = l1),
                    response.sel(line=l2),
                    line=l2,
                    spec_noise=s,
                    **kwargs_obs,
                )

                if isinstance(tmp, tuple):
                    mom, spec = tmp[0].assign_coords({"line": l2}), tmp[1].assign_coords({"line": l2})
                    moments_observable.append(mom)
                    fitted_spec.append(spec)
                else:
                    mom = tmp
                    moments_observable.append(mom)
        moments_observable = xr.concat(moments_observable, dim="line", coords="different", compat='equals')
        if len(line_channel_key) == 1:
            moments_observable = moments_observable.assign_coords(channel = ("line" ,[moments_observable.channel.data]) )
        if fitted_spec:
            fitted_spec = xr.concat(fitted_spec, dim="line", coords="different", compat='equals')
            if len(line_channel_key) == 1:
                fitted_spec = fitted_spec.assign_coords(channel = ("line" ,[fitted_spec.channel.data]) )
    else:
        if isinstance(line_channel_key, list):
            line_channel_key = line_channel_key[0]
        tmp = calculate_moments_gaussian(
            spectrum_observable,
            response,
            line=line_channel_key,
            spec_noise= spec_noise,
            **kwargs_obs,
        )
        if isinstance(tmp, tuple):
            moments_observable, fitted_spec = (
                tmp[0].assign_coords({"line": line_channel_key}),
                tmp[1].assign_coords({"line": line_channel_key}),
            )
            moments_observable = moments_observable.assign_coords(channel = ("line" ,[moments_observable.channel.data]) )
            fitted_spec = fitted_spec.assign_coords(channel = ("line" ,[fitted_spec.channel.data]) )
        else:
            moments_observable = tmp
            moments_observable = moments_observable.assign_coords(channel = ("line" ,[moments_observable.channel.data]) )

    add_history(moments_observable, locals(), gaussian_fitting_analysis)
    if fitted_spec:
        add_history(fitted_spec, locals(), gaussian_fitting_analysis)
        return moments_observable, fitted_spec
    return moments_observable


def masker(mask, moments, *, line_channel_list=None, thresholds_ph = None, thresholds_dn = None):
    """
    Generate the intensity mask based on threshold. Intended to be called by
    :meth:`get_outlier` function.

    Parameters
    ----------
    mask : `xarray.DataArray` or `str`
        Intensity mask for discarding points with low SNR in ground truth. This can take the following values:
            `None` : Apply no mask.
                     Return an array of 1s.
            `"auto"` : The intensity threshold is taken from the promised intensity counts with an exposure time of 1.8 seconds.
                    Mask is computed from the Contaminated or observable spectrum.
                    The line must be provided for this.
            `xarray.DataArray`: A mask array = 1 in locations to consider and `numpy.nan` in locations which must be discarded.
    moments : `xarray.Dataset`
        The moments dataset computed containing amplitude, velocity, linewidth and netflux.
    thresholds_ph: `dict`
        A dictionary consisting of intensity thresholds per channel. Thresholds are in units of ph.
    thresholds_dn: `dict`
        A dictionary consisting of intensity thresholds per channel. Thresholds are in units of DN.
    line_channel_list: list
        List of lines or channels under consideration.
        Defaults to None

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        Mask xarray.
        Will be `xarray.DataArray` for no mask, `xarray.Dataset` for "auto" and a `xarray.DataArray` or `xarray.Dataset` for a precomputed mask.
    """

    if mask is None:
        return xr.ones_like(moments)
    if mask == "auto":
        assert isinstance(line_channel_list, list)
        mask = moments.sel(line=line_channel_list).copy(deep=True)

        if thresholds_ph is None:
            thresholds_ph = NPHOT_THRESHOLD
        if thresholds_dn is None:
            thresholds_dn = NDN_THRESHOLD

        for line in line_channel_list:
            if moments.net_flux.attrs["units"] == str(u.ph):
                mask["amplitude"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask["velocity"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask["linewidth"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask["net_flux"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask.attrs[f"{line} Threshold"] = thresholds[str(moments.channel.sel(line=line).data)]
                mask.attrs[f"{line} Threshold units"] = str(u.ph)
            elif "DN" in moments.net_flux.attrs["units"]:
                mask["amplitude"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds_dn[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask["velocity"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds_dn[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask["linewidth"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds_dn[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask["net_flux"].loc[{"line": line}] = xr.where(
                    moments.net_flux.sel(line=line) > (thresholds_dn[str(moments.channel.sel(line=line).data)]),
                    1.0,
                    np.nan,
                )
                mask.attrs[f"{line} Threshold"] = thresholds_dn[str(moments.channel.sel(line=line).data)]
                mask.attrs[f"{line} Threshold units"] = "DN"

            if "exp_time" in moments.net_flux.attrs:
                mask.attrs["exposure time"] = moments.net_flux.attrs["exp_time"]
                mask.attrs["exposure time units"] = str(u.s)

        add_history(mask, locals(), masker)
        return mask
    if isinstance(mask, xr.DataArray | xr.Dataset):
        return mask
    msg = "Your mask is neither None, nor computed automatically, nor xarray Dataset or DataArray."
    raise ValueError(msg)


def get_outliers(
    moments_observable,
    moments_gt,
    line_channel_list,
    *,
    mask=None,
    mask_gt=None,
    mask_obs=None,
    factor_int=4.0,
):
    """
    Generate the outliers in velocity and line width for different lines over
    which the moments have been computed.

    Parameters
    ----------
    moments_observable:  `xarray.Dataset`
        The moments computed for the observable spectrum
    moments_gt: `xarray.Dataset`
        The moments computed for the ground truth spectrum
    line_channel_list: `list` or `str`
        The list of lines for which outliers need to be computed, or a string for one line.
    mask: `dict` or `str`
        Intensity mask for discarding points with low SNR in ground truth.
        This can take the following values:

        `None` : Apply no mask.
        "auto" : The intensity threshold is taken from the promised intensity counts with an exposure time of 1.8 seconds.
                Mask is computed from the Contaminated or observable spectrum.
        `xarray.DataArray` : A mask array  = 1 in locations to consider and np.nan in locations which must be discarded.

    Returns
    -------
    dict
        A dictionary containing the outlying % of points and the values themselves.
        Address the dictionary as: outlier[line or channel][velocity or linewidth][percent or values].
    """
    if mask is None:
        logger.debug("Mask is None, going ahead with no mask case.")
        if len(line_channel_list) > 1:
            mask = xr.concat(
                [
                    masker(mask, moments_observable.net_flux.sel(line=line_channel_list[0])).assign_coords({"line": line})
                    for line in line_channel_list
                ],
                dim="line",
                coords="different", compat='equals'
            )
        else:
            mask = xr.Dataset({line_channel_list[0]: masker(mask, moments_observable.net_flux)})
    else:
        assert isinstance(mask, xr.Dataset | xr.DataArray)
    if "line" in moments_observable.coords:
        if line_channel_list is str:
            line_channel_list = [line_channel_list]
        diffs = {
            key_lb: (moments_observable.sel(line=key_lb) - moments_gt.sel(line=key_lb)) * mask.sel(line=key_lb)
            for key_lb in line_channel_list
        }
    else:
        diffs = {line_channel_list: (moments_observable - moments_gt) * mask[line_channel_list]}
    param_list = ["velocity", "linewidth"]
    outliers = {}
    for k in line_channel_list:
        outliers[k] = {}
        diffs[k]["int_criteria"] = (
            moments_observable.sel(line=k)["net_flux"] / moments_gt.sel(line=k)["net_flux"]
        ) * mask.sel(line=k)["net_flux"]

        for p in param_list:
            low, high = CENTROID_UNCERT_PROMISED[k][p]
            mask_gt_k = mask_gt.sel(line=k) if mask_gt is not None else None
            mask_obs_k = mask_obs.sel(line=k) if mask_obs is not None else None
            percent, values = generate_outliers(
                diffs[k][p],
                low,
                high,
                mask=mask.sel(line=k),
                mask_gt=mask_gt_k,
                mask_obs=mask_obs_k,
                int_criteria=diffs[k]["int_criteria"],
                factor_int=factor_int,
            )
            outliers[k][p] = {"percent": percent, "values": values}
    return outliers