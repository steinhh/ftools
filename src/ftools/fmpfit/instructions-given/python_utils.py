"""
Functions whose scope is not limited to one part of the muse package.
"""

import datetime
import inspect
import string
from collections.abc import Callable
from pathlib import Path

import astropy.units as u
import numpy as np
import torch
import xarray as xr
from astropy.constants import c as speed_of_light
from scipy.interpolate import CubicSpline
from scipy.io import readsav

import muse
from muse import logger
from muse.data import OLD_FILTER_VALUES_FILE, VENDOR_DATA_FILE

__all__ = [
    "add_history",
    "calculate_moments",
    "deweight_func",
    "doppler_to_lambda",
    "generate_outliers",
    "lambda_to_doppler",
    "numpy_to_torch",
    "read_response",
    "sub_contaminants",
    "torch_to_numpy",
    "update_response",
    "weight_func",
]


def torch_to_numpy(torch_tensor, cuda_device: float | None = None):
    """
    Convert a `torch.Tensor` to a `numpy.ndarray`.

    Parameters
    ----------
    torch_tensor : `torch.Tensor`
        Torch tensor to convert.
    cuda_device : float, optional
        If provided, transfer the tensor from the GPU to the CPU before conversion.


    Returns
    -------
    `numpy.ndarray`
        The converted NumPy array.
    """
    if cuda_device is not None:
        with torch.cuda.device(f"cuda:{cuda_device}"):
            return torch_tensor.cpu().numpy()
    return torch_tensor.numpy()


def numpy_to_torch(numpy_array: np.ndarray, cuda_device: float | None = None):
    """
    Convert a `numpy.ndarray` to a `torch.Tensor`.

    Parameters
    ----------
    numpy_array : `numpy.ndarray`
        The array to convert.
    cuda_device : `float` or `None`, optional
        If provided, transfer the tensor to the specified CUDA device.

    Returns
    -------
    `torch.Tensor`
        The converted Torch tensor.
    """
    if cuda_device is not None:
        with torch.cuda.device(f"cuda:{cuda_device}"):
            return torch.tensor(numpy_array).cuda()
    return torch.tensor(numpy_array)


def lambda_to_doppler(wavelengths: np.ndarray, rest_wavelength: u.Quantity):
    """
    Convert wavelengths to Doppler shift in km/s.

    Parameters
    ----------
    wavelengths : `numpy.ndarray`
        Wavelength array.
    rest_wavelength : `astropy.units.Quantity`
        Rest wavelength.

    Returns
    -------
    `numpy.ndarray`
        Doppler shift in km/s.
    """
    return ((wavelengths / rest_wavelength - 1.0) * speed_of_light).to(u.km / u.s)


def doppler_to_lambda(doppler: np.ndarray, rest_wavelength: u.Quantity):
    """
    Convert Doppler shift (km/s) to wavelength.

    Parameters
    ----------
    doppler : `numpy.ndarray`
        Doppler shift array in km/s.
    rest_wavelength : `astropy.units.Quantity`
        Rest wavelength.

    Returns
    -------
    `numpy.ndarray`
        Wavelengths corresponding to the Doppler shifts.
    """
    return rest_wavelength * (1.0 + doppler / speed_of_light)


def lambda_to_doppler_xarray(resp):
    """
    Convert wavelengths to Doppler shift in km/s.

    Parameters
    ----------
    response : `xarray`
        include SG_wvl and line_wvl in coordinates.

    Returns
    -------
    `xarray`
        added Doppler shift coordinate in km/s.
    """
    response = resp.copy(deep=True)
    response.coords["dopp_vel"] = (response.coords["SG_wvl"] / response.coords["line_wvl"] - 1) * (
        speed_of_light.to(u.km / u.s)
    ).value
    response.coords["dopp_vel"].attrs["units"] = str(u.km / u.s)
    return response


def doppler_to_lambda_xarray(resp):
    """
    Convert Doppler shift in km/s to wavelengths in amgstrons.

    Parameters
    ----------
    response : `xarray`
        include dopp_vel and line_wvl in coordinates.

    Returns
    -------
    `xarray`
        added SG_wvl coordinate in amgstrons.
    """
    response = resp.copy(deep=True)
    response.coords["SG_wvl"] = response.coords["line_wvl"] * (
        1 + response.coords["dopp_vel"] / (speed_of_light.to(u.km / u.s)).value
    )
    response.coords["SG_wvl"].attrs["units"] = str(u.AA)
    return response


def conversion_ph2dn(wvl, gain=1):
    """
    Convert photons to DN or electrons.

    Parameters
    ----------
    wvl : `float`
        Wavelength in Angstroms.
    gain : `int`
        e->DN gain, by default 1.

    Returns
    -------
    conversion factor : `float`
    """
    return 12398.0 / wvl / 3.65 / gain


def raster_simulation_vdem(
    vdem: xr.Dataset,
    *,
    nsteps: int = 11,
    nslits: int | None = None,
    step_size: float = 0.40,
    pixel_size: float = 0.1666667,
) -> xr.Dataset:
    """
    Sample a VDEM at raster positions using interpolation.

    Parameters
    ----------
    vdem : `xarray.Dataset`
        VDEM as a function of x, y, logT, and velocity.
    nsteps : `int`, optional
        Number of raster steps, by default 11.
    nslits : `int` or None, optional
        Number of slits. Determined automatically if None.
    step_size : `float`, optional
        Raster step size in arcsec along x, by default 0.40.
    pixel_size : `float`, optional
        Pixel size in arcsec along y, by default 0.1666667.

    Returns
    -------
    `xarray.Dataset`
        The sampled VDEM.
    """
    if nslits is None:
        nslits = np.floor((vdem.x.max() - vdem.x.min()) / step_size / nsteps)
    nslits = min(nslits, 35)
    y = np.arange(vdem.y.min(), vdem.y.max() + 0.01 * pixel_size, pixel_size)
    exposures = []
    for r in range(nsteps):
        exposures.append(vdem.interp(x=(np.arange(nslits) * nsteps + r) * step_size, method="nearest", y=y))
        exposures[-1].x.data[:] = np.arange(nslits, dtype=int)
    raster = xr.concat(exposures, dim="step", coords="different", compat="equals")
    if "lgtaxis" in raster.dims:
        raster = raster.rename({"lgtaxis": "logT"})
    if "dopaxis" in raster.dims:
        raster = raster.rename({"dopaxis": "vdop"})
    raster = raster.rename({"x": "slit"})
    raster = raster.assign_coords(slit=np.arange(nslits, dtype=int))
    raster = raster.assign_coords(step=np.arange(nsteps))
    raster.attrs.update({"step_size": step_size, "step_size units": "arcsec"})

    add_history(raster, locals(), raster_simulation_vdem)

    return raster


def read_response(
    respfile: str,
    *,
    logT: np.ndarray = None,
    vdop: np.ndarray = None,
    slit: np.ndarray = None,
    logTmethod: np.ndarray = "nearest",
    vdopmethod: np.ndarray = "nearest",
    default_units: bool = True,
    gain: np.ndarray | None = None,
    **kwargs: dict | None,
) -> xr.Dataset:
    """
    Reads a response function into an `xarray.Dataset` interpolating if needed
    in vdop, and logT.

    Parameters
    ----------
    respfile : `str`
        Response function in Xarray readable format.
    logT : `array-like`, optional
        Temperature axis
    vdop : `array-like`, optional
        Velocity axis
    slit : `array-like`, optional
        Number of slits array of integers.
    logTmethod: `str`
        Interpolation method for logT, by default "nearest".
    vdopmethod: `str`
        Interpolation method for vdop, by default "nearest".
    kwargs : `dict`
        Keyword arguments to pass to `xarray.Dataset.assign_coords`.
        This is currently only used for the `LINE` attribute.
    default_units : `bool`
        If True, set default units for coordinates.
        vdop[km/s], SG_wvl[A], line_wvl[A], SG_resp[1e-27 ph cm^5/s]
    gain: `int`
        number of electron per DN, by default 10
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    `xarray.Dataset`
        The response function dataset.
    """
    r = xr.open_zarr(respfile) if respfile.split(".")[-1] == "zarr" or Path(respfile).is_dir() else xr.load_dataset(respfile)

    if logT is not None:
        loc_max = np.argmin(np.abs(logT.data - r.logT.max().data))
        if logT.max() > logT[loc_max]:
            logger.info("Response function is smaller than the VDEM in temp")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")

            logT = logT.where(logT <= logT[loc_max], drop=True)
        loc_min = np.argmin(np.abs(logT.data - r.logT.min().data))
        if logT.min() < logT[loc_min]:
            logger.info("Response function is smaller than the VDEM in temp")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")
            logT = logT.where(logT >= logT[loc_min], drop=True)
        if logTmethod == "nearest":
            r = r.sel(logT=logT, drop=True, method="nearest")
        else:
            r = r.interp(logT=logT, method=logTmethod)
            r["SG_resp"] = r.SG_resp.fillna(0)
            r["SG_resp"] = r.SG_resp.where(r.SG_resp > 0, 0)
        r["logT"] = logT
        r = r.assign_coords(logT=logT)

    if vdop is not None:
        loc_max = np.argmin(np.abs(vdop.data - r.vdop.max().data))
        if vdop.max() > vdop[loc_max]:
            logger.info("Response function is smaller than the VDEM in vdop")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")
            vdop = vdop.where(vdop <= vdop[loc_max], drop=True)
        loc_min = np.argmin(np.abs(vdop.data - r.vdop.min().data))
        if vdop.min() < vdop[loc_min]:
            logger.info("Response function is smaller than the VDEM in vdop")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")
            vdop = vdop.where(vdop >= vdop[loc_min], drop=True)
        if vdopmethod == "nearest":
            r = r.sel(vdop=vdop, drop=True, method="nearest")
        else:
            r = r.interp(vdop=vdop, method=vdopmethod)
            r["SG_resp"] = r.SG_resp.fillna(0)
            r["SG_resp"] = r.SG_resp.where(r.SG_resp > 0, 0)

        r["vdop"] = vdop
        r = r.assign_coords(vdop=vdop)

    if "channel" in r.dims:
        if "line" not in r.coords:
            r = r.assign_coords(line=("channel", r.channel.data))
    elif "line" not in r.dims:
        r = r.assign_coords(logT=logT, line=r.attrs.get("LINE", kwargs.get("LINE")))

    if slit is not None:
        r = r.sel(slit=np.arange(slit.max() + 1), drop=True, method="nearest")
    if "line_wvl" not in r:
        if r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL")) is None and "channel" in r.dims:
            r["line_wvl"] = r.channel
        else:
            r["line_wvl"] = r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL"))

    if "channel" not in r.dims and "line" not in r.dims:
        r = r.expand_dims("line")

    if gain is None:
        gain = np.array([10])
    r = r.assign_coords(gain=("channel", gain)) if "channel" in r.dims else r.assign_coords(gain=("line", gain))

    if default_units:
        if "vdop" in r.dims:
            r.vdop.attrs.update({"units": str(u.km / u.s)})
        r.SG_resp.attrs.update({"units": str(1e-27 * u.ph * u.cm**5 / u.s)})
        if "SG_wvl" in r:
            r.SG_wvl.attrs.update({"units": str(u.AA)})
        r.line_wvl.attrs.update({"units": str(u.AA)})

    return r


def update_response(response: xr.Dataset, vendor="AC"):
    """
    Update the MUSE instrument response with new vendor reflectance.

    This was designed to work only on the **COMBINED** response functions, not the individual ones.

    Parameters
    ----------
    response : `xarray.Dataset`
        The response of the MUSE instrument
    vendor : {"AM" | "AT" | "BT" | "AC"}, optional
        Vendor selection, by default "AC"
        Currently allowed vendors are: "AM", "AT", "BT", "AC"

    Returns
    -------
    `xarray.Dataset`
        Updated response dataset.

    Raises
    ------
    `ValueError`
        If the vendor is not allowed.
    """
    ALLOWED_VENDORS = ["AM", "AT", "BT", "AC"]
    if vendor not in ALLOWED_VENDORS:
        msg = f"Vendor {vendor} not allowed. Allowed vendors are {ALLOWED_VENDORS}"
        raise ValueError(msg)
    updated_response = response.copy(deep=True)
    vendor, model = vendor[0].upper(), vendor[1].upper()
    vendor_data = xr.open_dataset(VENDOR_DATA_FILE)
    old_filter_values = readsav(OLD_FILTER_VALUES_FILE)
    bands = {108: 0, 171: 1, 284: 2}
    for idx, band in enumerate(response.band.values):
        save_idx = bands[band]
        vendor_idx = f"{vendor}_{band}_{model}"
        old_wavelength, old_reflectivity = (
            old_filter_values["muse"][save_idx]["WAVE"],
            old_filter_values["muse"][save_idx]["MIRROR"],
        )
        nan_idx = np.isfinite(vendor_data[vendor_idx].values)
        new_wavelength, new_reflectivity = (
            vendor_data[vendor_idx].wavelength.values[nan_idx],
            vendor_data[vendor_idx].values[nan_idx],
        )
        old_model = CubicSpline(old_wavelength, old_reflectivity, extrapolate=False)
        new_model = CubicSpline(new_wavelength, new_reflectivity, extrapolate=False)
        for slit_idx in updated_response.slit.values:
            logger.info(f"Updating band {band} and slit {slit_idx}")
            for sgx_idx in updated_response.SG_xpixel.values:
                single_wavelength = updated_response.isel(line=idx).SG_wvl[slit_idx, sgx_idx]
                interpolated_old_reflectivity = old_model(single_wavelength)
                interpolated_new_reflectivity = new_model(single_wavelength)
                new_ratio = (interpolated_new_reflectivity / interpolated_old_reflectivity) ** 2
                updated_response["SG_resp"].loc[{"line": updated_response.line[idx]}][..., slit_idx, sgx_idx] *= new_ratio

    add_history(updated_response, locals(), update_response)

    return updated_response


def calculate_moments(
    spect: xr.Dataset,
    *,
    moment_dim: str = "vdop",
    vmax: float | None = None,
    label: float | None = "",
    vmask: xr.Dataset | None = None,
    vdop0_prox: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Compute the zeroth, first, and second moments from a spectrum.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        Input spectrum.
    moment_dim : `str`, optional
        Doppler shift axis name, by default "vdop".
    vmax : `float` or None, optional
        Maximum velocity for integration, by default None.
    label : `str`, optional
        Label for the output variables.
    vmask : `xarray.Dataset` or None, optional
        Mask for velocity.
    vdop0_prox : `xarray.Dataset`, optional
        Doppler shift proxy, e.g., from main line obtained by the
        SDC code, by default `None`.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing the moments.
    """
    from muse.transforms.transforms import reshape_x2rs

    spectrum = spect.copy(deep=True)
    # TODO: We should see about enforcing this?
    if "vdop" not in spectrum.variables:
        spectrum = lambda_to_doppler_xarray(spectrum)
    index_list = list(string.ascii_lowercase)
    einsum_str = ""
    vdop_dict = {}
    for ij, j in enumerate(spectrum.dopp_vel.dims):
        einsum_str += index_list[ij]
        vdop_dict[j] = index_list[ij]
    einsum_str += ","
    out_str = ""
    out_str_vmax = ""
    for ik, k in enumerate(spectrum.flux.dims):
        if k in spectrum.dopp_vel.dims:
            einsum_str += vdop_dict[k]
            out_str_vmax += vdop_dict[k]
        else:
            einsum_str += index_list[ij + ik + 1]
            out_str_vmax += index_list[ij + ik + 1]
        if k != moment_dim:
            out_str += vdop_dict[k] if k in spectrum.dopp_vel.dims else index_list[ij + ik + 1]
        logger.debug(f"{einsum_str}->{out_str}")

    if vmax is not None and vdop0_prox is not None:
        da = spectrum["dopp_vel"].copy(deep=True)
        mom1st_rs = reshape_x2rs(vdop0_prox["SDC main, 1st mom"].sel(line=["Fe XIX", "Fe IX", "Fe XV"]))
        x_vmax = xr.where(np.abs(da - mom1st_rs) > da.differentiate("SG_xpixel") * vmask, 0.0, 1.0)
        x_vmax = x_vmax.where(np.abs(da) < vmax, 0.0 * da)
        logger.debug(f"{einsum_str}->{out_str_vmax}", x_vmax.dims, spectrum.flux.dims)
        spec1 = spectrum.copy(deep=True)
        spec1["flux"] = x_vmax * spectrum.flux
    elif vmax is not None:
        da = spectrum["dopp_vel"].copy(deep=True)
        x_vmax = xr.where(np.abs(da) > vmax, 0.0 * da, 1.0 + 0.0 * da)
        logger.debug(f"{einsum_str}->{out_str_vmax}", x_vmax.dims, spectrum.flux.dims)
        spec1 = spectrum.copy(deep=True)
        spec1["flux"] = xr.DataArray(
            np.einsum(f"{einsum_str}->{out_str_vmax}", x_vmax, spectrum.flux),
            dims=spectrum.flux.dims,
        )
        if vmask is not None:
            spec1_new = spec1.copy(deep=True)
            spec1max = spec1.flux.argmax(dim=["SG_xpixel"])
            spec1_new["xpixels"] = spec1max["SG_xpixel"]
            spec1_new["xpixels"] = spec1_new.xpixels.expand_dims(
                {"SG_xpixel": np.size(spec1["SG_xpixel"].to_numpy())},
            ).copy(deep=True)
            spec1["flux"] = spec1.flux.where(np.abs(spec1_new.xpixels - spec1.coords["SG_xpixel"]) < vmask, 0)
    else:
        spec1 = spectrum.copy(deep=True)

    einsum_str = ""
    vdop_dict = {}
    for ij, j in enumerate(spec1.dopp_vel.dims):
        einsum_str += index_list[ij]
        vdop_dict[j] = index_list[ij]
    einsum_str += ","
    out_str = ""
    out_str_vmax = ""
    for ik, k in enumerate(spec1.flux.dims):
        if k in spec1.dopp_vel.dims:
            einsum_str += vdop_dict[k]
            out_str_vmax += vdop_dict[k]
        else:
            einsum_str += index_list[ij + ik + 1]
            out_str_vmax += index_list[ij + ik + 1]
        if k != moment_dim:
            out_str += vdop_dict[k] if k in spec1.dopp_vel.dims else index_list[ij + ik + 1]
        logger.debug(f"{einsum_str}->{out_str}")

    spec1["flux"] = spec1.flux.where(spec1.flux > 0, 0)
    zeroth = spec1.flux.sum(dim=moment_dim)
    first = np.einsum(f"{einsum_str}->{out_str}", spec1.dopp_vel.data, spec1.flux) / zeroth
    # Note that int(I (u-I1)^2 du)/I0 = (int(I u^2 du))/I0-I1^2
    second = np.sqrt(
        np.einsum(f"{einsum_str}->{out_str}", spec1.dopp_vel.data**2, spec1.flux) / zeroth - first**2,
    )
    out_dims = list(spectrum.flux.dims)
    out_dims = out_dims.remove(moment_dim)
    out_coords = list(spectrum.flux.coords)
    out_coords = out_coords.remove(moment_dim)
    moments = xr.Dataset()
    moments[f"{label}, 0th mom"] = xr.DataArray(zeroth, dims=out_dims, coords=out_coords)
    moments[f"{label}, 1st mom"] = xr.DataArray(first, dims=out_dims, coords=out_coords)
    moments[f"{label}, 2nd mom"] = xr.DataArray(second, dims=out_dims, coords=out_coords)
    moments.attrs = spectrum.attrs
    moments[f"{label}, 0th mom"].attrs = spec1.flux.attrs
    moments[f"{label}, 1st mom"].attrs["units"] = str(u.km / u.s)
    moments[f"{label}, 2nd mom"].attrs["units"] = str(u.km / u.s)
    add_history(moments, locals(), calculate_moments)
    return moments


def weight_func(
    spec: xr.Dataset,
    resp: xr.Dataset,
    *,
    fact: dict | None = None,
    consf: float = 1,
) -> xr.Dataset:
    """
    Normalizes or add weights to the spectrum and response for each spectral
    channel.

    Parameters
    ----------
    spec : `xarray.Dataset`
        Spectrum.
    resp : `xarray.Dataset`
        Response functions.
    fact : `dict`, optional
        Dictionary of the weight in each channel.
        By default, `None` which sets it to `{0: 1.0, 1: 0.4, 2: 0.5}`.
    consf : `float`, optional
        Constant apply equally to spectrum (not response functions), by default `1e27`.

    Returns
    -------
    spec : `xarray.Dataset`
        A new Dataset of the spectrum with the weighted values.
    resp : `xarray.Dataset`
        A new Dataset of the response with the weighted values.
    """
    if fact is None:
        fact = {0: 1.0, 1: 0.4, 2: 0.5}
    for iichannel, ichannel in enumerate(spec.coords["channel"]):
        spec.flux.loc[{"channel": ichannel}] = spec.flux.sel(channel=ichannel) / fact[iichannel] * consf
        resp.SG_resp.loc[{"channel": ichannel}] = resp.SG_resp.sel(channel=ichannel) / fact[iichannel]
    add_history(spec, locals(), weight_func)
    add_history(resp, locals(), weight_func)

    return spec, resp


def deweight_func(
    spec: xr.Dataset,
    resp: xr.Dataset,
    vdem: xr.Dataset,
    *,
    fact: dict | None = None,
    consf: float = 1,
):
    """
    Puts back the original values/units to the spectrum, response and
    disambiguated vdem.

    Parameters
    ----------
    spec : `xarray.Dataset`
        Spectrum.
    resp : `xarray.Dataset`
        Response functions.
    vdem : `xarray.Dataset`
        VDEM.
    fact : `dict`, optional
        Dictionary of the weight in each channel.
        By default, `None` which sets it to `{0: 1.0, 1: 0.4, 2: 0.5}`.
    consf : `float`, optional
        Constant apply equally to spectrum (not response functions), by default `1e27`.

    Returns
    -------
    spec : `xarray.Dataset`
        A new Dataset of the spectrum without the weighted values.
    resp : `xarray.Dataset`
        A new Dataset of the response without the weighted values.
    vdem : `xarray.Dataset`
        A new Dataset of the response without the weighted values.
    """
    if fact is None:
        fact = {0: 1.0, 1: 0.4, 2: 0.5}
    for iichannel, ichannel in enumerate(spec.coords["channel"]):
        spec.flux.loc[{"channel": ichannel}] = spec.flux.sel(channel=ichannel) * fact[iichannel] / consf
        resp.SG_resp.loc[{"channel": ichannel}] = resp.SG_resp.sel(channel=ichannel) * fact[iichannel]
    vdem.vdem.data = vdem.vdem.data / consf
    add_history(spec, locals(), deweight_func)
    add_history(resp, locals(), deweight_func)
    add_history(vdem, locals(), deweight_func)
    return spec, resp, vdem


def sub_positive(
    spec: xr.Dataset,
    subspec: xr.Dataset,
    *,
    factor: float = 1.0,
    channel: np.ndarray | list = None,
) -> xr.Dataset:
    """
    Subtract `subspec` from `spec`, setting any negative values to zero.

    Parameters
    ----------
    spec : `xarray.Dataset`
        Input spectrum.
    subspec : `xarray.Dataset`
        Spectrum. Typically only the contribution from the diffraction pattern.
    factor: `float`, optional
        Multiplicative factor for subtraction, by default 1.0.
    channel: `int array` or None, optional
        List of channels to process, by default (108, 171, 284).

    Returns
    -------
    `xarray.Dataset`
        Resulting spectrum after subtraction.
    """
    if channel is None:
        channel = (108, 171, 284)
    spec_out = spec.copy(deep=True)
    for ichannel in channel:
        spec_out.flux.loc[{"channel": ichannel}] = spec.flux.sel(channel=ichannel) - subspec.flux.sel(channel=ichannel) * factor
    spec_out["flux"].where(spec_out.flux < 0, 0)

    return spec_out


def sub_contaminants(spec: xr.Dataset, spec_dis: xr.Dataset, *, lines: list | None = None) -> xr.Dataset:
    """
    Subtract contaminants from a spectrum.

    Parameters
    ----------
    spec : `xarray.Dataset`
        MUSE spectrum.
    spec_dis : `xarray.Dataset`
        Disambiguated spectrum.
    lines: `list(str)`
        List of line names to subtract. If None, uses main lines from `spec_dis`.

    Returns
    -------
    `xarray.Dataset`
        Spectrum with contaminants subtracted.
    """
    if lines is None:
        lines = []
        for iline in spec_dis.line.data:
            if "missing" not in iline and "remaining" not in iline:
                lines.append(iline)
    idx_lines = [list(spec_dis.channel.coords["line"].values).index(i) for i in lines]
    channel = spec_dis.channel.isel(line=idx_lines).data
    spec_sub2 = xr.Dataset()
    count = 0
    for iimainlines, imainlines in enumerate(lines):
        for iilines, _ in enumerate(spec_dis.coords["line"]):
            if spec_dis.coords["channel"][iilines] == channel[iimainlines] and spec_dis.coords["line"][iilines] != imainlines:
                spec_dis_pos = xr.where(
                    spec_dis.flux.sel(line=spec_dis.coords["line"][iilines].data) > 0,
                    spec_dis.flux.sel(line=spec_dis.coords["line"][iilines].data),
                    0,
                )
                spec_dis_pos = spec_dis_pos.fillna(0)
                spec_sub = spec.flux.sel(channel=channel[iimainlines], drop=True) - spec_dis_pos.sum("slit")
                if count == 0:
                    spec_sub2["flux"] = spec_sub
                    if "line" in spec_sub2:
                        spec_sub2 = spec_sub2.drop_vars("line")
                    spec_sub3 = spec_sub2.expand_dims({"line": np.size(idx_lines)}).copy(deep=True)
                    spec_sub3.update({"line": ("line", np.array(lines))})
                    count = 1
                else:
                    spec_sub3["flux"].loc[{"line": imainlines}] = spec_sub
    spec_sub3["flux"] = spec_sub3.flux.where(spec_sub3.flux > 0, 0)
    if "vdop" in spec_dis:
        spec_sub3["vdop"] = spec_dis["vdop"]
    if "SG_wvl" in spec_sub3:
        spec_sub3 = spec_sub3.drop("SG_wvl")
    if "line_wvl" in spec_dis:
        spec_sub3["line_wvl"] = spec_dis["line_wvl"]
    spec_sub3.attrs.update(spec.attrs)
    if "units" in spec_dis.flux.attrs:
        spec_sub3.flux.attrs["units"] = spec_dis.flux.attrs["units"]
    spec_sub3 = spec_sub3.assign_coords(channel=("line", channel))
    add_history(spec_sub3, locals(), sub_contaminants)
    return spec_sub3


def add_history(ds: xr.Dataset, local_vars: dict, func: Callable) -> None:
    """
    Add a history entry to a dataset.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to update.
    local_vars : `dict`
        Local variables from the calling function.
    func : `Callable`
        Function being recorded in the history.
    """
    string_vals = []
    for arg, value in local_vars.items():
        if arg in inspect.signature(func).parameters:
            if isinstance(value, xr.Dataset | xr.DataArray | np.ndarray):
                if isinstance(value, np.ndarray) and value.shape in [(), (1,)]:
                    string_vals.append(f"{arg}={value.tolist()}")
                elif isinstance(value, xr.DataArray) and value.size == 1:
                    string_vals.append(f"{arg}={value.values.tolist()}")
                else:
                    string_vals.append(f"{arg}={arg}")
            else:
                string_vals.append(f"{arg}={value}")

    history_entry = f"{func.__name__}({', '.join(string_vals)})"
    if "HISTORY" in ds.attrs:
        if isinstance(ds.attrs["HISTORY"], list):
            ds.attrs["HISTORY"].append(history_entry)
        else:
            ds.attrs["HISTORY"] = [ds.attrs["HISTORY"], history_entry]
    else:
        ds.attrs["HISTORY"] = [history_entry]

    today = datetime.datetime.now(tz=datetime.timezone.utc)
    if "date created" in ds.attrs:
        ds.attrs["date modified"] = today.strftime("%d-%b-%Y")
    else:
        ds.attrs["date created"] = today.strftime("%d-%b-%Y")
    ds.attrs["version"] = muse.__version__


def generate_outliers(
    ds: xr.Dataset,
    low: float,
    high: float,
    mask=None,
    mask_gt=None,
    mask_obs=None,
    int_criteria=None,
    factor_int=4,
):
    """
    Compute statistics of points outside given limits in a data array.

    Intended to be used as:

    .. code-block:: python

       ds = main_line_only["velocity"] - main_line_with_contaminants["velocity"]
       low = -5  # km/s for Fe IX
       high = 5  # km/s for Fe IX.
       frac, values = generate_outliers(ds, low, high)

    If you want to mask the data with intensity threshold, perform the threshold and then compute ``ds``.

    Parameters
    ----------
    ds : `xarray.DataArray`
        Data array to analyze.
    low : `float`
        Lower limit.
    high : `float`
        Upper limit.
    mask : optional
        Mask for valid data.
    mask_gt : optional
        Ground truth mask.
    mask_obs : optional
        Observed mask.
    int_criteria : optional
        Intensity criteria.
    factor_int : `float`, optional
        Intensity factor, by default 4.

    Returns
    -------
    `float`
        Percentage of points outside limits.
    `numpy.ndarray`
        Points outside the limits.
    """
    outside_mask = xr.where(ds < low, 1.0, 0.0) + xr.where(ds > high, 1.0, 0.0)
    all_elements = xr.where(mask.net_flux.values > 0, 1.0, 0.0)
    if (mask_obs is not None) and (mask_gt is not None):
        mask_obs_temp = mask_obs.fillna(0.0)
        mask_gt_temp = mask_gt.fillna(0.0)
        outside_int_mask = xr.where((mask_obs_temp.net_flux.values != mask_gt_temp.net_flux.values), 1.0, 0.0)
        all_elements = xr.where((mask_obs_temp.net_flux.values != mask_gt_temp.net_flux.values), 1.0, all_elements)
        outside_mask_all = xr.where((outside_mask == 1), 1.0, outside_int_mask)
        outside_mask_all = xr.where((int_criteria > factor_int), 1.0, outside_mask_all)
    else:
        outside_mask_all = xr.where((int_criteria > factor_int), 1.0, outside_mask)
    outside_mask_all = xr.where((int_criteria < 1.0 / factor_int), 1.0, outside_mask_all)
    all_elements = all_elements.sum()
    if "x" in outside_mask_all.coords.dims and "y" in outside_mask_all.coords.dims:
        N_outside = outside_mask_all.sum(dim=["x", "y"]).values
    else:
        N_outside = outside_mask_all.sum().values
    frac = 100 * N_outside / all_elements
    values = outside_mask_all  # (ds * outside_mask_all).values
    return frac, values


def metric_auc(array, percent=68):
    """
    Compute the spread about the median for a given percentile.

    This function computes the spread about the median of the
    distribution. For a Gaussian distribution, mu-sigma ( = st) to
    mu+sigma ( = ed) cover 68% of points. Hence, we compute 0.5*(ed-st),
    which should be perfectly equal to the stddev of a Gaussian, and
    provides a measure of spread for a non-Gaussian distribution.

    Parameters
    ----------
    array : `numpy.ndarray`
        Input array.
    percent : `float`, optional
        Percentile to cover, by default 68.

    Returns
    -------
    `float`
        Proxy sigma (spread) value.
    """
    if np.isnan(array).all() is True:
        return np.nan
    a_copy = array[~np.isnan(array)]
    a_copy = np.sort(a_copy)
    N = len(a_copy)
    i_med = np.argmin(np.abs(a_copy - np.median(a_copy)))
    p_34 = int(N * percent / 200)
    st, ed = a_copy[i_med - p_34], a_copy[i_med + p_34]
    return 0.5 * (ed - st)


def find_sigma_iterative(array, percentile=68, tol=1e-3, astep=0.1):
    """
    Find symmetric limits about zero containing a given percentile of points.

    Parameters
    ----------
    array : `numpy.ndarray`
        Input 1D array.
    percentile : `float`, optional
        Percentile to cover, by default 68.
    tol : `float`, optional
        Tolerance for convergence, by default 1e-3.
    astep : `float`, optional
        Step size for limit adjustment, by default 0.1.

    Returns
    -------
    `float`
        Symmetric limit value.
    """
    array = array.ravel()
    array = np.sort(array)
    a_start = 0.1
    N = len(array)
    nobreak = True
    while nobreak:
        n_frac = np.searchsorted(array, a_start) - np.searchsorted(array, -a_start)
        n_frac = n_frac / N
        if n_frac * 100 < (percentile - tol):
            a_start = a_start + astep
        elif n_frac * 100 > (percentile + tol):
            a_start = a_start - astep
        else:
            nobreak = False
    return a_start


def find_sigma_auc(parameter_array: xr.DataArray, percentile=68, method="percentile", **kwargs):
    """
    Estimate symmetric bounds containing a given percentile of values.

    We need to estimate the value a such that:
    array >= -a : array <=a contains 68% of all values.

    The dumb way of doing this is shown in `find_sigma_iterative`,
    This function starts off with a, and iteratively increases it to converge to a symmetric bound.

    However, if we consider the absolute values of the array and find the 68th percentile,
    it is going to provide us a symmetric bound incorporating both the limits.

    These two methods were compared, and they give similar result.

    Parameters
    ----------
    parameter_array : `xr.DataArray`
        A data array containing the differences, or the set of values for which the limits are to be calculated.
    percentile : `float`
        Percentile to cover, by default 68.
    method: `str`
        Method to use for limit finding, by default "percentile".
        Can be either `percentile` or `iterative`.
        `percentile`: Takes the absolute value of the array, and finds the 68th percentile. Since the absolute value
        contains both positive and negative values, it is a symmetric measure.
        `iterative`: Here, we start from a guess limit L, and compute the fraction of points from [-L,L] for the
        array in consideration. We iteratively update L till a convergence is reached. This method
        is only to test out the correctness of method #1, and it not recommended to be used.
    **kwargs
        Additional arguments for iterative method.

    Returns
    -------
    `float`
        Symmetric bound value.
    """
    from copy import deepcopy

    array = deepcopy(parameter_array.data)
    array = array[~np.isnan(array)]

    # If net_flux, subtract 1 as the quantity is expected to be centered around 1
    if "net_flux" in parameter_array.name:
        array = array - 1.0

    if np.size(array) < 2:
        return None
    if method == "iterative":
        kwargs["percentile"] = percentile
        limit = find_sigma_iterative(array, **kwargs)
    elif method == "percentile":
        array = np.abs(array)
        limit = np.percentile(array, percentile)
    else:
        msg = "Invalid method"
        raise ValueError(msg)
    return limit


def reduced_chisquare(obs, pred, stddev, dof, axis=None):
    """
    Calculate the reduced chi-square value for a given set of observations,
    predictions, standard deviations, and degrees of freedom.

    Parameters
    ----------
    obs : `array`
        Observed values.
    pred : `array`
        Predicted values.
    stddev: `array`, float
        The standard deviations of the observations.
    dof : `int`
        The degrees of freedom.
    axis: `int`, optional
        Axis along which to sum, by default None.

    Returns
    -------
    `float`
        Reduced chi-square value.
    """
    if axis is None:
        return np.sum(np.square(obs - pred) / (dof * stddev**2))
    return np.sum(np.square(obs - pred) / (dof * stddev**2), axis=axis)


def confusion_masked(
    moments: xr.Dataset,
    spec_toplines: xr.Dataset,
    channel: np.ndarray | list = None,
    lines: list | None = None,
):
    """
    Calculate contaminant ratios and dominant lines per pixel using a mask.

    Parameters
    ----------
    moments : `xarray.Dataset`
        Dataset containing moments and mask.
    spec_toplines : `xarray.Dataset`
        Topline spectra.
    channel : `array` or None, optional
        Channels to process, by default all in `moments`.
    lines : `list` or None, optional
        Lines to process, by default all in `moments`.

    Returns
    -------
    `xarray.Dataset`
        Masked spectra and ratios.
    `xarray.Dataset`
        Identity maps for dominant slit and line per pixel.
    """
    from muse.transforms.transforms import reshape_rs2x, reshape_x2rs

    if channel is None:
        if lines is None:
            lines = moments.line.data
            channel = moments.channel.data
        else:
            channel = moments.sel(line=lines).channel.data

    # load mask
    moments = reshape_x2rs(moments)
    GFIT_mask = moments.GFIT_mask.copy(deep=True)
    GFIT_mask = GFIT_mask.rename({"slit": "slit_gfat"})
    GFIT_mask = GFIT_mask.swap_dims({"line": "channel"})

    spec_test = xr.Dataset()
    spec_test["flux"] = xr.DataArray(
        np.ones(
            (
                spec_toplines.y.size,
                spec_toplines.line.size,
                spec_toplines.step.size,
                spec_toplines.slit.size,
                GFIT_mask.slit_gfat.size,
            )
        ),
        dims=["y", "line", "step", "slit", "slit_gfat"],
        coords={
            "y": spec_toplines.y.data,
            "line": spec_toplines.line.data,
            "step": spec_toplines.step.data,
            "slit": spec_toplines.slit.data,
            "slit_gfat": GFIT_mask.slit_gfat.data,
        },
    )
    # create masked spectra using GFIT mask-> SUM over wavelength(spec(y,step,wavelength)*mask(y,step,wavelength))

    for istep in spec_toplines.step.data:
        for _iiy, it in enumerate(spec_toplines.y.data):
            for ii_line, _iline in enumerate(spec_toplines.line.data):
                GFIT_mask_channel = GFIT_mask.sel(y=it, step=istep, channel=spec_toplines.isel(line=ii_line).channel)
                if GFIT_mask_channel.channel.size > 1:
                    GFIT_mask_channel = GFIT_mask_channel.isel(channel=0)
                spec_test["flux"].loc[{"step": istep, "y": it}][ii_line] = (
                    spec_toplines.flux.sel(y=it, step=istep).isel(line=ii_line) * GFIT_mask_channel
                ).sum(dim="SG_xpixel")
    spec_test = spec_test.assign_coords(channel=spec_toplines.channel)

    spec_test_to_x = xr.Dataset()
    # reshape from slit, step into x and y
    spec_test_to_x["flux"] = reshape_rs2x((spec_test.flux.sum(dim="slit")).rename({"slit_gfat": "slit"}))

    # array to identify contaminant slit location in contaminant tool
    spec_test_to_x["identity_slitmax_x"] = reshape_rs2x(
        (spec_test.flux.argmax(dim="slit")).rename({"slit_gfat": "slit"})
    )  # convert to x-y space

    identity_maps = xr.Dataset()
    identity_maps["slit_map"] = xr.DataArray(
        np.ones((spec_test_to_x.y.size, spec_test_to_x.x.size, np.size(channel))),
        dims=["y", "x", "channel"],
        coords={
            "y": spec_test_to_x.y.data,
            "x": spec_test_to_x.x.data,
            "channel": channel,
        },
    )
    identity_maps["line_map"] = xr.DataArray(
        np.ones((spec_test_to_x.y.size, spec_test_to_x.x.size, np.size(channel))),
        dims=["y", "x", "channel"],
        coords={
            "y": spec_test_to_x.y.data,
            "x": spec_test_to_x.x.data,
            "channel": channel,
        },
    )
    spec_test_to_x_sw = spec_test_to_x.swap_dims({"line": "channel"})
    for ii_channel, ichannel in enumerate(channel):
        ratio_channel = spec_test_to_x_sw.sel(channel=ichannel)
        ratio_channel = ratio_channel.swap_dims({"channel": "line"})

        # This if is because we have two times the same line ... With the new upcoming Jake's version this should be remove
        if spec_test_to_x.flux.sel(line=lines[ii_channel]).line.size > 1:
            ratio = (ratio_channel.flux) / spec_test_to_x.flux.sel(line=lines[ii_channel]).isel(line=0)  # "Fe IX 171.073")
        else:
            ratio = (ratio_channel.flux) / spec_test_to_x.flux.sel(line=lines[ii_channel])
        ratio = ratio.assign_coords(channel=("line", [ichannel] * ratio.coords["line"].size))

        if "line_wvl" in ratio.coords:
            ratio = ratio.drop_vars("line_wvl")

        if ii_channel == 0:
            ratio_all = ratio.copy(deep=True)

        if ii_channel != 0:
            ratio_all = xr.concat([ratio_all, ratio], dim="line", coords="different", compat="equals")

    spec_test_to_x["ratio"] = ratio_all

    spec_test_to_x_sw = spec_test_to_x.swap_dims({"line": "channel"})

    # create slit map and line map, where do each dominant contaminant in each pixel come from ?
    for ichannel in channel:
        spec_test_to_x_channel = spec_test_to_x_sw.sel(channel=ichannel)
        spec_test_to_x_channel = spec_test_to_x_channel.fillna(0.0)
        spec_test_to_x_channel = spec_test_to_x_channel.swap_dims({"channel": "line"})
        for ix in spec_test_to_x_channel.x.data:
            for it in spec_test_to_x_channel.y.data:
                line_index = (
                    spec_test_to_x_channel.ratio.sel(x=ix, method="nearest").sel(y=it, method="nearest").argmax(dim="line")
                )
                identity_maps["slit_map"].loc[{"x": ix, "y": it, "channel": ichannel}] = spec_test_to_x.identity_slitmax_x.sel(
                    x=ix, y=it, method="nearest"
                ).isel(line=line_index)
                identity_maps["line_map"].loc[{"x": ix, "y": it, "channel": ichannel}] = line_index

    return spec_test_to_x, identity_maps
