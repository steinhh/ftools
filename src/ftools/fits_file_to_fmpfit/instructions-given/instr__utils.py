import os
import time
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import named_arrays as na
import numexpr as ne
import numpy as np
import xarray as xr
from astropy.constants import c as speed_of_light

from muse import logger
from muse.synthesis.synthesis import conversion_ph2dn, conversion_units
from muse.utils.utils import add_history

__all__ = ["filter_mesh_parameters", "sum_lines_per_channel", "sum_lines_slits_per_channel"]


def filter_mesh_parameters(*, preflightcore: bool = False) -> dict:
    """
    Returns the mesh parameters for the filter.

    Parameters
    ----------
    preflightcore : `bool`, optional
        Use the preflight core, by default `False`.

    Returns
    -------
    `dict`
        The mesh parameters.

    Notes
    -----
    ``focal_length`` : Effective focal length of SG.
    ``fp_ccd_dist`` : Distance from focal plane to ccd.
    ``CDELT`` : Sample MUSE mesh.
    """
    return {
        108 * u.angstrom: {
            "error_angle_arm": [0.02, 0.02, 0.02, 0.02] * u.deg,
            "angle_arm": [50.0, 40, -40.0, -50] * u.deg,
            "spacing_e": 16.26 * u.pixel,
            "mesh_pitch": 363.0 * u.um,
            "mesh_width": 34.0 * u.um,
            "spacing_fp": 0.377 * u.pixel,
            "width": (0.962 if preflightcore else 4.5) * (0.6 * u.arcsec) ** (-2),
            "focal_length": 16.09 * u.m,
            "fp_ccd_dist": 34.44 * u.cm,
            "CDELT": [0.4, 0.167] * u.arcsec,
            "wavelength": 108 * u.angstrom,
        },
        171 * u.angstrom: {
            "angle_arm": [50.0, -40.0] * u.deg,
            "error_angle_arm": [0.02, 0.02] * u.deg,
            "spacing_e": 16.26 * u.pixel,
            "mesh_pitch": 363.0 * u.um,
            "mesh_width": 34.0 * u.um,
            "spacing_fp": 0.377 * u.pixel,
            "width": (0.962 if preflightcore else 4.5) * (0.6 * u.arcsec) ** (-2),
            "focal_length": 16.09 * u.m,
            "fp_ccd_dist": 34.44 * u.cm,
            "CDELT": [0.4, 0.167] * u.arcsec,
            "wavelength": 171 * u.angstrom,
        },
        193 * u.angstrom: {
            "angle_arm": [49.82, 39.57, -40.12, -50.37] * u.deg,
            "error_angle_arm": [0.02, 0.02, 0.03, 0.04] * u.deg,
            "spacing_e": 18.39 * u.pixel,
            "mesh_pitch": 363.0 * u.um,
            "mesh_width": 34.0 * u.um,
            "spacing_fp": 0.425 * u.pixel,
            "width": (1.512 if preflightcore else 4.5) * u.pixel,
            "CDELT": [0.143, 0.143] * u.arcsec,
        },
        284 * u.angstrom: {
            "angle_arm": [50.0, -40.0] * u.deg,
            "error_angle_arm": [0.02, 0.02] * u.deg,
            "spacing_e": 16.26 * u.pixel,
            "mesh_pitch": 363.0 * u.um,
            "mesh_width": 34.0 * u.um,
            "spacing_fp": 0.377 * u.pixel,
            "width": (0.962 if preflightcore else 4.5) * (0.6 * u.arcsec) ** (-2),
            "focal_length": 16.09 * u.m,
            "fp_ccd_dist": 34.44 * u.cm,
            "CDELT": [0.4, 0.167] * u.arcsec,
            "wavelength": 284 * u.angstrom,
        },
    }


def chianti_gofnt_linelist(
    temperature: xr.DataArray,
    density: xr.DataArray = None,
    pressure: xr.DataArray = None,
    abundance: str | None = None,
    wavelength_range=None,
    minimum_abundance=1e-5,
    elementList=None,
    ionList=None,
) -> xr.Dataset:
    """
    Generate a line list with GOFNT using ChiantiPy.

    Parameters
    ----------
    temperature : `xarray.DataArray`
        Temperature array.
    density : `xarray.DataArray`, optional
        Density array.
    pressure : `xarray.DataArray`, optional
        Pressure array.
    abundance : `str` or `None`, optional
        Abundance name.
    wavelength_range : `tuple` or `None`, optional
        Wavelength range.
    minimum_abundance : `float` or `None`, optional
        Minimum abundance, by default 1e-5.
    elementList : `list` or `None`, optional
        List of elements.
    ionList : `list` or `None`, optional
        List of ions.

    Returns
    -------
    `xarray.Dataset`
        Line list with GOFNT and related information.
    """
    try:
        import ChiantiPy.core as ch
    except ImportError:
        msg = "ChiantiPy required for this function"
        raise ImportError(msg) from None

    if ionList is not None and minimum_abundance is not None:
        logger.warning("Having minimum_abundance, the ionList will be ignored")

    if np.any(density):
        if np.any(pressure):
            msg = "Cannot specify both density and pressure"
            logger.error(msg)

        if isinstance(density, xr.DataArray) and isinstance(temperature, xr.DataArray):
            temp_xr, dens_xr = xr.broadcast(temperature, density)

            temp = temp_xr.data.reshape(-1)
            dens = dens_xr.data.reshape(-1)
        extra_coord_name = density.dims[0]
        extra_coord = np.log10(density.data)
    if np.any(pressure):
        if np.any(density):
            msg = "Cannot specify both density and pressure"
            logger.error(msg)

        if isinstance(pressure, xr.DataArray) and isinstance(temperature, xr.DataArray):
            density = pressure / temperature
            temp_xr = temperature.broadcast_like(density)

            dens = density.data.reshape(-1)
            temp = temp_xr.data.reshape(-1)
        extra_coord_name = pressure.dims[0]
        extra_coord = pressure.data

    s = ch.bunch(
        temp,
        dens,
        wavelength_range,
        em=1.0,
        abundance=abundance,
        allLines=True,
        keepIons=True,
        minAbund=minimum_abundance,
        ionList=ionList,
        elementList=elementList,
    )

    observed = xr.DataArray(
        s.Intensity["obs"] == "Y",
        dims="trans_index",
    )

    intensity = s.Intensity["intensity"]
    intensity = intensity.reshape((*temp_xr.data.shape, 1, -1))
    line_list_gofnt = xr.DataArray(
        data=intensity,
        dims=(*temp_xr.dims, "abundance", "trans_index"),
        coords={"logT": np.log10(temperature), extra_coord_name: extra_coord, "abundance": np.array([abundance])},
    )

    line_list_ions = xr.DataArray(
        data=s.Intensity["ionS"],
        dims="trans_index",
    )

    line_list_wvs = xr.DataArray(
        data=s.Intensity["wvl"],
        dims="trans_index",
    )
    pretty1 = xr.DataArray(
        data=s.Intensity["pretty1"],
        dims="trans_index",
    )
    pretty2 = xr.DataArray(
        data=s.Intensity["pretty2"],
        dims="trans_index",
    )
    lvl1 = xr.DataArray(
        data=s.Intensity["lvl1"],
        dims="trans_index",
    )
    lvl2 = xr.DataArray(
        data=s.Intensity["lvl2"],
        dims="trans_index",
    )
    spectroscopic = xr.DataArray(
        data=np.array([s.IonInstances[ion].Spectroscopic for ion in s.Intensity["ionS"]]),
        dims="trans_index",
    )
    z = xr.DataArray(
        data=np.array([s.IonInstances[ion].Z for ion in s.Intensity["ionS"]]),
        dims="trans_index",
    )
    logT_max = np.log10(temperature[{"logT": line_list_gofnt.argmax(dim="logT")}])
    line_list = xr.Dataset(
        {
            "IonStr": line_list_ions,
            "wvl": line_list_wvs,
            "gofnt": line_list_gofnt,
            "pretty1": pretty1,
            "pretty2": pretty2,
            "lvl1": lvl1,
            "lvl2": lvl2,
            "logT_max": logT_max,
            "spectroscopic": spectroscopic,
            "z": z,
            "observed": observed,
        }
    )

    full_line_name = line_list.spectroscopic.astype(object) + " " + line_list.wvl.astype(str).astype(object)
    line_list["full_name"] = full_line_name

    # cut down linelist to match wavelength range
    in_range = (line_list.wvl > wavelength_range[0]) * (line_list.wvl < wavelength_range[1])

    return line_list.isel(trans_index=in_range)


def bin_repeat_lines(
    line_list: xr.Dataset,
    full_line_name=None,
) -> xr.Dataset:
    """
    Bin and sum repeated lines in a line list.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        Line list dataset.
    full_line_name : `list` or `None`, optional
        List of full line names to check for repeats.

    Returns
    -------
    `xarray.Dataset`
        Line list with repeated lines binned and summed.
    """
    new_line_list = []
    processed_repeats = []
    for i, line in enumerate(line_list.full_name.data):
        if line in full_line_name:
            if line not in processed_repeats:
                repeats = line_list.full_name.data == line
                if repeats.sum() > 1:
                    processed_repeats.append(line)
                    repeat_line = line_list.isel(trans_index=repeats)
                    summed_gofnt = repeat_line.gofnt.sum("logT").data
                    j = summed_gofnt.argmax()
                    new_line = repeat_line.isel(trans_index=[j])
                    new_line["gofnt"].data = repeat_line.gofnt.sum("trans_index", keepdims=True).data
                    new_line_list.append(new_line)
            else:
                pass
        else:
            new_line_list.append(line_list.isel(trans_index=[i]))

    return xr.concat(new_line_list, dim="trans_index", coords="different", compat="equals")


def create_resp_func(
    line_list: xr.Dataset,
    instr_width: float = 0,
    normalization: float = 1e-27,
    method: str = "linear",
    wvlo: float | None = None,
    vdop: u.Quantity | np.ndarray | list = None,
    wvlr: np.ndarray | list = None,
    dma: float = 4.9,
    num_wv_bins: int | None = None,
    effective_area: xr.Dataset | None = None,
    num_lines_keep: int = 2,
    verbose=False,
    band="",
) -> xr.Dataset:
    """
    Computes a response function as a function of velocity and temperature.

    Parameters
    ----------
    line_list : `xr.Dataset`
        Line list, can be created with `create_resp_line_list`.
    instr_width : `float`, optional
        Instrumental width sigma in Angstroms, by default 0.
    normalization : `float`, optional
        Normalization in the response function, by default 1e-27.
    method : `str`, optional
        Interpolation method, default 'linear'.
    vdop : `array-like`, optional
        Doppler axis array in km/s.
    wvlr : `array-like`, optional
        Wavelength range to select the spectral line.
    dma : `float`, optional
        Wavelength pixel size in mÅ, by default 4.9.
    num_wv_bins : `int` or `None`, optional
        Number of wavelength bins.
    effective_area : `xr.Dataset` or `None`, optional
        Effective area, can be generated with `create_eff_area_xarray`.
    num_lines_keep : `int`, optional
        Number of lines to keep, by default 2.
    verbose : `bool`, optional
        Verbose output, by default False.
    band : `str`, optional
        Band label.

    Returns
    -------
    `xarray.Dataset`
        Response function with temperature, velocity and wavelength axis.
    """
    try:
        import periodictable as pt
    except ImportError:
        msg = "periodictable required for this function"
        raise ImportError(msg) from None

    if num_lines_keep == 0:
        line = None

    CC = const.c
    CC_kms = CC.to(u.km / u.s).value
    CC_ms = CC.to(u.m / u.s).value  # Light speed (m/s)
    KB = const.k_B.to(u.J / u.K).value  # Boltzman constant (J/K = kg m^2/K/s^2)
    MP = const.m_p.to(u.kg).value  # Proton mass (kg)

    if wvlr is None:
        wvlr = [line_list.wvl.min().data, line_list.wvl.max().data]

    if num_wv_bins:
        dnus = np.linspace(wvlr[0], wvlr[1], num=num_wv_bins)
        dims = ("wavelength", *wvlr[0].dims) if isinstance(wvlr[0], xr.DataArray) else "wavelength"
        dnus = xr.DataArray(dnus, dims=dims)
    else:
        dnus = np.arange(wvlr[0], wvlr[1] + dma / 1e3, dma / 1e3)  # [A]
        dnus = xr.DataArray(
            dnus,
            dims="wavelength",
        )

    if vdop is not None:
        if isinstance(vdop, u.Quantity):
            vdop = vdop.to(u.km / u.s).value
        else:
            pass

        vdop = xr.DataArray(
            data=vdop,
            dims="vdop",
            coords={"vdop": vdop},
        )

        wvlo = line_list["wvl"] * (1 + vdop / CC_kms)

    else:
        wvlo = line_list["wvl"]

    # logic for single line response function
    if "trans_index" not in line_list.dims:
        line_list = line_list.expand_dims("trans_index")

    resp = []
    num_lines = line_list.sizes["trans_index"]
    summed_lines_included = False
    for i in range(num_lines):
        start_time = time.perf_counter()

        wv_0 = wvlo.isel(trans_index=i)

        Z = line_list.z.isel(trans_index=i)
        mass_xr = Z.copy()
        mass = np.array([pt.elements[z].mass for z in Z.data.reshape(-1)])
        mass_xr.data = mass.reshape(mass_xr.data.shape) * MP

        thermal_velocity = np.sqrt(KB * 10**line_list.logT / mass_xr)
        dopp_width_th = wv_0 * thermal_velocity / CC_ms  # (AA) sigma
        dopp_width = np.sqrt(dopp_width_th**2 + instr_width**2)  # (AA)
        shift = dnus - wv_0  # (AA)
        # pre broadcast everything for ne.evaluate
        shift = shift.broadcast_like(line_list.gofnt.isel(trans_index=0))
        dopp_width, shift = xr.broadcast(dopp_width, shift)

        goft = line_list.gofnt.isel(trans_index=i).broadcast_like(dopp_width)
        norm = np.sqrt(2 * np.pi)

        if i < num_lines_keep:
            resp_temp = ne.evaluate("goft / normalization * exp(-0.5 * (shift / dopp_width)**2) / norm / dopp_width")  # (1/AA)
            resp_temp = xr.DataArray(resp_temp, dims=goft.dims)
            resp_temp = resp_temp.expand_dims("line")
            line = (
                line_list.spectroscopic.isel(trans_index=i).astype(object)
                + " "
                + line_list.wvl.isel(trans_index=i).astype(str).astype(object)
            )
            resp_temp = resp_temp.assign_coords({"line": line.squeeze().expand_dims("line").data})

            # This is silly, but Juan wants it. line_wvl for summed lines will be equal to the first line in the list
            # If the line_list has been sorted ahead of time, it will be the brightest line.
            resp_temp = resp_temp.assign_coords(
                line_wvl=("line", line_list.isel(trans_index=0).squeeze().wvl.expand_dims("line").data)
            )

            resp.append(resp_temp)
        else:
            if i == num_lines_keep:
                summed_lines_included = True
                resp_contam_temp = ne.evaluate("goft / normalization * exp(-0.5 * (shift / dopp_width)**2) / norm / dopp_width")

            resp_contam_temp = ne.evaluate(
                "goft / normalization * exp(-0.5 * (shift / dopp_width)**2) / norm / dopp_width + resp_contam_temp"
            )

        end_time = time.perf_counter()
        time_remaining = (num_lines - i) * (end_time - start_time)
        if verbose:
            pass

    if summed_lines_included:
        resp_contam_temp = xr.DataArray(resp_contam_temp, dims=goft.dims)
        resp_contam_temp = resp_contam_temp.expand_dims("line")
        # # overwrite line coord with correct label
        contam_line_label = xr.DataArray(
            [f"{band} remaining {line_list.sizes['trans_index'] - num_lines_keep} lines"], dims="line"
        )

        resp_contam_temp = resp_contam_temp.assign_coords(line=contam_line_label)
        resp_contam_temp = resp_contam_temp.assign_coords({"line_wvl": line_list.isel(trans_index=0).wvl.expand_dims("line")})

        if line is not None:
            resp_contam_temp["line"] = contam_line_label.broadcast_like(line)
        else:
            resp_contam_temp["line"] = contam_line_label

        if resp:
            resp.append(resp_contam_temp)
        else:
            resp = resp_contam_temp

    resp = xr.concat(resp, dim="line", coords="different", compat="equals")

    ds = xr.Dataset()
    ds["SG_resp"] = resp
    for icoords in line_list.isel(trans_index=0).coords:
        ds = ds.assign_coords(**{icoords: line_list.coords[icoords]})
    if vdop is not None:
        ds = ds.assign_coords({"vdop": vdop})

    sg_attrs = ds.SG_resp.attrs

    if effective_area is not None:
        interp = effective_area.interp(wavelength=dnus, method=method)
        ds["SG_resp"] = ds.SG_resp * interp
        ds.SG_resp.attrs.update(sg_attrs)
        ds.SG_resp.attrs["units"] = str(normalization * u.erg * u.cm**5 / u.s / u.sr / u.angstrom)

    ds.coords["SG_wvl"] = dnus

    if num_lines_keep > 0:
        ds.coords["SG_wvl"] = ds.coords["SG_wvl"].expand_dims({"line": num_lines_keep})
    ds.coords["SG_wvl"].attrs.update({"units": str(u.AA)})

    add_history(ds, locals(), create_resp_func)

    return ds


def create_resp_func_nullvel(
    line_list: xr.Dataset,
    *,
    instr_width: float = 0,
    normalization: float = 1e-27,
    abundance: str = "sun_photospheric_1998_grevesse",
    method: str = "linear",
    sum_lines: bool = False,
    missing_line: str = "missing main line",
    temp: np.ndarray | list = None,
    dma: float = 4.9,
    dens: np.ndarray | list = None,
    pres: float | None = None,
    wvlr: np.ndarray | list = None,
    wvlo: float | None = None,
    effective_area: xr.Dataset | None = None,
    goft_max: float | None = None,
    npix: int = 6,
) -> xr.Dataset:
    """
    Computes a response function with SG as a function temperature assuming
    zero Doppler velocity.

    Parameters
    ----------
    line_list: `xr.Dataset`
        line list, can be created with `create_resp_line_list`
    instr_width : `float`,
        Instrumental width sigma in Amstrongs, by default 0  TBD: add options for formulae
    normalization: `float`
        Normalization in the response function, by default 1e-27
    abundance : `str`
        Abundance, by default sun_photospheric_1998_grevesse.
    method : `str`
        Interpolation method, default 'linear'
    sum_lines : `bool`
        Sum all lines in the list, by default False
    missing_line: `str`
        String to add in the line coords when sum_lines is true, e.g. "all lines", by
        default assumes that the main line is removed from the list, i.e., "missing main line"
    temp : `float array`,
        Temperature array in K, by default 10.0 ** np.arange(5,6.5,0.1)
    dma : `float`,
        wavelength pixel size in mÅ, by default 14.7 / 3
    dens : `float`,
        Density, by default 1e9
    pres : `float` or `None`, optional
        Pressure.
    wvlr : `float array`,
        wavelength range to select the spectral line, by default
        it will select the minimum (and maximum) of rest wavelength position of
        all the lines minus (plus) the limit of vel in Amgstrongs.
    wvlo : `float` or `None`, optional
        Central wavelength.
    effective_area : `xr.Dataset`
        effective area, this can be generated with `create_eff_area_xarray`
    goft_max: `float` or `None`, optional
        Cut off for lower goft_max than max(GOFT).
    npix : `int`, optional
        Number of extra pixels from the longest and shortest wavelengths from the line list.

    Returns
    -------
    `xarray.Dataset`
        Response function with temperature, velocity and wavelength axis.
    """
    try:
        import ChiantiPy.core as ch
        import ChiantiPy.tools.io as chio
        import periodictable as pt
        import roman
    except ImportError:
        msg = "ChiantiPy, periodictable and roman required for this function"
        raise ImportError(msg) from None

    CC_ms = speed_of_light.to(u.m / u.s).value  # Light speed (m/s)
    CC_As = CC_ms * 1e10  # Light speed (AA/s)
    KB = 1.3806488e-23  # Boltzman constant (J/K = kg m^2/K/s^2)
    MP = 1.67262178e-27  # Proton mass (kg)

    if temp is None:
        temp = 10.0 ** np.arange(5, 6.5, 0.1)
    if dens is None:
        dens = 1e9
    if wvlr is None:
        wvlr = [line_list.wvl.min().data, line_list.wvl.max().data]

    wvl_center = (wvlr[1] - wvlr[0]) / 2.0 + wvlr[0]
    dnus = np.arange(wvlr[0] - npix * dma / 1e3, wvlr[1] + npix * dma / 1e3, dma / 1e3) - wvl_center  # [A]

    ion_list = list(set(line_list.IonStr.data))

    line_list_included = []
    wvlo_list = []
    for i_iionstr, iionstr in enumerate(ion_list):
        if np.size(dens) > 1:
            denst = np.tile([dens], [np.size(temp), 1])
            tempt = np.tile(np.array([temp]).T, [1, np.size(dens)])
            denst = np.reshape(denst, (np.size(denst)))
            tempt = np.reshape(tempt, (np.size(tempt)))
        elif pres is not None:
            denst = pres / temp
            tempt = temp
        else:
            denst = dens
            tempt = temp

        ion = ch.ion(iionstr, temperature=tempt, eDensity=denst, abundance=abundance)

        # Calculating the emissivities for the specified line.
        # It does not include elemental abundance or ionization fraction.
        ion.emiss()  # (erg/s/sr)

        if np.size(dens) > 1:
            denst = np.reshape(denst, (np.size(temp), np.size(dens)))
            tempt = np.reshape(tempt, (np.size(temp), np.size(dens)))

        # Finding the spectral line.
        wvl_em = np.array(ion.Emiss["wvl"]) / line_list.attrs["order"]
        wvl_list = line_list.where(line_list.IonStr == ion.IonStr).wvl.dropna(dim="trans_index")
        iwvlo_list = [iii for iii, ii in enumerate(wvl_em) if ii in wvl_list]

        for iiwvlo, iwvlo in enumerate(iwvlo_list):
            wvlo = ion.Emiss["wvl"][np.squeeze(iwvlo)] / line_list.attrs["order"]  # (AA)
            wvlo_list.append(wvlo)
            ion_level = iionstr.split("_")[1]
            if ion_level[-1] == "d":
                ion_level = ion_level[:-1]
            ion_name = iionstr.split("_")[0][0].capitalize() + iionstr.split("_")[0][1:]
            line_list_included.append(f"{ion_name} {roman.toRoman(int(ion_level))} {wvlo!s}")
            if np.size(dens) > 1:
                emiss = np.reshape(
                    ion.Emiss["emiss"][np.squeeze(iwvlo)],
                    (np.size(temp), np.size(dens)),
                )
                IoneqOne = np.reshape(ion.IoneqOne, (np.size(temp), np.size(dens)))
            else:
                emiss = ion.Emiss["emiss"][np.squeeze(iwvlo)]
                IoneqOne = ion.IoneqOne
            gofnt = np.squeeze(ion.Abundance * IoneqOne / denst * emiss)

            if goft_max is None or np.max(gofnt) > goft_max:
                try:
                    awgt = pt.elements[ion.Z].ion[1].mass
                except Exception:  # NOQA: BLE001
                    awgt = pt.elements[ion.Z].mass

                dopp_width_th = wvlo / CC_ms * np.sqrt(KB * temp / awgt / MP)  # (AA) sigma
                dopp_width = np.sqrt(dopp_width_th**2 + instr_width**2)  # (AA)

                shift = dnus - (wvlo - wvl_center)  # (AA)

                shift_g = np.einsum("a,c->ac", shift, 1.0 / dopp_width)  # (adimensional)
                phi = np.exp(-0.5 * shift_g * shift_g) / np.sqrt(
                    2 * np.pi
                )  # (adimensional)  exp {dlambda^2 / [2 dopp_width^2 sqrt (2 pi)]}
                phi_g = np.einsum("ac,c->ac", phi, 1.0 / dopp_width)  # (1/AA)

                einsum_str = "ac,cd->acd" if np.size(dens) > 1 else "ac,c->ac"
                resp_temp = np.einsum(einsum_str, phi_g, gofnt / normalization)  # (1e-27 erg cm^3/s/sr/A)

                # No integrate the wvl pixel, i.e., (1e-27 erg cm^3/s/sr/A)
                if iiwvlo == 0 and i_iionstr == 0:
                    resp = resp_temp if sum_lines else [resp_temp]
                elif sum_lines:
                    resp += resp_temp
                else:
                    resp = np.append(resp, [resp_temp], axis=0)

    dims = ["SG_xpixel", "logT"]
    coords = {"logT": np.log10(temp), "SG_xpixel": dnus + wvl_center}

    if np.size(dens) > 1:
        dims = np.append(dims, "logD")
        coords["logD"] = np.log10(dens)
    if not sum_lines:
        dims = np.append("line", dims)
        if np.size(line_list_included) == 1:
            line_list_included[0] = f"{line_list_included[0].split(' ')[0]} {line_list_included[0].split(' ')[1]}"
        coords["line"] = line_list_included
    else:
        coords["line"] = f"{line_list.band.data} {missing_line}"

    ds = xr.Dataset()
    ds["SG_resp"] = xr.DataArray(
        resp,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Response function",
            "units": str(normalization * u.erg * u.cm**3 / u.s / u.sr / u.angstrom),
            "abundance": abundance,
            "instr_width": instr_width,  # AA
        },
    )
    sg_attrs = ds.SG_resp.attrs

    if np.size(denst) == 1:
        ds["SG_resp"].attrs["logD"] = np.log10(dens)
    elif pres is not None:
        ds["SG_resp"].attrs["logPe"] = np.log10(pres)

    if effective_area is not None:
        ds["SG_resp"] = ds.SG_resp * effective_area.eff_area.isel(band=0).interp(SG_xpixel=dnus + wvl_center, method=method)
        ds.SG_resp.attrs.update(sg_attrs)
        ds.SG_resp.attrs["units"] = str(normalization * u.erg * u.cm**5 / u.s / u.sr / u.angstrom)

        ds["effective_area"] = xr.DataArray(
            effective_area.isel(band=0).eff_area.interp(SG_xpixel=dnus + wvl_center, method=method).data,
            dims=["SG_xpixel"],
            attrs={
                "description": "effective area",
                "units": str(u.cm**2),
            },
        )
        if not sum_lines:
            ds = ds.assign_coords(band=("line", np.tile(effective_area.band.data, np.size(line_list_included))))

    ds.coords["SG_wvl"] = xr.DataArray(dnus + wvl_center, dims=["SG_xpixel"])
    ds.coords["SG_wvl"].attrs.update({"units": str(u.AA)})

    if "line" not in ds.dims:
        ds = ds.expand_dims({"line": 1})
    if not sum_lines:
        ds = ds.assign_coords({"line_wvl": ("line", wvlo_list)})
    elif wvlo is not None:
        ds = ds.assign_coords({"line_wvl": ("line", [wvlo])})
    else:
        ds = ds.assign_coords({"line_wvl": ("line", [wvl_center])})

    ds.line_wvl.attrs.update({"units": str(u.AA)})
    ds.SG_xpixel.attrs.update({"units": str(u.AA)})
    ds.attrs["Chianti version"] = chio.versionRead()
    ds.attrs["abundance"] = abundance

    add_history(ds, locals(), create_resp_func_nullvel)

    return ds


def add_vdop_to_nullvel(
    sg_ds: xr.Dataset,
    *,
    vdop: np.ndarray | list = None,
    effective_area=None,
    normalization: float = 1e-27,
    method: str = "linear",
) -> xr.Dataset:
    """
    Expands the response function by adding doppler shift axis.

    Parameters
    ----------
    sg_ds: `xr.Dataset`
        Response function with SG0 without Doppler velocity axis, i.e.,
        assuming zero Doppler velocity, can be created with `create_resp_line_list`
    vdop : `float array`,
        Doppler velocity axis array in km/s, by default np.arange(-1e3,1e3,10)
    effective_area : `xr.Dataset` or `None`, optional
        Effective area.
    normalization : `float`, optional
        Normalization in the response function, by default 1e-27.
    method : `str`, optional
        Interpolation method, default 'linear'.

    Returns
    -------
    `xarray.Dataset`
        Response function with temperature, Doppler velocity and wavelength axis.
    """
    if vdop is None:
        vdop = np.arange(-1000, 1000, 10)
    dsv = sg_ds.copy(deep=True)
    dsv = dsv.expand_dims(dim={"vdop": np.size(vdop)})
    dsv = dsv.assign_coords(vdop=("vdop", vdop)).copy(deep=True)
    CC_ms = speed_of_light.to(u.m / u.s).value  # Light speed (m/s)

    for ii in vdop:
        if effective_area is None:
            dsv["SG_resp"].loc[{"vdop": ii}] = sg_ds.SG_resp.interp(
                SG_xpixel=sg_ds.SG_xpixel - sg_ds.SG_xpixel * ii / CC_ms * 1e3,
                method=method,
                kwargs={"fill_value": 0.0},
            )
        else:
            dsv["SG_resp"].loc[{"vdop": ii}] = sg_ds.SG_resp.interp(
                SG_xpixel=sg_ds.SG_xpixel - sg_ds.SG_xpixel * ii / CC_ms * 1e3,
                method=method,
                kwargs={"fill_value": 0.0},
            ) * effective_area.eff_area.isel(band=0).interp(SG_xpixel=sg_ds.SG_xpixel, method=method)
        dsv.coords["SG_wvl"].loc[{"vdop": ii}] = sg_ds.SG_xpixel.data - sg_ds.SG_xpixel.data * ii / CC_ms * 1e3

    if effective_area is not None:
        effective_area_temp = effective_area.isel(band=0).expand_dims(dim={"vdop": vdop})
        effective_area_temp = effective_area_temp.interp(
            SG_xpixel=sg_ds.SG_xpixel - sg_ds.SG_xpixel * ii / CC_ms * 1e3, method=method
        )
        for ii in vdop:
            effective_area_temp["eff_area"].loc[{"vdop": ii}] = (
                effective_area.eff_area.isel(band=0)
                .interp(SG_xpixel=sg_ds.SG_xpixel - sg_ds.SG_xpixel * ii / CC_ms * 1e3, method=method)
                .data
            )
        dsv["effective_area"] = xr.DataArray(
            effective_area_temp.eff_area.data,
            dims=["vdop", "SG_xpixel"],
            attrs={
                "description": "effective area",
                "units": str(u.cm**2),
            },
        )

    dsv = dsv.assign_coords({"line_wvl": ("line", sg_ds.line_wvl.data)})
    dsv.line_wvl.attrs.update({"units": str(u.AA)})

    dsv.SG_resp.attrs["units"] = str(normalization * u.erg * u.cm**5 / u.s / u.sr / u.angstrom)
    add_history(dsv, locals(), add_vdop_to_nullvel)

    return dsv


def create_resp_func_ci(
    line_list: xr.Dataset,
    *,
    normalization: float = 1e-27,
    abundance: str = "sun_photospheric_1998_grevesse",
    method: str = "linear",
    sum_lines: bool = False,
    temp: np.ndarray | list = None,
    dens: np.ndarray | list = None,
    pres: float | None = None,
    effective_area: xr.Dataset | None = None,
    goft_max: float | None = None,
    wvlo: float | None = None,
    units: str | None = None,
    dx_pix: float = 0.4,
    dy_pix: float = 0.4,
    gain: int = 1,
    **kwargs,
) -> xr.Dataset:
    """
    Computes a response function as a function of velocity and temperature.

    Parameters
    ----------
    line_list : `xr.Dataset`
        Line list, can be created with `create_resp_line_list`.
    normalization : `float`, optional
        Normalization in the response function, by default 1e-27.
    abundance : `str`, optional
        Abundance, by default "sun_photospheric_1998_grevesse".
    method : `str`, optional
        Interpolation method, default 'linear'.
    sum_lines : `bool`, optional
        Sum all lines in the list, by default False.
    temp : `array-like`, optional
        Temperature array in K.
    dens : `array-like`, optional
        Density.
    pres : `float` or `None`, optional
        Pressure.
    effective_area : `xr.Dataset` or `None`, optional
        Effective area.
    goft_max : `float` or `None`, optional
        Cut off for lower goft_max than max(GOFT).
    wvlo : `float` or `None`, optional
        Central wavelength.
    units : `str` or `None`, optional
        Output units.
    dx_pix : `float`, optional
        Pixel size along x in arcsec, by default 0.4.
    dy_pix : `float`, optional
        Pixel size along y in arcsec, by default 0.4.
    gain : `int`, optional
        Gain value, by default 1.

    Returns
    -------
    `xarray.Dataset`
        Response function with temperature.
    """
    try:
        import ChiantiPy.core as ch
        import ChiantiPy.tools.io as chio
        import roman
    except ImportError:
        msg = "ChiantiPy and roman required for this function"
        raise ImportError(msg) from None

    if temp is None:
        temp = 10.0 ** np.arange(5, 6.5, 0.1)
    if dens is None:
        dens = 1e9

    ion_list = list(set(line_list.IonStr.data))

    line_list_included = []
    wvlo_list = []
    for i_iionstr, iionstr in enumerate(ion_list):
        if np.size(dens) > 1:
            denst = np.tile([dens], [np.size(temp), 1])
            tempt = np.tile(np.array([temp]).T, [1, np.size(dens)])
            denst = np.reshape(denst, (np.size(denst)))
            tempt = np.reshape(tempt, (np.size(tempt)))
        elif pres is not None:
            denst = pres / temp
            tempt = temp
        else:
            denst = dens
            tempt = temp
        ion = ch.ion(iionstr, temperature=tempt, eDensity=denst, abundance=abundance)

        # Calculating the emissivities for the specified line.
        # It does not include elemental abundance or ionization fraction.
        ion.emiss()  # (erg/s/sr)

        if np.size(dens) > 1:
            denst = np.reshape(denst, (np.size(temp), np.size(dens)))
            tempt = np.reshape(tempt, (np.size(temp), np.size(dens)))

        # Finding the spectral line.
        wvl_em = np.array(ion.Emiss["wvl"])
        wvl_list = line_list.where(line_list.IonStr == ion.IonStr).wvl.dropna(dim="trans_index")
        iwvlo_list = [iii for iii, ii in enumerate(wvl_em) if ii in wvl_list]

        for iiwvlo, iwvlo in enumerate(iwvlo_list):
            wvlo = ion.Emiss["wvl"][np.squeeze(iwvlo)]  # (AA)
            wvlo_list.append(wvlo)
            ion_level = iionstr.split("_")[1]
            if ion_level[-1] == "d":
                ion_level = ion_level[:-1]
            line_list_included.append(iionstr.split("_")[0] + " " + roman.toRoman(int(ion_level)) + " " + str(wvlo))
            if np.size(dens) > 1:
                emiss = np.reshape(
                    ion.Emiss["emiss"][np.squeeze(iwvlo)],
                    (np.size(temp), np.size(dens)),
                )
                IoneqOne = np.reshape(ion.IoneqOne, (np.size(temp), np.size(dens)))
            else:
                emiss = ion.Emiss["emiss"][np.squeeze(iwvlo)]
                IoneqOne = ion.IoneqOne
            gofnt = np.squeeze(ion.Abundance * IoneqOne / denst * emiss)

            if (iiwvlo == 0) and (i_iionstr == 0):
                resp = np.zeros_like(gofnt)

            ph2dn = 1.0
            if units is not None:
                units_temp = units.replace("DN", str(u.ph)) if "DN" in units else units
                conversion_factor, add_units = conversion_units(
                    wvlo,
                    old_units=str(normalization * u.erg * u.cm**5 / u.s / u.sr),
                    new_units=units_temp,
                    dx_pix=dx_pix,
                    dy_pix=dy_pix,
                )
                if "ph" in units:
                    gain = 1
                else:
                    ph2dn = conversion_ph2dn(wvlo, gain=gain)
                conversion_factor = conversion_factor * ph2dn

            else:
                conversion_factor = 1.0
            gofnt *= conversion_factor

            if goft_max is None or np.max(gofnt) > goft_max:
                resp_temp = gofnt / normalization  # (1e-27 erg cm^3/s/sr/A)
                if effective_area is not None:
                    resp_temp = (
                        resp_temp
                        * effective_area.eff_area.isel(band=0)
                        .interp(
                            SG_xpixel=wvlo,
                            method=method,
                            kwargs={"fill_value": 0.0},
                        )
                        .data
                    )
                if sum_lines:
                    resp += resp_temp
                elif np.shape(resp) == np.shape(resp_temp):
                    resp = [resp_temp]  # integrate the wvl pixel (1e-27 erg cm^3/s/sr)
                else:
                    resp = np.append(
                        resp,
                        [resp_temp],
                        axis=0,
                    )  # integrate the wvl pixel (1e-27 erg cm^3/s/sr)

    coords = {"logT": np.log10(temp)}

    if not sum_lines:
        dims = ["line", "logT"]
        coords["line"] = line_list_included
    else:
        dims = ["logT"]

    if np.size(dens) > 1:
        dims = np.append(dims, "logD")
        coords["logD"] = np.log10(dens)

    ds = xr.Dataset()
    ds["SG_resp"] = xr.DataArray(
        resp,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Response function",
            "units": str(normalization * u.erg * u.cm**3 / u.s / u.sr),
            "abundance": abundance,
        },
    )

    if np.size(dens) == 1:
        ds["SG_resp"].attrs["logD"] = np.log10(dens)
    elif pres is not None:
        ds["SG_resp"].attrs["logPe"] = np.log10(pres)

    if effective_area is not None:
        if units is None:
            ds.SG_resp.attrs["units"] = str(normalization * u.erg * u.cm**5 / u.s / u.sr)
        else:
            ds.SG_resp.attrs["units"] = units

        ds["effective_area"] = effective_area.eff_area
        ds.effective_area.attrs.update(effective_area.attrs)
    if not sum_lines:
        ds.coords["SG_wvl"] = xr.DataArray(wvlo_list, dims=["line"])
        ds.coords["SG_wvl"].attrs.update({"units": str(u.AA)})
        ds = ds.assign_coords({"line_wvl": ("line", wvlo_list)})
    else:
        if wvlo is None:
            wvlo = np.mean(wvlo_list)
        ds = ds.assign_coords({"line_wvl": ("line", [wvlo])})

    ds.line_wvl.attrs.update({"units": str(u.AA)})
    ds.attrs["Chianti version"] = chio.versionRead()
    ds.attrs["abundance"] = abundance
    add_history(ds, locals(), create_resp_func_ci)

    return ds


def create_eff_area_xarray(
    eff_area: np.ndarray,
    wvl: np.ndarray,
    band: list,
    mirror=None,
    ccd=None,
    inst_filter=None,
    grating=None,
    contam=None,
    attrs=None,
) -> xr.Dataset:
    """
    Create effective area from numpy array into an xarray.

    Parameters
    ----------
    eff_area : `numpy.ndarray`
        Effective area as a function of wavelength.
    wvl : `numpy.ndarray`
        Wavelength.
    band : `list`
        Spectral band list.
    mirror : `numpy.ndarray`, optional
        Mirror response as a function of wavelength.
    ccd : `numpy.ndarray`, optional
        CCD response as a function of wavelength.
    inst_filter : `numpy.ndarray`, optional
        Filter response as a function of wavelength.
    grating : `numpy.ndarray`, optional
        Grating response as a function of wavelength.
    contam : `numpy.ndarray`, optional
        Contamination response as a function of wavelength.
    attrs : `dict`, optional
        Attributes to be added.

    Returns
    -------
    `xarray.Dataset`
        Effective area as a function of wavelength axis.
    """
    nwvl = np.size(wvl)
    SG_xpixel = np.arange(nwvl)
    effective_area = xr.Dataset()
    effective_area["eff_area"] = xr.DataArray(
        np.reshape(eff_area, (np.size(SG_xpixel), np.size(band))),
        dims=["wavelength", "band"],
        coords={"wavelength": wvl, "band": band},
        attrs={
            "description": "Effective area",
            "units": str(u.cm**2),
        },
    )
    if mirror is not None:
        effective_area["mirror"] = xr.DataArray(
            np.reshape(mirror, (np.size(SG_xpixel), np.size(band))),
            dims=["wavelength", "band"],
            coords={"wavelength": wvl, "band": band},
            attrs={
                "description": "Mirror",
                "units": "cm2",
            },
        )
    if inst_filter is not None:
        effective_area["filter"] = xr.DataArray(
            np.reshape(inst_filter, (np.size(SG_xpixel), np.size(band))),
            dims=["wavelength", "band"],
            coords={"wavelength": wvl, "band": band},
            attrs={
                "description": "thin Al filter response",
                "units": "cm2",
            },
        )
    if ccd is not None:
        effective_area["ccd"] = xr.DataArray(
            np.reshape(ccd, (np.size(SG_xpixel), np.size(band))),
            dims=["wavelength", "band"],
            coords={"wavelength": wvl, "band": band},
            attrs={
                "description": "CCD response",
                "units": "cm2",
            },
        )
    if grating is not None:
        effective_area["grating"] = xr.DataArray(
            np.reshape(grating, (np.size(SG_xpixel), np.size(band))),
            dims=["wavelength", "band"],
            coords={"wavelength": wvl, "band": band},
            attrs={
                "description": "grating response",
                "units": "cm2",
            },
        )
    if contam is not None:
        effective_area["contam"] = xr.DataArray(
            np.reshape(contam, (np.size(SG_xpixel), np.size(band))),
            dims=["wavelength", "band"],
            coords={"wavelength": wvl, "band": band},
            attrs={
                "description": "contam response",
                "units": "cm2",
            },
        )
    if attrs is not None:
        effective_area.attrs.update(attrs)

    effective_area.wavelength.attrs["units"] = str(u.AA)

    add_history(effective_area, locals(), create_eff_area_xarray)

    return effective_area


def convert_resp2muse_resp(
    response: xr.Dataset,
    muse_response: xr.Dataset,
    band_index: int,
    method: str = "linear",
) -> xr.Dataset:
    """
    Convert a response function in a MUSE response function, i.e., adding
    slits. No additional instrumental effects are included.

    Parameters
    ----------
    response : `xarray.Dataset`
        Response function with temperature, velocity and wavelength axis.
    muse_response : `xarray.Dataset`,
        MUSE xarray response function (including several bands/lines).
    band_index : `int`,
        Index of the band of the muse_response.
    method : `str`,
        Interpolation method, by default linear.

    Returns
    -------
    `xarray.Dataset`
        Response function including MUSE slits.
    """
    if "band" not in muse_response.dims:
        muse_response = muse_response.swap_dims({"line": "band"})
    if "line" not in response.dims:
        response = response.swap_dims({"band": "line"})
    resp2 = response.copy(deep=True)
    for _ in np.arange(np.size(muse_response.coords["slit"]) - 1):
        resp2 = xr.concat([resp2, response.copy(deep=True)], dim="slit", coords="different", compat="equals")
    resp2.coords["slit"] = muse_response.coords["slit"].values

    resp2 = resp2.interp(
        SG_xpixel=np.linspace(
            resp2.SG_xpixel.min().data,
            resp2.SG_xpixel.max().data,
            np.size(muse_response.SG_xpixel.values),
        ),
        method=method,
    )

    for ii in resp2.coords["slit"]:
        resp2["SG_resp"].loc[{"slit": ii}] = (
            resp2.SG_resp.sel(slit=ii)
            .interp(SG_xpixel=muse_response.SG_wvl.isel(band=band_index, slit=ii.data).data, method=method)
            .values
        )

    resp2["SG_resp"] = resp2.SG_resp.fillna(0)
    resp2["SG_resp"] = resp2.SG_resp.where(resp2.SG_resp > 0, 0)
    da_per_pix = (
        muse_response.SG_wvl.isel(band=band_index, slit=int(resp2.coords["slit"].max() / 2))[1]
        - muse_response.SG_wvl.isel(band=band_index, slit=int(resp2.coords["slit"].max() / 2))[0]
    )
    resp2["SG_resp"] = resp2.SG_resp * da_per_pix
    resp2.SG_resp.attrs["units"] = str(response.SG_resp.attrs["units"] * u.angstrom)

    resp2["SG_xpixel"] = muse_response.SG_xpixel.data
    if "band" in resp2.SG_wvl.dims:
        resp2["SG_wvl"].loc[{"band": resp2.coords["band"].data[0]}] = muse_response.SG_wvl.isel(band=band_index)
    else:
        resp2["SG_wvl"] = muse_response.SG_wvl.isel(band=band_index)
    resp2["line_wvl"] = muse_response.isel(band=band_index).line_wvl

    add_history(resp2, locals(), convert_resp2muse_resp)

    return resp2


def convert_resp2muse_sgresp(
    response: xr.Dataset,
    nslits=35,
    dma_per_pix=14.7,  # mA/pix
    pixel_per_slit=26.53,
    npix=1024,
    wvlo=170.62314,
    verbose=False,
    loop=True,
) -> xr.Dataset:
    """
    Convert a response function in a response function (wavelength , i.e.,
    adding slits. No additional instrumental effects are included.

    Parameters
    ----------
    response : `xarray.Dataset`
        Response function with temperature, velocity and wavelength axis.
    nslits : `int`, optional
        Number of MUSE slits, by default 35.
    dma_per_pix : `float`, optional
        1e-3 Angstroms per pixel, by default 14.7.
    pixel_per_slit : `float`, optional
        Number of pixels between slits, by default 26.53.
    npix : `int`, optional
        Number of pixels in the output, by default 1024.
    wvlo : `float`, optional
        Wavelength position of the first slit at SG_xpixel=0, by default 170.62314.
    verbose : `bool`, optional
        Verbose output, by default False.
    loop : `bool`, optional
        Whether to loop over temperature for interpolation, by default True.

    Returns
    -------
    `xarray.Dataset`
        Response function including MUSE slits.
    """
    resp = response.copy(deep=True)

    # build nslits into DataArray for broadcasting
    slit_num = xr.DataArray(
        np.arange(nslits),
        dims="slit",
    )

    wvl = np.linspace(wvlo, wvlo + npix * dma_per_pix / 1e3, npix)
    wvl = xr.DataArray(
        wvl,
        dims=("wavelength", *wvlo.dims),
    )

    fsr = (
        pixel_per_slit * dma_per_pix / 1e3
    )  # Actually, what is the formula, assuming a dma_per_pix, one can find out what is the number of slits per pixel, how?
    wvl_slits = wvl - (slit_num * fsr)

    # shove into named_arrays for fast interpolation
    wvl_slits_na = na.ScalarArray(wvl_slits.data, axes=wvl_slits.dims)
    sg_xpixel_na = na.ScalarArray(resp.wavelength.data, axes=resp.wavelength.dims)
    sg_resp_na = na.ScalarArray(resp.SG_resp.data, axes=resp.SG_resp.dims)

    # prepare to interpolate
    coordinates_input = sg_xpixel_na
    coordinates_output = wvl_slits_na
    values_input = sg_resp_na
    method = "multilinear"
    axis_input = "wavelength"
    axis_output = "wavelength"

    _weights, shape_input, shape_output = na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
    )

    if loop:
        results = []
        num_temps = values_input.shape["logT"]
        for i in range(num_temps):
            start_time = time.perf_counter()
            result = na.regridding.regrid_from_weights(
                weights=_weights,
                shape_input=shape_input,
                shape_output=shape_output,
                values_input=values_input[{"logT": i}],
            )
            results.append(result)

            end_time = time.perf_counter()
            time_remaining = (num_temps - i) * (end_time - start_time)
            if verbose:
                pass

        results = na.concatenate([result.add_axes("logT") for result in results], axis="logT")
    else:
        results = na.regridding.regrid_from_weights(
            weights=_weights,
            shape_input=shape_input,
            shape_output=shape_output,
            values_input=values_input,
        )
    resp = response.drop_dims("wavelength")

    resp["SG_resp"] = xr.DataArray(
        results.ndarray,
        dims=results.axes,
    )

    # multiply by pixel width in angstroms
    resp["SG_resp"] *= dma_per_pix / 1e3
    resp["SG_resp"].attrs["units"] = str(response.SG_resp.attrs["units"] * u.angstrom)

    resp.coords["SG_wvl"] = wvl_slits

    resp = resp.assign_coords({"SG_xpixel": xr.DataArray(np.arange(npix), dims="SG_xpixel")})
    resp = resp.assign_coords({"slit": slit_num})

    resp = resp.rename({"wavelength": "SG_xpixel"})

    add_history(resp, locals(), convert_resp2muse_sgresp)

    return resp


def convert_resp2muse_ciresp(response: xr.Dataset, verbose=False, loop=True) -> xr.Dataset:
    """
    Convert a wavelength cube to a MUSE CI response.

    Parameters
    ----------
    response : `xarray.Dataset`
        Response function with temperature, velocity and wavelength axis.


    Returns
    -------
    `xarray.Dataset`
        Response function for MUSE CI
    """
    resp = response.copy(deep=True)

    wavelength_range = resp.wavelength.max() - resp.wavelength.min()
    wavelength_size = resp.wavelength.size

    resp = resp.sum("wavelength")

    # multiply by pixel width in angstroms
    resp["SG_resp"] *= wavelength_range / wavelength_size
    resp["SG_resp"].attrs["units"] = str(response.SG_resp.attrs["units"] * u.angstrom)

    resp = resp.assign_coords({"SG_xpixel": xr.DataArray(np.arange(1), dims="SG_xpixel")})

    add_history(resp, locals(), convert_resp2muse_ciresp)

    return resp


def convert_resp2muse_resp_scr(
    response: xr.Dataset,
    nslits=35,
    dma_per_pix=14.7,  # mA/pix
    pixel_per_slit=26.53,
    npix=1024,
    wvlo=170.62314,
    method: str = "linear",
) -> xr.Dataset:
    """
    Convert a response function in a MUSE response function, i.e., adding
    slits. No additional instrumental effects are included.

    Parameters
    ----------
    response : `xarray.Dataset`
        Response function with temperature, velocity and wavelength axis.
    nslits : `int`, optional
        Number of MUSE slits, by default 35.
    dma_per_pix : `float`, optional
        1e-3 Angstroms per pixel, by default 14.7.
    pixel_per_slit : `float`, optional
        Number of pixels between slits, by default 26.53.
    npix : `int`, optional
        Number of pixels in the output, by default 1024.
    wvlo : `float`, optional
        Wavelength position of the first slit at SG_xpixel=0, by default 170.62314.
    method : `str`, optional
        Interpolation method, by default "linear".

    Returns
    -------
    `xarray.Dataset`
        Response function including MUSE slits.
    """
    if "line" not in response.dims:
        response = response.swap_dims({"band": "line"})
    resp2 = response.copy(deep=True)
    for _ in np.arange(nslits - 1):
        resp2 = xr.concat([resp2, response.copy(deep=True)], dim="slit", coords="different", compat="equals")

    if np.size(dma_per_pix) > 1:
        resp3 = resp2.copy(deep=True)
        for _ in np.arange(len(dma_per_pix) - 1):
            resp3 = xr.concat([resp3, resp2.copy(deep=True)], dim="dma", coords="different", compat="equals")
        resp2 = resp3
        resp2.coords["dma"] = dma_per_pix
    resp2.coords["slit"] = np.arange(nslits)

    if np.size(dma_per_pix) > 1:
        for dma in dma_per_pix:
            wvl = np.arange(wvlo, wvlo + (npix - 0.9) * dma / 1e3, dma / 1e3)
            fsr = (
                pixel_per_slit * dma / 1e3
            )  # Actually, what is the formula, assuming a dma_per_pix, one can find out what is the number of slits per pixel, how?

            wvl_slits = np.zeros((nslits, npix))

            for slit_num in np.arange(nslits):
                wvl_slits[slit_num, :] = wvl - (slit_num) * fsr
            resp2_cut = resp2.isel(SG_xpixel=np.arange(npix))
            for ii in resp2.coords["slit"]:
                resp2_cut["SG_resp"].loc[{"slit": ii, "dma": dma}] = (
                    resp2.SG_resp.sel(slit=ii, dma=dma).interp(SG_xpixel=wvl_slits[ii, :], method=method).values
                )
                resp2_cut["SG_wvl"].loc[{"slit": ii, "dma": dma}] = wvl_slits[ii, :]

    else:
        wvl = np.arange(wvlo, wvlo + (npix - 0.9) * dma_per_pix / 1e3, dma_per_pix / 1e3)
        fsr = (
            pixel_per_slit * dma_per_pix / 1e3
        )  # Actually, what is the formula, assuming a dma_per_pix, one can find out what is the number of slits per pixel, how?

        wvl_slits = np.zeros((nslits, npix))

        for slit_num in np.arange(nslits):
            wvl_slits[slit_num, :] = wvl - (slit_num) * fsr
        resp2_cut = resp2.isel(SG_xpixel=np.arange(npix))
        for ii in resp2.coords["slit"]:
            resp2_cut["SG_resp"].loc[{"slit": ii}] = (
                resp2.SG_resp.sel(slit=ii).interp(SG_xpixel=wvl_slits[ii, :], method=method).values
            )
            resp2_cut["SG_wvl"].loc[{"slit": ii}] = wvl_slits[ii, :]
    # Integrate the wvl pixel
    resp2_cut["SG_resp"] = resp2_cut.SG_resp * dma_per_pix / 1e3
    resp2_cut.SG_resp.attrs["units"] = str(response.SG_resp.attrs["units"] * u.angstrom)

    resp2_cut["SG_resp"] = resp2_cut.SG_resp.fillna(0)
    resp2_cut["SG_resp"] = resp2_cut.SG_resp.where(resp2_cut.SG_resp > 0, 0)
    resp2_cut.coords["SG_xpixel"] = np.arange(npix)
    resp2_cut = resp2_cut.assign_coords(line_wvl=("line", response.line_wvl.data))

    add_history(resp2_cut, locals(), convert_resp2muse_resp_scr)

    return resp2_cut


def add_wvl_slits(line_list, num_slits: int = 35, fsr: float = 0.390) -> xr.DataArray:
    """
    Add slit-dependent wavelengths to a line list.

    For each transition in the input line_list, this function computes the wavelength
    as seen through each slit, assuming a fixed free spectral range (FSR) between slits.
    It creates a new DataArray 'wvl_slits' with shape (num_slits, n_transitions), where
    each entry is the wavelength for a given slit and transition.

    Parameters
    ----------
    line_list : xr.DataArray
        The input line list, must have a 'wvl' coordinate and 'trans_index' dimension.
    num_slits : int, optional
        Number of slits to consider (default: 35).
    fsr : float, optional
        Free spectral range between slits in Angstroms (default: 0.390).

    Returns
    -------
    xr.DataArray
        The input line_list with an added 'wvl_slits' variable, containing the slit-dependent wavelengths.
    """
    wvl_slits = np.zeros((num_slits, line_list.coords["trans_index"].size))
    dims = ["slit", "trans_index"]
    coords = {"slit": np.arange(num_slits), "trans_index": line_list.coords["trans_index"].data}
    for slit_num in np.arange(num_slits):
        wvl_slits[slit_num, :] = np.squeeze(line_list.wvl.data) + slit_num * fsr

    line_list.coords["wvl_slits"] = xr.DataArray(
        wvl_slits,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Wavelength considering the various slits",
            "units": str(u.AA),
        },
    )
    return line_list


def create_resp_line_list(
    effective_area: xr.Dataset,
    temin: float = 6e4,
    temax: float = 1e9,
    abundance: str = "sun_photospheric_2009_asplund",
    eDensity: float = 1e9,
    fsr: float = 0.39,
    num_slits: int = 35,
    goft_max: float = 0,
    ionstr=None,
    wvlr=None,
    order=1.0,
):
    """
    Find the list of spectral lines with GOFT at the formation temperature
    (Tmax) and corresponding response function from an effective area.

    Parameters
    ----------
    effective_area : `xarray.Dataset`
        effective area dataset.
    temin : `float`, optional
        Minimum temperature for Tmax, by default 6e4.
    temax : `float`, optional
        Maximum temperature for Tmax, by default 1e9.
    abundance : `str`, optional
        Abundance, by default "sun_photospheric_2009_asplund".
    eDensity : `float`, optional
        Electron density, by default 1e9.
    fsr : `float`, optional
        Slit range in Angstroms, by default 0.39.
    num_slits : `int`, optional
        Number of slits, by default 35.
    goft_max : `float`, optional
        Selects lines with GOFT larger than this value, by default 0.
    ionstr : `list` or `None`, optional
        List of ion strings to use. If None, all ions are considered.
    wvlr : `list` or `None`, optional
        Wavelength range to consider. If None, uses the effective area range.
    order : `float`, optional
        Spectral order, by default 1.0.

    Returns
    -------
    `xarray.Dataset`
        Spectral line information data matching from an effective area.
    """
    try:
        import ChiantiPy.core as ch
        import ChiantiPy.tools.io as chio
    except ImportError:
        msg = "ChiantiPy required for this function"
        raise ImportError(msg) from None

    if not Path.exists(Path(os.environ["XUVTOP"]) / "abundance" / str(abundance + ".abund")):
        logger.warning(f"Abundance file {abundance + '.abund'} not found. Returning.")
        return None

    ionInfo = chio.masterListInfo()
    line_list = []
    tmax = []
    for ii in ionInfo:
        if "wmin" in ionInfo[ii]:
            if "tIoneqMax" in ionInfo[ii]:
                if (
                    ionInfo[ii]["tIoneqMax"] > temin
                    and ionInfo[ii]["wmin"] < np.max(effective_area.SG_xpixel.data)
                    and ionInfo[ii]["wmax"] > np.min(effective_area.SG_xpixel.data)
                    and ionInfo[ii]["tIoneqMax"] < temax
                ):
                    line_list.append(ii)
                    tmax.append(ionInfo[ii]["tIoneqMax"])
                else:
                    logger.debug("missing tIoneqMax in " + ii)
            elif ionInfo[ii]["wmin"] < np.max(effective_area.SG_xpixel.data) and ionInfo[ii]["wmax"] > np.min(
                effective_area.SG_xpixel.data
            ):
                line_list.append(ii)
            else:
                logger.debug("missing wmin " + ii)
        else:
            logger.debug("something wrong with " + ii)

    tempt = sorted(set(np.round(tmax)))

    if ionstr is None:
        ion = [0] * np.size(line_list)
        for iii, ii in enumerate(line_list):
            ion[iii] = ch.ion(ii, temperature=tempt, eDensity=eDensity, abundance=abundance)
    else:
        ion = [0] * np.size(ionstr)
        for iii, ii in enumerate(ionstr):
            ion[iii] = ch.ion(ii, temperature=tempt, eDensity=eDensity, abundance=abundance)

    Spectroscopic = []
    IonStr = []
    wvl = []
    gofnt = []
    tmax = []
    for _iiion, iion in enumerate(ion):
        try:
            Fails = False
            iion.emiss()
        except Exception:  # NOQA: BLE001
            logger.debug("Fails running emiss for", iion.IonStr)
            Fails = True
        if not Fails:
            if wvlr is None:
                wvlr = [np.min(effective_area.SG_xpixel.data), np.max(effective_area.SG_xpixel.data)]
            iwvlo_list = [
                np.where(np.abs(np.array(iion.Emiss["wvl"]) - (wvlr[0] + (wvlr[1] - wvlr[0]) / 2)) < (wvlr[1] - wvlr[0]) / 2)
            ]

            if np.size(iwvlo_list) > 1:
                for _iiwvlo, iwvlo in enumerate(np.squeeze(iwvlo_list)):
                    wvlo = iion.Emiss["wvl"][np.squeeze(iwvlo)] / order  # (AA)
                    emiss = iion.Emiss["emiss"][np.squeeze(iwvlo)]
                    IoneqOne = iion.IoneqOne
                    gofnt_temp = np.squeeze(iion.Abundance * IoneqOne / eDensity * emiss)
                    if np.max(gofnt_temp) > goft_max:
                        Spectroscopic.append(iion.Spectroscopic)
                        IonStr.append(iion.IonStr)
                        wvl.append(wvlo)
                        gofnt.append(np.max(gofnt_temp))
                        tmax.append(tempt[np.squeeze(np.where(gofnt_temp == np.max(gofnt_temp)))])
            elif np.size(iwvlo_list) == 1:
                wvlo = iion.Emiss["wvl"][np.squeeze(iwvlo_list)] / order  # (AA)
                emiss = iion.Emiss["emiss"][np.squeeze(iwvlo_list)]
                IoneqOne = iion.IoneqOne
                gofnt_temp = np.squeeze(iion.Abundance * IoneqOne / eDensity * emiss)
                if np.max(gofnt_temp) > goft_max:
                    Spectroscopic.append(iion.Spectroscopic)
                    IonStr.append(iion.IonStr)
                    wvl.append(wvlo)
                    gofnt.append(np.max(gofnt_temp))
                    tmax.append(tempt[np.squeeze(np.where(gofnt_temp == np.max(gofnt_temp)))])

    dims = ["trans_index"]
    coords = {"trans_index": np.arange(np.size(gofnt))}
    ds = xr.Dataset()
    ds["gofnt"] = xr.DataArray(
        gofnt,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Response function",
            "units": str(u.erg * u.cm**3 / u.s / u.sr),
        },
    )
    ds["wvl"] = xr.DataArray(
        wvl,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Wavelength",
            "units": str(u.AA),
        },
    )
    ds["Tmax"] = xr.DataArray(
        tmax,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Formation temperature",
            "units": str(u.K),
        },
    )
    ds["Spectroscopic"] = xr.DataArray(
        Spectroscopic,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Pretty Ion string",
        },
    )
    ds["IonStr"] = xr.DataArray(
        IonStr,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Ion string",
        },
    )
    ds["resp_func"] = ds.gofnt.copy(deep=True)
    ds["resp_func"].loc[{"trans_index": np.arange(np.size(gofnt))}] *= (
        effective_area.interp(SG_xpixel=ds.wvl.data).eff_area.isel(band=0).data
    )

    wvl_slits = np.zeros((num_slits, np.size(gofnt)))
    dims = ["slit", "trans_index"]
    coords = {"slit": np.arange(num_slits), "trans_index": np.arange(np.size(gofnt))}
    for slit_num in np.arange(num_slits):
        wvl_slits[slit_num, :] = wvl + slit_num * fsr

    ds["wvl_slits"] = xr.DataArray(
        wvl_slits,
        dims=dims,
        coords=coords,
        attrs={
            "description": "Wavelength considering the various slits",
            "units": str(u.AA),
        },
    )

    ds.attrs["Chianti version"] = chio.versionRead()
    ds.attrs["abundance"] = abundance
    if ds.slit.size > 1:
        ds.attrs["FSR in mA"] = fsr
    ds.attrs["eDensity"] = eDensity
    ds.attrs["order"] = order
    ds["effective_area"] = effective_area.eff_area
    ds.effective_area.attrs.update(effective_area.attrs)
    add_history(ds, locals(), create_resp_line_list)

    return ds.dropna(dim="trans_index")


def ci2sg_response_format(
    ci_response: xr.Dataset,
    sg_response: xr.Dataset,
    line: str = "All",
    wavelength: float = 193.0,
    slit_sep: int = 27,
    method: str = "linear",
) -> xr.Dataset:
    """
    Convert MUSE CI response function in a SG MUSE response function, i.e.,
    adding slits to add in the SDC code.

    Parameters
    ----------
    ci_response : `xarray.Dataset`
        Response function with temperature of the MUSE CI.
    sg_response : `xarray.Dataset`,
        MUSE SG xarray response function (including several bands/lines).
    line : `str`,
        Main ion in the CI.
    wavelength: `float`, optional
        Wavelength of the main line in the CI. Default is 193.0.
    slit_sep: `int`, optional
        Separation between slits. Default is 27.
    method : `str`, optional
        Interpolation method. Default is "linear".

    Returns
    -------
    `xarray.Dataset`
        Response function including MUSE slits for the CI.
    """
    nslits = np.size(sg_response.slit.data)
    nsg_xpixels = np.size(sg_response.SG_xpixel.data)
    nvdop = np.size(sg_response.vdop.data)

    if "line" not in ci_response.dims:
        ci_response = ci_response.expand_dims(
            line=1,
        )
        logger.debug("missing line information, added default 193")

    if "channel" not in ci_response.coords:
        ci_response = ci_response.assign_coords({"channel": ("line", [int(wavelength)])})

    if "SG_xpixel" in ci_response.dims:
        ci_response = ci_response.sum(dim="SG_xpixel")

    ci_response_exp = ci_response.expand_dims(
        vdop=nvdop,
        slit=nslits,
        SG_xpixel=nsg_xpixels,
    )
    ci_response_exp["SG_resp"] = ci_response_exp.SG_resp * 0

    for ii in range(nslits):
        ci_response_exp["SG_resp"].loc[{"slit": ii, "SG_xpixel": slit_sep + slit_sep * ii}] = ci_response.SG_resp.expand_dims(
            vdop=nvdop
        )

    ci_response_exp = ci_response_exp.interp(logT=sg_response.logT, method=method)

    ci_response_exp = ci_response_exp.assign_coords(vdop=sg_response.vdop.data)
    ci_response_exp = ci_response_exp.assign_coords(slit=sg_response.slit.data)
    ci_response_exp = ci_response_exp.assign_coords(SG_xpixel=sg_response.SG_xpixel.data)
    ci_response_exp["SG_wvl"] = xr.DataArray(
        np.reshape(sg_response.SG_wvl.isel(line=0).to_numpy(), (1, nslits, nsg_xpixels)),
        dims={"line": 1, "slit": nslits, "SG_xpixel": nsg_xpixels},
    )

    ci_response_exp["line_wvl"] = sg_response.line_wvl
    ci_response_exp["line_wvl"].loc[{"line": ci_response_exp.isel(line=0).coords["line"]}] = ci_response.channel.values[0]
    ci_response_exp["line"] = ci_response.channel.values
    ci_response_exp["logT"] = sg_response.logT  # Danger, the logT needs to be interpolated
    add_history(ci_response_exp, locals(), ci2sg_response_format)

    return ci_response_exp


def ci2sg_vdop_response_format(
    ci_response: xr.Dataset,
    sg_response: xr.Dataset,
    line: str = "All",
    wavelength: float = 193.0,
    slit_sep: int = 27,
    method: str = "linear",
) -> xr.Dataset:
    """
    Convert MUSE CI response function in a SG MUSE response function, i.e.,
    adding slits to add in the SDC code.

    Parameters
    ----------
    ci_response : `xarray.Dataset`
        Response function with temperature of the MUSE CI.
    sg_response : `xarray.Dataset`
        MUSE SG xarray response function (including several channels/lines).
    line : `str`, optional
        Main ion in the CI. Default is "All".
    wavelength : `float`, optional
        Wavelength of the main line in the CI. Default is 193.0.
    slit_sep : `int`, optional
        Separation between slits. Default is 27.
    method : `str`, optional
        Interpolation method. Default is "linear".

    Returns
    -------
    `xarray.Dataset`
        Response function including MUSE slits for the CI.
    """
    nslits = np.size(sg_response.slit.data)
    nsg_xpixels = np.size(sg_response.SG_xpixel.data)

    if "line" not in ci_response.dims:
        ci_response = ci_response.expand_dims(
            line=1,
        )
        logger.debug("missing line information, added default 193")

    if "channel" not in ci_response.coords:
        ci_response = ci_response.assign_coords({"channel": ("line", [int(wavelength)])})

    if "SG_xpixel" in ci_response.dims:
        ci_response = ci_response.sum(dim="SG_xpixel")

    ci_response_exp = ci_response.expand_dims(
        slit=nslits,
        SG_xpixel=nsg_xpixels,
    )
    ci_response_exp["SG_resp"] = ci_response_exp.SG_resp * 0

    for ii in range(nslits):
        ci_response_exp["SG_resp"].loc[{"slit": ii, "SG_xpixel": slit_sep + slit_sep * ii}] = ci_response.SG_resp

    ci_response_exp = ci_response_exp.interp(logT=sg_response.logT, method=method)

    ci_response_exp = ci_response_exp.assign_coords(slit=sg_response.slit.data)
    ci_response_exp = ci_response_exp.assign_coords(SG_xpixel=sg_response.SG_xpixel.data)
    ci_response_exp.coords["SG_wvl"] = xr.DataArray(
        np.reshape(sg_response.SG_wvl.isel(line=0).to_numpy(), (1, nslits, nsg_xpixels)),
        dims={"line": 1, "slit": nslits, "SG_xpixel": nsg_xpixels},
    )

    ci_response_exp["line_wvl"] = sg_response.line_wvl
    ci_response_exp["line_wvl"].loc[{"line": ci_response_exp.isel(line=0).coords["line"]}] = ci_response.channel.values[0]
    ci_response_exp["line"] = ci_response.channel.values
    ci_response_exp["logT"] = sg_response.logT  # Danger, the logT needs to be interpolated
    add_history(ci_response_exp, locals(), ci2sg_vdop_response_format)

    return ci_response_exp


def sum_lines_per_channel(spec: xr.Dataset):
    """
    Sums all the lines within a channel for the spectrum or response functions.

    Parameters
    ----------
    spec : `xarray.Dataset`
        Spectrum or response function.

    Returns
    -------
    `xarray.Dataset`, `xarray.Dataset`
        xarray with all the spectral lines added together per channel.
    """
    spec_temp = spec.copy(deep=True)
    if "channel" not in spec.dims:
        logger.info("Swapping lines with channel was required")
        spec_temp = spec_temp.swap_dims({"line": "channel"})
    if "gain" in spec_temp.coords:
        gain_list = []

    channel_list = []
    line_wvl_list = []
    for kk, k in enumerate(spec_temp.channel):
        if k not in channel_list:
            line_wvl_list.append(spec_temp.line_wvl[kk])
            if kk == 0:
                SG_wvl_list = spec_temp.SG_wvl.isel(channel=kk)
            else:
                SG_wvl_list = xr.concat(
                    [SG_wvl_list, spec_temp.SG_wvl.isel(channel=kk)], dim="channel", coords="different", compat="equals"
                )
            channel_list.append(k.data)
            if "gain" in spec_temp.coords:
                gain_list.append(spec_temp.gain[kk])

    for ik, k in enumerate(channel_list):
        if ik == 0:
            try:
                specsum = spec_temp.sel(channel=k).mean(dim=["channel"])
            except Exception:  # NOQA: BLE001
                specsum = spec_temp.sel(channel=k)

        elif spec_temp.sel(channel=k).coords["channel"].size > 1:
            specsum = xr.concat(
                [specsum, spec_temp.sel(channel=k).mean(dim=["channel"])], dim="channel", coords="different", compat="equals"
            )
        else:
            temp = spec_temp.sel(channel=k).expand_dims(channel=1)
            if "line" in temp:
                temp = temp.drop_vars("line")
            if "gain" in temp:
                temp = temp.drop_vars("gain")
            if "channel" not in specsum.dims:
                specsum = specsum.expand_dims(channel=1)
            if "channel" in specsum.coords:
                specsum = specsum.drop_vars("channel")
            if "channel" in temp.coords:
                temp = temp.drop_vars("channel")
            specsum = xr.concat([specsum, temp], dim="channel", coords="different", compat="equals")

    specsum = specsum.assign_coords(channel=("channel", channel_list))
    specsum = specsum.assign_coords(line_wvl=("channel", line_wvl_list))
    specsum = specsum.assign_coords(SG_wvl=SG_wvl_list)
    if "gain" in spec.coords:
        specsum = specsum.assign_coords(gain=("channel", gain_list))

    for k in channel_list:
        if spec_temp.sel(channel=k).coords["channel"].size > 1:
            if "SG_resp" in specsum:
                specsum["SG_resp"].loc[{"channel": k}] = (
                    specsum.SG_resp.sel(channel=k) * spec_temp.sel(channel=k).coords["channel"].size
                )
            if "flux" in specsum:
                specsum["flux"].loc[{"channel": k}] = (
                    specsum.flux.sel(channel=k) * spec_temp.sel(channel=k).coords["channel"].size
                )
            if "SG_wvl" in specsum:
                specsum["SG_wvl"].loc[{"channel": k}] = spec_temp.SG_wvl.sel(channel=k)[0]
    if "line_wvl" in specsum.data_vars:
        specsum.line_wvl.attrs = spec_temp.line_wvl.attrs
    if "flux" in specsum.data_vars:
        specsum.flux.attrs.update(spec_temp.flux.attrs)
    if "SG_resp" in specsum.data_vars:
        specsum.SG_resp.attrs.update(spec_temp.SG_resp.attrs)
    if "x" in specsum.dims:
        specsum.x.attrs.update(spec_temp.x.attrs)
    if "y" in specsum.dims:
        specsum.y.attrs.update(spec_temp.y.attrs)
    if "SG_wvl" in specsum.dims:
        specsum.SG_wvl.attrs.update(spec_temp.SG_wvl.attrs)
    specsum.attrs.update(spec_temp.attrs)

    add_history(specsum, locals(), sum_lines_per_channel)
    return specsum


def sum_lines_slits_per_channel(spec: xr.Dataset):
    """
    Sums all the lines and slits within a channel for the spectrum or response
    functions.

    Parameters
    ----------
    spec : `xarray.Dataset`
        Spectrum or response function.

    Returns
    -------
    `xarray.Dataset`, `xarray.Dataset`
        xarray with all the spectral lines added together per channel and slits.
    """
    if "channel" not in spec.dims:
        spec = spec.swap_dims({"line": "channel"})

    channel_list = []
    line_wvl_list = []
    if "gain" in spec.coords:
        gain_list = []

    for kk, k in enumerate(spec.channel):
        if k not in channel_list:
            channel_list.append(k.data)
            line_wvl_list.append(spec.line_wvl[kk])
            if "gain" in spec.coords:
                gain_list.append(spec.gain[kk])

    for ik, k in enumerate(channel_list):
        if ik == 0:
            if "slit" not in spec.dims:
                specsum = (
                    spec.sel(channel=k).sum(dim=["channel"])
                    if spec.sel(channel=k).coords["channel"].size > 1
                    else spec.sel(channel=k)
                )
            elif spec.sel(channel=k).coords["channel"].size > 1:
                specsum = spec.sel(channel=k).sum(dim=["slit", "channel"])
            else:
                specsum = spec.sel(channel=k).sum(dim=["slit"])

        elif "slit" in spec.dims:
            if spec.sel(channel=k).coords["channel"].size > 1:
                specsum = xr.concat(
                    [specsum, spec.sel(channel=k).sum(dim=["slit", "channel"])],
                    dim="channel",
                    coords="different",
                    compat="equals",
                )
            else:
                temp = spec.sel(channel=k).sum(dim=["slit"]).expand_dims(channel=1)
                if "line" in temp:
                    temp = temp.drop_vars("line")
                if "gain" in temp:
                    temp = temp.drop_vars("gain")
                if "channel" not in specsum.dims:
                    specsum = specsum.expand_dims(channel=1)
                if "channel" in specsum.coords:
                    specsum = specsum.drop_vars("channel")
                if "channel" in temp.coords:
                    temp = temp.drop_vars("channel")
                specsum = xr.concat([specsum, temp], dim="channel", coords="different", compat="equals")
        elif spec.sel(channel=k).coords["channel"].size > 1:
            specsum = xr.concat(
                [specsum, spec.sel(channel=k).sum(dim=["channel"])], dim="channel", coords="different", compat="equals"
            )
        else:
            temp = spec.sel(channel=k).expand_dims(channel=1)
            if "line" in temp:
                temp = temp.drop_vars("line")
            if "gain" in temp:
                temp = temp.drop_vars("gain")
            if "channel" in temp:
                temp = temp.drop_vars("channel")
            specsum = xr.concat([specsum, temp], dim="channel", coords="different", compat="equals")

    specsum = specsum.assign_coords(channel=("channel", channel_list))
    specsum = specsum.assign_coords(line_wvl=("channel", line_wvl_list))

    if "gain" in spec.coords:
        specsum = specsum.assign_coords(gain=("channel", gain_list))
    if "SG_wvl" in specsum.dims:
        specsum = specsum.drop("SG_wvl")
    if "SG_wvl" in specsum:
        specsum = specsum.drop_vars("SG_wvl")

    specsum.attrs.update(spec.attrs)
    if "flux" in specsum.data_vars:
        specsum.flux.attrs.update(spec.flux.attrs)
    if "SG_resp" in specsum.data_vars:
        specsum.SG_resp.attrs.update(spec.SG_resp.attrs)
    if "line_wvl" in specsum.data_vars:
        specsum.line_wvl.attrs.update(spec.line_wvl.attrs)
    if "x" in specsum.dims:
        specsum.x.attrs.update(spec.x.attrs)
    if "y" in specsum.dims:
        specsum.y.attrs.update(spec.y.attrs)

    add_history(specsum, locals(), sum_lines_slits_per_channel)
    return specsum


def blooming(image0, fwd=120000):
    """
    Simulate blooming in a 2D image along the y-axis for a given full well
    depth (FWD).

    Parameters
    ----------
    image0 : `np.ndarray`
        2D image.
    fwd : `float`, optional
        The data value where blooming starts (full well depth). Default is 120000.

    Returns
    -------
    `np.ndarray`
        The bloomed image.
    `np.ndarray`
        Boolean map indicating bloomed pixels.
    """
    image = image0.T.copy()

    # aia ccd has max 290000
    # sdl ccd has max 120000.

    ncol, ny = np.shape(image)
    bloommap = image * 0.0

    for i in range(ncol):
        excessbank = 0

        while np.max(image[i, :]) > fwd:
            # find the peak intensity along the column
            m = np.argmax(image[i, :])

            # at this point, note that this pixel is impacted by blooming, calculate the excess electrons, including any left over from
            # the last calculation, and set the intensity in this pixel of the fwd.
            bloommap[i, m] = 1
            excess = image[i, m] - fwd + excessbank
            excessbank = 0
            image[i, m] = fwd

            # have we hit the edge of the detector?  If so, just move on.
            if (m + 1 == ny) or (m - 1 == 0):
                excessbank = excess
            # are the pixels above/below able to receive charge?  If so, split the charge between them both
            elif ((image[i, m - 1] < fwd) and (image[i, m + 1] < fwd)) or ((image[i, m - 1] > fwd) and (image[i, m + 1] > fwd)):
                image[i, m - 1] = image[i, m - 1] + round(excess / 2.0)
                image[i, m + 1] = image[i, m + 1] + round(excess / 2.0)
                bloommap[i, m + 1] = 1
                bloommap[i, m - 1] = 1
            # Is only one pixel able to receive charge?  If so, it gets all the charge
            elif ((image[i, m - 1] < fwd) and (image[i, m + 1] >= fwd)) or ((image[i, m - 1] > fwd) and (image[i, m + 1] == fwd)):
                image[i, m - 1] = image[i, m - 1] + excess
                bloommap[i, m - 1] = 1
            elif ((image[i, m + 1] < fwd) and (image[i, m - 1] >= fwd)) or ((image[i, m + 1] > fwd) and (image[i, m - 1] == fwd)):
                image[i, m + 1] = image[i, m + 1] + excess
                bloommap[i, m + 1] = 1
            # if both pixels above and below have already been fixed to fwd, just add the excess to the bank and try to deal with it next time
            elif (image[i, m - 1] == fwd) and (image[i, m + 1] == fwd):
                excessbank = excess
            else:
                logger.warning("oh! No...")

    return image.T, bloommap.T


def blooming_ci_xr(ci_obsdata, axis="y", fwd=120000, channel=None):
    """
    Bloom a 2D image for a full well depth (FWD) for CI data.

    Parameters
    ----------
    ci_obsdata : `xr.Dataset`
        Observed data with CI.
    axis : `str`, optional
        Axis to apply blooming, either "x" or "y". Default is "y".
    fwd : `float`, optional
        The data value where blooming starts (full well depth). Default is 120000.
    channel : `list` or `None`, optional
        List of channels to apply blooming.

    Returns
    -------
    `xr.Dataset`
        The bloomed data.
    """
    ci_bloomed = ci_obsdata.copy(deep=True)
    for ichannel in channel:
        if axis == "y":
            bloomed = blooming(ci_obsdata.flux.sel(channel=ichannel).isel(SG_xpixel=0).data.T, fwd=fwd)
            ci_bloomed["flux"].loc[{"channel": ichannel, "SG_xpixel": 0}] = bloomed[0].T
        else:
            bloomed = blooming(ci_obsdata.flux.isel(channel=ichannel).isel(SG_xpixel=0).data, fwd=fwd)
            ci_bloomed["flux"].loc[{"channel": ichannel, "SG_xpixel": 0}] = bloomed[0]
    return ci_bloomed
