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
        functkw = {'x': xdata, 'y': ydata, 'error': y_sigma}
        parinfo = [{'value':p0[i], 'fixed':0, 'limited':[1,1],
            'limits':[bounds[0][i],bounds[1][i]]} for i in range(len(p0))]

        mp_ = mpfit(deviates,parinfo=parinfo, functkw=functkw, xtol=1.0E-6, ftol=1.0E-6, gtol=1.0E-6,
                        maxiter=2000, quiet=1)
        popt = mp_.params
        dof = len(xdata) - len(p0)  # No. of observations  - no. of fitted params.
        chisquare = reduced_chisquare(ydata, fgaussian(xdata, *popt), y_sigma, dof)
        popt = np.append(popt, chisquare)
        popt = np.append(popt, 0)
    return popt
