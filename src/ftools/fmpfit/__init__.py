"""
fmpfit: Levenberg-Marquardt least-squares minimization

Python wrapper for MPFIT (MINPACK-1 Least Squares Fitting Library in C)
Provides efficient curve fitting with parameter constraints.
"""

import numpy as np
from . import fmpfit_ext


class MPFitResult:
    """
    Result object returned by fmpfit_py()
    
    Attributes
    ----------
    best_params : ndarray
        Best-fit parameter values
    bestnorm : float
        Final chi-square value
    orignorm : float
        Starting chi-square value
    niter : int
        Number of iterations performed
    nfev : int
        Number of function evaluations
    status : int
        Status code (0 = success)
    npar : int
        Number of parameters
    nfree : int
        Number of free parameters
    npegged : int
        Number of pegged parameters
    nfunc : int
        Number of data points
    resid : ndarray
        Final residuals
    xerror : ndarray
        Parameter uncertainties (1-sigma)
    covar : ndarray
        Covariance matrix (npar x npar)
    """
    def __init__(self, best_params, bestnorm, orignorm, niter, nfev, status,
                 npar, nfree, npegged, nfunc, resid, xerror, covar):
        self.best_params = best_params
        self.bestnorm = bestnorm
        self.orignorm = orignorm
        self.niter = niter
        self.nfev = nfev
        self.status = status
        self.npar = npar
        self.nfree = nfree
        self.npegged = npegged
        self.nfunc = nfunc
        self.resid = resid
        self.xerror = xerror
        self.covar = covar
    
    def __repr__(self):
        return (f"MPFitResult(status={self.status}, niter={self.niter}, "
                f"bestnorm={self.bestnorm:.6e})")


def fmpfit_py(deviate_type, parinfo=None, functkw=None, 
              xtol=1.0e-6, ftol=1.0e-6, gtol=1.0e-6, 
              maxiter=2000, quiet=1):
    """
    Levenberg-Marquardt least-squares minimization
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian
    parinfo : list of dict
        Parameter info, each dict contains:
        - 'value': float, initial parameter value
        - 'limits': [lower, upper], parameter bounds
        - 'fixed': int (optional), 1=fixed, 0=free (default)
    functkw : dict
        Function keywords containing:
        - 'x': ndarray, independent variable
        - 'y': ndarray, dependent variable
        - 'error': ndarray, measurement uncertainties
    xtol : float, optional
        Relative tolerance in parameter values (default: 1e-6)
    ftol : float, optional
        Relative tolerance in chi-square (default: 1e-6)
    gtol : float, optional
        Orthogonality tolerance (default: 1e-6)
    maxiter : int, optional
        Maximum iterations (default: 2000)
    quiet : int, optional
        Suppress output: 1=quiet, 0=verbose (default: 1)
    
    Returns
    -------
    MPFitResult
        Object containing fit results and diagnostics
    
    Examples
    --------
    >>> import numpy as np
    >>> from ftools import fmpfit
    >>> x = np.linspace(-5, 5, 100)
    >>> y = 2.5 * np.exp(-0.5*((x-1.0)/0.8)**2) + np.random.normal(0, 0.1, 100)
    >>> error = np.ones_like(y) * 0.1
    >>> parinfo = [
    ...     {'value': 1.0, 'limits': [0.0, 10.0]},   # amplitude
    ...     {'value': 0.0, 'limits': [-5.0, 5.0]},   # mean
    ...     {'value': 1.0, 'limits': [0.1, 5.0]}     # sigma
    ... ]
    >>> functkw = {'x': x, 'y': y, 'error': error}
    >>> result = fmpfit.fmpfit_py(0, parinfo=parinfo, functkw=functkw)
    >>> print(result.best_params)
    """
    # Validate inputs
    if functkw is None:
        raise ValueError("functkw must be provided")
    if parinfo is None:
        raise ValueError("parinfo must be provided")
    
    # Extract data from functkw
    if 'x' not in functkw or 'y' not in functkw or 'error' not in functkw:
        raise ValueError("functkw must contain 'x', 'y', and 'error'")
    
    x = np.asarray(functkw['x'], dtype=np.float64)
    y = np.asarray(functkw['y'], dtype=np.float64)
    error = np.asarray(functkw['error'], dtype=np.float64)
    
    # Validate data shapes
    if x.ndim != 1 or y.ndim != 1 or error.ndim != 1:
        raise ValueError("x, y, and error must be 1D arrays")
    if len(x) != len(y) or len(x) != len(error):
        raise ValueError("x, y, and error must have the same length")
    
    mpoints = len(x)
    npar = len(parinfo)
    
    # Extract initial parameter values and bounds
    p0 = np.zeros(npar, dtype=np.float64)
    bounds = np.zeros((npar, 2), dtype=np.float64)
    
    for i, pinfo in enumerate(parinfo):
        if 'value' not in pinfo:
            raise ValueError(f"parinfo[{i}] must contain 'value'")
        if 'limits' not in pinfo:
            raise ValueError(f"parinfo[{i}] must contain 'limits'")
        
        p0[i] = float(pinfo['value'])
        limits = pinfo['limits']
        if len(limits) != 2:
            raise ValueError(f"parinfo[{i}]['limits'] must have 2 elements")
        bounds[i, 0] = float(limits[0])
        bounds[i, 1] = float(limits[1])
        
        # Validate bounds
        if bounds[i, 0] >= bounds[i, 1]:
            raise ValueError(f"parinfo[{i}]: lower bound must be < upper bound")
    
    # Call C extension
    result_dict = fmpfit_ext.fmpfit(
        x, y, error, p0, bounds,
        int(mpoints), int(npar), int(deviate_type),
        float(xtol), float(ftol), float(gtol),
        int(maxiter), int(quiet)
    )
    
    # Create result object
    return MPFitResult(
        best_params=result_dict['best_params'],
        bestnorm=result_dict['bestnorm'],
        orignorm=result_dict['orignorm'],
        niter=result_dict['niter'],
        nfev=result_dict['nfev'],
        status=result_dict['status'],
        npar=result_dict['npar'],
        nfree=result_dict['nfree'],
        npegged=result_dict['npegged'],
        nfunc=result_dict['nfunc'],
        resid=result_dict['resid'],
        xerror=result_dict['xerror'],
        covar=result_dict['covar']
    )


__all__ = ['fmpfit_py', 'MPFitResult']
