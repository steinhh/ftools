"""
fmpfit: Levenberg-Marquardt least-squares minimization

Python wrapper for MPFIT (MINPACK-1 Least Squares Fitting Library in C)
Provides efficient curve fitting with parameter constraints.
"""

import os
import numpy as np
from . import fmpfit_f64_ext
from . import fmpfit_f32_ext
from . import fmpfit_f64_block_ext
from . import fmpfit_f32_block_ext


def get_include():
    """Return the directory containing mpfit header and source files.
    
    This allows other packages to compile C extensions that use mpfit.
    
    Returns
    -------
    str
        Path to directory containing cmpfit-1.5/ subdirectory with mpfit.h and mpfit.c
    
    Example
    -------
    In another package's setup.py:
    
        from ftoolss.fmpfit import get_include
        
        Extension(
            "mypackage.myext",
            sources=["myext.c", os.path.join(get_include(), "cmpfit-1.5", "mpfit.c")],
            include_dirs=[get_include()],
        )
    """
    return os.path.dirname(__file__)


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
        Parameter uncertainties (1-sigma, unscaled - assumes input errors are correct)
    
    xerror_scipy : ndarray
        Parameter uncertainties computed using full Hessian inverse (scipy-style).
        This is directly comparable to scipy.optimize.curve_fit errors, even when
        parameters hit their bounds. Computed internally by mpfit using the
        user-provided deviate callback function.
    covar : ndarray
        Covariance matrix (npar x npar)
    c_time : float
        Time spent in C extension (seconds)
    """
    def __init__(self, best_params, bestnorm, orignorm, niter, nfev, status,     # NOSONAR
                 npar, nfree, npegged, nfunc, resid, xerror,                    # NOSONAR
                 xerror_scipy, covar, c_time=0.0):                              # NOSONAR
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
        self.xerror_scipy = xerror_scipy
        self.covar = covar
        self.c_time = c_time
    
    def __repr__(self):
        return (f"MPFitResult(status={self.status}, niter={self.niter}, "
                f"bestnorm={self.bestnorm:.6e})")


def fmpfit_f64_pywrap(deviate_type, parinfo=None, functkw=None, #NOSONAR
                      xtol=1.0e-6, ftol=1.0e-6, gtol=1.0e-6, 
                      maxiter=2000, quiet=1):
    """
    Levenberg-Marquardt least-squares minimization (float64)
    
    Uses analytical derivatives (Jacobian) for the Gaussian model internally,
    which provides faster and more accurate convergence than finite differences.
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian (uses analytical derivatives)
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
    >>> from ftoolss.fmpfit import fmpfit_f64_pywrap
    >>> x = np.linspace(-5, 5, 100)
    >>> y = 2.5 * np.exp(-0.5*((x-1.0)/0.8)**2) + np.random.normal(0, 0.1, 100)
    >>> error = np.ones_like(y) * 0.1
    >>> parinfo = [
    ...     {'value': 1.0, 'limits': [0.0, 10.0]},   # amplitude
    ...     {'value': 0.0, 'limits': [-5.0, 5.0]},   # mean
    ...     {'value': 1.0, 'limits': [0.1, 5.0]}     # sigma
    ... ]
    >>> functkw = {'x': x, 'y': y, 'error': error}
    >>> result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
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
    
    # Call C extension with timing (mpoints/npar inferred from array shapes)
    import time
    t_start = time.perf_counter()
    result_dict = fmpfit_f64_ext.fmpfit_f64(
        x, y, error, p0, bounds,
        int(deviate_type),
        float(xtol), float(ftol), float(gtol),
        int(maxiter), int(quiet)
    )
    t_end = time.perf_counter()
    c_time = t_end - t_start
    
    # Create result object (f64 wrapper)
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
        xerror_scipy=result_dict['xerror_scipy'],
        covar=result_dict['covar'],
        c_time=c_time
    )


def fmpfit_f32_pywrap(deviate_type, parinfo=None, functkw=None, xtol=1e-6, #NOSONAR
                      ftol=1e-6, gtol=1e-6, maxiter=2000, quiet=True):
    """MPFIT wrapper function for float32 (single precision) fitting.
    Uses analytical derivatives (Jacobian) for the Gaussian model internally,
    which provides faster and more accurate convergence than finite differences.

    Same as fmpfit_f64_pywrap but uses float32 precision internally for faster computation
    and lower memory usage.
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian (uses analytical derivatives)
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
        Relative tolerance in parameter values (default: 1e-10)
    ftol : float, optional
        Relative tolerance in chi-square (default: 1e-10)
    gtol : float, optional
        Orthogonality tolerance (default: 1e-10)
    maxiter : int, optional
        Maximum iterations (default: 200)
    quiet : int, optional
        Suppress output: 1=quiet, 0=verbose (default: 1)
    
    Returns
    -------
    MPFitResult
        Object containing fit results and diagnostics
    """
    # Validate inputs
    if functkw is None:
        raise ValueError("functkw must be provided")
    if parinfo is None:
        raise ValueError("parinfo must be provided")
    
    # Extract data from functkw
    if 'x' not in functkw or 'y' not in functkw or 'error' not in functkw:
        raise ValueError("functkw must contain 'x', 'y', and 'error'")
    
    x = np.asarray(functkw['x'], dtype=np.float32)
    y = np.asarray(functkw['y'], dtype=np.float32)
    error = np.asarray(functkw['error'], dtype=np.float32)
    
    # Validate data shapes
    if x.ndim != 1 or y.ndim != 1 or error.ndim != 1:
        raise ValueError("x, y, and error must be 1D arrays")
    if len(x) != len(y) or len(x) != len(error):
        raise ValueError("x, y, and error must have the same length")
    
    npar = len(parinfo)
    
    # Extract initial parameter values and bounds
    p0 = np.zeros(npar, dtype=np.float32)
    bounds = np.zeros((npar, 2), dtype=np.float32)
    
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
    
    # Call C extension with timing (mpoints/npar inferred from array shapes)
    import time
    t_start = time.perf_counter()
    result_dict = fmpfit_f32_ext.fmpfit_f32(
        x, y, error, p0, bounds,
        int(deviate_type),
        float(xtol), float(ftol), float(gtol),
        int(maxiter), int(quiet)
    )
    t_end = time.perf_counter()
    c_time = t_end - t_start
    
    # Create result object (f32 wrapper)
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
        xerror_scipy=result_dict['xerror_scipy'],
        covar=result_dict['covar'],
        c_time=c_time
    )


def fmpfit_f64_block_pywrap(deviate_type, x, y, error, p0, bounds, #NOSONAR
                            xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000, quiet=1):
    """
    Levenberg-Marquardt least-squares minimization for multiple spectra (float64)
    
    Fits multiple spectra independently in a single call, using analytical derivatives
    (Jacobian) for the Gaussian model internally.
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian (uses analytical derivatives)
    x : ndarray, shape (n_spectra, n_data_points)
        Independent variable for each spectrum. Data points are contiguous per spectrum.
    y : ndarray, shape (n_spectra, n_data_points)
        Dependent variable (measured values) for each spectrum.
    error : ndarray, shape (n_spectra, n_data_points)
        Measurement uncertainties for each spectrum.
    p0 : ndarray, shape (n_spectra, n_params)
        Initial parameter guesses for each spectrum.
    bounds : ndarray, shape (n_spectra, n_params, 2)
        Parameter bounds [min, max] for each parameter of each spectrum.
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
    dict
        Dictionary with fit results for all spectra:
        - 'best_params': shape (n_spectra, n_params) - best-fit parameters
        - 'bestnorm': shape (n_spectra,) - final chi-square values
        - 'orignorm': shape (n_spectra,) - initial chi-square values
        - 'niter': shape (n_spectra,) - iterations performed
        - 'nfev': shape (n_spectra,) - function evaluations
        - 'status': shape (n_spectra,) - status codes
        - 'npar': shape (n_spectra,) - number of parameters
        - 'nfree': shape (n_spectra,) - number of free parameters
        - 'npegged': shape (n_spectra,) - number of pegged parameters
        - 'nfunc': shape (n_spectra,) - number of data points
        - 'resid': shape (n_spectra, n_data_points) - final residuals
        - 'xerror': shape (n_spectra, n_params) - parameter uncertainties (unscaled)
        - 'xerror_scipy': shape (n_spectra, n_params) - uncertainties using full Hessian (scipy-style)
        - 'covar': shape (n_spectra, n_params, n_params) - covariance matrices
    
    Examples
    --------
    >>> import numpy as np
    >>> from ftoolss.fmpfit import fmpfit_f64_block_pywrap
    >>> n_spectra, n_points, n_params = 100, 5, 3
    >>> x = np.tile(np.linspace(-2, 2, n_points), (n_spectra, 1))
    >>> # ... generate y, error, p0, bounds ...
    >>> result = fmpfit_f64_block_pywrap(0, x, y, error, p0, bounds)
    >>> print(result['best_params'].shape)  # (100, 3)
    """
    # Convert to contiguous float64 arrays
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    error = np.ascontiguousarray(error, dtype=np.float64)
    p0 = np.ascontiguousarray(p0, dtype=np.float64)
    bounds = np.ascontiguousarray(bounds, dtype=np.float64)
    
    # Validate shapes
    if x.ndim != 2 or y.ndim != 2 or error.ndim != 2:
        raise ValueError("x, y, and error must be 2D arrays with shape (n_spectra, n_data_points)")
    if p0.ndim != 2:
        raise ValueError("p0 must be a 2D array with shape (n_spectra, n_params)")
    if bounds.ndim != 3:
        raise ValueError("bounds must be a 3D array with shape (n_spectra, n_params, 2)")
    
    n_spectra, mpoints = x.shape
    _, npar = p0.shape
    
    if y.shape != (n_spectra, mpoints) or error.shape != (n_spectra, mpoints):
        raise ValueError("x, y, and error must have the same shape")
    if p0.shape[0] != n_spectra:
        raise ValueError("p0 must have n_spectra rows")
    if bounds.shape != (n_spectra, npar, 2):
        raise ValueError("bounds must have shape (n_spectra, n_params, 2)")
    
    # Call C extension (dimensions inferred from array shapes)
    import time
    t_start = time.perf_counter()
    result_dict = fmpfit_f64_block_ext.fmpfit_f64_block(
        x, y, error, p0, bounds,
        int(deviate_type),
        float(xtol), float(ftol), float(gtol),
        int(maxiter), int(quiet)
    )
    t_end = time.perf_counter()
    result_dict['c_time'] = t_end - t_start
    
    return result_dict


def fmpfit_f32_block_pywrap(deviate_type, x, y, error, p0, bounds, #NOSONAR
                            xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000, quiet=1):
    """
    Levenberg-Marquardt least-squares minimization for multiple spectra (float32)
    
    Fits multiple spectra independently in a single call, using analytical derivatives
    (Jacobian) for the Gaussian model internally. Uses single precision for faster
    computation and lower memory usage.
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian (uses analytical derivatives)
    x : ndarray, shape (n_spectra, n_data_points)
        Independent variable for each spectrum. Data points are contiguous per spectrum.
    y : ndarray, shape (n_spectra, n_data_points)
        Dependent variable (measured values) for each spectrum.
    error : ndarray, shape (n_spectra, n_data_points)
        Measurement uncertainties for each spectrum.
    p0 : ndarray, shape (n_spectra, n_params)
        Initial parameter guesses for each spectrum.
    bounds : ndarray, shape (n_spectra, n_params, 2)
        Parameter bounds [min, max] for each parameter of each spectrum.
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
    dict
        Dictionary with fit results for all spectra (same as fmpfit_f64_block_pywrap)
    """
    # Convert to contiguous float32 arrays
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    error = np.ascontiguousarray(error, dtype=np.float32)
    p0 = np.ascontiguousarray(p0, dtype=np.float32)
    bounds = np.ascontiguousarray(bounds, dtype=np.float32)
    
    # Validate shapes
    if x.ndim != 2 or y.ndim != 2 or error.ndim != 2:
        raise ValueError("x, y, and error must be 2D arrays with shape (n_spectra, n_data_points)")
    if p0.ndim != 2:
        raise ValueError("p0 must be a 2D array with shape (n_spectra, n_params)")
    if bounds.ndim != 3:
        raise ValueError("bounds must be a 3D array with shape (n_spectra, n_params, 2)")
    
    n_spectra, mpoints = x.shape
    _, npar = p0.shape
    
    if y.shape != (n_spectra, mpoints) or error.shape != (n_spectra, mpoints):
        raise ValueError("x, y, and error must have the same shape")
    if p0.shape[0] != n_spectra:
        raise ValueError("p0 must have n_spectra rows")
    if bounds.shape != (n_spectra, npar, 2):
        raise ValueError("bounds must have shape (n_spectra, n_params, 2)")
    
    # Call C extension (dimensions inferred from array shapes)
    import time
    t_start = time.perf_counter()
    result_dict = fmpfit_f32_block_ext.fmpfit_f32_block(
        x, y, error, p0, bounds,
        int(deviate_type),
        float(xtol), float(ftol), float(gtol),
        int(maxiter), int(quiet)
    )
    t_end = time.perf_counter()
    result_dict['c_time'] = t_end - t_start
    
    return result_dict


def fmpfit_block_pywrap(deviate_type, x, y, error, p0, bounds, dtype=None, #NOSONAR
                        xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000, quiet=1):
    """
    Unified Levenberg-Marquardt least-squares minimization for multiple spectra
    
    Fits multiple spectra independently in a single call, using analytical derivatives
    (Jacobian) for the Gaussian model internally. Automatically selects float32 or
    float64 precision based on input data dtype or explicit dtype parameter.
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian (uses analytical derivatives)
    x : ndarray, shape (n_spectra, n_data_points)
        Independent variable for each spectrum.
    y : ndarray, shape (n_spectra, n_data_points)
        Dependent variable (measured values) for each spectrum.
    error : ndarray, shape (n_spectra, n_data_points)
        Measurement uncertainties for each spectrum.
    p0 : ndarray, shape (n_spectra, n_params)
        Initial parameter guesses for each spectrum.
    bounds : ndarray, shape (n_spectra, n_params, 2)
        Parameter bounds [min, max] for each parameter of each spectrum.
    dtype : numpy dtype, optional
        Force specific precision: np.float32 or np.float64.
        If None (default), infers from input 'y' array dtype.
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
    dict
        Dictionary with fit results for all spectra (same as fmpfit_f64_block_pywrap)
    
    Examples
    --------
    >>> import numpy as np
    >>> from ftoolss.fmpfit import fmpfit_block_pywrap
    >>> n_spectra, n_points, n_params = 100, 5, 3
    >>> x = np.tile(np.linspace(-2, 2, n_points), (n_spectra, 1))
    >>> # ... generate y, error, p0, bounds ...
    >>> result = fmpfit_block_pywrap(0, x, y, error, p0, bounds)  # auto-select dtype
    >>> result32 = fmpfit_block_pywrap(0, x, y, error, p0, bounds, dtype=np.float32)
    """
    # Determine dtype from input or explicit parameter
    if dtype is None:
        y_arr = np.asarray(y)
        if y_arr.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64  # Default to float64 for all other types
    
    # Dispatch to appropriate wrapper
    if dtype == np.float32:
        return fmpfit_f32_block_pywrap(deviate_type, x, y, error, p0, bounds,
                                       xtol=xtol, ftol=ftol, gtol=gtol,
                                       maxiter=maxiter, quiet=quiet)
    else:
        return fmpfit_f64_block_pywrap(deviate_type, x, y, error, p0, bounds,
                                       xtol=xtol, ftol=ftol, gtol=gtol,
                                       maxiter=maxiter, quiet=quiet)


def fmpfit_pywrap(deviate_type, parinfo=None, functkw=None, dtype=None, #NOSONAR
                  xtol=1e-6, ftol=1e-6, gtol=1e-6, maxiter=2000, quiet=1):
    """
    Unified Levenberg-Marquardt least-squares minimization (auto dtype selection)
    
    Uses analytical derivatives (Jacobian) for the Gaussian model internally,
    which provides faster and more accurate convergence than finite differences.
    
    Automatically selects float32 or float64 precision based on input data dtype
    or explicit dtype parameter.
    
    Parameters
    ----------
    deviate_type : int
        Model type: 0 = Gaussian (uses analytical derivatives)
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
    dtype : numpy dtype, optional
        Force specific precision: np.float32 or np.float64.
        If None (default), infers from input 'y' array dtype.
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
    >>> from ftoolss.fmpfit import fmpfit_pywrap
    >>> x = np.linspace(-5, 5, 100)
    >>> y = 2.5 * np.exp(-0.5*((x-1.0)/0.8)**2) + np.random.normal(0, 0.1, 100)
    >>> error = np.ones_like(y) * 0.1
    >>> parinfo = [
    ...     {'value': 1.0, 'limits': [0.0, 10.0]},   # amplitude
    ...     {'value': 0.0, 'limits': [-5.0, 5.0]},   # mean
    ...     {'value': 1.0, 'limits': [0.1, 5.0]}     # sigma
    ... ]
    >>> functkw = {'x': x, 'y': y, 'error': error}
    >>> # Auto-select dtype from input
    >>> result = fmpfit_pywrap(0, parinfo=parinfo, functkw=functkw)
    >>> # Force float32 precision
    >>> result32 = fmpfit_pywrap(0, parinfo=parinfo, functkw=functkw, dtype=np.float32)
    """
    # Validate inputs
    if functkw is None:
        raise ValueError("functkw must be provided")
    if parinfo is None:
        raise ValueError("parinfo must be provided")
    if 'y' not in functkw:
        raise ValueError("functkw must contain 'y'")
    
    # Determine dtype from input or explicit parameter
    if dtype is None:
        y_arr = np.asarray(functkw['y'])
        if y_arr.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64  # Default to float64 for all other types
    
    # Dispatch to appropriate wrapper
    if dtype == np.float32:
        return fmpfit_f32_pywrap(deviate_type, parinfo=parinfo, functkw=functkw,
                                 xtol=xtol, ftol=ftol, gtol=gtol,
                                 maxiter=maxiter, quiet=quiet)
    else:
        return fmpfit_f64_pywrap(deviate_type, parinfo=parinfo, functkw=functkw,
                                 xtol=xtol, ftol=ftol, gtol=gtol,
                                 maxiter=maxiter, quiet=quiet)


__all__ = ['fmpfit_pywrap', 'fmpfit_f64_pywrap', 'fmpfit_f32_pywrap', 
           'fmpfit_block_pywrap', 'fmpfit_f64_block_pywrap', 'fmpfit_f32_block_pywrap',
           'MPFitResult']
