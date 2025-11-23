#!/usr/bin/env python
"""
Unit tests for fmpfit extension

Tests the Python wrapper and C extension interface.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ftools import fmpfit
from ftools.fmpfit import MPFitResult


def test_fmpfit_import():
    """Test that fmpfit can be imported"""
    assert fmpfit is not None
    assert callable(fmpfit)
    assert MPFitResult is not None


def test_fmpfit_basic_call():
    """Test basic fmpfit call with valid inputs"""
    x = np.linspace(-5, 5, 50, dtype=np.float64)
    y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5)**2)
    error = np.ones_like(y) * 0.1
    
    parinfo = [
        {'value': 1.0, 'limits': [0.0, 10.0]},
        {'value': 0.0, 'limits': [-5.0, 5.0]},
        {'value': 1.0, 'limits': [0.1, 5.0]}
    ]
    
    functkw = {'x': x, 'y': y, 'error': error}
    
    result = fmpfit(0, parinfo=parinfo, functkw=functkw)
    
    # Check result object attributes
    assert hasattr(result, 'best_params')
    assert hasattr(result, 'bestnorm')
    assert hasattr(result, 'orignorm')
    assert hasattr(result, 'niter')
    assert hasattr(result, 'nfev')
    assert hasattr(result, 'status')
    assert hasattr(result, 'npar')
    assert hasattr(result, 'nfree')
    assert hasattr(result, 'npegged')
    assert hasattr(result, 'nfunc')
    assert hasattr(result, 'resid')
    assert hasattr(result, 'xerror')
    assert hasattr(result, 'covar')


def test_fmpfit_result_shapes():
    """Test that result arrays have correct shapes"""
    x = np.linspace(-5, 5, 50, dtype=np.float64)
    y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5)**2)
    error = np.ones_like(y) * 0.1
    
    npar = 3
    parinfo = [
        {'value': 1.0, 'limits': [0.0, 10.0]},
        {'value': 0.0, 'limits': [-5.0, 5.0]},
        {'value': 1.0, 'limits': [0.1, 5.0]}
    ]
    
    functkw = {'x': x, 'y': y, 'error': error}
    
    result = fmpfit(0, parinfo=parinfo, functkw=functkw)
    
    assert result.best_params.shape == (npar,)
    assert result.xerror.shape == (npar,)
    assert result.covar.shape == (npar, npar)
    assert result.resid.shape == (len(x),)
    assert result.npar == npar
    assert result.nfunc == len(x)


def test_fmpfit_missing_functkw():
    """Test that missing functkw raises ValueError"""
    parinfo = [{'value': 1.0, 'limits': [0.0, 10.0]}]
    
    with pytest.raises(ValueError, match="functkw must be provided"):
        fmpfit(0, parinfo=parinfo)


def test_fmpfit_missing_parinfo():
    """Test that missing parinfo raises ValueError"""
    x = np.linspace(-5, 5, 50)
    y = np.ones_like(x)
    error = np.ones_like(y) * 0.1
    functkw = {'x': x, 'y': y, 'error': error}
    
    with pytest.raises(ValueError, match="parinfo must be provided"):
        fmpfit(0, functkw=functkw)


def test_fmpfit_incomplete_functkw():
    """Test that incomplete functkw raises ValueError"""
    x = np.linspace(-5, 5, 50)
    y = np.ones_like(x)
    parinfo = [{'value': 1.0, 'limits': [0.0, 10.0]}]
    
    # Missing 'error'
    functkw = {'x': x, 'y': y}
    with pytest.raises(ValueError, match="functkw must contain"):
        fmpfit(0, parinfo=parinfo, functkw=functkw)


def test_fmpfit_array_length_mismatch():
    """Test that mismatched array lengths raise ValueError"""
    x = np.linspace(-5, 5, 50)
    y = np.ones(40)  # Different length
    error = np.ones(50) * 0.1
    parinfo = [{'value': 1.0, 'limits': [0.0, 10.0]}]
    functkw = {'x': x, 'y': y, 'error': error}
    
    with pytest.raises(ValueError, match="must have the same length"):
        fmpfit(0, parinfo=parinfo, functkw=functkw)


def test_fmpfit_invalid_bounds():
    """Test that invalid bounds raise ValueError"""
    x = np.linspace(-5, 5, 50)
    y = np.ones_like(x)
    error = np.ones_like(y) * 0.1
    
    # Lower bound >= upper bound
    parinfo = [{'value': 1.0, 'limits': [10.0, 5.0]}]
    functkw = {'x': x, 'y': y, 'error': error}
    
    with pytest.raises(ValueError, match="lower bound must be < upper bound"):
        fmpfit(0, parinfo=parinfo, functkw=functkw)


def test_fmpfit_result_repr():
    """Test MPFitResult __repr__ method"""
    x = np.linspace(-5, 5, 50)
    y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5)**2)
    error = np.ones_like(y) * 0.1
    
    parinfo = [
        {'value': 1.0, 'limits': [0.0, 10.0]},
        {'value': 0.0, 'limits': [-5.0, 5.0]},
        {'value': 1.0, 'limits': [0.1, 5.0]}
    ]
    
    functkw = {'x': x, 'y': y, 'error': error}
    result = fmpfit(0, parinfo=parinfo, functkw=functkw)
    
    repr_str = repr(result)
    assert 'MPFitResult' in repr_str
    assert 'status' in repr_str
    assert 'niter' in repr_str
    assert 'bestnorm' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
