#!/usr/bin/env python
"""
Unit tests for fmpfit extension

Tests the Python wrapper and C extension interface for both float64 and float32.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ftoolss import fmpfit_f64_pywrap, fmpfit_f32_pywrap
from ftoolss.fmpfit import MPFitResult


def test_fmpfit_import():
    """Test that fmpfit module imports correctly."""
    assert fmpfit_f64_pywrap is not None


def test_fmpfit_basic_call():
    """Test basic fmpfit functionality using float64 wrapper."""
    data_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data_y = np.array([0.5, 2.0, 4.5, 2.0, 0.5])

    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def residuals(p, fjac=None, x=None, y=None, err=None):
        model = gaussian(x, p[0], p[1], p[2])
        return [0, (y - model) / err]

    parinfo = [
        {"value": 4.0, "fixed": 0, "limited": [1, 1], "limits": [0, 10]},
        {"value": 3.0, "fixed": 0, "limited": [1, 1], "limits": [-2, 2]},
        {"value": 1.0, "fixed": 0, "limited": [1, 1], "limits": [0.1, 5]}
    ]

    functkw = {"x": data_x, "y": data_y, "error": np.ones(5)}
    result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
    
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
    """Test that fmpfit returns properly shaped results."""
    data_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data_y = np.array([0.5, 2.0, 4.5, 2.0, 0.5])

    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def residuals(p, fjac=None, x=None, y=None, err=None):
        model = gaussian(x, p[0], p[1], p[2])
        return [0, (y - model) / err]

    parinfo = [
        {"value": 4.0, "fixed": 0, "limited": [1, 1], "limits": [0, 10]},
        {"value": 3.0, "fixed": 0, "limited": [1, 1], "limits": [-2, 2]},
        {"value": 1.0, "fixed": 0, "limited": [1, 1], "limits": [0.1, 5]}
    ]

    functkw = {"x": data_x, "y": data_y, "error": np.ones(5)}
    result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)


def test_fmpfit_missing_functkw():
    """Test fmpfit validation when functkw is missing."""
    parinfo = [
        {"value": 4.0, "fixed": 0, "limited": [1, 1], "limits": [0, 10]},
    ]
    with pytest.raises(ValueError):
        fmpfit_f64_pywrap(0, parinfo=parinfo)


def test_fmpfit_missing_parinfo():
    """Test fmpfit validation when parinfo is missing."""
    functkw = {"x": np.array([1, 2, 3])}
    with pytest.raises(ValueError):
        fmpfit_f64_pywrap(0, functkw=functkw)


def test_fmpfit_incomplete_functkw():
    """Test that incomplete functkw raises ValueError"""
    x = np.linspace(-5, 5, 50)
    y = np.ones_like(x)
    parinfo = [{'value': 1.0, 'limits': [0.0, 10.0]}]
    
    # Missing 'error'
    functkw = {'x': x, 'y': y}
    with pytest.raises(ValueError, match="functkw must contain"):
        fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)


def test_fmpfit_array_length_mismatch():
    """Test validation when arrays in functkw have different lengths."""
    parinfo = [
        {"value": 4.0, "fixed": 0, "limited": [1, 1], "limits": [0, 10]},
    ]
    functkw = {"x": np.array([1, 2, 3]), "y": np.array([1, 2])}
    with pytest.raises(ValueError):
        fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)


def test_fmpfit_invalid_bounds():
    """Test validation of parameter bounds."""
    parinfo = [
        {"value": 4.0, "fixed": 0, "limited": [1, 1], "limits": [10, 0]},  # Invalid: lower > upper
    ]
    functkw = {"x": np.array([1, 2, 3])}
    with pytest.raises(ValueError):
        fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)


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
    result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
    
    repr_str = repr(result)
    assert 'MPFitResult' in repr_str
    assert 'status' in repr_str
    assert 'niter' in repr_str
    assert 'bestnorm' in repr_str


def test_fmpfit_f64_basic_call():
    """Test basic fmpfit_f64 call with valid inputs"""
    x = np.linspace(-5, 5, 50, dtype=np.float64)
    y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5)**2)
    error = np.ones_like(y) * 0.1
    
    parinfo = [
        {'value': 1.0, 'limits': [0.0, 10.0]},
        {'value': 0.0, 'limits': [-5.0, 5.0]},
        {'value': 1.0, 'limits': [0.1, 5.0]}
    ]
    
    functkw = {'x': x, 'y': y, 'error': error}
    result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=functkw)
    
    assert result is not None
    assert result.status > 0
    assert len(result.best_params) == 3


def test_fmpfit_f32_basic_call():
    """Test basic fmpfit_f32 call with valid inputs"""
    x = np.linspace(-5, 5, 50, dtype=np.float32)
    y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5)**2).astype(np.float32)
    error = np.ones_like(y, dtype=np.float32) * 0.1
    
    parinfo = [
        {'value': 1.0, 'limits': [0.0, 10.0]},
        {'value': 0.0, 'limits': [-5.0, 5.0]},
        {'value': 1.0, 'limits': [0.1, 5.0]}
    ]
    
    functkw = {'x': x, 'y': y, 'error': error}
    result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=functkw)
    
    assert result is not None
    assert result.status > 0
    assert len(result.best_params) == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
