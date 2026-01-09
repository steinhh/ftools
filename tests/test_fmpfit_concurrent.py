import threading
import numpy as np
import pytest
import os
import sys

# Ensure package import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/')

from ftoolss import fmpfit_f64_pywrap, fmpfit_f32_pywrap


def make_functkw(dtype):
    x = np.linspace(-5, 5, 50, dtype=dtype)
    y = (2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5)**2)).astype(dtype)
    error = np.ones_like(y, dtype=dtype) * dtype(0.1)
    return {'x': x, 'y': y, 'error': error}


def worker_f64(results, idx):
    try:
        parinfo = [
            {'value': 1.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-5.0, 5.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]}
        ]
        result = fmpfit_f64_pywrap(0, parinfo=parinfo, functkw=make_functkw(np.float64))
        results[idx] = ('ok', result.status if result is not None else None)
    except Exception as e:
        results[idx] = ('error', repr(e))


def worker_f32(results, idx):
    try:
        parinfo = [
            {'value': 1.0, 'limits': [0.0, 10.0]},
            {'value': 0.0, 'limits': [-5.0, 5.0]},
            {'value': 1.0, 'limits': [0.1, 5.0]}
        ]
        result = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=make_functkw(np.float32))
        results[idx] = ('ok', result.status if result is not None else None)
    except Exception as e:
        results[idx] = ('error', repr(e))


@pytest.mark.parametrize('nthreads', [2, 4, 8])
def test_concurrent_fmpfit_threads(nthreads):
    """Spawn multiple threads running fmpfit concurrently (both precisions)."""
    threads = []
    results = [None] * (2 * nthreads)

    # Start f64 workers
    for i in range(nthreads):
        t = threading.Thread(target=worker_f64, args=(results, i))
        t.start()
        threads.append(t)

    # Start f32 workers
    for i in range(nthreads):
        t = threading.Thread(target=worker_f32, args=(results, nthreads + i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Assert no thread raised exceptions and all returned positive status
    errors = [r for r in results if r is not None and r[0] == 'error']
    assert not errors, f"Errors in threads: {errors}"

    oks = [r for r in results if r is not None and r[0] == 'ok']
    assert len(oks) == len(results), f"Not all threads finished successfully: {results}"
