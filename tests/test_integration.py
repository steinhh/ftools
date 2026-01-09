import numpy as np

from ftoolss import fmedian, fsigma


def test_smoke_integration_basic():
    """Basic smoke test that runs both fmedian and fsigma together.

    This ensures the packaged, return-style APIs for both extensions work
    in a small processing pipeline.
    """
    rng = np.random.default_rng(12345)
    arr = rng.normal(0.0, 1.0, (32, 32)).astype(np.float64)

    med = fmedian(arr, (1, 1), 1)
    sig = fsigma(arr, (1, 1), 1)

    # Basic sanity checks
    assert med.shape == arr.shape
    assert sig.shape == arr.shape
    assert med.dtype == np.float64 and sig.dtype == np.float64
    assert np.all(np.isfinite(med)) and np.all(np.isfinite(sig))
    assert np.all(sig >= 0.0)

    # Ensure the pipeline continues to accept fmedian output
    sig2 = fsigma(med, (1, 1), 1)
    assert sig2.shape == arr.shape
