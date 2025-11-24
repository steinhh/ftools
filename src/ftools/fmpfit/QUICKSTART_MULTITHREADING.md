# Quick Start: Testing fmpfit_f32 Multithreading

## TL;DR

The fmpfit_f32_ext achieves **4.1x speedup with 6 threads** on your 12-core system.

## Quick Test

```bash
# Verify correctness (30 seconds)
python src/ftools/fmpfit/benchmark_f32_multithreading.py --verify-only

# Full benchmark (2 minutes)
python src/ftools/fmpfit/benchmark_f32_multithreading.py

# Test thread safety (5 seconds)
pytest tests/test_fmpfit_concurrent.py -v
```

## Results Summary

48 fits × 10,000 points each:

```
Threads   Time     Speedup    Efficiency
   1      0.103s   0.97×      97%
   2      0.054s   1.85×      93%
   4      0.030s   3.37×      84%  ? Recommended
   6      0.024s   4.14×      69%  ? Maximum speedup
   8      0.025s   4.03×      50%
```

## When to Use Threading

- ? Fits with >1,000 data points (>0.5ms each)
- ? Batch processing many sources
- ? Quick fits <100 points (overhead dominates)

## Usage Example

```python
from concurrent.futures import ThreadPoolExecutor
from ftools.fmpfit import fmpfit_f32_wrap

# Use 4 threads (optimal balance)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fit_function, datasets))
```

## Files

- `benchmark_f32_multithreading.py` - Main benchmark script
- `MULTITHREADING_BENCHMARK.md` - Detailed analysis
- `MULTITHREADING_TEST_SUMMARY.md` - Complete test report
- `plot_multithreading_speedup.py` - Generate plots

## Verification

All correctness tests passed:

- ? Bit-exact results (seq vs threaded)
- ? No race conditions
- ? All pytest concurrent tests passed
