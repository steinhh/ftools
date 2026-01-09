#!/usr/bin/env python3
"""
Comprehensive Benchmark: All fgaussian variants

Compares performance of:
  - f64 scalar
  - f32 scalar
  - f64 accelerate
  - f32 accelerate

All speedups are relative to f64 scalar (1.0x baseline).

Usage:
    python benchmark_all_variants.py [--rebuild-both]
"""

import argparse
import os
import sys
import time
import subprocess
import numpy as np


def get_module_build_info():
    """Check if current build uses Accelerate or scalar."""
    # This function is unreliable because Python caches modules
    # Just return a placeholder
    return "(verification skipped - using subprocesses for isolation)"


def rebuild_extensions(force_scalar=False):
    """Rebuild C extensions with specified configuration."""
    env = os.environ.copy()
    if force_scalar:
        env["FORCE_SCALAR"] = "1"
        print("Building with FORCE_SCALAR=1 (scalar version)...")
    else:
        env.pop("FORCE_SCALAR", None)
        print("Building with Accelerate framework (default)...")

    # Compute repository root relative to this script
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # CRITICAL: Remove build directory to force complete rebuild
    subprocess.run(["rm", "-rf", "build"], cwd=repo_root, capture_output=True)
    
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Build failed with return code {result.returncode}")
        return False

    print("Build successful!")
    
    # Verify compilation flags were used
    if force_scalar:
        if "-DFORCE_SCALAR" in result.stderr:
            print("  ? Verified -DFORCE_SCALAR compilation flag")
    
    # Verify the build
    import importlib
    # Clear module cache
    mods_to_delete = [m for m in sys.modules.keys() if 'ftools' in m]
    for m in mods_to_delete:
        del sys.modules[m]
    
    build_type = get_module_build_info()
    print(f"Build verification: {build_type}")
    
    return True


def _run_benchmarks_subprocess(sizes):
    """Run benchmarks in isolated subprocess to avoid module cache issues."""
    import json
    import tempfile
    
    # Use repository root so subprocesses run from project root
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    results = {}
    
    for n in sizes:
        # Use fixed iterations based on array size for better reproducibility
        # Don't use too many iterations or timing becomes noisy
        if n < 1000:
            num_iter = 15000
        elif n < 10000:
            num_iter = 2000
        elif n < 50000:
            num_iter = 1000
        else:
            num_iter = 500
        
        # Create a temporary Python script to run in subprocess
        script = f"""
import numpy as np
import time
from ftoolss.fgaussian import fgaussian_f32_ext, fgaussian_f64_ext

n = {n}
num_iter = {num_iter}
x_f32 = np.linspace(-10, 10, n, dtype=np.float32)
x_f64 = np.linspace(-10, 10, n, dtype=np.float64)
i0, mu, sigma = 2.5, 1.5, 3.0

# Warm up
for _ in range(10):
    fgaussian_f32_ext.fgaussian_f32(x_f32, float(i0), float(mu), float(sigma))
    fgaussian_f64_ext.fgaussian_f64(x_f64, i0, mu, sigma)

# Benchmark f64 (take median of 3 runs)
times_f64 = []
for _ in range(3):
    start = time.perf_counter()
    for _ in range(num_iter):
        fgaussian_f64_ext.fgaussian_f64(x_f64, i0, mu, sigma)
    times_f64.append((time.perf_counter() - start) / num_iter * 1e6)
time_f64 = sorted(times_f64)[1]  # Take median

# Benchmark f32 (take median of 3 runs)
times_f32 = []
for _ in range(3):
    _i0 = float(i0)
    _mu = float(mu)
    _sigma = float(sigma)
    start = time.perf_counter()
    for _ in range(num_iter):
        fgaussian_f32_ext.fgaussian_f32(x_f32, _i0, _mu, _sigma)
    times_f32.append((time.perf_counter() - start) / num_iter * 1e6)
time_f32 = sorted(times_f32)[1]  # Take median

print(f"{{time_f64}} {{time_f32}}")
"""
        
        # Set up environment with PYTHONPATH to find ftools module
        # Prepend src directory to ensure we find the correct ftoolss package
        env = os.environ.copy()
        src_path = os.path.join(cwd, 'src')
        env['PYTHONPATH'] = src_path
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode == 0:
            try:
                parts = result.stdout.strip().split()
                time_f64 = float(parts[0])
                time_f32 = float(parts[1])
                results[n] = {"f64": time_f64, "f32": time_f32}
                print(f"  N={n}: f64={time_f64:.3f}?s, f32={time_f32:.3f}?s")
            except (ValueError, IndexError) as e:
                print(f"  N={n}: Parse error - {e}")
        else:
            print(f"  N={n}: Error - {result.stderr[:100]}")
    
    return results


def benchmark_size(n, num_iterations=1000):
    """Benchmark for a specific array size, returning all four variants."""
    # Force fresh import
    mods_to_delete = [m for m in sys.modules.keys() if 'ftools' in m]
    for mod in mods_to_delete:
        del sys.modules[mod]
    
    from ftoolss.fgaussian import fgaussian_f32_ext, fgaussian_f64_ext
    
    # Setup
    x_f32 = np.linspace(-10, 10, n, dtype=np.float32)
    x_f64 = np.linspace(-10, 10, n, dtype=np.float64)
    i0, mu, sigma = 2.5, 1.5, 3.0

    # Warm up
    for _ in range(5):
        fgaussian_f32_ext.fgaussian_f32(x_f32, float(i0), float(mu), float(sigma))
        fgaussian_f64_ext.fgaussian_f64(x_f64, i0, mu, sigma)

    # Benchmark f64
    start = time.perf_counter()
    for _ in range(num_iterations):
        fgaussian_f64_ext.fgaussian_f64(x_f64, i0, mu, sigma)
    time_f64 = (time.perf_counter() - start) / num_iterations * 1e6

    # Benchmark f32
    _i0 = float(i0)
    _mu = float(mu)
    _sigma = float(sigma)
    start = time.perf_counter()
    for _ in range(num_iterations):
        fgaussian_f32_ext.fgaussian_f32(x_f32, _i0, _mu, _sigma)
    time_f32 = (time.perf_counter() - start) / num_iterations * 1e6

    return time_f64, time_f32
def run_benchmark_all_variants():
    """Run benchmark for all four variants with adaptive N range."""
    print()
    print("=" * 110)
    print("fgaussian Comprehensive Benchmark: All Variants (100-100k points)")
    print("=" * 110)
    print()

    # Generate size list by geometric doubling starting at 5 up to 100000
    sizes = [1,2,3,4,5,6,7,8,9,10]
    n = 10
    while n < 10000:
        sizes.append(n)
        n *= 2

    # Phase 1: Rebuild with FORCE_SCALAR and collect scalar results
    print("Phase 1: Building and benchmarking SCALAR version...")
    if not rebuild_extensions(force_scalar=True):
        sys.exit(1)
    
    print("Running scalar version benchmarks...")
    results_scalar = _run_benchmarks_subprocess(sizes)

    # Phase 2: Rebuild with accelerate and collect those results
    print("\nPhase 2: Building and benchmarking ACCELERATE version...")
    if not rebuild_extensions(force_scalar=False):
        sys.exit(1)

    print("Running Accelerate version benchmarks...")
    results_accelerate = _run_benchmarks_subprocess(sizes)

    # Print results
    print()
    print("=" * 110)
    print("Results (all speedups relative to f64 scalar baseline):")
    print("=" * 110)
    print()
    print(f"{'N':<8} {'f64 Scalar':<18} {'f32 Scalar':<18} {'f64 Accel':<18} {'f32 Accel':<18}")
    print(f"{'':8} {'speedup':<18} {'speedup':<18} {'speedup':<18} {'speedup':<18}")
    print("-" * 110)

    for n in sizes:
        if n not in results_scalar or n not in results_accelerate:
            continue

        # Get baseline (f64 scalar)
        baseline = results_scalar[n]["f64"]

        # Calculate all speedups relative to f64 scalar baseline
        f64_scalar_speedup = 1.0  # Always the reference
        f32_scalar_speedup = baseline / results_scalar[n]["f32"]
        f64_accel_speedup = baseline / results_accelerate[n]["f64"]
        f32_accel_speedup = baseline / results_accelerate[n]["f32"]

        print(
            f"{n:<8} "
            f"{f64_scalar_speedup:<18.2f} "
            f"{f32_scalar_speedup:<18.2f} "
            f"{f64_accel_speedup:<18.2f} "
            f"{f32_accel_speedup:<18.2f}"
        )

    print("=" * 110)
    print()

    # Summary statistics
    print("Summary Statistics:")
    print()

    f32_scalar_speedups = [
        baseline / results_scalar[n]["f32"]
        for n in sizes
        if n in results_scalar
        for baseline in [results_scalar[n]["f64"]]
    ]
    f64_accel_speedups = [
        baseline / results_accelerate[n]["f64"]
        for n in sizes
        if n in results_accelerate
        for baseline in [results_scalar[n]["f64"]]
    ]
    f32_accel_speedups = [
        baseline / results_accelerate[n]["f32"]
        for n in sizes
        if n in results_accelerate
        for baseline in [results_scalar[n]["f64"]]
    ]

    print(f"f32 Scalar:   {np.mean(f32_scalar_speedups):.2f}x baseline on average")
    print(f"f64 Accelerate: {np.mean(f64_accel_speedups):.2f}x baseline on average")
    print(f"f32 Accelerate: {np.mean(f32_accel_speedups):.2f}x baseline on average")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark of all fgaussian variants"
    )
    parser.add_argument(
        "--rebuild-both",
        action="store_true",
        help="Force rebuild of both versions",
    )

    args = parser.parse_args()
    # Change to repository root (relative to this script location)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    os.chdir(repo_root)

    if args.rebuild_both:
        print("Rebuilding scalar version first...")
        if not rebuild_extensions(force_scalar=True):
            sys.exit(1)

    run_benchmark_all_variants()


if __name__ == "__main__":
    main()
