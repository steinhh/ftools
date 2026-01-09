#!/usr/bin/env python3
"""Test if subprocess fresh import works correctly"""

import subprocess
import sys
import os

# Compute repository root relative to this script so file still works in new location
cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Phase 1: Build scalar
print("Phase 1: Building scalar...")
env = os.environ.copy()
env["FORCE_SCALAR"] = "1"
subprocess.run(["rm", "-rf", "build"], cwd=cwd, capture_output=True)
result = subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    env=env, cwd=cwd, capture_output=True, text=True
)
print("Build done, return code:", result.returncode)

# Run benchmark in fresh subprocess
print("\nRunning benchmark in subprocess (should be scalar)...")
script = """
import numpy as np
import time
from ftoolss.fgaussian import fgaussian_f64_ext

n = 100000
num_iter = 100
x = np.linspace(-100, 100, n, dtype=np.float64)
i0, mu, sigma = 1.0, 0.0, 10.0

for _ in range(5):
    fgaussian_f64_ext.fgaussian_f64(x, i0, mu, sigma)

start = time.perf_counter()
for _ in range(num_iter):
    fgaussian_f64_ext.fgaussian_f64(x, i0, mu, sigma)
elapsed_us = (time.perf_counter() - start) / num_iter * 1e6
print(f"SCALAR SUBPROCESS: {elapsed_us:.1f}")
"""

result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=cwd, capture_output=True, text=True
)
print("Output:", result.stdout.strip())
if result.returncode != 0:
    print("Error:", result.stderr[:200])

# Phase 2: Build accelerate
print("\nPhase 2: Building accelerate...")
env = os.environ.copy()
env.pop("FORCE_SCALAR", None)
subprocess.run(["rm", "-rf", "build"], cwd=cwd, capture_output=True)
result = subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    env=env, cwd=cwd, capture_output=True, text=True
)
print("Build done, return code:", result.returncode)

# Run benchmark in fresh subprocess
print("\nRunning benchmark in subprocess (should be accelerate)...")
script = """
import numpy as np
import time
from ftoolss.fgaussian import fgaussian_f64_ext

n = 100000
num_iter = 100
x = np.linspace(-100, 100, n, dtype=np.float64)
i0, mu, sigma = 1.0, 0.0, 10.0

for _ in range(5):
    fgaussian_f64_ext.fgaussian_f64(x, i0, mu, sigma)

start = time.perf_counter()
for _ in range(num_iter):
    fgaussian_f64_ext.fgaussian_f64(x, i0, mu, sigma)
elapsed_us = (time.perf_counter() - start) / num_iter * 1e6
print(f"ACCELERATE SUBPROCESS: {elapsed_us:.1f}")
"""

result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=cwd, capture_output=True, text=True
)
print("Output:", result.stdout.strip())
if result.returncode != 0:
    print("Error:", result.stderr[:200])
