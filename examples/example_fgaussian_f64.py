#!/usr/bin/env python
"""
Example usage of fgaussian_f64 (float64 version)

Demonstrates the float64 version of the Gaussian profile computation.
This version accepts float64 input arrays for compatibility with existing code.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from ftoolss.fgaussian import fgaussian_f64

# Example parameters
i0 = 1.0      # Peak intensity
mu = 0.0      # Center position
sigma = 1.5   # Width parameter

# Create float64 input array
x = np.linspace(-10, 10, 1000, dtype=np.float64)

# Compute Gaussian profile
profile = fgaussian_f64(x, i0, mu, sigma)

print(f"Input array shape: {x.shape}, dtype: {x.dtype}")
print(f"Output array shape: {profile.shape}, dtype: {profile.dtype}")
print(f"\nPeak value: {profile.max():.10f}")
print(f"Peak location: x[{profile.argmax()}] = {x[profile.argmax()]:.3f}")

# Compare with NumPy reference implementation
profile_numpy = i0 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
max_diff = np.abs(profile - profile_numpy).max()
print(f"\nMaximum difference vs NumPy: {max_diff:.2e}")

# Show some sample values
print(f"\nSample values:")
indices = [0, 250, 500, 750, 999]
for i in indices:
    print(f"  x[{i:3d}] = {x[i]:6.2f}  ->  profile = {profile[i]:.6f}")
