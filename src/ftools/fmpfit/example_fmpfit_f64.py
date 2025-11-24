#!/usr/bin/env python
"""
Example usage of fmpfit_f64 extension

Demonstrates fitting a Gaussian model to synthetic noisy data using the
MPFIT Levenberg-Marquardt algorithm with parameter constraints (float64).
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftools import fmpfit_f64_wrap

# Generate synthetic Gaussian data
np.random.seed(42)
x = np.linspace(-5, 5, 100)

# True parameters: amplitude=2.5, mean=1.0, sigma=0.8
true_params = [2.5, 1.0, 0.8]
y_true = true_params[0] * np.exp(-0.5 * ((x - true_params[1]) / true_params[2])**2)
noise = np.random.normal(0, 0.1, len(x))
y = y_true + noise
error = np.ones_like(y) * 0.1

# Initial parameter guesses
p0 = [1.5, 0.5, 1.0]  # Deliberately off from true values

# Parameter bounds
bounds = [
    [0.0, 10.0],   # amplitude: 0 to 10
    [-5.0, 5.0],   # mean: -5 to 5
    [0.1, 5.0]     # sigma: 0.1 to 5
]

# Create parinfo structure
parinfo = [
    {'value': p0[i], 'limits': bounds[i]} 
    for i in range(len(p0))
]

# Create functkw
functkw = {'x': x, 'y': y, 'error': error}

print("=" * 70)
print("FMPFIT Example: Gaussian Fitting")
print("=" * 70)
print(f"\nData points: {len(x)}")
print(f"Parameters: {len(p0)}")
print(f"\nTrue parameters:    {true_params}")
print(f"Initial guesses:    {p0}")
print(f"Parameter bounds:   {bounds}")

# Call fmpfit_py
print("\nCalling fmpfit_wrap()...")
result = fmpfit_f64_wrap(
    deviate_type=0,  # 0 = Gaussian model
    parinfo=parinfo,
    functkw=functkw,
    xtol=1.0e-6,
    ftol=1.0e-6,
    gtol=1.0e-6,
    maxiter=2000,
    quiet=1
)

print("\nResults:")
print("-" * 70)
print(f"Status:             {result.status}")
print(f"Iterations:         {result.niter}")
print(f"Function evals:     {result.nfev}")
print(f"Initial chi^2:      {result.orignorm:.6e}")
print(f"Final chi^2:        {result.bestnorm:.6e}")
print(f"\nParameters:         {result.npar}")
print(f"Free parameters:    {result.nfree}")
print(f"Pegged parameters:  {result.npegged}")
print(f"Data points:        {result.nfunc}")

print("\nBest-fit parameters:")
for i, (param, err) in enumerate(zip(result.best_params, result.xerror)):
    print(f"  p[{i}] = {param:10.6f} +/- {err:.6f}")

print(f"\nCovariance matrix shape: {result.covar.shape}")
print(f"Residuals shape:         {result.resid.shape}")
print("=" * 70)
