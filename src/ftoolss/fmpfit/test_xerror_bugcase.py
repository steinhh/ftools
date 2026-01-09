#!/usr/bin/env python
"""
Test case extracted from real MUSE data where xerror_scipy bug appears.
This case has linewidth hitting upper bound (200).
"""

import numpy as np
from scipy.optimize import curve_fit

try:
    import ftoolss
    from ftoolss.fmpfit import fmpfit_f32_pywrap
    print(f"ftools version: {ftoolss.__version__}")
except ImportError:
    print("ERROR: ftools not installed")
    raise


def gaussian(x, amp, vel, lw):
    """Gaussian model: amp * exp(-0.5 * ((x - vel) / lw)^2)"""
    return amp * np.exp(-0.5 * ((x - vel) / lw) ** 2)


# Bug case extracted from real MUSE data processing
xdata = np.array([-81.21540537932414, -40.60770268966206, 0.0, 40.60770268966209, 81.21540537932415], dtype=np.float32)
ydata = np.array([0.8990826606750488, 0.9491528868675232, 0.9999998211860657, 0.9884452223777771, 0.9523726105690002], dtype=np.float32)
y_sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
p0 = [1.0, 0.0, 69.1378943769105]
bounds_lower = [0.9, -81.21540537932412, 59.137894376910495]
bounds_upper = [1.1, 81.21540537932412, 200.0]

print("\n" + "="*80)
print("Bug case: Linewidth at upper bound")
print("="*80)
print(f"n_points: {len(xdata)}, dof: {len(xdata) - 3}")

# Fit with fmpfit
fa = {'x': xdata, 'y': ydata, 'error': y_sigma}
parinfo = [
    {'value': p0[0], 'limits': [bounds_lower[0], bounds_upper[0]]},
    {'value': p0[1], 'limits': [bounds_lower[1], bounds_upper[1]]},
    {'value': p0[2], 'limits': [bounds_lower[2], bounds_upper[2]]},
]

mp_ = fmpfit_f32_pywrap(0, parinfo=parinfo, functkw=fa,
                        xtol=1.0E-6, ftol=1.0E-6, gtol=1.0E-6,
                        maxiter=2000, quiet=1)

print(f"\nfmpfit results:")
print(f"  params: amp={mp_.best_params[0]:.6f}, vel={mp_.best_params[1]:.6f}, lw={mp_.best_params[2]:.6f}")
print(f"  xerror_scipy: {mp_.xerror_scipy}")
print(f"  bestnorm (chi2): {mp_.bestnorm:.6f}")

# Fit with scipy for reference
try:
    popt_scipy, pcov_scipy = curve_fit(
        gaussian, xdata, ydata, p0=p0, sigma=y_sigma,
        bounds=(bounds_lower, bounds_upper),
        absolute_sigma=False  # Match fmpfit's chi2 scaling
    )
    perr_scipy = np.sqrt(np.diag(pcov_scipy))
    print(f"\nscipy curve_fit results:")
    print(f"  params: amp={popt_scipy[0]:.6f}, vel={popt_scipy[1]:.6f}, lw={popt_scipy[2]:.6f}")
    print(f"  errors: {perr_scipy}")
except Exception as e:
    print(f"\nscipy fit failed: {e}")

# Compare
print("\n" + "-"*40)
print("Comparison:")
print(f"  fmpfit xerror_scipy[1] (vel error): {mp_.xerror_scipy[1]:.6f}")
try:
    print(f"  scipy error[1] (vel error): {perr_scipy[1]:.6f}")
    ratio = mp_.xerror_scipy[1] / perr_scipy[1] if perr_scipy[1] > 0 else float('inf')
    print(f"  Ratio fmpfit/scipy: {ratio:.2f}")
except NameError:
    pass
print("="*80)
