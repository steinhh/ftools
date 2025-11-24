#!/usr/bin/env python3
"""
Plot multithreading speedup curves for fmpfit_f32

Generates speedup and efficiency plots from benchmark data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark results (60 fits, 10,000 points each, 12-core system)
threads = np.array([1, 2, 4, 6, 8, 10, 12])
time_seconds = np.array([0.160, 0.068, 0.037, 0.030, 0.030, 0.035, 0.030])
baseline = 0.126  # Sequential time

speedup = baseline / time_seconds
efficiency = speedup / threads * 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Speedup plot
ax1.plot(threads, speedup, 'o-', linewidth=2, markersize=8, label='Actual')
ax1.plot(threads, threads, '--', linewidth=1, color='gray', label='Ideal (linear)')
ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Speedup (x)', fontsize=12)
ax1.set_title('fmpfit_f32 Multithreading Speedup', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(threads)

# Add value labels
for i, (t, s) in enumerate(zip(threads, speedup)):
    ax1.text(t, s + 0.15, f'{s:.2f}x', ha='center', fontsize=9)

# Efficiency plot
ax2.plot(threads, efficiency, 's-', linewidth=2, markersize=8, color='orangered')
ax2.axhline(y=100, linestyle='--', linewidth=1, color='gray', label='100% efficiency')
ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Efficiency (%)', fontsize=12)
ax2.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(threads)
ax2.set_ylim([0, 110])

# Add value labels
for i, (t, e) in enumerate(zip(threads, efficiency)):
    ax2.text(t, e + 3, f'{e:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fmpfit_f32_multithreading_speedup.png', dpi=150, bbox_inches='tight')
print("Saved plot to: fmpfit_f32_multithreading_speedup.png")

# Print summary statistics
print("\n" + "="*60)
print("SPEEDUP SUMMARY")
print("="*60)
print(f"{'Threads':>8} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12}")
print("-"*60)
for t, time, sp, eff in zip(threads, time_seconds, speedup, efficiency):
    print(f"{t:8d} {time:10.3f} {sp:10.2f}x {eff:11.1f}%")
print("\nBaseline (sequential): {:.3f} s".format(baseline))
print(f"Best speedup: {speedup.max():.2f}x at {threads[speedup.argmax()]} threads")
print(f"Best efficiency: {efficiency.max():.1f}% at {threads[efficiency.argmax()]} thread(s)")
