#!/usr/bin/env python3
"""
Example program demonstrating the use of fmedian_ext module.

This program creates a sample 2D array with some noise and applies
the filtered median function to smooth it.
"""

import numpy as np

from ftoolss.fmedian import fmedian

def main():
    print("=" * 60)
    print("Filtered Median Example")
    print("=" * 60)
    
    # Create a sample input array (10x10) with float64 type
    print("\n1. Creating sample input array (10x10)...")
    input_array = np.array([
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [10, 20, 20, 20, 20, 20, 20, 20, 20, 10],
        [10, 20, 30, 30, 30, 30, 30, 30, 20, 10],
        [10, 20, 30, 40, 40, 40, 40, 30, 20, 10],
        [10, 20, 30, 40, 100, 40, 40, 30, 20, 10],  # 100 is an outlier
        [10, 20, 30, 40, 40, 40, 40, 30, 20, 10],
        [10, 20, 30, 30, 30, 30, 30, 30, 20, 10],
        [10, 20, 20, 20, 20, 20, 20, 20, 20, 10],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [5, 10, 10, 10, 10, 10, 10, 10, 10, 5],
    ], dtype=np.float64)
    
    print("Input array:")
    print(input_array)
    
    # Define filter parameters
    xsize = 3      # Window size in x direction (must be odd)
    ysize = 3      # Window size in y direction (must be odd)

    print("\n2. Applying filtered median with parameters:")
    print(f"   - Window size: ({xsize} x {ysize})")

    # Call the fmedian function (exclude_center controls whether center is skipped)
    exclude_center = 1
    output_array = fmedian(input_array, xsize, ysize, exclude_center=exclude_center)
    
    print("\n3. Output array (filtered median):")
    print(output_array)
    
    # Show the difference (particularly for the outlier)
    print("\n4. Difference (input - output):")
    difference = input_array.astype(np.float64) - output_array
    print(difference)
    
    # Highlight where the filter made significant changes
    print("\n5. Analysis:")
    significant_changes = np.abs(difference) > 5
    if np.any(significant_changes):
        print(f"   Significant changes (|diff| > 5) at {np.sum(significant_changes)} locations:")
        coords = np.argwhere(significant_changes)
        for coord in coords[:5]:  # Show first 5
            y, x = coord
            print(f"   Position ({y}, {x}): {input_array[y, x]} -> {output_array[y, x]:.2f} (diff: {difference[y, x]:.2f})")
        if len(coords) > 5:
            print(f"   ... and {len(coords) - 5} more")
    else:
        print("   No significant changes detected.")
    
    # Re-run with the center pixel included in the neighborhood
    print("\n6. Re-running filter (center pixel included)...")
    # Example: include the center pixel this time (exclude_center=0)
    exclude_center = 0
    output_array2 = fmedian(input_array, xsize, ysize, exclude_center=exclude_center)
    print("Output array (second run, exclude_center=0 -> center included):")
    print(output_array2)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
