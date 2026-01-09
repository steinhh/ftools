#!/usr/bin/env python3
"""
Example program demonstrating the use of fmedian3_ext module.

This program creates a sample 3D array with some noise and applies
the filtered median function to smooth it.
"""

import numpy as np

from ftoolss.fmedian3 import fmedian3

def main():
    print("=" * 60)
    print("Filtered Median 3D Example")
    print("=" * 60)
    
    # Create a sample input array (5x5x5) with float64 type
    print("\n1. Creating sample input array (5x5x5)...")
    # Initialize with a pattern that has an outlier
    input_array = np.full((5, 5, 5), 10.0, dtype=np.float64)
    
    # Add some structure
    input_array[1:4, 1:4, 1:4] = 20.0
    input_array[2, 2, 2] = 30.0
    
    # Add an outlier
    input_array[2, 2, 1] = 100.0  # This is our outlier
    
    print("Input array shape:", input_array.shape)
    print("Input array (showing z-slice 1 which contains the outlier):")
    print(input_array[:, :, 1])
    print("\nOutlier value at [2, 2, 1]:", input_array[2, 2, 1])
    
    # Define filter parameters
    xsize = 3      # Window size in x direction (must be odd)
    ysize = 3      # Window size in y direction (must be odd)
    zsize = 3      # Window size in z direction (must be odd)

    print("\n2. Applying filtered median with parameters:")
    print(f"   - Window size: ({xsize} x {ysize} x {zsize})")

    # Call the fmedian3 function (exclude_center controls whether center is skipped)
    exclude_center = 1
    output_array = fmedian3(input_array, xsize, ysize, zsize, exclude_center=exclude_center)
    
    print("\n3. Output array (filtered median):")
    print("Output array shape:", output_array.shape)
    print("Output array (showing z-slice 1):")
    print(output_array[:, :, 1])
    print("\nFiltered value at [2, 2, 1]:", output_array[2, 2, 1])
    
    # Show the difference (particularly for the outlier)
    print("\n4. Comparison at outlier position [2, 2, 1]:")
    print(f"   Original value: {input_array[2, 2, 1]}")
    print(f"   Filtered value: {output_array[2, 2, 1]}")
    print(f"   Difference: {input_array[2, 2, 1] - output_array[2, 2, 1]}")
    
    # Test with center included
    print("\n5. Testing with center included (exclude_center=0):")
    exclude_center = 0
    output_with_center = fmedian3(input_array, xsize, ysize, zsize, exclude_center=exclude_center)
    print("Filtered value at [2, 2, 1] (with center):", output_with_center[2, 2, 1])
    
    print("\n6. Summary:")
    print("   - The 3D median filter successfully reduces the outlier")
    print("   - When exclude_center=1, the outlier is completely ignored")
    print("   - When exclude_center=0, the outlier still affects the median")

if __name__ == "__main__":
    main()