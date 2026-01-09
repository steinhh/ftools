#!/usr/bin/env python3
"""
Example program demonstrating the use of the ``fsigma3`` extension module.

The script creates a sample 3D array with an injected outlier and computes
the local standard deviation (sigma) around each voxel.
"""

import numpy as np

from ftoolss.fsigma3 import fsigma3

def main():
    print("=" * 60)
    print("Filtered Sigma 3D Example")
    print("=" * 60)
    
    # Create a sample input array (5x5x5) with float64 type
    print("\n1. Creating sample input array (5x5x5)...")
    # Initialize with a pattern that has an outlier
    input_array = np.full((5, 5, 5), 10.0, dtype=np.float64)
    
    # Add some structure with varying values
    input_array[1:4, 1:4, 1:4] = 20.0
    input_array[2, 2, 2] = 30.0
    
    # Add an outlier to create high local variance
    input_array[2, 2, 1] = 100.0  # This is our outlier
    
    print("Input array shape:", input_array.shape)
    print("Input array (showing z-slice 1 which contains the outlier):")
    print(input_array[:, :, 1])
    print("\nOutlier value at [2, 2, 1]:", input_array[2, 2, 1])
    
    # Define filter parameters
    xsize = 3      # Window size in x direction (must be odd)
    ysize = 3      # Window size in y direction (must be odd)
    zsize = 3      # Window size in z direction (must be odd)

    print("\n2. Applying filtered sigma with parameters:")
    print(f"   - Window size: ({xsize} x {ysize} x {zsize})")

    # Call the fsigma3 function (exclude_center controls whether center is skipped)
    exclude_center = 1
    output_array = fsigma3(input_array, xsize, ysize, zsize, exclude_center=exclude_center)
    
    print("\n3. Output array (local sigma values):")
    print("Output array shape:", output_array.shape)
    print("Output array (showing z-slice 1):")
    print(np.round(output_array[:, :, 1], 2))  # Round for readability
    print(f"\nSigma value at [2, 2, 1]: {output_array[2, 2, 1]:.3f}")
    
    # Test with center included
    print("\n4. Testing with center included (exclude_center=0):")
    exclude_center = 0
    output_with_center = fsigma3(input_array, xsize, ysize, zsize, exclude_center=exclude_center)
    print(f"Sigma value at [2, 2, 1] (with center): {output_with_center[2, 2, 1]:.3f}")
    
    # Show values around the outlier position
    print("\n5. Local sigma values around the outlier:")
    print("z-slice 0 (around outlier):")
    print(np.round(output_array[:, :, 0], 2))
    print("z-slice 1 (contains outlier):")
    print(np.round(output_array[:, :, 1], 2))
    print("z-slice 2 (around outlier):")
    print(np.round(output_array[:, :, 2], 2))
    
    print("\n6. Summary:")
    print("   - High sigma values indicate areas with large local variation")
    print("   - The outlier creates elevated sigma values in its neighborhood")
    print("   - exclude_center=1 removes the outlier from sigma calculation")
    print("   - exclude_center=0 includes the outlier in sigma calculation")

if __name__ == "__main__":
    main()