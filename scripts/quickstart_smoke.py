"""Small smoke test for the Quickstart snippets.

This script assumes you've built the extensions in-place with:

    python3 setup.py build_ext --inplace

It tries two imports:
 - preferred: from fmedian.fmedian_ext import fmedian
 - convenience: from ftoolss import fmedian, fsigma

and runs a tiny call to `fmedian` to ensure the extension is callable.
"""

import numpy as np

from ftoolss import fmedian

print("Running quickstart smoke test (using ftoolss imports)...")
print("Imported fmedian from ftoolss")

# create a small test array
a = np.arange(25.0, dtype=np.float64).reshape(5, 5)

out = fmedian(a, (3, 3), exclude_center=0)
print("fmedian (via ftoolss) call succeeded. sample output[2,2] =", out[2,2])

print("quickstart smoke test completed successfully")
