from __future__ import annotations

import io
import os
import sys
from setuptools import setup, find_packages, Extension


def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme = os.path.join(here, "README.md")
    if os.path.exists(readme):
        with io.open(readme, "r", encoding="utf8") as fh:
            return fh.read()
    return "ftoolss: C-based local image filters (fmedian, fsigma)"


# Try to obtain numpy include dir; setup_requires ensures numpy is available when
# building via setuptools, but fall back gracefully to an empty list at import-time.
include_dirs = []
try:
    import numpy as _np

    include_dirs = [_np.get_include()]
except Exception:
    # numpy may not yet be available at import time; setup_requires=['numpy'] will
    # request it at build time.
    include_dirs = []

# Check for FORCE_SCALAR environment variable to disable Accelerate framework
force_scalar = os.environ.get("FORCE_SCALAR", "0") == "1"
fgaussian_extra_compile_args = ["-O3"]
if force_scalar:
    fgaussian_extra_compile_args.append("-DFORCE_SCALAR")

# No platform-specific link arguments needed
fgaussian_extra_link_args = []
if sys.platform == "darwin":
    # macOS: use Accelerate framework
    fgaussian_extra_link_args = ["-framework", "Accelerate"]
else:
    # Linux/other platforms: explicitly link math library for vectorized functions
    fgaussian_extra_link_args = ["-lm", "-lmvec"]

ext_modules = [
    Extension(
        "ftoolss.fmedian.fmedian_ext",
        sources=[os.path.join("src", "ftoolss", "fmedian", "fmedian_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftoolss.fsigma.fsigma_ext",
        sources=[os.path.join("src", "ftoolss", "fsigma", "fsigma_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftoolss.fmedian3.fmedian3_ext",
        sources=[os.path.join("src", "ftoolss", "fmedian3", "fmedian3_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftoolss.fsigma3.fsigma3_ext",
        sources=[os.path.join("src", "ftoolss", "fsigma3", "fsigma3_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftoolss.fgaussian.fgaussian_f32_ext",
        sources=[os.path.join("src", "ftoolss", "fgaussian", "fgaussian_f32_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=fgaussian_extra_compile_args,
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fgaussian.fgaussian_f64_ext",
        sources=[os.path.join("src", "ftoolss", "fgaussian", "fgaussian_f64_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=fgaussian_extra_compile_args,
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fgaussian.fgaussian_jacobian_f32_ext",
        sources=[os.path.join("src", "ftoolss", "fgaussian", "fgaussian_jacobian_f32_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=fgaussian_extra_compile_args,
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fgaussian.fgaussian_jacobian_f64_ext",
        sources=[os.path.join("src", "ftoolss", "fgaussian", "fgaussian_jacobian_f64_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=fgaussian_extra_compile_args,
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fgaussian.fgaussian_jacobian_f64_f32_ext",
        sources=[os.path.join("src", "ftoolss", "fgaussian", "fgaussian_jacobian_f64_f32_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=fgaussian_extra_compile_args,
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fmpfit.fmpfit_f64_ext",
        sources=[
            os.path.join("src", "ftoolss", "fmpfit", "fmpfit_f64_ext.c"),
            os.path.join("src", "ftoolss", "fmpfit", "cmpfit-1.5", "mpfit.c"),
        ],
        include_dirs=include_dirs + [os.path.join("src", "ftoolss", "fmpfit")],
        extra_compile_args=["-O3"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fmpfit.fmpfit_f32_ext",
        sources=[
            os.path.join("src", "ftoolss", "fmpfit", "fmpfit_f32_ext.c"),
            os.path.join("src", "ftoolss", "fmpfit", "cmpfit-1.5", "mpfit.c"),
        ],
        include_dirs=include_dirs + [os.path.join("src", "ftoolss", "fmpfit")],
        extra_compile_args=["-O3", "-DMPFIT_FLOAT"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fmpfit.fmpfit_f64_block_ext",
        sources=[
            os.path.join("src", "ftoolss", "fmpfit", "fmpfit_f64_block_ext.c"),
            os.path.join("src", "ftoolss", "fmpfit", "cmpfit-1.5", "mpfit.c"),
        ],
        include_dirs=include_dirs + [os.path.join("src", "ftoolss", "fmpfit")],
        extra_compile_args=["-O3"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftoolss.fmpfit.fmpfit_f32_block_ext",
        sources=[
            os.path.join("src", "ftoolss", "fmpfit", "fmpfit_f32_block_ext.c"),
            os.path.join("src", "ftoolss", "fmpfit", "cmpfit-1.5", "mpfit.c"),
        ],
        include_dirs=include_dirs + [os.path.join("src", "ftoolss", "fmpfit")],
        extra_compile_args=["-O3", "-DMPFIT_FLOAT"],
        extra_link_args=fgaussian_extra_link_args,
    ),
]


setup(
    name="ftoolss",
    # Version 5 introduced fmpfit
    # Version 5.2 introduces xerror_scipy (different handling of bounded params)
    # Version 6 changed name to ftoolss (ftools, Stein's) for publishing to PiPY
    version="6.0.26",
    description="Small C extensions for local image filters (fmedian, fsigma)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    # Include mpfit C source files so other packages can compile against them
    package_data={
        "ftoolss.fmpfit": [
            "cmpfit-1.5/mpfit.h",
            "cmpfit-1.5/mpfit.c",
            "cmpfit-1.5/DISCLAIMER",
            "gaussian_deviate.c",
        ],
    },
    ext_modules=ext_modules,
    setup_requires=["numpy>=1.20"],
    install_requires=["numpy>=1.20", "scipy>=1.7"],
    extras_require={
        "test": ["pytest>=6.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
