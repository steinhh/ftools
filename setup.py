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
    return "ftools: C-based local image filters (fmedian, fsigma)"


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

# Platform-specific settings for fgaussian
fgaussian_extra_link_args = []
if sys.platform == "darwin":
    # macOS: use Accelerate framework
    fgaussian_extra_link_args = ["-framework", "Accelerate"]
# On Linux/other platforms, no special linking needed (uses standard math library)

ext_modules = [
    Extension(
        "ftools.fmedian.fmedian_ext",
        sources=[os.path.join("src", "ftools", "fmedian", "fmedian_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftools.fsigma.fsigma_ext",
        sources=[os.path.join("src", "ftools", "fsigma", "fsigma_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftools.fmedian3.fmedian3_ext",
        sources=[os.path.join("src", "ftools", "fmedian3", "fmedian3_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftools.fsigma3.fsigma3_ext",
        sources=[os.path.join("src", "ftools", "fsigma3", "fsigma3_ext.c")],
        include_dirs=include_dirs,
    ),
    Extension(
        "ftools.fgaussian.fgaussian_f32_ext",
        sources=[os.path.join("src", "ftools", "fgaussian", "fgaussian_f32_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftools.fgaussian.fgaussian_f64_ext",
        sources=[os.path.join("src", "ftools", "fgaussian", "fgaussian_f64_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftools.fgaussian.fgaussian_jacobian_f32_ext",
        sources=[os.path.join("src", "ftools", "fgaussian", "fgaussian_jacobian_f32_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftools.fgaussian.fgaussian_jacobian_f64_ext",
        sources=[os.path.join("src", "ftools", "fgaussian", "fgaussian_jacobian_f64_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=fgaussian_extra_link_args,
    ),
    Extension(
        "ftools.fgaussian.fgaussian_jacobian_f64_f32_ext",
        sources=[os.path.join("src", "ftools", "fgaussian", "fgaussian_jacobian_f64_f32_ext.c")],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=fgaussian_extra_link_args,
    ),
]


setup(
    name="ftools",
    version="4.0.24",
    description="Small C extensions for local image filters (fmedian, fsigma)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=ext_modules,
    setup_requires=["numpy>=1.20"],
    install_requires=["numpy>=1.20"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
