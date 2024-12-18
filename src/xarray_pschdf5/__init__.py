"""
Copyright (c) 2024 Kai Germaschewski. All rights reserved.

xarray-pschdf5: XArray reader for PSC HDF5 data
"""

from __future__ import annotations

import pathlib

from ._version import version as __version__

sample_dir = pathlib.Path(__file__).parent / "sample"

__all__ = ["__version__", "sample_dir"]
