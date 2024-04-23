from __future__ import annotations

import importlib.metadata

import xarray_pschdf5 as m


def test_version():
    assert importlib.metadata.version("xarray_pschdf5") == m.__version__
