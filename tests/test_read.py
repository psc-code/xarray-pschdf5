from __future__ import annotations

import xarray as xr

import xarray_pschdf5


def test_read_sample():
    xr.open_dataset(xarray_pschdf5.sample_dir / "sample_xdmf.iof.004020.xdmf")


def test_read_sample_all():
    ds = xr.open_dataset(xarray_pschdf5.sample_dir / "sample_xdmf.iof.xdmf")
    print(ds)
    assert set(ds.coords) == set({"time", "lats", "longs", "colats", "mlts"})
    assert ds.pot.dims == ("time", "lats", "longs")
