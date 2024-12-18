from __future__ import annotations

import pathlib
from collections import OrderedDict
from typing import Any, ClassVar

import h5py
import numpy as np
import xarray as xr
from pugixml import pugi
from xarray.backends import BackendEntrypoint


class PscHdf5Entrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        return pschdf5_open_dataset(filename_or_obj, drop_variables=drop_variables)

    open_dataset_parameters: ClassVar[Any] = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj) -> bool:
        filename_or_obj = pathlib.Path(filename_or_obj)

        return filename_or_obj.suffix == ".xdmf"

    description = "XArray reader for PSC HDF5 data"

    url = "https://link_to/your_backend/documentation"  # FIXME


def pschdf5_open_dataset(filename_or_obj, *, drop_variables=None):
    filename_or_obj = pathlib.Path(filename_or_obj)
    dirname = filename_or_obj.parent
    meta = read_xdmf(filename_or_obj)
    meta["run"] = "RUN"
    grids = meta["grids"]

    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    elif drop_variables is None:
        drop_variables = []
    drop_variables = set(drop_variables)

    vars = dict()
    assert len(grids) == 1
    for _, grid in grids.items():
        for fldname, fld in grid["fields"].items():
            if fldname in drop_variables:
                continue

            data_dims = fld["dims"]
            data_path = fld["path"]
            h5_filename, h5_path = fld["path"].split(":")
            h5_file = h5py.File(dirname / h5_filename)
            data = h5_file[h5_path][:].T

            data_attrs = dict(path=data_path)
            match len(data_dims):
                case 2:
                    dims = ("lats", "longs")
                case 3:
                    dims = ("x", "y", "z")
            vars[fldname] = xr.DataArray(data=data, dims=dims, attrs=data_attrs)

    match grid["topology"]["type"]:
        case "3DCoRectMesh":
            coords = {
                "xyz"[d]: make_crd(
                    grid["topology"]["dims"][d],
                    grid["geometry"]["origin"][d],
                    grid["geometry"]["spacing"][d],
                )
                for d in range(3)
            }
        case "2DSMesh":
            coords = {}

    attrs = dict(run=meta["run"], time=meta["time"])

    ds = xr.Dataset(vars, coords=coords, attrs=attrs)
    #    ds.set_close(my_close_method)

    return ds


def make_crd(dim, origin, spacing):
    return origin + np.arange(0.5, dim) * spacing


def _parse_dimensions_attr(node):
    attr = node.attribute("Dimensions")
    return tuple(int(d) for d in attr.value().split(" "))


def _parse_geometry_origin_dxdydz(geometry):
    geo = dict()
    for child in geometry.children():
        if child.attribute("Name").value() == "Origin":
            geo["origin"] = np.asarray(
                [float(x) for x in child.text().as_string().split(" ")]
            )[::-1]

        if child.attribute("Name").value() == "Spacing":
            geo["spacing"] = np.asarray(
                [float(x) for x in child.text().as_string().split(" ")]
            )[::-1]
    return geo


def _parse_geometry_xyz(geometry):
    geo = dict()
    data_item = geometry.child("DataItem")
    assert data_item.attribute("Format").value() == "XML"
    dims = _parse_dimensions_attr(data_item)
    data = np.loadtxt(data_item.text().as_string().splitlines())
    geo = {"data_item": data.reshape(dims)}
    return geo


def read_xdmf(filename):
    doc = pugi.XMLDocument()
    result = doc.load_file(filename)
    if not result:
        raise f"parse error: status={result.status} description={result.description()}"

    grid_collection = doc.child("Xdmf").child("Domain").child("Grid")
    assert grid_collection.attribute("GridType").value() == "Collection"
    grid_time = grid_collection.child("Time")
    assert grid_time.attribute("Type").value() == "Single"
    time = grid_time.attribute("Value").value()
    rv = {}
    rv["time"] = time
    rv["grids"] = {}
    for node in grid_collection.children():
        if node.name() == "Grid":
            grid = {}
            grid_name = node.attribute("Name").value()
            topology = node.child("Topology")
            dims = _parse_dimensions_attr(topology)[::-1]
            grid["topology"] = {
                "type": topology.attribute("TopologyType").value(),
                "dims": dims,
            }

            geometry = node.child("Geometry")
            match geometry.attribute("GeometryType").value():
                case "Origin_DxDyDz":
                    grid["geometry"] = _parse_geometry_origin_dxdydz(geometry)
                case "XYZ":
                    grid["geometry"] = _parse_geometry_xyz(geometry)

            flds = OrderedDict()
            for child in node.children():
                if child.name() != "Attribute":
                    continue

                fld = child.attribute("Name").value()
                item = child.child("DataItem")
                fld_dims = _parse_dimensions_attr(item)[::-1]
                assert np.all(fld_dims == dims)
                assert item.attribute("Format").value() == "HDF"
                path = item.text().as_string().strip()
                flds[fld] = {"path": path, "dims": dims}

            grid["fields"] = flds
            rv["grids"][grid_name] = grid

    return rv
