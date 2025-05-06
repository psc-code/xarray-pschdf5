from __future__ import annotations

import dataclasses
import os
import pathlib
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, ClassVar

import h5py
import numpy as np
import xarray as xr
from pugixml import pugi
from typing_extensions import override
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.core.datatree import DataTree
from xarray.core.types import ReadBuffer


class PscHdf5Entrypoint(BackendEntrypoint):
    """XArray backend entrypoint for PSC HDF5 data"""

    description = "XArray reader for PSC HDF5 data"
    # url = "https://link_to/your_backend/documentation"

    open_dataset_parameters: ClassVar[Any] = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj) -> bool:
        if not isinstance(filename_or_obj, str | os.PathLike):
            return False

        filename_or_obj = pathlib.Path(filename_or_obj)

        return filename_or_obj.suffix == ".xdmf"

    @override
    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale: bool = True,  # pylint: disable=unused-argument
        decode_times: bool = True,  # pylint: disable=unused-argument
        concat_characters: bool = True,  # pylint: disable=unused-argument
        decode_coords: bool = True,  # pylint: disable=unused-argument
        drop_variables: str | Iterable[str] | None = None,
        use_cftime: bool | None = None,  # pylint: disable=unused-argument
        decode_timedelta: bool | None = None,  # pylint: disable=unused-argument
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        return pschdf5_open_dataset(filename_or_obj, drop_variables=drop_variables)

    @override
    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        **kwargs: Any,
    ) -> DataTree:
        raise NotImplementedError()


@dataclasses.dataclass
class VariableInfo:
    """Class for keeping track of per-variable info."""

    shape: tuple
    dims: tuple
    times: list[np.datetime64] = dataclasses.field(default_factory=list)
    paths: list[str] = dataclasses.field(default_factory=list)


def pschdf5_open_dataset(filename_or_obj, *, drop_variables=None):
    filename_or_obj = pathlib.Path(filename_or_obj)
    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    elif drop_variables is None:
        drop_variables = []
    drop_variables = set(drop_variables)

    dirname = filename_or_obj.parent
    meta = read_xdmf(filename_or_obj)

    var_infos = {}
    n_times = len(meta)
    for spatial in meta:
        grids = spatial["grids"]
        assert len(grids) == 1
        _, grid = next(iter(grids.items()))
        for fldname, fld in grid["fields"].items():
            if fldname in drop_variables:
                continue

            if fldname not in var_infos:
                var_infos[fldname] = VariableInfo(
                    shape=(n_times,) + fld["dims"], dims=_make_dims(fld)
                )
            time = np.datetime64("2000-01-01T00:00:00") + np.timedelta64(
                int(spatial["time"]) * 1000000000, "ns"
            )
            var_infos[fldname].times.append(time)
            var_infos[fldname].paths.append(fld["path"])

    _, var_info = next(iter(var_infos.items()))
    coords = _make_coords(grid, var_info.times)

    vars = {}  # pylint: disable=redefined-builtin
    for name, info in var_infos.items():
        da = xr.DataArray(data=np.empty(info.shape), dims=info.dims)
        for it, path in enumerate(info.paths):
            h5_filename, h5_path = path.split(":")
            h5_file = h5py.File(dirname / h5_filename)
            da[it, :, :] = h5_file[h5_path]

        vars[name] = da

    attrs = {}
    ds = xr.Dataset(vars, coords=coords, attrs=attrs)
    #    ds.set_close(my_close_method)

    return ds  # noqa: RET504


def _make_dims(fld):
    match len(fld["dims"]):
        case 2:
            return ("time", "lats", "longs")
        case 3:
            return ("time", "z", "y", "x")


def _make_coords(grid, times):  # ("topology"), grid["geometry"]):
    dims = grid["topology"]["dims"]
    match grid["topology"]["type"]:
        case "3DCoRectMesh":
            coords = {
                "xyz"[d]: (
                    "xyz"[d],
                    make_crd(
                        dims[d],
                        grid["geometry"]["origin"][d],
                        grid["geometry"]["spacing"][d],
                    ),
                )
                for d in range(3)
            }
        case "2DSMesh":
            coords = {
                "lats": ("lats", np.linspace(90, -90, dims[0])),
                "longs": ("longs", np.linspace(-180, 180, dims[1])),
                "colats": ("lats", np.linspace(0, 180, dims[0])),
                "mlts": ("longs", np.linspace(0, 24, dims[1])),
            }

    coords["time"] = ("time", np.asarray(times))
    return coords


def make_crd(dim, origin, spacing):
    return origin + np.arange(0.5, dim) * spacing


def _parse_dimensions_attr(node):
    attr = node.attribute("Dimensions")
    return tuple(int(d) for d in attr.value().split(" "))


def _parse_geometry_origin_dxdydz(geometry):
    geo = {}
    for child in geometry.children():
        if child.attribute("Name").value() == "Origin":
            geo["origin"] = np.asarray(
                [float(x) for x in child.text().as_string().split(" ")]
            )

        if child.attribute("Name").value() == "Spacing":
            geo["spacing"] = np.asarray(
                [float(x) for x in child.text().as_string().split(" ")]
            )
    return geo


def _parse_geometry_xyz(geometry):
    data_item = geometry.child("DataItem")
    assert data_item.attribute("Format").value() == "XML"
    dims = _parse_dimensions_attr(data_item)
    data = np.loadtxt(data_item.text().as_string().splitlines())
    return {"data_item": data.reshape(dims)}


def _parse_temporal_collection(filename, grid_collection):
    temporal = []
    for node in grid_collection.children():
        href = node.attribute("href").value()
        doc = pugi.XMLDocument()  # pylint: disable=c-extension-no-member
        result = doc.load_file(filename.parent / href)
        if not result:
            msg = f"parse error: status={result.status} description={result.description()}"
            raise RuntimeError(msg)

        temporal.append(
            _parse_spatial_collection(doc.child("Xdmf").child("Domain").child("Grid"))
        )

    return temporal


def _parse_spatial_collection(grid_collection):
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
            dims = _parse_dimensions_attr(topology)
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
                fld_dims = _parse_dimensions_attr(item)
                assert np.all(fld_dims == dims)
                assert item.attribute("Format").value() == "HDF"
                path = item.text().as_string().strip()
                flds[fld] = {"path": path, "dims": dims}

            grid["fields"] = flds
            rv["grids"][grid_name] = grid

    return rv


def read_xdmf(filename):
    doc = pugi.XMLDocument()  # pylint: disable=c-extension-no-member
    result = doc.load_file(filename)
    if not result:
        msg = f"parse error: status={result.status} description={result.description()}"
        raise RuntimeError(msg)

    grid_collection = doc.child("Xdmf").child("Domain").child("Grid")
    assert grid_collection.attribute("GridType").value() == "Collection"
    match grid_collection.attribute("CollectionType").value():
        case "Spatial":
            return [_parse_spatial_collection(grid_collection)]
        case "Temporal":
            return _parse_temporal_collection(filename, grid_collection)
        case _:
            raise RuntimeError()
