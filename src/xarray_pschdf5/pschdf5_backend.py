from collections import OrderedDict
from pugixml import pugi
import h5py
import xarray as xr
from xarray.backends import BackendEntrypoint
import numpy as np
import os


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

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        if filename_or_obj.endswith(".xdmf"):
            return True

        return False

    description = "XArray reader for PSC HDF5 data"

    url = "https://link_to/your_backend/documentation"  # FIXME


def pschdf5_open_dataset(filename_or_obj, *, drop_variables=None):
    dirname, basename = os.path.split(filename_or_obj)
    meta = read_xdmf(filename_or_obj)
    meta["run"] = "RUN"
    grids = meta["grids"]

    vars = dict()
    assert len(grids) == 1
    for gridname, grid in grids.items():
        for fldname, fld in grid["fields"].items():
            data_dims = fld["dims"]
            data_path = fld["path"]
            h5_filename, h5_path = fld['path'].split(':')
            h5_file = h5py.File(dirname + '/' + h5_filename)
            data = h5_file[h5_path][:].T

            data_attrs = dict(path=data_path)
            vars[fldname] = xr.DataArray(
                data=data,
                dims=["x", "y", "z"],
                attrs=data_attrs)

    coords = {"xyz"[d]: make_crd(grid['topology']['dims'][d], grid['geometry']
                                 ['origin'][d], grid['geometry']['spacing'][d]) for d in range(3)}

    attrs = dict(run=meta["run"], time=meta["time"])

    ds = xr.Dataset(vars, coords=coords, attrs=attrs)
#    ds.set_close(my_close_method)

    return ds


def make_crd(dim, origin, spacing):
    return origin + np.arange(.5, dim) * spacing


def read_xdmf(filename):
    doc = pugi.XMLDocument()
    result = doc.load_file(filename)
    if not result:
        print('parse error: status=%r description=%r' %
              (result.status, result.description()))
        assert False

    grid_collection = doc.child('Xdmf').child('Domain').child('Grid')
    assert grid_collection.attribute('GridType').value() == 'Collection'
    grid_time = grid_collection.child('Time')
    assert grid_time.attribute('Type').value() == 'Single'
    time = grid_time.attribute('Value').value()
    rv = {}
    rv['time'] = time
    rv['grids'] = {}
    for node in grid_collection.children():
        if node.name() == 'Grid':
            grid = {}
            grid_name = node.attribute('Name').value()
            topology = node.child('Topology')
            # assert topology.attribute('TopologyType').value() == '3DCoRectMesh'
            dims = topology.attribute('Dimensions').value()
            dims = np.asarray([int(d) - 1 for d in dims.split(" ")])[::-1]
            grid['topology'] = {'type': topology.attribute(
                'TopologyType').value(), 'dims': dims}

            geometry = node.child('Geometry')
            assert geometry.attribute(
                'GeometryType').value() == 'Origin_DxDyDz'
            for child in geometry.children():
                if child.attribute('Name').value() == 'Origin':
                    origin = np.asarray(
                        [float(x) for x in child.text().as_string().split(" ")])[::-1]
                if child.attribute('Name').value() == 'Spacing':
                    spacing = np.asarray(
                        [float(x) for x in child.text().as_string().split(" ")])[::-1]

            grid['geometry'] = {'origin': origin, 'spacing': spacing}

            flds = OrderedDict()
            for child in node.children():
                if child.name() != 'Attribute':
                    continue

                fld = child.attribute('Name').value()
                item = child.child('DataItem')
                fld_dims = np.asarray([int(d) for d in item.attribute(
                    'Dimensions').value().split(" ")])[::-1]
                assert np.all(fld_dims == dims)
                assert item.attribute('Format').value() == 'HDF'
                path = item.text().as_string()
                flds[fld] = {'path': path, 'dims': dims}

            grid['fields'] = flds
            rv['grids'][grid_name] = grid

    return rv
