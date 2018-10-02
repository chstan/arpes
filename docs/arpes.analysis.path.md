# arpes.analysis.path module

**arpes.analysis.path.discretize\_path(path:
xarray.core.dataset.Dataset, n\_points=None, scaling=None)**

> Shares logic with slice\_along\_path :param path: :param n\_points:
> :return:

**arpes.analysis.path.path\_from\_points(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
symmetry\_points\_or\_interpolation\_points)**

> Acceepts a list of either tuples or point references. Point references
> can be string keys to *.attrs\[‘symmetry\_points’\]* This is the same
> behavior as *analysis.slice\_along\_path* and underlies the logic
> there. :param data: :param
> symmetry\_points\_or\_interpolation\_points: :return:
