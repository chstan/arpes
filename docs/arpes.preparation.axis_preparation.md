# arpes.preparation.axis\_preparation module

**arpes.preparation.axis\_preparation.flip\_axis(arr:
xarray.core.dataarray.DataArray, axis\_name, flip\_data=True)**

**arpes.preparation.axis\_preparation.normalize\_dim(arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
dim\_or\_dims, keep\_id=False)**

> Normalizes the intensity so that all values along arr.sum(dims other
> than those in `dim_or_dims`) have the same value. The function
> normalizes so that the average value of cells in the output is 1.
> :param dim\_name: :return:

**arpes.preparation.axis\_preparation.dim\_normalizer(dim\_name)**

**arpes.preparation.axis\_preparation.transform\_dataarray\_axis(f,
old\_axis\_name: str, new\_axis\_name: str, new\_axis, dataset:
xarray.core.dataarray.DataArray, prep\_name, transform\_spectra=None,
remove\_old=True)**

**arpes.preparation.axis\_preparation.normalize\_total(data:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\])**

**arpes.preparation.axis\_preparation.sort\_axis(data:
xarray.core.dataarray.DataArray, axis\_name)**
