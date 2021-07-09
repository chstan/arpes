"""Note: this used to be a single function in xarray_extensions but was refactored out.

TODO This was done to better support simultaneous traversal of collections, but is 
still in process.

Transform has similar semantics to matrix multiplication, the dimensions of the
output can grow or shrink depending on whether the transformation is size preserving,
grows the data, shinks the data, or leaves in place.

As an example, let us suppose we have a function which takes the mean and
variance of the data:

f: [dimension], coordinate_value -> [{'mean', 'variance'}]

And a dataset with dimensions [X, Y]

Then calling

data.G.transform('X', f)

maps to a dataset with the same dimension X but where Y has been replaced by
the length 2 label {'mean', 'variance'}. The full dimensions in this case are
['X', {'mean', 'variance'}].

Please note that the transformed axes always remain in the data because they
are iterated over and cannot therefore be modified.

The transform function `transform_fn` must accept the coordinate of the
marginal at the currently iterated point.

if isinstance(self._obj, xr.Dataset):
    raise TypeError(
        "transform can only work on xr.DataArrays for now because of "
        "how the type inference works"
    )

dest = None
for coord, value in self.iterate_axis(axes):
    new_value = transform_fn(value, coord, *args, **kwargs)

    if dest is None:
        new_value = transform_fn(value, coord, *args, **kwargs)

        original_dims = [d for d in self._obj.dims if d not in value.dims]
        original_shape = [self._obj.shape[self._obj.dims.index(d)] for d in original_dims]
        original_coords = {k: v for k, v in self._obj.coords.items() if k not in value.dims}

        full_shape = original_shape + list(new_value.shape)

        new_coords = original_coords
        new_coords.update({k: v for k, v in new_value.coords.items() if k not in original_coords})
        new_dims = original_dims + list(new_value.dims)
        dest = xr.DataArray(
            np.zeros(full_shape, dtype=dtype or new_value.data.dtype),
            coords=new_coords,
            dims=new_dims,
        )

    dest.loc[coord] = new_value

return dest
"""
