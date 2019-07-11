# arpes.utilities.funcutils module

**class arpes.utilities.funcutils.Debounce(period)**

> Bases: `object`
> 
> **reset()**

**arpes.utilities.funcutils.lift\_dataarray\_to\_generic(f)**

> A functorial decorator that lifts functions with the signature
> 
> (xr.DataArray, \>\>*\<\<args, \>\>*\*\<\<kwargs) -\> xr.DataArray
> 
> to one with signature
> 
> A = typing.Union\[xr.DataArray, xr.Dataset\] (A, \>\>*\<\<args,
> \>\>*\*\<\<kwargs) -\> A
> 
> i.e. one that will operate either over xr.DataArrays or xr.Datasets.
> :param f: :return:

**arpes.utilities.funcutils.iter\_leaves(tree, is\_leaf=None)**

> Iterates across the leaves of a nested dictionary. Whether a
> particular piece of data counts as a leaf is controlled by the
> predicate *is\_leaf*. By default, all nested dictionaries are
> considered not leaves, i.e. an item is a leaf if and only if it is not
> a dictionary.
> 
> Iterated items are returned as key value pairs.
> 
> As an example, you can easily flatten a nested structure with
> *dict(leaves(data))*
> 
>   - Parameters
>     
>       - **tree** –
>       - **is\_leaf** –
> 
>   - Returns
