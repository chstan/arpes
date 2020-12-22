arpes.utilities.funcutils module
================================

**class arpes.utilities.funcutils.Debounce(period)**

> Bases: `object`
>
> **reset()**

**arpes.utilities.funcutils.cycle(sequence)**

**arpes.utilities.funcutils.group\_by(grouping, sequence)**

**arpes.utilities.funcutils.iter\_leaves(tree: Dict\[str, Any\],
is\_leaf: Optional\[Callable\] = None) -&gt; Iterator\[Tuple\[str,
numpy.ndarray\]\]**

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
> Parameters  
> -   **tree** –
> -   **is\_leaf** –
>
> Returns  

**arpes.utilities.funcutils.lift\_dataarray\_to\_generic(f)**

> A functorial decorator that lifts functions with the signature
>
> (xr.DataArray, &gt;&gt;*&lt;&lt;args, &gt;&gt;*\*&lt;&lt;kwargs) -&gt;
> xr.DataArray
>
> to one with signature
>
> A = typing.Union\[xr.DataArray, xr.Dataset\] (A,
> &gt;&gt;*&lt;&lt;args, &gt;&gt;*\*&lt;&lt;kwargs) -&gt; A
>
> i.e. one that will operate either over xr.DataArrays or xr.Datasets.
> :param f: :return:
