# arpes.provenance module

Provides data provenance for PyARPES. Most analysis routines built into
PyARPES support provenance. Of course, Python is a dynamic language and
nothing can be done to prevent the experimenter from circumventing the
provenance scheme.

All the same, between analysis notebooks and the data provenenace
provided by PyARPES, we provide an environment with much higher standard
for reproducible analysis than many other current analysis environments.

This provenenace record is automatically exported when using the built
in plotting utilities. Additionally, passing *used\_data* to the PyARPES
*savefig* wrapper allows saving provenance information even for bespoke
plots created in a Jupyter cell.

PyARPES also makes it easy to opt into data provenance for new analysis
functions by providing convenient decorators. These decorators inspect
data passed at runtime to look for and update provenance entries on
arguments and return values.

**arpes.provenance.attach\_id(data)**

> Ensures that an ID is attached to a piece of data, if it does not
> already exist. IDs are generated at the time of identification in an
> analysis notebook. Sometimes a piece of data is created from nothing,
> and we might need to generate one for it on the spot. :param data:
> :return:

**arpes.provenance.provenance(child\_arr:
xarray.core.dataarray.DataArray, parent\_arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset,
List\[Union\[xarray.core.dataarray.DataArray,
xarray.core.dataset.Dataset\]\]\], record, keep\_parent\_ref=False)**

> Function that updates the provenance for a piece of data with a single
> parent.
> 
>   - Parameters
>     
>       - **child\_arr** –
>       - **parent\_arr** –
>       - **record** –
>       - **keep\_parent\_ref** –
> 
>   - Returns

**arpes.provenance.provenance\_from\_file(child\_arr:
Union\[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset\],
file, record)**

> Builds a provenance entry for a dataset corresponding to loading data
> from a file. This is used by data loaders at the start of an analysis.
> :param child\_arr: :param file: :param record: :return:

**arpes.provenance.provenance\_multiple\_parents(child\_arr: (\<class
'xarray.core.dataarray.DataArray'\>, \<class
'xarray.core.dataset.Dataset'\>), parents, record,
keep\_parent\_ref=False)**

> Similar to *provenance* updates the data provenance information for
> data with multiple sources or “parents”. For instance, if you
> normalize a piece of data “X” by a metal reference “Y”, then the
> returned data would list both “X” and “Y” in its history.
> 
>   - Parameters
>     
>       - **child\_arr** –
>       - **parents** –
>       - **record** –
>       - **keep\_parent\_ref** –
> 
>   - Returns

**arpes.provenance.save\_plot\_provenance(plot\_fn)**

> A decorator that automates saving the provenance information for a
> particular plot. A plotting function creates an image or movie
> resource at some location on the filesystem.
> 
> In order to hook into this decorator appropriately, because there is
> no way that I know of of temporarily overriding the behavior of the
> open builtin in order to monitor for a write.
> 
>   - Parameters  
>     **plot\_fn** – A plotting function to decorate
> 
>   - Returns

**arpes.provenance.update\_provenance(what, record\_args=None,
keep\_parent\_ref=False)**

> Provides a decorator that promotes a function to one that records data
> provenance.
> 
>   - Parameters
>     
>       -   - **what** – Description of what transpired, to put into
>             the  
>             record.
>     
>       -   - **record\_args** – Unused presently, will allow recording
>             args  
>             into record.
>     
>       -   - **keep\_parent\_ref** – Whether to keep a pointer to the  
>             parents in the hierarchy or not.
> 
>   - Returns  
>     decorator
