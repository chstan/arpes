# arpes.plotting.annotations module

**arpes.plotting.annotations.annotate\_cuts(ax, data, plotted\_axes,
include\_text\_labels=False,**kwargs)\*\*

> Example: annotate\_cuts(ax, conv, \[‘kz’, ‘ky’\], hv=80)
> 
>   - Parameters
>     
>       - **ax** –
>       - **data** –
>       - **plotted\_axes** –
>       - **include\_text\_labels** –
>       - **kwargs** –
> 
>   - Returns

**arpes.plotting.annotations.annotate\_point(ax, location, label,
delta=None,**kwargs)\*\*

**arpes.plotting.annotations.annotate\_experimental\_conditions(ax,
data, desc, show=False, orientation='top',**kwargs)\*\*

> Renders information about the experimental conditions onto a set of
> axes, also adjust the axes limits and hides the axes.
> 
> data should be the dataset described, and desc should be one of
> 
> ‘temp’, ‘photon’, ‘photon polarization’, ‘polarization’, or a number
> to act as a spacer in units of the axis coordinates
> 
> or a list of such items
> 
>   - Parameters
>     
>       - **ax** –
>       - **data** –
>       - **desc** –
> 
>   - Returns
