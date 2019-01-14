# arpes.preparation.tof\_preparation module

**arpes.preparation.tof\_preparation.build\_KE\_coords\_to\_time\_pixel\_coords(dataset:
xarray.core.dataset.Dataset,
interpolation\_axis)**

**arpes.preparation.tof\_preparation.build\_KE\_coords\_to\_time\_coords(dataset:
xarray.core.dataset.Dataset, interpolation\_axis)**

> Geometric transform assumes pixel -\> pixel transformations, so we
> need to get the index associated to the appropriate timing value
> :param dataset: :param interpolation\_axis: :return:

**arpes.preparation.tof\_preparation.process\_DLD(dataset:
xarray.core.dataset.Dataset)**

**arpes.preparation.tof\_preparation.process\_SToF(dataset:
xarray.core.dataset.Dataset)**

> This isn’t the best unit conversion function because it doesn’t
> properly take into account the Jacobian of the coordinate conversion.
> This can be fixed by multiplying each channel by the appropriate
> ammount, but it might still be best to use the alternative method.
> 
>   - Parameters  
>     **dataset** –
> 
>   - Returns
