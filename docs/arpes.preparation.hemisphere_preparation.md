# arpes.preparation.hemisphere\_preparation module

**arpes.preparation.hemisphere\_preparation.infer\_center\_pixel(arr:
xarray.core.dataarray.DataArray)**

**arpes.preparation.hemisphere\_preparation.stitch\_maps(arr:
xarray.core.dataarray.DataArray, arr2: xarray.core.dataarray.DataArray,
dimension='beta')**

> Stitches together two maps by appending and potentially dropping
> frames in the first dataset.
> 
> This is useful for beamline work when the beam is lost or in L-ARPES
> if laser output is blocked for part of a scan and a subsequent scan
> was taken to repair the problem. :param arr: Incomplete map :param
> arr2: completion of first map :return:
