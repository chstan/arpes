# arpes.utilities.conversion.remap\_manipulator module

**arpes.utilities.conversion.remap\_manipulator.remap\_coords\_to(arr,
reference\_arr)**

> This needs to be thought out a bit more, namely to take into account
> better the manipulator scan degree of freeedom.
> 
> Produces coords which provide the scan cut path for the array `arr` as
> seen in the coordinate system defined by the manipulator location in
> `reference_arr`. This is useful for plotting locations of cuts in a
> FS.
> 
> This code also assumes that a hemispherical analyzer was used, because
> it uses the coordinate ‘phi’. :param arr: Scan that represents a cut
> we would like to understand :param reference\_arr: Scan providing the
> desired destination coordinates :return: Coordinates dict providing
> the path cut by the dataset `arr`
