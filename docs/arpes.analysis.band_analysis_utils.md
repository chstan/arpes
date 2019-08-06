# arpes.analysis.band\_analysis\_utils module

**class arpes.analysis.band\_analysis\_utils.ParamType(value, stderr)**

> Bases: `tuple`
> 
> `stderr`
> 
> > Alias for field number 1
> 
> `value`
> 
> > Alias for field number 0

**arpes.analysis.band\_analysis\_utils.param\_getter(param\_name,
safe=True)**

> Constructs a function to extract a parameter value by name. Useful to
> extract data from inside an array of *lmfit.ModelResult* instances.
> 
>   - Parameters  
>     **param\_name** – Parameter name to retrieve. If you performed a
>     composite model fit,
> 
> make sure to include the prefix. :param safe: Guards against NaN
> values. This is typically desirable but sometimes it is advantageous
> to have NaNs fail an analysis quickly.
:return:

**arpes.analysis.band\_analysis\_utils.param\_stderr\_getter(param\_name,
safe=True)**

> Constructs a function to extract a parameter value by name. Useful to
> extract data from inside an array of *lmfit.ModelResult* instances.
> 
>   - Parameters  
>     **param\_name** – Parameter name to retrieve. If you performed a
>     composite model fit,
> 
> make sure to include the prefix. :param safe: Guards against NaN
> values. This is typically desirable but sometimes it is advantageous
> to have NaNs fail an analysis quickly. :return:
