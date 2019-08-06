# arpes.endstations.igor\_utils module

**arpes.endstations.igor\_utils.shim\_wave\_note(path)**

> Hack to read the corrupted wavenote out of the h5 files that Igor has
> been producing. h5 dump still produces the right value, so we use it
> from the command line in order to get the value of the note.
> 
> This is not necessary unless you are trying to read HDF files exported
> from Igor (the preferred way before we developed an Igor data loading
> plugin).
> 
>   - Parameters  
>     **path** – Location of the file
> 
>   - Returns
