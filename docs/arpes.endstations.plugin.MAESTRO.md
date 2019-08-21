# arpes.endstations.plugin.MAESTRO module

**class arpes.endstations.plugin.MAESTRO.MAESTROMicroARPESEndstation**

> Bases: `arpes.endstations.plugin.MAESTRO.MAESTROARPESEndstationBase`
> 
> Implements data loading at the microARPES endstation of ALS’s MAESTRO.
> 
> `ALIASES = ['BL7', 'BL7.0.2', 'ALS-BL7.0.2', 'MAESTRO']`
> 
> `ANALYZER_INFORMATION = {'analyzer': 'R4000' ... ular_deflectors':
> True}`
> 
> `ATTR_TRANSFORMS = {'SF_SLITN': <functi ... ESEndstation.<lambda>>}`
> 
> `MERGE_ATTRS = {'mcp_voltage': None ... ', 'undulator_z': None}`
> 
> `PRINCIPAL_NAME = 'ALS-BL7'`
> 
> `RENAME_COORDS = {'X': 'x', 'Y': 'y', 'Z': 'z'}`
> 
> `RENAME_KEYS = {'LMOTOR0': 'x', 'LM ... onic', 'mono_eV': 'hv'}`

**class arpes.endstations.plugin.MAESTRO.MAESTRONanoARPESEndstation**

> Bases: `arpes.endstations.plugin.MAESTRO.MAESTROARPESEndstationBase`
> 
> Implements data loading at the nanoARPES endstation of ALS’s MAESTRO.
> 
> `ALIASES = ['BL7-nano', 'BL7.0.2-nano', 'ALS-BL7.0.2-nano',
> 'MAESTRO-nano']`
> 
> `ANALYZER_INFORMATION = {'analyzer': 'DA-30' ... lar_deflectors':
> False}`
> 
> `ATTR_TRANSFORMS = {'SF_SLITN': <functi ... ESEndstation.<lambda>>}`
> 
> `ENSURE_COORDS_EXIST = ['long_x', 'long_y', ... _y',
> 'physical_long_z']`
> 
> `MERGE_ATTRS = {'beta': 0, 'mcp_vol ... ', 'undulator_z': None}`
> 
> `PRINCIPAL_NAME = 'ALS-BL7-nano'`
> 
> `RENAME_COORDS = {'Optics Stage': 'op ... long_y', 'Z': 'long_z'}`
> 
> `RENAME_KEYS = {'EPU_E': 'undulator ... : 'undulator_harmonic'}`
> 
> **postprocess\_final(data: xarray.core.dataset.Dataset, scan\_desc:
> dict = None)**
> 
> `static unwind_serptentine(data: xarray.core.dataset.Dataset) ->
> xarray.core.dataset.Dataset`
> 
> > MAESTRO supports a serpentine (think snake the computer game) scan
> > mode to minimize the motion time for coarsely locating samples.
> > Unfortunately, the DAQ just dumps the raw data, so we have to unwind
> > it ourselves. :param data: :return:
> 
> `static update_hierarchical_coordinates(data:
> xarray.core.dataset.Dataset)`
> 
> > Nano-ARPES endstations often have two sets of spatial coordinates, a
> > long-range piezo inertia or stepper stage, sometimes outside vacuum,
> > and a fast, high resolution piezo scan stage that may or may not be
> > based on piezo inertia (“slip-stick”) type actuators.
> > 
> > Additionally, any spatially imaging experiments like PEEM or the
> > transmission operating mode of hemispherical analyzers have two
> > spatial coordinates, the one on the manipulator and the imaged axis.
> > In these cases, this imaged axis will always be treated in the same
> > role as the high-resolution motion axis of a nano-ARPES system.
> > 
> > Working in two coordinate systems is frustrating, and it makes
> > comparing data cumbersome. In PyARPES x,y,z is always the total
> > inferrable coordinate value, i.e. (+/- long range +/- high
> > resolution) as appropriate. You can still access the underlying
> > coordinates in this case as *long\_{dim}* and *short\_{dim}*.
> > 
> >   - Parameters  
> >     **data** –
> > 
> >   - Returns
