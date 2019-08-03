# Adding Support for Beamlines or Lab Facilities

**Note:** This is an advanced section, you can skip this unless you need or want to extend PyARPES to cover
more data formats. 

One of the overarching design goals of PyARPES is to provide a completely uniform, pragmatic, 
and understandable approach to loading and performing common analyses of ARPES data. Practically,
this consists of at least

1. Programmatic or automated data loading + saving, including from networked locations
2. Data assuming the same ultimate format and structure, where this is reasonable (excluding SARPES)
3. k-space conversion "just working", without requiring additional effort from either
   the PyARPES authors or users once implemented
4. Backward compatibility with Igor Pro experiments and binary data types
   
In order to meet these constraints in light of the large variety of data formats, PyARPES 
first requires that data is normalized to a single data type (NetCDF) which is well supported by
`xarray` our data primitive of choice. This normalization step is performed automatically and 
without user intervention aside from invocation of `prepare_raw_files()`. 

Internally, `prepare_raw_files` invokes a particular data loading plugin on the basis of the value of
the `location` [spreadsheet column](/analysis-spreadsheets). This value should match the 
`PRINCIPAL_NAME` or one of the `ALIASES` of a plugin. The matched plugin will be used to actually 
load the data.

All plugins are loaded at configuration time (IPython kernel startup or invocation of `arpes.setup`) from
`arpes.endstations.plugin`. You can add more at runtime by calling
 
```python
arpes.endstations.add_endstation(MyPlugin)
```

## How a Plugin Loads Data

In order to load data, a plugin need only support the function `load(scan_description: dict, **kwargs)`.
In practice though, most plugins subclass `arpes.endstations.EndstationBase`, whose `load` function
looks like this:

```python
def load(self, scan_desc: dict=None, **kwargs):
        """
        Loads a scan from a single file or a sequence of files.

        :param scan_desc:
        :param kwargs:
        :return:
        """
        resolved_frame_locations = self.resolve_frame_locations(scan_desc)
        resolved_frame_locations = [f if isinstance(f, str) else str(f) 
                                    for f in resolved_frame_locations]

        frames = [self.load_single_frame(fpath, scan_desc, **kwargs) 
                  for fpath in resolved_frame_locations]
        frames = [self.postprocess(f) for f in frames]
        concatted = self.concatenate_frames(frames, scan_desc)
        concatted = self.postprocess_final(concatted, scan_desc)

        if 'id' in scan_desc:
            concatted.attrs['id'] = scan_desc['id']

        return concatted
```

The core steps are:

1. Find associated files (corresponding to the whole dataset or to "frames" of the dataset) 
   with `resolve_frame_locations`
2. Load each frame individually with `load_single_frame`
3. Perform some additional work on each frame with `postprocess`
4. Concatenate the frames with `concatenate_frames`
5. Perform some final work on the constructed dataset with `postprocess_final`
6. Add an `id`

The reason for the "frames" concept is that some beamlines split datasets up over many files 
(MERLIN at the ALS, as an example), while others produce just one. In the case that only one file is 
present, `concatenate_frames` will return just this data.

## Writing Your Own Plugin

To write your own plugin to be included in PyARPES, make a file containing a single class, subclassing
`EndstationBase`. If it represents a instrument with a hemispherical electron analyzer, 
subclass as well `HemisphericalEndstation`. If it is associated with a synchrotron, subclass 
`SynchrotronEndstation`.

```python
class MySamplePlugin(SynchrotronEndstation, EndstationBase, HemisphericalEndstation):
    # use this plugin for any data associated with the locations "AMAZING-ARPES-LAB", 
    # "Best lab", or "AAL" 
    PRINCIPAL_NAME = 'AMAZING-ARPES-LAB'
    ALIASES = ['Best lab', 'AAL',]
    
    RENAME_KEYS = {
        # Our LabView software weirdly calls the temperature "ThermalEnergy", and 
        # "SFE_0" is the spectrometer center binding energy 
        'ThermalEnergy': 'temp',
        'SFE_0': 'binding_offset',
    }
    
    def load_single_frame(self, frame_path: str=None, scan_desc: dict=None, **kwargs):
        # data loading logic here...
        pass
```

In the above, you should fill in `load_single_frame` so that it returns a `xr.Dataset` with a 
`spectrum` data variable. For examples of how the actual loading code might look, have a look at the 
definitions of the currently implemented plugins in `merlin.py` (SES binary multiframe format), 
`MAESTRO.py` (FITS single frame format), and `ALG_main.py` (FITS single frame format).

Finally, ensure your plugin is exported in your module's `__all__` attribute

```python
__all__ = ('MySamplePlugin',)
```

### Renaming attributes

`RENAME_KEYS` can be used to rename attributes in the event that your VIs or spectrometer drivers 
produce. In the example above, we rename "ThermalEnergy" to "temp" and "SFE_0" to "binding_offset".

You can include as many of these key renamings as you like, in addition to the standard ones performed
automatically.