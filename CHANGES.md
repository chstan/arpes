# Changes

Changes are listed with most recent versions at the top.


## 1.1.0 (2019-08-11)

### New:

1. Add a self-check utility for debugging installs, `import arpes; arpes.check()`
2. PyARPES can generate scan directives to make working at beamlines or nanoARPES endstations simpler. You 
   can now export a region or boundary of a region from a PyARPES analysis to a (first pass) LabView compatible 
   scan specification. For now this consists of a coordinate list and optional spectrum declaration.
3. `local_config.py` now has a programmatic interface in `arpes.config.override_settings`. 
4. Add `arpes.utilities.collections.deep_update`

### Changed:

1. Documentation overhaul, focusing on legibility for new users and installation instructions 

### Fixed:

1. Version requirements on `lmfit` are now correct after Nick added `SplitLorentzian` xarray compatible models    


## 1.0.2 (2019-08-08)

### New:

1. Moved to CI/CD on Azure Pipelines (https://dev.azure.com/lanzara-group/PyARPES)
2. Tests available for data loading and some limited analysis routines

### Changed:

1. Lanzara group Main Chamber data loading code will set a photon energy of 5.93 eV 
on all datasets by default

### Fixed:

1. `arpes.analysis.derivative.dn_along_axis` now properly accepts a smoothing function (`smooth_fn`) with the 
signature `xr.DataArray -> xr.DataArray`.

## 1.0.0 (June 2019)

### New:

1. First official release. API should be largely in place around most of PyARPES.