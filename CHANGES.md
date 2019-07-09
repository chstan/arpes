# Changes

Changes are listed with most recent versions at the top.

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
