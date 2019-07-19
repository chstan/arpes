# Changes

Changes are listed with most recent versions at the top.

## 1.2.0 (2019-08-18)

### New:

1. Ship example data so that people can try what is in the documentation
   immediately after installing
2. Users can now load data directly, i.e. without a spreadsheet, with
   `load_without_dataset`, in the future this will support matches based
   on the current working directory.
3. Users are better warned when spreadsheets are not in the correct format.
   Spreadsheet loading is also generally more permissive, see below.


### Changed:

1. Added more tests, especially around data loading, spreadsheet loading
   and normalization.

### Fixed:

1. Spreadsheet loading no longer relatively silently fails due to whitespace in column names,
   we might nevertheless consider doing more significant cleaning of data at the very initial
   stages of spreadsheet loading.
2. Spreadsheet loading now appropriately uses safe_read universally. `modern_clean_xlsx_dataset`
   is functionally deprecated, but will stay in at least for a little while I consider its removal.
3. Spreadsheet loading now appropriately handles files with 'cleaned' in their name.
4. Spreadsheet writing will not include the index and therefore an unnamed column when saving to disk.


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
