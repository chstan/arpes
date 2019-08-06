# Changes

Changes are listed with most recent versions at the top.

Dates are in YYYY/MM/DD format.

Primary (X.-.-) version numbers are used to denote backwards incompatibilities
between versions, while minor (-.X.-) numbers primarily indicate new
features and documentation.

## 2.1.1 (2019-08-06)

### New

1. Improved type annotations
2. Slightly safer data loading in light of plugins: no need to call `load_plugins()` manually.

### Changed

### Fixed

1. Data moved to a location where it is available in PyPI builds


## 2.1.0 (2019-08-06)

### New:

1. Improved API documentation.
2. Most recent interative plot context is saved to `arpes.config.CONFIG['CURRENT_CONTEXT']`.
   This allows simple and transparent recovery in case you forget to save the context and
   performed a lot of work in an interactive session.
   Additionally, this means that matplotlib interactive tools should work transparently,
   as the relevant widgets are guaranteed to be kept in memory.
3. Improved provenance coverage for builtins.

### Changed:

1. Metadata reworked to a common format accross all endstations. 
   This is now documented appropriately with the data model. 

### Fixed:

1. MBS data loader now warns about unsatisfiable attributes and
   produces otherwise correct coordinates in the PyARPES format.
2. Some improvements made in the ANTARES data loader, still not as high
   quality as I would like though.

## 2.0.0 (2019-07-31)

### New:

1. Major rework in order to provide a consistent angle convention
2. New momentum space conversion widget allows
   setting offsets interactively
3. Fermi surface conversion functions now allow azimuthal rotations
4. New `experiment` module contains primitives for exporting
   scan sequences. This is an early addition towards being able
   to perform ARPES experiments from inside PyARPES.

   1. As an example: After conducting nano-XPS, you can use PCA to
      select your sample region and export a scan sequnce just over the
      sample ROI or over the border between your sample and another area.

### Changed:

1. All loaded data comes with all angles and positions as coordinates
2. All loaded data should immediately convert to momentum space
   without issue (though normal emission is not guaranteed!)
3. Documentation changes to reflect these adjustments to the data model


### Fixed:

1. Documentation link in README.rst is now correct.

## 1.2.0 (2019-07-18)

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


## 1.1.0 (2019-07-11)

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


## 1.0.2 (2019-07-08)

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
