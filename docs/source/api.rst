.. currentmodule:: arpes

#############
API reference
#############

This page documents the PyARPES API by generation from signatures and
docstrings. You can use this and the source code to supplement the remainder 
of the PyARPES documentation.

Please note that this is not a complete API documentation page. Less used and 
internal APIs are not documented here.

Data-loading functions
======================

.. autosummary::
   :toctree: generated/

   io.load_data
   io.load_example_data
   io.example_data

Momentum Conversion
===================

Small-Angle Approximated and Volumetric Related
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   utilities.conversion.core.convert_to_kspace
   utilities.conversion.forward.convert_coordinate_forward
   utilities.conversion.forward.convert_through_angular_point
   utilities.conversion.forward.convert_through_angular_pair

Exact Forward Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   utilities.conversion.forward.convert_coordinates_to_kspace_forward


Utilities
~~~~~~~~~

.. autosummary::

   utilities.conversion.fast_interp.Interpolator
   utilities.conversion.bounds_calculations.full_angles_to_k
   utilities.conversion.remap_manipulator.remap_coords_to

Conversion Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~

You do not need to call these directly, but they are good references for anyone 
looking to implement a different kind of coordinate transform.

.. autosummary::

   utilities.conversion.base.CoordinateConverter
   utilities.conversion.kx_ky_conversion.ConvertKp
   utilities.conversion.kx_ky_conversion.ConvertKxKy
   utilities.conversion.kz_conversion.ConvertKpKz


General Analysis
================

Axis Methods
~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.general.rebin
   arpes.analysis.general.symmetrize_axis
   arpes.analysis.general.condense
   arpes.preparation.axis_preparation.normalize_dim
   arpes.preparation.axis_preparation.sort_axis

Experimental Resolution Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.resolution.total_resolution_estimate
   arpes.analysis.resolution.thermal_broadening_estimate
   arpes.analysis.resolution.beamline_resolution_estimate
   arpes.analysis.resolution.analyzer_resolution_estimate
   


Self-Energy
~~~~~~~~~~~

.. autosummary::

   arpes.analysis.self_energy.to_self_energy
   arpes.analysis.self_energy.fit_for_self_energy
   arpes.analysis.self_energy.estimate_bare_band
   arpes.analysis.self_energy.quasiparticle_lifetime
   


Spin-ARPES Related
~~~~~~~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.sarpes.to_intensity_polarization
   arpes.analysis.sarpes.to_up_down
   arpes.analysis.sarpes.normalize_sarpes_photocurrent


Fermi Edge Related
~~~~~~~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.gap.symmetrize
   arpes.analysis.gap.determine_broadened_fermi_distribution

Derivatives
~~~~~~~~~~~

.. autosummary::

   arpes.analysis.derivative.dn_along_axis
   arpes.analysis.derivative.curvature
   arpes.analysis.derivative.minimum_gradient

Smoothing and Filtering
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.filters.gaussian_filter_arr
   arpes.analysis.filters.boxcar_filter_arr
   arpes.analysis.savitzky_golay.savitzky_golay

Deconvolution
~~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.deconvolution.make_psf1d
   arpes.analysis.deconvolution.deconvolve_ice
   arpes.analysis.deconvolution.deconvolve_rl

Masks
~~~~~

.. autosummary::

   arpes.analysis.mask.apply_mask
   arpes.analysis.mask.apply_mask_to_coords
   arpes.analysis.mask.polys_to_mask
   arpes.analysis.mask.raw_poly_to_mask

Fermi Surface Pockets
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   arpes.analysis.pocket.curves_along_pocket
   arpes.analysis.pocket.edcs_along_pocket
   arpes.analysis.pocket.radial_edcs_along_pocket
   arpes.analysis.pocket.pocket_parameters
   

Background Removal
~~~~~~~~~~~~~~~~~~

Shirley Backgrounds
-------------------

.. autosummary::
   :toctree: generated/

   analysis.shirley.calculate_shirley_background
   analysis.shirley.remove_shirley_background

Convex Hull Backgrounds
-----------------------

.. autosummary::
   :toctree: generated/
   
   analysis.background.calculate_background_hull
   analysis.background.remove_background_hull

Incoherent Backgrounds
----------------------

.. autosummary::
   :toctree: generated/
   
   corrections.background.remove_incoherent_background


Array Alignment
~~~~~~~~~~~~~~~

Subpixel Correlation Based Alignment
------------------------------------

.. autosummary::
   :toctree: generated/

   analysis.align.align


Machine Learning
================

ML Based Decompositions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.decomposition.decomposition_along
   analysis.decomposition.pca_along
   analysis.decomposition.nmf_along
   analysis.decomposition.factor_analysis_along
   analysis.decomposition.ica_along

Interactive Decomposition Browsers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   widgets.pca_explorer

Deep Learning and PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for interpretation of results akin to those in fast.ai.

.. autosummary::
   :toctree: generated/

   deep_learning.interpret.Interpretation
   deep_learning.interpret.InterpretationItem

IO tools.

.. autosummary::
   :toctree: generated/

   deep_learning.io.from_portable_bin
   deep_learning.io.to_portable_bin

Transform pipelines.

.. autosummary::
   :toctree: generated/
   
   deep_learning.transforms.Identity
   deep_learning.transforms.ReversibleLambda
   deep_learning.transforms.ComposeBoth

Data Generation
~~~~~~~~~~~~~~~

Here are tools for simulating ARPES from theoretical models, including 
finite statistics, resolution modeling, and detector effects.

.. autosummary::
   :toctree: generated/

   simulation.sample_from_distribution
   simulation.cloud_to_arr
   simulation.SpectralFunctionMFL
   simulation.SpectralFunctionPhaseCoherent

   simulation.DetectorEffect
   simulation.NonlinearDetectorEffect


Data Loading Plugins
====================

.. autosummary::
   :toctree: generated/

   endstations.plugin.ALG_main.ALGMainChamber
   endstations.plugin.ALG_spin_ToF.SpinToFEndstation
   endstations.plugin.ANTARES.ANTARESEndstation
   endstations.plugin.Elettra_spectromicroscopy.SpectromicroscopyElettraEndstation
   endstations.plugin.example_data.ExampleDataEndstation
   endstations.plugin.fallback.FallbackEndstation
   endstations.plugin.HERS.HERSEndstation
   endstations.plugin.igor_export.IgorExportEndstation
   endstations.plugin.igor_plugin.IgorEndstation
   endstations.plugin.kaindl.KaindlEndstation
   endstations.plugin.MAESTRO.MAESTROMicroARPESEndstation
   endstations.plugin.MAESTRO.MAESTRONanoARPESEndstation
   endstations.plugin.MBS.MBSEndstation
   endstations.plugin.merlin.BL403ARPESEndstation
   endstations.plugin.SToF_DLD.SToFDLDEndstation


Curve Fitting
=============

Broadcast Fitting
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   fits.utilities.broadcast_model

General Curve Fitting Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   fits.utilities.result_to_hints

Models
~~~~~~

.. autosummary::
   :toctree: generated/
   :recursive:
   
   fits.fit_models

Helpful Methods for Setting Initial Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   xarray_extensions.SelectionToolAccessor.first_exceeding
   xarray_extensions.SelectionToolAccessor.max_in_window

Working with Broadcast Fit Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available as methods via ``.F`` accessor.

.. autosummary::
   :toctree: generated/
   :recursive:
   
   xarray_extensions.ARPESDatasetFitToolAccessor
   xarray_extensions.ARPESFitToolsAccessor

NorthStar Laue
==============

Loading Laue data from the NorthStar DAQ program

.. autosummary::
   :toctree: generated/
   
   laue.load_laue

Plotting
========

Interactive Utilities: Qt Based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   plotting.qt_tool.qt_tool
   plotting.qt_ktool.ktool
   plotting.fit_tool.fit_tool
   plotting.bz_tool.bz_tool

   plotting.basic_tools.path_tool
   plotting.basic_tools.mask_tool
   plotting.basic_tools.bkg_tool
   plotting.basic_tools.det_window_tool

General Utilities/Matplotlib Quality of Life
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An incomplete list of useful plotting utilities from ``arpes.utilities.plotting``.

.. autosummary::
   :toctree: generated/

   plotting.utils.path_for_plot
   plotting.utils.savefig
   plotting.utils.simple_ax_grid
   plotting.utils.invisible_axes
   plotting.utils.no_ticks
   plotting.utils.remove_colorbars
   plotting.utils.frame_with
   plotting.utils.unchanged_limits
   plotting.utils.plot_arr
   plotting.utils.imshow_mask
   plotting.utils.swap_xaxis_side
   plotting.utils.swap_yaxis_side
   plotting.utils.swap_axis_sides
   plotting.utils.latex_escape
   plotting.utils.fancy_labels
   plotting.utils.sum_annotation
   plotting.utils.summarize

Interactive Utilities: Bokeh Based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   plotting.dyn_tool.dyn
   plotting.dyn_tool.DynamicTool

   plotting.fit_inspection_tool.FitCheckTool
   plotting.mask_tool.MaskTool
   plotting.path_tool.PathTool
   plotting.band_tool.BandTool
   plotting.interactive.ImageTool

Interactive Utilities: Matplotlib Based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   widgets.pick_rectangles
   widgets.pick_points
   widgets.pca_explorer
   widgets.kspace_tool
   widgets.fit_initializer

Stack Plots
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   plotting.stack_plot.stack_dispersion_plot
   plotting.stack_plot.flat_stack_plot
   plotting.stack_plot.offset_scatter_plot

Spin-ARPES Plots
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   plotting.spin.spin_polarized_spectrum
   plotting.spin.spin_colored_spectrum
   plotting.spin.spin_difference_spectrum

Count-based and ToF Plots
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   plotting.tof.plot_with_std
   plotting.tof.scatter_with_std


Reference Plots
~~~~~~~~~~~~~~~

.. autosummary::

   plotting.spatial.plot_spatial_reference
   plotting.spatial.reference_scan_spatial


Curve Fitting Plots
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   plotting.fits.plot_fit
   plotting.fits.plot_fits
   plotting.parameter.plot_parameter

False Color Plots
~~~~~~~~~~~~~~~~~

.. autosummary::
   plotting.false_color.false_color_plot

Plotting with Brillouin Zones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   plotting.bz.bz_plot
   plotting.bz.plot_data_to_bz
   plotting.bz.overplot_standard
   plotting.bz.plot_plane_to_bz

Plot Annotations
~~~~~~~~~~~~~~~~

.. autosummary::

   plotting.annotations.annotate_cuts
   plotting.annotations.annotate_point
   plotting.annotations.annotate_experimental_conditions