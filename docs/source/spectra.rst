The PyARPES Data Model and Conventions
======================================

Coordinate conventions
----------------------

The PyARPES coordinate representation is summarized in the figure below.
Three spatial coordinates specify the manipulator translation relative
to a fixed origin. Two manipulator angles (:math:`\theta` and
:math:`\beta`) specify the orientation of the sample normal relative to
the analyzer axis. A final azimuthal angle :math:`\chi` specifies the
rotation of the sample face.

These six angles are not enough to fully specify the photocurrent
though, because the analyzer observes an angular cut which can be
sometimes independently manipulated. The angle along the analyzer slit
is always labelled as :math:`\phi`, while the angle perpendicular to
this one is labelled as :math:`\psi`. This :math:`\psi` angle will be
familiar to those using a deflector that allows recording Fermi surfaces
without sample motion. Finally, some analyzers allow rotation along the
analyzer axis. This is the :math:`\alpha` angle. As a convention we will
take :math:`\alpha=0` when the the slit of the analyzer is in the x-z
plane.

.. figure:: _static/angle-conventions.png
   :alt: Hemispherical Analyzer Angular Conventions

   Hemispherical Analyzer Angular Conventions

Hierarchal Spatial Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nano-ARPES endstations often have two sets of spatial coordinates, a
long-range piezo inertia or stepper stage, sometimes outside vacuum, and
a fast, high resolution piezo scan stage that may or may not be based on
piezo inertia (“slip-stick”) type actuators.

Additionally, any spatially imaging experiments like PEEM or the
transmission operating mode of hemispherical analyzers have two spatial
coordinates, the one on the manipulator and the imaged axis. In these
cases, this imaged axis will always be treated in the same role as the
high-resolution motion axis of a nano-ARPES system.

Working in two coordinate systems is frustrating, and it makes comparing
data cumbersome. In PyARPES x,y,z is always the total inferrable
coordinate value, i.e. (+/- long range +/- high resolution) as
appropriate. You can still access the underlying coordinates in this
case as ``long_{dim}`` and ``short_{dim}``.

ARPES Metadata
--------------

Convenient and consistent metadata conventions are essential in data
analysis. Without consistent conventions, even basic analysis has to be
done essentially manually and cannot easily be reused. Furthermore,
offering convenient conventions for metadata reduces friction inherent
in starting analysis for on data from new beamlines.

PyARPES offers strong metadata support for experiments conducted at
synchrotron beamlines and those performed with lasers. Ultimately, the
quality of metadata you get is limited by what is included in the data
by the DAQ software.

To help mitigate this, all spreadsheet columns are attached to the
appropriate data, allowing you to manually specify metadata that is not
otherwise recorded by default.

Units
^^^^^

Spatial and angular coordinates are reported in millimeters and radians
respectively. Temperatures are everywhere recorded in Kelvin. Relative
times are reported in seconds. Currents are recorded in nanoamp unit.
Pressures are recorded in torr. Potentials are recorded in volts. Laser
pulse durations and other pump-probe quantities are reported in
picoseconds. Energies are reported in electron volts. Fluences are
reported in units of micro-Joules per square centimeter. Frequencies are
reported in Hz.

Deviations from these units are reported as relevant below.

Coordinates
~~~~~~~~~~~

Coordinates are the most important metadata available. PyARPES
guarantees that every piece of data loaded through the provided plugins
contain the photon and binding energies as well as all of the six
analyzer and sample angular coordinates. The physical sample position in
millimeter units is also provided.

.. figure:: _static/coords-info.png
   :alt: Example Coordinates

   Example Coordinates

Scan Information
~~~~~~~~~~~~~~~~

This includes coarse information about the scan that was performed to
collect this ARPES data. You can access it at ``.S.scan_info``

.. figure:: _static/scan-info.png
   :alt: Scan Metadata

   Scan Metadata

A Note on Polarizations
^^^^^^^^^^^^^^^^^^^^^^^

In order to be able to represent elliptical polarizations, PyARPES
reports photon polarizations in (rotation angle, phase angle) format.

Experiment Information
~~~~~~~~~~~~~~~~~~~~~~

You can access the experimental conditions with the
``.S.experiment_info`` accessor.

.. figure:: _static/experiment-info.png
   :alt: Experiment Metadata

   Experiment Metadata

Analyzer Settings
~~~~~~~~~~~~~~~~~

.. figure:: _static/analyzer-info.png
   :alt: Analyzer Metadata

   Analyzer Metadata

Beamline Settings
~~~~~~~~~~~~~~~~~

Metadata about the beamline is collected under ``.S.beamline_info``

.. figure:: _static/beamline-info.png
   :alt: Beamline Metadata

   Beamline Metadata

Data Acquisition Settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata about data acquisition settings is collected under
``.S.daq_info``. Because DAQ information depends highly on the
implementation of DAQ programs, this collection of metadata varies
somewhat across beamlines and sources.

.. figure:: _static/daq-info.png
   :alt: DAQ Metadata

   DAQ Metadata

Laser/Pump-Probe Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/laser-info.png
   :alt: Laser Metadata

   Laser Metadata

Sample Information
~~~~~~~~~~~~~~~~~~

.. figure:: _static/sample-info.png
   :alt: Sample Metadata

   Sample Metadata

The Sign of the Binding Energy
------------------------------

PyARPES makes a choice to represent the binding energy as negative below
the Fermi level, this is opposite of the physical value (you can think
of it instead as the photoelectron kinetic energy but offset to align
zero energy at the Fermi level). Nevertheless, it has the advantage of
making math simpler, and orienting plots and figures in an aesthetically
pleasing way.
