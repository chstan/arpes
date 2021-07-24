A Quick Introduction to ARPES
=============================

**A**\ ngle-**R**\ esolved **P**\ hoto\ **e**\ mission
**S**\ pectroscopy is an experimental technique based on several
refinements of the photoelectric effect initially observed by Heinrich
Hertz in 1887. When a monochromatic beam of photons of energy
:math:`h\nu` are incident upon a sample, measurement of the electron’s
kinetic energy and exit angle gives information about the momentum and
energy (“band structure”) of the electron state in the studied material.

Directly, ARPES gives the binding energy (:math:`\text{E}_\text{b}`) of
the emitted electrons and the components of momentum parallel to the
sample surface (:math:`\textbf{k}_\parallel`). Crystalline translational
symmetry is broken by the vacuum interface, so less information is
available for the out of plane momentum (:math:`\textbf{k}_\text{z}`)
without varying the incident photon energy.

Experiment
----------

.. container:: flex flex-column items-center

   .. raw:: html

      <figure class="flex-item figure">

   .. raw:: html

      <figcaption>

   Figure 1. ARPES experimental geometry with hemispherical energy
   analyzer

   .. raw:: html

      </figcaption>

   .. raw:: html

      </figure>

From simple energy and momentum conservation arguments, we can explain
the process by which ARPES makes available the electronic band
structure. In typical ARPES experiments, a hemispherical energy analyzer
simultaneously measures the intensity distribution of photoelectrons
along a one dimensional linecut of angle—defined by an entrance slit on
the analyzer—and resolved by the photoelectron kinetic energy. Briefly,
photoelectrons emitted from different angles (green, blue, and orange in
the above depiction) are incident at different locations on the entrance
slit and follow different circular trajectories between the analyzer
plates. Meanwhile, more and less energetic electrons follow longer and
shorter paths (depicted as a spread in blue, green, and orange curves)
due to their different bend radii in the constant E-field produced by
the analyzer. The analyzer therefore spatially filters the electrons to
produce an image of the photoemitted beam. A final electron sensitive
detector, typically a channel plate paired with a phosphor screen + CCD
or delay line + timing hardware, turns the spatially filtered electron
signal into a digital one.

If a photoemitted electron leaves the sample with an energy
:math:`\text{E}_\text{kin}` at an angle :math:`(\theta, \phi)` to the
sample normal, :math:`\hat{\textbf{z}}`, the binding energy and
:math:`\textbf{k}_\parallel` are given by conservation:

.. math::

   \text{E}_{\text{b}} = \underbrace{h\nu}_\text{known} - W - \underbrace{\text{E}_\text{kin}}_\text{measured}

.. math::

   \textbf{p}_\parallel = \sqrt{2 m \text{E}_\text{kin}}\sin{\theta} \left(\cos \phi \hat{\mathbf{x}} + \sin \phi \hat{\mathbf{y}} \right).

The sample workfunction :math:`W`, giving the difference between the
Fermi and vacuum levels, may or may not be known. The Fermi edge of a
metallic sample (actual or a metal reference), nevertheless links the
kinetic energy to the electron binding energy.

By turning the sample, :math:`\hat{\textbf{z}}` can be scanned
away from the analyzer entrance, making available the photoemission
intensity over all values of :math:`\textbf{k}_\parallel`. Consideration
must be given to the difference between experimentally measured angles
and the spherical polar angles relative to
:math:`\hat{\textbf{z}}`, and this is where we will turn our
attention in the next section on `momentum
conversion </momentum-conversion>`__.

To obtain high-quality data, ARPES experiments are conducted in an
ultra-high vacuum chamber, typically better than
:math:`1\cdot10^{-10} \text{torr}`, which minimizes surface
contamination and interactions between the photoemitted electrons and
any potential interference between the emission and detection processes.
Additionally, ARPES experiments are often performed at cryogenic
temperatures to minimize thermal broadening of the data. This capability
also allows for the study of high-temperature superconductors below
their critical temperatures, where the electrons take on a fundamentally
different structure.
