Fermi Surfaces
==============

You can access the Fermi surface associated to a given dataset with
``.S.fermi_surface``, which will give the Fermi surface integrated in a
reasonable range (30 millivolts) of the chemical potential.

You can use this to rapidly plot Fermi surfaces

.. figure:: _static/manual-fs.png
   :alt: Making a Fermi surface manually

   Making a Fermi surface manually

Alternatively, you can use
``arpes.plotting.dispersion.labeled_fermi_surface`` to get a Fermi
surface that optionally includes the labeled high symmetry points.

.. figure:: _static/labeled-fs.png
   :alt: A labeled Fermi surface

   A labeled Fermi surface

You can also `add annotations manually </annotations>`__.
