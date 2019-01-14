# Fermi Surfaces

You can access the Fermi surface associated to a given dataset with 
`.S.fermi_surface`, which will give the Fermi surface integrated in a 
reasonable range (30 millivolts) of the chemical potential.

You can use this to rapidly plot Fermi surfaces

![Making a Fermi surface manually](static/manual-fs.png)

Alternatively, you can use 
`arpes.plotting.dispersion.labeled_fermi_surface` to get a Fermi surface 
that optionally includes the labeled high symmetry points.

![A labeled Fermi surface](static/labeled-fs.png)

You can also [add annotations manually](/annotations).