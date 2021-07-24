Migration Guide
===============

Migrating to PyARPES v3
~~~~~~~~~~~~~~~~~~~~~~~

You no longer need to provide data spreadsheets. See the documentation at :doc:`loading-data` for details
on the the data loading API.

Many improvements have been made to performance. For the most part, these changes are completely 
transparent, as in momentum conversion which is 10-50x faster than in PyARPES v2. However, PyARPES
v3 uses multiprocessing for large groups of curve fits, through the `parallel=True/False` kwarg to 
`arpes.fits.utilities.broadcast_model`. If you do not want to use parallel curve fitting, simply pass
`False` to this kwarg when you do your curve fitting.

A downside to parallel curve fitting is that there is a substantial memory overhead: about 200MB / core
on your computer. As most high-core computers also have more memory headroom, we felt this an appropriate 
default behavior. Again, you can avoid the overhead with the parallelization kwarg.

For more detailed changes see the :doc:`CHANGELOG`.