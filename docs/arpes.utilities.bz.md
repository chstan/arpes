# arpes.utilities.bz module

TODO: Standardize this module around support for some other library that
has proper Brillouin zone plotting, like in ASE.

This module also includes tools for masking regions of data against
Brillouin zones.

**arpes.utilities.bz.bz\_symmetry(flat\_symmetry\_points)**

**arpes.utilities.bz.bz\_cutter(symmetry\_points, reduced=True)**

> TODO UNFINISHED :param symmetry\_points: :param reduced: :return:

**arpes.utilities.bz.reduced\_bz\_selection(data)**

**arpes.utilities.bz.reduced\_bz\_axes(data)**

**arpes.utilities.bz.reduced\_bz\_mask(data,**kwargs)\*\*

**arpes.utilities.bz.reduced\_bz\_poly(data, scale\_zone=False)**

**arpes.utilities.bz.reduced\_bz\_axis\_to(data, S, include\_E=False)**

**arpes.utilities.bz.reduced\_bz\_E\_mask(data, S, e\_cut,
scale\_zone=False)**

**arpes.utilities.bz.axis\_along(data, S)**

> Determines which axis lies principally along the direction G-\>S.
> :param data: :param S: :return:

**arpes.utilities.bz.hex\_cell(a=1, c=1)**

**arpes.utilities.bz.hex\_cell\_2d(a=1)**

**arpes.utilities.bz.orthorhombic\_cell(a=1, b=1, c=1)**

**arpes.utilities.bz.process\_kpath(paths, cell, special\_points=None)**
