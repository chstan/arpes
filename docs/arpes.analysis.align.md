arpes.analysis.align module
===========================

This module contains methods that get unitful alignments of one array
against another. This is very useful for determining spectral shifts
before doing serious curve fitting analysis or similar.

Implementations are included for each of 1D and 2D arrays, but this
could be simply extended to ND if we need to. I doubt that this is
necessary and don’t mind the copied code too much at the present.

**arpes.analysis.align.align(a, b,**kwargs)\*\*

**arpes.analysis.align.align1d(a, b, subpixel=True)**

> Returns the unitful offset of b in a for 1D arrays :param a: :param b:
> :param subpixel: :return:

**arpes.analysis.align.align2d(a, b, subpixel=True)**

> Returns the unitful offset of b in a for 2D arrays :param a: :param b:
> :param subpixel: :return:
