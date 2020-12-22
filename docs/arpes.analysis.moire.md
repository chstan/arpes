arpes.analysis.moire module
===========================

arpes.moire includes some tools for analyzing moirés and data on moiré
heterostructures in particular.

All of the moirés discussed here are on hexagonal crystal systems.

**arpes.analysis.moire.angle\_between\_vectors(a, b)**

**arpes.analysis.moire.calc\_commensurate\_moire\_cell(underlayer\_a,
overlayer\_a, relative\_angle=0, swap\_angle=False)**

> Calculates nearly commensurate moire unit cells for two hexagonal
> lattices :return:

**arpes.analysis.moire.calculate\_bz\_vertices\_from\_direct\_cell(cell)**

**arpes.analysis.moire.generate\_other\_lattice\_points(a, b, ratio,
order=1, angle=0)**

**arpes.analysis.moire.generate\_segments(grouped\_points, a, b)**

**arpes.analysis.moire.higher\_order\_commensurability(lattice\_constant\_ratio,
order=2, angle\_range=None)**

> Unfinished
>
> Parameters  
> -   **lattice\_constant\_ratio** –
> -   **order** –
> -   **angle\_range** –
>
> Returns  

**arpes.analysis.moire.minimum\_distance(pts, a, b)**

**arpes.analysis.moire.mod\_points\_to\_lattice(pts, a, b)**

**arpes.analysis.moire.plot\_simple\_moire\_unit\_cell(underlayer\_a,
overlayer\_a, relative\_angle, ax=None, offset=True,
swap\_angle=False)**

> Plots a digram of a moiré unit cell. :return:

**arpes.analysis.moire.unique\_points(pts)**
