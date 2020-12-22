arpes.models.band module
========================

**class arpes.models.band.AffineBackgroundBand(label,
display\_label=None, data=None)**

> Bases:
>
> **property fit\_cls**

**class arpes.models.band.BackgroundBand(label, display\_label=None,
data=None)**

> Bases:
>
> **property fit\_cls**

**class arpes.models.band.Band(label, display\_label=None, data=None)**

> Bases: `object`
>
> **property amplitude**
>
> **band\_energy(coordinates)**
>
> **property band\_width**
>
> **property center**
>
> **property center\_stderr**
>
> **property coords**
>
> **property dims**
>
> **property display\_label**
>
> **property fermi\_velocity**
>
> **property fit\_cls**
>
> **get\_dataarray(var\_name, clean=True)**
>
> **property indexes**
>
> **property self\_energy**
>
> **property sigma**
>
> **property velocity**

**class arpes.models.band.FermiEdgeBand(label, display\_label=None,
data=None)**

> Bases:
>
> **property fit\_cls**

**class arpes.models.band.MultifitBand(label, display\_label=None,
data=None)**

> Bases:
>
> Convenience class that reimplements reading data out of a composite
> fit result
>
> **get\_dataarray(var\_name, clean=True)**

**class arpes.models.band.VoigtBand(label, display\_label=None,
data=None)**

> Bases:
>
> **property fit\_cls**
