# arpes.models.band module

**class arpes.models.band.Band(label, display\_label=None, data=None)**

> Bases: `object`
> 
> `amplitude`
> 
> **band\_energy(coordinates)**
> 
> `band_width`
> 
> `center`
> 
> `center_stderr`
> 
> `coords`
> 
> `dims`
> 
> `display_label`
> 
> `fermi_velocity`
> 
> `fit_cls`
> 
> **get\_dataarray(var\_name, clean=True)**
> 
> `indexes`
> 
> `self_energy`
> 
> `sigma`
> 
> `velocity`

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
> `fit_cls`

**class arpes.models.band.BackgroundBand(label, display\_label=None,
data=None)**

> Bases:
> 
> `fit_cls`

**class arpes.models.band.FermiEdgeBand(label, display\_label=None,
data=None)**

> Bases:
> 
> `fit_cls`

**class arpes.models.band.AffineBackgroundBand(label,
display\_label=None, data=None)**

> Bases:
> 
> `fit_cls`
