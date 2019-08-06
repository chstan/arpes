# arpes.optics module

Some utilities for optics and optical design. This and the utilities in
arpes.laser should maybe be grouped into a separate place.

We don’t do any astigmatism aware calculations here, which might be of
practical utility.

Things to offer in the future:

1.    - Calculating beam characteristics (beam divergence, focus, etc)  
        from a sequence of knife edge tests or beam profiler images in
        order to facilitate common tasks like beam collimation,
        focusing, or preparing a new Tr-ARPES setup.

2.    - Nonlinear optics utilities including damage thresholds to
        allow  
        simpler design of harmonic generation for Tr-ARPES.

**arpes.optics.waist(wavelength, z, z\_R)**

**arpes.optics.waist\_R(waist\_0, m\_squared=1)**

**arpes.optics.rayleigh\_range(wavelength, waist, m\_squared=1)**

**arpes.optics.lens\_transfer(s, f, rayleigh\_range, m\_squared=1)**

> Produces s’’ :param s: :param f: :param f\_p: :param m\_squared:
> :return:

**arpes.optics.magnification(s, f, rayleigh\_range, m\_squared=1)**

> Calculates the magnification offered by a lens system. :param s:
> :param f: :param rayleigh\_range: :param m\_squared: :return:

**arpes.optics.waist\_from\_divergence(wavelength,
half\_angle\_divergence)**

**arpes.optics.waist\_from\_rr(wavelength, rayleigh\_range)**
