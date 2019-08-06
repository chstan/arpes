# arpes.simulation module

Phenomenological models and detector simulation for accurate modelling
of ARPES data.

Currently we offer relatively rudimentary detecotor modeling, mostly
providing only nonlinearity and some stubs, but a future release will
provide reasonably accurate modeling of the trapezoidal effect in
hemispherical analyzers, fixed mode artifacts, dust, and more.

Additionally we offer the ability to model the detector response at the
level of individual electron events using a point spread or more
complicated response. This allows the creation of reasonably realistic
spectra for testing new analysis techniques or working on machine
learning based approaches that must be robust to the shortcomings of
actual ARPES data.

**arpes.simulation.apply\_psf\_to\_point\_cloud(point\_cloud, shape,
sigma=None)**

> Takes a point cloud and turns it into a spectrum. Finally, smears it
> by a gaussian PSF given through the *sigma* parameter.
> 
> In the future, we should also allow for specifying a particular PSF.
> :param point\_cloud: :param shape: :param sigma: :return:

**arpes.simulation.sample\_from\_distribution(distribution, N=5000)**

> Given a probability distribution in ND modeled by an array providing
> the PDF, sample individual events coming from this PDF.
> 
>   - Parameters
>     
>       - **distribution** –
>       - **N** –
> 
>   - Returns

**class arpes.simulation.SpectralFunction(k=None, omega=None, T=None)**

> Bases: `object`
> 
> Model for a band with self energy.
> 
> **bare\_band()**
> 
> **fermi\_dirac(omega)**
> 
> **imag\_self\_energy()**
> 
> **measured\_spectral\_function()**
> 
> **occupied\_spectral\_function()**
> 
> **real\_self\_energy()**
> 
> > Default to Kramers-Kronig
> 
> **sampled\_spectral\_function(n\_electrons=50000, n\_cycles=1,
> psf=None)**
> 
> **self\_energy()**
> 
> **spectral\_function()**

**class arpes.simulation.DetectorEffect**

> Bases: `object`
> 
> Detector effects are callables that map a spectrum into a new
> transformed one. This might be used to imprint the image of a grid,
> dust, or impose detector nonlinearities.

**class arpes.simulation.SpectralFunctionBSSCO(k=None, omega=None,
T=None, delta=None, gamma\_s=None, gamma\_p=None)**

> Bases:
> 
> Implements the spectral function for BSSCO as reported in
> PhysRevB.57.R11093 and explored in ["Collapse of superconductivity in
> cuprates via ultrafast quenching of phase
> coherence"](https://arxiv.org/pdf/1707.02305.pdf).
> 
> **self\_energy()**
> 
> **spectral\_function()**

**class arpes.simulation.SpectralFunctionMFL(k=None, omega=None, T=None,
a=None, b=None)**

> Bases:
> 
> Implements the Marginal Fermi Liquid spectral function, more or less.
> 
> **imag\_self\_energy()**

**class arpes.simulation.SpectralFunctionPhaseCoherent(k=None,
omega=None, T=None, delta=None, gamma\_s=None, gamma\_p=None)**

> Bases:
> 
> **self\_energy()**

**class arpes.simulation.NonlinearDetectorEffect(gamma=None,
nonlinearity=None)**

> Bases:
> 
> Implements power law detector nonlinearities.

**class arpes.simulation.FixedModeDetectorEffect(spacing=None,
periodic='hex', detector\_efficiency=None)**

> Bases:
> 
> Implements a grid or pore structure of an MCP or field termination
> mesh. Talk to Danny or Sam about getting hyperuniform point cloud
> distributions to use for the pore structure.
> 
> `detector_imprint`

**class arpes.simulation.DustDetectorEffect**

> Bases:
> 
> TODO, dust.

**class arpes.simulation.TrapezoidalDetectorEffect**

> Bases:
> 
> TODO model that phi(pixel) is also a function of binding energy, i.e.
> that the detector has severe aberrations at low photoelectron kinetic
> energy (high retardation ratio).

**class arpes.simulation.WindowedDetectorEffect**

> Bases:
> 
> TODO model the finite width of the detector window as recorded on a
> camera.
