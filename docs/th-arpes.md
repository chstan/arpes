# A Quick Introduction to ARPES

**A**ngle-**R**esolved **P**hoto**e**mission **S**pectroscopy is an experimental technique 
based on several refinements of the photoelectric effect initially observed by Heinrich Hertz 
in 1887. When a monochromatic beam of photons of energy $h\nu$ are incident upon a sample, 
measurement of the electron's kinetic energy and exit angle gives information about the momentum and 
energy ("band structure") of the electron state in the studied material. 

Directly, ARPES gives the binding energy ($\text{E}_\text{b}$) of the emitted electrons and the 
components of momentum parallel to the sample surface ($\textbf{k}_\parallel$). 
Crystalline translational symmetry is broken by the vacuum interface, so less 
information is available for the out of plane momentum ($\textbf{k}_\text{z}$) 
without varying the incident photon energy.

## Experiment

<div class="flex flex-column items-center">
  <figure class="flex-item figure">
    <img style="flex-align: center; max-width: 30em; margin-left: 10em;" class="flex-item" src="static/hemisphere-horizontal.png" alt="Hemispherical Energy Analyzer"/>
    <figcaption><strong>Figure 1.</strong> ARPES experimental geometry with hemispherical energy analyzer</figcaption>
  </figure>
</div>

From simple energy and momentum conservation arguments, we can explain the process by which ARPES
makes available the electronic band structure. In typical ARPES experiments, a hemispherical energy analyzer
simultaneously measures the intensity distribution of photoelectrons along a one dimensional linecut
of angle&mdash;defined by an entrance slit on the analyzer&mdash;and resolved by the photoelectron kinetic energy.
Briefly, photoelectrons emitted from different angles (green, blue, and orange in the above depiction) are 
incident at different locations on the entrance slit and follow different circular trajectories between 
the analyzer plates. Meanwhile, more and less energetic electrons follow longer and shorter paths
(depicted as a spread in blue, green, and orange curves) due to their different bend radii in the constant E-field 
produced by the analyzer. The analyzer therefore spatially filters the electrons to produce an image of the 
photoemitted beam. A final electron sensitive detector, typically a channel plate paired with a phosphor
screen + CCD or delay line + timing hardware, turns the spatially filtered electron signal into a digital one. 

If a photoemitted electron leaves the sample with an energy $\text{E}_\text{kin}$ at 
an angle $(\theta, \phi)$ to the sample normal, $\hat{\text{\textbf{z}}}$, the 
binding energy and $\textbf{k}_\parallel$ are given by conservation:

$$
\text{E}_{\text{b}} = \underbrace{h\nu}_\text{known} - W - \underbrace{\text{E}_\text{kin}}_\text{measured}
$$

$$
\textbf{p}_\parallel = \sqrt{2 m \text{E}_\text{kin}}\sin{\theta} \left(\cos \phi \hat{\mathbf{x}} + \sin \phi \hat{\mathbf{y}} \right).
$$

The sample workfunction $W$, giving the difference between the 
Fermi and vacuum levels, may or may not be known. The Fermi edge of a metallic sample 
(actual or a metal reference), nevertheless links the kinetic energy to the electron binding energy.

By turning the sample, $\hat{\text{\textbf{z}}}$ can be scanned away from the analyzer entrance,
making available the photoemission intensity over all values 
of $\textbf{k}_\parallel$. Consideration must be given 
to the difference between experimentally measured angles and the spherical polar angles relative to
$\hat{\text{\textbf{z}}}$, and this is where we will turn our attention in the next section on 
[momentum conversion](/momentum-conversion). 

To obtain high-quality data, ARPES experiments are conducted in an ultra-high vacuum 
chamber, typically better than $1\cdot10^{-10} \text{torr}$, which minimizes surface contamination and 
interactions between the photoemitted electrons and any potential interference between the emission and detection 
processes. Additionally, ARPES experiments are often performed at cryogenic temperatures to minimize 
thermal broadening of the data. This capability also allows for the study of 
high-temperature superconductors below their critical temperatures, where the electrons 
take on a fundamentally different structure.


