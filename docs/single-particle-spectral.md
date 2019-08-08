# Understanding ARPES: the Single Particle Spectral Function

## Single + Many Body Physics

ARPES provides a direct measurement of the angle resolved photocurrent 
$I(\textbf{k}, \omega)$. This can be expressed in terms of a 
dipole interaction matrix element $M$, the Fermi-Dirac 
distribution for state occupancy, and the single-particle spectral function $A(\mathbf{k}, \omega)$:

$$
I(\textbf{k}, \omega) = M\left(\textbf{k}, \omega, \mathbf{k}\cdot\mathbf{A}\right)f(\omega)A(\mathbf{k}, \omega).
$$

Although matrix element contribution can in general be difficult to disentangle, it is typically 
slowly varying in momentum and photon energy, and therefore ARPES can be thought of as measuring also 
$A(\textbf{k}, \omega)$. 

$$
\textbf{A}(\mathbf{k}, \omega) = -\frac{1}{\pi}
\frac{\textcolor{blue}{\Sigma''(\textcolor{black}{\mathbf{k}, \omega})}}
{[\omega - \hspace{-0.25em}\underbrace{\textcolor{green}{\epsilon_0(\textcolor{black}{\mathbf{k}})}}_{\textcolor{green}{\text{Bare band}}}\hspace{-0.25em} - \hspace{-0.4em}\underbrace{\textcolor{red}{\Sigma'\left(\textcolor{black}{\mathbf{k},\omega}\right)}}_{\textcolor{red}{\text{E-renormalization}}}\hspace{-0.9em}]^2 + 
[\underbrace{\textcolor{blue}{\Sigma''\left(\textcolor{black}{\mathbf{k},\omega}\right)}}_{
\textcolor{blue}{\text{Lifetimes}}
}]^2}
$$

From this quantity much can be extracted, including single-particle
<span style="color: green;">band structure</span>, 
<span style="color: red;">energy renormalization by interaction</span>, and measurement
of the
<span style="color: blue;">quasiparticle lifetimes</span>.
ARPES therefore makes available momentum resolved information on electron-electron and electron-boson 
interaction in real material systems.

Additionally, a significant amount can be learned about the sample physics from the strength of the 
dipole matrix element $M$. The value of $\mathbf{A}\cdot\mathbf{k}$ can be tuned independently of the experimental 
geometry by varying the photon polarization&mdash;often discretely at synchrotrons or continuously with lasers&mdash;and
offers insight into the orbital character of the material band structure. In some cases, full information about the 
matrix elements at fixed experimental geometry can be recorded. Many new ARPES experiments
are able to record a large solid angle, $> 0.2 \text{ sr}$, without moving the sample at all either 
by physically moving the analyzer rather than the sample, by virtually moving the slit of the analyzer using 
electrostatic deflectors, or by using detectors which measure simultaneously two axes in momentum space. This latter 
class includes most notably **p**hoto-**e**mission **e**lectron **m**icroscopes (PEEMs) and 
**a**ngle-**r**esoved **t**ime-**o**f-**f**light analyzers (ARToFs).

## Laser ARPES

From the conservation equations, we can see that given a fixed angular resolution of $\Delta \phi$
of our analyzer, we may obtain a higher momentum resolution for smaller photoelectron kinetic energy, 
$\Delta\mathbf{k}\propto\sqrt{\text{E}_\text{kin}}\Delta\phi$.
UV and VUV lasers based on frequency doubling in BBO or KBBF and operating at very high photon fluxes with narrow spectral 
linewidths of a few meV allow ARPES studies of materials at a much higher resolution than is possible
at synchrotron sources.

Pulsed laser sources also enable ARPES studies of materials perturbed out of equilibrium by a strong visible 
to IR pulse, at the penalty of larger bandwidth $\Delta \nu$ and therefore worse energy 
resolution. For a bandwidth limited pulse, the ARPES probe pulse duration is constrained by 
the time bandwidth product 

$$
\Delta \tau (\text{fs}) \Delta E (\text{meV}) \approx 1823.
$$  

As it is relatively typical to desire ARPES resolutions somewhat better than $50\text{ meV}$,
pump-probe ARPES experiments are often conducted with temporal resolutions worse than $50\text{ fs}$ or so,
but still fast enough to resolve the quasiparticle recovery and phonon-mediated changes to the band structure. 