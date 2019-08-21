# Understanding ARPES: Momentum Conversion

The geometry we discussed in the introduction gives a complete picture of the experiment, 
but is not practical. For an experimenter, the situation is more complicated: while
for a single **E**nergy **D**istribution **C**urve (EDC) these equations can be directly solved, 
calculating the emission angle is not always straightforward because of the number of degrees 
of freedom on the six-axis sample goniometers which have become the standard for photoemission. 
Essentially all photoemission experiments now record
very high dimensional datasets either due to capable analyzers, or by scanning 
goniometer angles. As a result, to transform evenly-spaced volumetric
data in angle to evenly-spaced data in momentum, we actually require the inverse transform
from momentum-space to angle-space, at which point we can interpolate the data based on the local
structure in the recorded angle-resolved photocurrent.  

In the most general case we record the electron velocity with three angles (you can read more about how
PyARPES sets angle conventions and about its data format [here](/spectra)): $\phi$, the angle along the analyzer
slit in the case of hemispheres, $\psi$ the angle perpendicular to a hemispherical 
analyzer’s slit, and $\alpha$ a rotation angle about the spectrometer axis. In
our convention, horizontal slit hemispherical analyzers measure always with
$\alpha = 0$, while vertical slit analyzers measure with $\alpha = \pi/2$. 
The photoelectron velocity that the analyzer records, $\textbf{v}_\text{a}$ can be 
written as

$$
    \textbf{v}_\text{a} = \left[\begin{matrix}
           \cos\alpha\cos\psi\sin\phi - \sin\alpha\sin\psi \\
           \sin\alpha\cos\psi\sin\phi + \cos\alpha\sin\psi \\
           \cos\phi\cos\psi
         \end{matrix}\right]
$$


We need the velocity in the sample coordinate system, as
these are the ones that can be related to the crystal momentum. 
A six-axis ARPES goniometer implements three rotations about the sample 
normal ($\chi$), about an axis perpendicular to the cryostat ($\beta$), and finally
one around the cryostat axis ($\theta$). Depending on the design of the manipulator,
the order of the $\beta$ and $\theta$ rotations may be reversed. Finally, rotations from
the cryostat to sample normal coordinates $(\chi', \beta',\theta')$ must be performed if they arise
due to unintended or intentional offsets of the crystal and cryostat normal
vectors. Because these are often small we will absorb them into $(\chi, \beta,\theta)$ by the 
small angle approximation. 

$$
\textbf{v}_\text{s} = \text{R}(\chi,\hat{\textbf{z}})\text{R}(\beta,\hat{\textbf{x}})\text{R}
(\theta,\hat{\textbf{y}})\textbf{v}_\text{a}
$$
 
Finally, depending on the coordinates underlying the recorded ARPES
data, these equations must be inverted from the appropriate velocities to the
scanned analyzer. As an example, this can be done directly after small angle approximation in the
case of a hemispherical analyzer which has perpendicular electron deflectors
with $\alpha = 0$

$$
\begin{aligned}
\left(\phi - \theta\right) &\approx \arcsin\left(\frac{
\left(\text{R}\left(\chi,\hat{\textbf{z}}\right)\hat{\textbf{k}}\right)\cdot\hat{\textbf{x}}}
{\sqrt{1 - \left(\left(\text{R}\left(\chi, \hat{\textbf{z}}\right)\hat{\textbf{k}}\right)\cdot\hat{\textbf{y}}\right)^2}}\right) \\

\left(\psi - \beta\right) &\approx \arcsin\left(\left(\text{R}\left(\chi, \hat{\textbf{z}}\right)\hat{\textbf{k}}\right)
\cdot\hat{\textbf{y}}\right) \\
\end{aligned}
$$

The inverse transforms can be calculated for the remainder of the geometries, 
using one or both in-plane equations for fixed photon energy and
using the out-of-plane expression when the incident photon energy is varied. 
So long as the hardware for a particular experiment does not change
dramatically, these equations once calculated can be applied directly to data.

You can find a relatively complete implementation of these transforms in the small angle 
approximation picture <a href="/static/mathematica-coordinates.pdf">in this Mathematica notebook</a>.