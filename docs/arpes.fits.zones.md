arpes.fits.zones module
=======================

Contains models and utilities for “curve fitting” the forward momentum
conversion function. This allows for rapid fine tuning of the angle
offsets used in momentum conversion.

If the user calculates, places, or curve fits for the high symmetry
locations in the data $H\_i(phi,psi, heta,eta,chi)$, these can be used
as waypoints to find a set of $Deltaphi$, $Delta heta$, $Deltachi$, etc.
that minimize

$$ sum\_i ext{min}\_j &gt;&gt;^2 $$

where $S\_j$ enumerates the high symmetry points of the known Brillouin
zone, and $ ext{P}$ is the function that maps forwards from angle to
momenta. This can also be used to calculate moiré information, but using
the (many) available high symmetry points of the moiré superlattice to
get a finer estimate of relative angle alignment, lattice
incommensuration, and strain than is possible using the constituent
lattices and Brillouin zones.
