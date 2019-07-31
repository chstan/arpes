# The PyARPES Data Model and Conventions

## Coordinate conventions

The PyARPES coordinate representation is summarized in the figure below. 
Three spatial coordinates specify the manipulator translation relative 
to a fixed origin. Two manipulator angles ("theta" and "beta") specify the
orientation of the sample normal relative to the analyzer axis. A final azimuthal 
angle "chi" specifies the rotation of the sample face.

These six angles are not enough to fully specify the photocurrent though,
because the analyzer observes an angular cut which can be sometimes independently
manipulated. The angle along the analyzer slit is always labelled as "phi",
while the angle perpendicular to this one is labelled as "psi". This psi angle 
will be familiar to those using a deflector that allows recording Fermi surfaces
without sample motion. Finally, some analyzers allow rotation
along the analyzer axis. This is the "alpha" angle. As a convention
we will take alpha=0 when the the slit of the analyzer is in the x-z plane.


![Hemispherical Analyzer Angular Conventions](static/angle-conventions.png)


# The Sign of the Binding Energy

PyARPES makes a choice to represent the binding energy as negative below the 
Fermi level, this is opposite of the physical value (you can think of it instead
as the photoelectron kinetic energy but offset to choose zero kinetic energy at the 
Fermi level), but it has the advantage of making math simpler, and orienting plots
and figures in an aesthetically pleasing way.