# Advanced Topics in Curve Fitting

## Some Introduction

The data that we work with in physics can often be described by a 
model. 

$$x \rightarrow f(x)$$

where we can say that the function $f$ generates the observed data over some range of
physical interest $x$. As a concrete example, in photoemission over the binding energy inveral $x : [25, 40]\\text{eV}$
the density of states might be well modeled by a function $f$ which describes the locations
of the tungsten 4f core level peaks and their spin-orbit splitting.

As physicists though, finding the modeling function $f$ is aided by the observation that many physical phenomena are similar:
in $WS_2$ the locations of the 4f peaks differ from the peak locations in $WTe_2$ slightly because of different
chemical bonding, but the doublet structure and spin-orbit splitting remain as characteristic features.

This prompts us to describe the model function $f$ as a member of a family parameterized by some variables $\\sigma$.
Our model describes the data now by $x \rightarrow f(x; \\sigma)$.