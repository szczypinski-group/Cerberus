"""Measured equilibria from NMR (SOP 4.4): spectra -> conversion -> log K.

Reads Bruker 1D 1H experiments with nmrglue, integrates the aldehyde and
imine signals in the ppm ranges given in <branch>/<experimental_dir>/
samples.toml, and writes one <reaction key>.toml per reaction there: the
measured log K the benchmark and production analyses compare against.

nmr           read a Bruker experiment (processed or raw), ppm axis, integrals
equilibrium   samples.toml -> conversion -> K -> <reaction key>.toml
"""
