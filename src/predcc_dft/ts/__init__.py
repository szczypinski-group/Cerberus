"""TS module (SOP 10): transition states, rate constants and kinetics.

Starts from the trunk best.xyz like the two branches and never writes outside
ts/. For every imine condensation of the selection, both steps (addition to
the hemiaminal, dehydration to imine + water) are searched with autodE, once
per proton shuttle of [ts] shuttles, with the step's own forming/breaking
bonds; each TS is checked with an ORCA Freq and turned into rate constants;
a microkinetic model then gives the concentrations over time under the
experiment's conditions ([ts.conditions]: 50 mM, 2 mol% TFA, water produced
by the reaction, 24 h). Every molecule and every TS has one TOML file with
all its paths and numbers (ts/species/<label>/ts.toml, channel.toml).

mechanism   atom-mapped reactant and product complexes per step and shuttle
geometry    3D complexes and conformers (RDKit + MMFF94)
engine      hand-over to autodE: direction, fragments, active bonds
autode_job  the autodE search run inside a SLURM task (imports autodE)
tasks       ORCA and autodE task folders: write and judge
stage       job arrays per stage; ts.toml, channel.toml, reaction.toml
kinetics    mass-action ODE model (SciPy BDF)
report      barriers, rates, concentrations over time, summary tables
"""
