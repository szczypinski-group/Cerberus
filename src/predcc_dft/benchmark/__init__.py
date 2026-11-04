"""Benchmark branch (SOP Part II, 5.1 steps 8-11, 6, 8).

Which functional and basis set reproduces DLPNO-CCSD(T)/CBS (and experiment)
best?

Starts from the trunk's best.xyz and never writes outside benchmark/:

reference   DLPNO-CCSD(T)/CBS + every solvation model     (steps 8, 10)
grid        every method of benchmark/methods.toml          (step 9)
report      species/reaction tables, errors, ranking        (step 11, 8)
ranking     MAE, SD, ... per method, linkage class, level   (8)
candidate   production_candidate.toml for `predcc promote`  (8.1)
"""
