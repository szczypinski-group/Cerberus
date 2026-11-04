"""SOP 5.1 steps 8-10: shared building blocks for single points on best.xyz.

single_point  write, check and collect one ORCA single point (record.toml)
reference     step 8: DLPNO-CCSD(T)/CBS jobs and extrapolation
solvation     step 10: gas + solvation-model jobs, dG_solv
thermo        trunk free-energy terms reused by every method

Used by both branches (benchmark/, production/); which methods run, and where
results are stored, is decided there.
"""
