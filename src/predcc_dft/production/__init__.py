"""Production branch (SOP Part V, 12): one locked method, many molecules.

One functional, basis set and solvation model (production/production.toml).

Starts from the same trunk best.xyz as the benchmark, never runs the method
grid or DLPNO-CCSD(T), and never writes outside production/:

stage    the locked single point + solvation per molecule, v<version>/
report   G, dG, log K tables per production version
"""
