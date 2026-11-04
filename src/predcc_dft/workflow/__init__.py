"""How the levels are run: commands, the SLURM round driver, progress.

cli           the `predcc` command
pipeline      one round: local steps, write and submit every ready stage
status        per-molecule progress table
orca_inputs   one-off ORCA inputs for pilots, with the pipeline's settings
"""
