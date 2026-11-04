"""SOP 5.1 steps 4-7: from seed geometries to one checked best geometry.

conformers    steps 4-5: CREST per seed, CENSO on the pooled ensemble
dft_opt       step 5, alternative: plain ORCA optimisations of CREST conformers
frequencies   step 6: r2SCAN-3c/CPCM Opt+Freq, retries
sanity        step 7: structure check, isomer populations, best.xyz
"""
