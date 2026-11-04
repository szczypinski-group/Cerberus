"""SOP 5.1 step 11, 2.7, 4.4: branch-neutral analysis of computed energies.

assembly   species G and reaction dE / dG / ddG / log K from energy terms
errors     experimental K, signed errors against a reference or experiment
tables     tidy CSV writing

Both branches use these; which energies go in (a method grid with a DLPNO
reference, or one locked method) is decided by benchmark/ and production/.
"""
