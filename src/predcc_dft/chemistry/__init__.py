"""Chemistry definitions shared by several SOP levels.

identifiers  SOP 2.2/2.3: canonical SMILES, InChI(Key), compound/reaction keys
smarts       SOP 5.2: aldehyde, nucleophile-class and reaction SMARTS
geometry     acylhydrazone amide rotamer (syn/anti) from 3D coordinates
"""

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
