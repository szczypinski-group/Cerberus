"""SMARTS definitions (SOP 5.2): reactive groups, classes, reactions."""

from __future__ import annotations

from rdkit import Chem

# Reactive aldehyde carbon (formaldehyde excluded: needs a carbon neighbour).
ALDEHYDE_SMARTS = "[CX3H1:1](=O)[#6]"

# Nucleophile classes, keyed on the reacting NH2 (atom map 1). Mutually
# exclusive by the atom bonded to the NH2. Order matters only for reporting.
NUCLEOPHILE_CLASSES = {
    "primary_amine": "[NX3H2;!$(N[#6]=[O,S,N]):1][#6]",
    "hydroxylamine": "[NX3H2:1][OX2H1]",
    "alkoxyamine": "[NX3H2:1][OX2][#6]",
    "hydrazine": "[NX3H2:1][NX3;!$(N[#6]=[O,S]);!$(N[SX4](=O)=O)]",
    "hydrazide": "[NX3H2:1][NX3H1][CX3]=[OX1]",
    # generated products only (azine route)
    "hydrazone": "[NX3H2:1][NX2]=[CX3]",
}

CLASS_TO_LINKAGE = {
    "primary_amine": "imine",
    "hydroxylamine": "oxime",
    "alkoxyamine": "oxime",
    "hydrazine": "hydrazone",
    "hydrazide": "acylhydrazone",
    "hydrazone": "azine",
}

# Reaction SMARTS (SOP 5.2). Map 1 = carbonyl C, map 3 = reacting N.
# Unmapped template atoms (the carbonyl O) are removed -> water.
REACTION_SMARTS = {
    "imine": (
        "[CX3H1:1](=O)[#6:2].[NX3H2;!$(N[#6]=[O,S,N]):3][#6:4]"
        ">>[#6:2][CX3H1:1]=[N:3][#6:4]"
    ),
    "oxime": (
        "[CX3H1:1](=O)[#6:2].[NX3H2:3][OX2:4]>>[#6:2][CX3H1:1]=[N:3][O:4]"
    ),
    "hydrazone": (
        "[CX3H1:1](=O)[#6:2]."
        "[NX3H2:3][NX3;!$(N[#6]=[O,S]);!$(N[SX4](=O)=O):4]"
        ">>[#6:2][CX3H1:1]=[N:3][N:4]"
    ),
    "acylhydrazone": (
        "[CX3H1:1](=O)[#6:2].[NX3H2:3][NX3H1:4][CX3:5]=[O:6]"
        ">>[#6:2][CX3H1:1]=[N:3][N:4][C:5]=[O:6]"
    ),
    "azine": (
        "[CX3H1:1](=O)[#6:2].[NX3H2:3][NX2:4]=[CX3:5]"
        ">>[#6:2][CX3H1:1]=[N:3][N:4]=[C:5]"
    ),
}

# Amide rotamer dihedral O=C-N-N in acylhydrazones (map: O, C, N(H), N=).
ACYLHYDRAZONE_DIHEDRAL_SMARTS = "[OX1]=[CX3][NX3H1][NX2]=[CX3]"

WATER_SMILES = "O"


def match_sites(mol: Chem.Mol, smarts: str) -> list[int]:
    """Indices of the atom mapped 1 in every match of smarts (unique)."""
    patt = Chem.MolFromSmarts(smarts)
    mapped = [a.GetIdx() for a in patt.GetAtoms() if a.GetAtomMapNum() == 1]
    pos = mapped[0] if mapped else 0
    return sorted(
        {m[pos] for m in mol.GetSubstructMatches(patt, uniquify=False)}
    )
