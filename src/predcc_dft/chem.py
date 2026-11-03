"""Shared chemistry: standardisation, identifiers, class and product SMARTS."""

from __future__ import annotations

import hashlib

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

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


class InputError(ValueError):
    """Molecule rejected at step 1, with a reason."""


def standardise(
    smiles: str, strip_salts: bool = True, neutralise: bool = True
) -> tuple[Chem.Mol, list[str]]:
    """Parse, keep the largest fragment, neutralise, sanitise.

    Returns (mol, notes).
    """
    notes: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InputError("unparseable SMILES")
    if len(Chem.GetMolFrags(mol)) > 1:
        if not strip_salts:
            raise InputError("multiple fragments")
        chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        removed = Chem.MolToSmiles(mol)
        mol = chooser.choose(mol)
        notes.append(f"salt/fragments stripped from {removed}")
    if neutralise and Chem.GetFormalCharge(mol) != 0:
        before = Chem.MolToSmiles(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        if Chem.MolToSmiles(mol) != before:
            notes.append(f"neutralised from {before}")
    Chem.SanitizeMol(mol)
    return mol, notes


def canonical(mol: Chem.Mol, isomeric: bool = True) -> str:
    """Canonical SMILES without explicit hydrogens."""
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=isomeric)


def inchikey(mol: Chem.Mol) -> str:
    """Standard InChIKey."""
    return Chem.MolToInchiKey(Chem.RemoveHs(mol))


def inchi(mol: Chem.Mol) -> str:
    """Standard InChI."""
    return Chem.MolToInchi(Chem.RemoveHs(mol))


def compound_key(key: str) -> str:
    """First InChIKey block: groups E/Z isomers of one molecule (SOP 2.2)."""
    return key.split("-")[0]


def properties(mol: Chem.Mol) -> dict:
    """Formula, weight, size, flexibility, charge and multiplicity."""
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": round(Descriptors.MolWt(mol), 4),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "charge": Chem.GetFormalCharge(mol),
        "multiplicity": 1
        + sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
    }


def match_sites(mol: Chem.Mol, smarts: str) -> list[int]:
    """Indices of the atom mapped 1 in every match of smarts (unique)."""
    patt = Chem.MolFromSmarts(smarts)
    mapped = [a.GetIdx() for a in patt.GetAtoms() if a.GetAtomMapNum() == 1]
    pos = mapped[0] if mapped else 0
    return sorted(
        {m[pos] for m in mol.GetSubstructMatches(patt, uniquify=False)}
    )


def reaction_key(
    prefix: str, reactants: list[str], products: list[str]
) -> tuple[str, str]:
    """Deterministic reaction key from compound keys (SOP 2.3)."""

    def side(keys: list[str]) -> str:
        counts: dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        return "+".join(
            f"{n}*{k}" if n != 1 else k for k, n in sorted(counts.items())
        )

    canonical_string = f"{side(reactants)}>>{side(products)}"
    digest = hashlib.sha256(canonical_string.encode()).hexdigest()[:16]
    return f"{prefix}-{digest}", canonical_string
