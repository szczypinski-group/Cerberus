"""Identifiers (SOP 2.2, 2.3): canonical SMILES, InChI/InChIKey, keys."""

from __future__ import annotations

import hashlib

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


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
