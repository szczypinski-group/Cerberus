"""Geometry helpers shared by coordinate generation and structure checks."""

from __future__ import annotations

from rdkit import Chem

from predcc_dft.chemistry.smarts import ACYLHYDRAZONE_DIHEDRAL_SMARTS


def amide_rotamer_atoms(mol: Chem.Mol) -> tuple[int, int, int, int] | None:
    """Atoms of the acylhydrazone O=C-N-N dihedral, if any."""
    match = mol.GetSubstructMatch(
        Chem.MolFromSmarts(ACYLHYDRAZONE_DIHEDRAL_SMARTS)
    )
    return tuple(match[:4]) if match else None


def rotamer_label(dihedral_deg: float) -> str:
    """Return syn (|dihedral| < 90 deg) or anti."""
    return "syn" if abs(dihedral_deg) < 90.0 else "anti"
