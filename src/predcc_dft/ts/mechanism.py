"""Stationary points of each mechanistic step as atom-mapped molecular graphs.

SOP 10.2: condensation runs in two steps through a hemiaminal:

    addition      aldehyde + amine        -> hemiaminal        (C-N forms)
    dehydration   hemiaminal              -> imine + water     (C-O breaks)

In both, one proton moves from the amine N-H to an oxygen. Without a shuttle
it moves directly (a strained 4-membered TS); with a shuttle it is relayed:
the shuttle takes the N-H proton on its acceptor atom and gives its own
proton to the oxygen, and comes out unchanged (SOP 10.2, 'proton shuttles').

Shuttles (one per TS here):
    water     H2O              accepts on O, gives an O-H
    amine     a second amine   accepts on N, gives an N-H
    tfa       CF3COOH          accepts on C=O, gives the O-H  (8-ring relay)
    tfa_salt  R-NH3+ CF3COO-   accepts on O-, gives an N+-H; leaves as
                               amine + TFA (the acid-base pair re-forms fast)

The reactant and product complexes of a step share one atom order, as the
NEB path search requires; product bond orders and charges are re-derived
from the edited connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

STEPS = ("addition", "dehydration")
SHUTTLES = ("none", "water", "amine", "tfa", "tfa_salt")

ALDEHYDE = Chem.MolFromSmarts("[CX3H1:1](=[OX1:2])[#6]")
AMINE = Chem.MolFromSmarts("[NX3H2;!$(N[#6]=[O,S,N]):1][#6]")
HEMIAMINAL = Chem.MolFromSmarts("[CX4H1:1]([OX2H1:2])[NX3H1:3]")
TFA_SMILES = "OC(=O)C(F)(F)F"
WATER_SMILES = "O"


class MechanismError(ValueError):
    """A step cannot be built for these molecules."""


@dataclass
class Transfer:
    """One hydrogen moving from atom src to atom dst (complex indices)."""

    h: int
    src: int
    dst: int


@dataclass
class Step:
    """Reactant and product complex of one step, sharing atom indices."""

    name: str
    shuttle: str
    reactant: Chem.Mol
    product: Chem.Mol
    forming: list[tuple[int, int]] = field(default_factory=list)
    breaking: list[tuple[int, int]] = field(default_factory=list)
    transfers: list[Transfer] = field(default_factory=list)

    @property
    def charge(self) -> int:
        """Total charge of the complex."""
        return Chem.GetFormalCharge(self.reactant)

    def fragments(self, which: str) -> list[str]:
        """Canonical SMILES of the separate molecules of one side."""
        mol = self.reactant if which == "reactant" else self.product
        return sorted(
            Chem.MolToSmiles(Chem.RemoveHs(f))
            for f in Chem.GetMolFrags(mol, asMols=True)
        )


def _with_h(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MechanismError(f"unparseable SMILES {smiles}")
    return Chem.AddHs(mol)


def _protonated(amine_smiles: str) -> Chem.Mol:
    """R-NH3+ of a primary amine (on its reacting NH2)."""
    mol = Chem.RWMol(Chem.MolFromSmiles(amine_smiles))
    n = mol.GetSubstructMatch(AMINE)[0]
    atom = mol.GetAtomWithIdx(n)
    atom.SetFormalCharge(1)
    atom.SetNumExplicitHs(3)
    atom.SetNoImplicit(True)
    Chem.SanitizeMol(mol)
    return Chem.AddHs(mol)


def _h_on(mol: Chem.Mol, idx: int) -> int:
    for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            return nb.GetIdx()
    raise MechanismError(f"atom {idx} carries no hydrogen")


def _combine(parts: list[Chem.Mol]) -> tuple[Chem.Mol, list[int]]:
    """One molecule of several fragments, plus each fragment's offset."""
    combined, offsets, n = parts[0], [0], parts[0].GetNumAtoms()
    for part in parts[1:]:
        offsets.append(n)
        combined = Chem.CombineMols(combined, part)
        n += part.GetNumAtoms()
    combined = Chem.Mol(combined)
    Chem.SanitizeMol(combined)
    return combined, offsets


def _shuttle(name: str, amine_smiles: str) -> tuple[Chem.Mol, int, int, int]:
    """Shuttle molecule and its (acceptor, donor, donated H) indices."""
    if name == "water":
        mol = _with_h(WATER_SMILES)
        return mol, 0, 0, _h_on(mol, 0)
    if name == "amine":
        mol = _with_h(amine_smiles)
        n = mol.GetSubstructMatch(AMINE)[0]
        return mol, n, n, _h_on(mol, n)
    if name == "tfa":
        mol = _with_h(TFA_SMILES)
        acc = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[CX3](=[OX1:1])[OX2H1]")
        )
        don = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1:1]")
        )
        return mol, acc[1], don[2], _h_on(mol, don[2])
    if name == "tfa_salt":
        ammonium = _protonated(amine_smiles)
        carboxylate = _with_h("[O-]C(=O)C(F)(F)F")
        mol, (o_a, o_c) = _combine([ammonium, carboxylate])
        n = mol.GetSubstructMatch(Chem.MolFromSmarts("[NX4+]"))[0]
        o_minus = (
            o_c
            + carboxylate.GetSubstructMatch(Chem.MolFromSmarts("[OX1-]"))[0]
        )
        return mol, o_minus, n, _h_on(mol, n)
    raise MechanismError(f"unknown shuttle {name!r}; use one of {SHUTTLES}")


def _assign_orders(rw: Chem.RWMol, charge: int) -> Chem.Mol:
    """Bond orders and formal charges from connectivity alone."""
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(i), 0.0, 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineBondOrders(mol, charge=charge, embedChiral=False)
    mol.RemoveAllConformers()
    Chem.SanitizeMol(mol)
    return mol


def _relay(
    rw: Chem.RWMol,
    h_n: int,
    n: int,
    o: int,
    shuttle: tuple[int, int, int] | None,
) -> list[Transfer]:
    """Move the N-H proton to O, directly or through the shuttle."""
    rw.RemoveBond(n, h_n)
    if shuttle is None:
        rw.AddBond(o, h_n, Chem.BondType.SINGLE)
        return [Transfer(h_n, n, o)]
    acc, don, h_s = shuttle
    rw.AddBond(acc, h_n, Chem.BondType.SINGLE)
    rw.RemoveBond(don, h_s)
    rw.AddBond(o, h_s, Chem.BondType.SINGLE)
    return [Transfer(h_n, n, acc), Transfer(h_s, don, o)]


def _parts(shuttle: str, amine_smiles: str) -> tuple[list[Chem.Mol], tuple]:
    if shuttle == "none":
        return [], ()
    mol, acc, don, h = _shuttle(shuttle, amine_smiles)
    return [mol], (acc, don, h)


def addition(aldehyde_smiles: str, amine_smiles: str, shuttle: str) -> Step:
    """Aldehyde + amine (+ shuttle) -> hemiaminal (+ shuttle)."""
    ald, amine = _with_h(aldehyde_smiles), _with_h(amine_smiles)
    if not ald.HasSubstructMatch(ALDEHYDE) or not amine.HasSubstructMatch(
        AMINE
    ):
        raise MechanismError("needs an aldehyde and a primary amine")
    extra, s = _parts(shuttle, amine_smiles)
    reactant, offs = _combine([ald, amine, *extra])
    c, o = ald.GetSubstructMatch(ALDEHYDE)[:2]
    n = offs[1] + amine.GetSubstructMatch(AMINE)[0]
    h_n = _h_on(reactant, n)
    rw = Chem.RWMol(reactant)
    rw.AddBond(c, n, Chem.BondType.SINGLE)
    sh = tuple(offs[2] + i for i in s) if s else None
    transfers = _relay(rw, h_n, n, o, sh)  # type: ignore[arg-type]
    product = _assign_orders(rw, Chem.GetFormalCharge(reactant))
    return Step(
        "addition", shuttle, reactant, product, [(c, n)], [], transfers
    )


def dehydration(
    hemiaminal_smiles: str, amine_smiles: str, shuttle: str
) -> Step:
    """Hemiaminal (+ shuttle) -> imine + water (+ shuttle)."""
    hemi = _with_h(hemiaminal_smiles)
    match = hemi.GetSubstructMatch(HEMIAMINAL)
    if not match:
        raise MechanismError("not a hemiaminal of a primary amine")
    extra, s = _parts(shuttle, amine_smiles)
    reactant, offs = _combine([hemi, *extra])
    c, o, n = match
    h_n = _h_on(reactant, n)
    rw = Chem.RWMol(reactant)
    rw.RemoveBond(c, o)
    sh = tuple(offs[1] + i for i in s) if s else None
    transfers = _relay(rw, h_n, n, o, sh)  # type: ignore[arg-type]
    product = _assign_orders(rw, Chem.GetFormalCharge(reactant))
    return Step(
        "dehydration", shuttle, reactant, product, [], [(c, o)], transfers
    )


def hemiaminal_smiles(aldehyde_smiles: str, amine_smiles: str) -> str:
    """The hemiaminal (carbinolamine) of an aldehyde and an amine."""
    step = addition(aldehyde_smiles, amine_smiles, "none")
    return max(step.fragments("product"), key=len)


def shuttle_species(shuttle: str, amine_smiles: str) -> list[str]:
    """Separate molecules a shuttle is made of, as SMILES (none: [])."""
    if shuttle == "none":
        return []
    if shuttle == "water":
        return [WATER_SMILES]
    if shuttle == "amine":
        return [Chem.MolToSmiles(Chem.MolFromSmiles(amine_smiles))]
    if shuttle == "tfa":
        return [Chem.MolToSmiles(Chem.MolFromSmiles(TFA_SMILES))]
    if shuttle == "tfa_salt":  # one species: the ion pair
        mol, *_ = _shuttle("tfa_salt", amine_smiles)
        return [Chem.MolToSmiles(Chem.RemoveHs(mol))]
    raise MechanismError(f"unknown shuttle {shuttle!r}")


# Kinetic species (see predcc_dft.ts.kinetics) a shuttle brings in and
# leaves as: the ion pair S comes in as one species and leaves as N + T.
_SHUTTLE_IN_OUT = {
    "none": ([], []),
    "water": (["W"], ["W"]),
    "amine": (["N"], ["N"]),
    "tfa": (["T"], ["T"]),
    "tfa_salt": (["S"], ["N", "T"]),
}


def kinetic_species(step: str, shuttle: str) -> tuple[list[str], list[str]]:
    """Reactant and product kinetic species of one step and shuttle."""
    s_in, s_out = _SHUTTLE_IN_OUT[shuttle]
    if step == "addition":
        return ["A", "N", *s_in], ["H", *s_out]
    if step == "dehydration":
        return ["H", *s_in], ["I", "W", *s_out]
    raise MechanismError(f"unknown step {step!r}; use one of {STEPS}")


def shuttle_atoms(step: Step) -> list[int]:
    """Atoms of the shuttle in the complex (empty without a shuttle)."""
    if step.shuttle == "none":
        return []
    frags = Chem.GetMolFrags(step.reactant)
    substrate = 2 if step.name == "addition" else 1
    return sorted(i for f in frags[substrate:] for i in f)
