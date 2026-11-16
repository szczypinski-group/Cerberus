"""3D reactant and product complexes of a step, and connectivity checks.

Both ends of a NEB path must hold the same atoms in the same order and lie
close in space. They are made from one embedding of a pre-reactive complex
in which the reacting atoms are already near each other (distance-geometry
bounds on the forming bonds and on each transferred proton's destination).
That geometry is relaxed twice with MMFF94 and distance restraints:

    reactant   reactant bonds, forming bonds held apart (2.7-2.9 A),
               transferred protons hydrogen-bonded to their destination
    product    product bonds, breaking bonds held apart (2.7-2.9 A),
               transferred protons hydrogen-bonded back to their source

:func:`bonds_from_xyz` / :func:`matches` compare a geometry's bonds with a
molecular graph (used to check the complexes).
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDistGeom, rdForceFieldHelpers
from rdkit.DistanceGeometry import DoTriangleSmoothing

from predcc_dft.protocol.io import Atoms, format_xyz
from predcc_dft.ts.mechanism import Step

PRE_FORMING = (2.0, 2.3)  # A, pre-reactive complex (embedding only)
PRE_TRANSFER = (1.6, 1.9)  # H...acceptor through a shuttle
PRE_DIRECT = (1.9, 2.5)  # H...O of the strained direct (4-ring) transfer
APART = (2.7, 2.9)  # forming / breaking bonds held open in the ends
HBOND = (1.75, 2.05)  # proton hydrogen-bonded to its next / previous atom
HBOND_DIRECT = (2.2, 2.6)  # same, direct transfer (geometry allows no less)


class EmbedError(RuntimeError):
    """No 3D complex could be built for a step."""


def _bounds(step: Step, scale: float) -> object:
    mol = step.reactant
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    direct = step.shuttle == "none"

    def put(i: int, j: int, lo: float, hi: float) -> None:
        i, j = min(i, j), max(i, j)
        bm[i][j] = hi * scale  # upper triangle: upper bound
        bm[j][i] = lo  # lower triangle: lower bound

    for i, j in step.forming:
        put(i, j, *PRE_FORMING)
    for t in step.transfers:
        put(t.h, t.dst, *(PRE_DIRECT if direct else PRE_TRANSFER))
    if not DoTriangleSmoothing(bm):
        raise EmbedError("inconsistent distance bounds")
    return bm


def _embed(
    mol: Chem.Mol, links: list[tuple[int, int]], bm: object, seed: int
) -> Chem.Conformer | None:
    """Distance-geometry embedding of mol with a bounds matrix.

    RDKit ignores a bounds matrix for a molecule of several fragments, so
    the fragments are joined by the restrained pairs (links) for the
    embedding only; the bounds themselves come from the real molecule.
    """
    joined = Chem.RWMol(mol)
    for i, j in links:
        if joined.GetBondBetweenAtoms(i, j) is None:
            joined.AddBond(i, j, Chem.BondType.ZERO)
    joined.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(joined)
    if len(Chem.GetMolFrags(joined)) > 1:
        raise EmbedError("restraints do not join every fragment")
    params: Any = rdDistGeom.EmbedParameters()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.enforceChirality = False
    params.SetBoundsMat(bm)
    if rdDistGeom.EmbedMolecule(joined, params) != 0:
        return None
    return Chem.Conformer(joined.GetConformer())


def _core(step: Step, seed: int) -> Chem.Conformer:
    """Pre-reactive complex: reactant graph, reacting atoms close."""
    links = list(step.forming) + [(t.h, t.dst) for t in step.transfers]
    for attempt, scale in enumerate((1.0, 1.15, 1.3)):
        try:
            bm = _bounds(step, scale)
        except EmbedError:
            continue
        conf = _embed(step.reactant, links, bm, seed + attempt)
        if conf is not None:
            return conf
    raise EmbedError(f"could not embed the {step.name}/{step.shuttle} complex")


def _relax(
    mol: Chem.Mol,
    coords: Chem.Conformer,
    restraints: list[tuple[int, int, tuple[float, float]]],
) -> Chem.Mol:
    """MMFF94 (UFF fallback) on mol from coords, with distance restraints."""
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    out.AddConformer(Chem.Conformer(coords), assignId=True)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(out)
    if props is not None:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            out, props, ignoreInterfragInteractions=False
        )
        for i, j, (lo, hi) in restraints:
            ff.MMFFAddDistanceConstraint(i, j, False, lo, hi, 200.0)
    else:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
            out, ignoreInterfragInteractions=False
        )
        for i, j, (lo, hi) in restraints:
            ff.UFFAddDistanceConstraint(i, j, False, lo, hi, 200.0)
    ff.Initialize()
    ff.Minimize(maxIts=2000)
    return out


def atoms(mol: Chem.Mol) -> Atoms:
    """Element and coordinates of every atom of the first conformer."""
    conf = mol.GetConformer()
    return [
        (
            a.GetSymbol(),
            conf.GetAtomPosition(a.GetIdx()).x,
            conf.GetAtomPosition(a.GetIdx()).y,
            conf.GetAtomPosition(a.GetIdx()).z,
        )
        for a in mol.GetAtoms()
    ]


def complexes(step: Step, seed: int = 67) -> tuple[str, str]:
    """Xyz texts of the reactant and product complex (same atom order)."""
    core = _core(step, seed)
    hb = HBOND_DIRECT if step.shuttle == "none" else HBOND
    reactant = _relax(
        step.reactant,
        core,
        [(i, j, APART) for i, j in step.forming]
        + [(t.h, t.dst, hb) for t in step.transfers],
    )
    product = _relax(
        step.product,
        reactant.GetConformer(),
        [(i, j, APART) for i, j in step.breaking]
        + [(t.h, t.src, hb) for t in step.transfers],
    )
    label = f"{step.name} {step.shuttle}"
    return (
        format_xyz(atoms(reactant), f"reactant complex: {label}"),
        format_xyz(atoms(product), f"product complex: {label}"),
    )


def conformers(smiles: str, n: int, seed: int = 67) -> list[str]:
    """Up to n MMFF-optimised conformers of a molecule as xyz texts.

    An ion pair (ammonium carboxylate) is kept together by holding one N-H
    of the ammonium hydrogen-bonded to the anionic oxygen.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    pair = _ion_pair(mol)
    restraints = [(pair[0], pair[1], HBOND)] if pair else []
    confs: list[Chem.Conformer] = []
    if pair is None:
        params: Any = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        cids = rdDistGeom.EmbedMultipleConfs(mol, 3 * n, params)
        confs = [Chem.Conformer(mol.GetConformer(c)) for c in cids]
    else:
        bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
        h, o = min(pair), max(pair)
        bm[h][o], bm[o][h] = HBOND[1], HBOND[0]
        DoTriangleSmoothing(bm)
        for k in range(3 * n):
            conf = _embed(mol, [pair], bm, seed + k)
            if conf is not None:
                confs.append(conf)
    if not confs:
        raise EmbedError(f"could not embed {smiles}")
    found = []
    for conf in confs:
        relaxed = _relax(mol, conf, restraints)
        found.append((_energy(relaxed), relaxed))
    found.sort(key=lambda x: x[0])
    kept: list[tuple[float, Chem.Mol]] = []
    for energy, m in found:  # drop near-duplicates (0.1 kcal/mol)
        if all(abs(energy - e) > 0.1 for e, _ in kept):
            kept.append((energy, m))
        if len(kept) == n:
            break
    return [format_xyz(atoms(m), smiles) for _, m in kept]


def _energy(mol: Chem.Mol) -> float:
    """MMFF94 (UFF fallback) energy of the first conformer, kcal/mol."""
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ff = (
        rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
        if props is not None
        else rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
    )
    return float(ff.CalcEnergy())


def _ion_pair(mol: Chem.Mol) -> tuple[int, int] | None:
    """(H on N+, O-) of an ammonium carboxylate pair, else None."""
    if len(Chem.GetMolFrags(mol)) < 2:
        return None
    n = mol.GetSubstructMatch(Chem.MolFromSmarts("[NX4+][H]"))
    o = mol.GetSubstructMatch(Chem.MolFromSmarts("[OX1-]"))
    return (n[1], o[0]) if n and o else None


def bonds_from_xyz(text: str, charge: int, factor: float) -> set[frozenset]:
    """Atom pairs bonded in an xyz geometry (covalent radii x factor)."""
    mol = Chem.MolFromXYZBlock(text)
    if mol is None:
        raise ValueError("unreadable xyz")
    rdDetermineBonds.DetermineConnectivity(
        mol, charge=charge, covFactor=factor
    )
    return {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
    }


def graph_bonds(mol: Chem.Mol) -> set[frozenset]:
    """Atom pairs bonded in a molecular graph."""
    return {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
    }


def matches(
    found: set[frozenset], expected: set[frozenset], ignore: set[int]
) -> list[str]:
    """Differences between two bond sets, leaving out bonds of ignored atoms.

    ignore: atoms whose bonds are not compared.
    """
    diff = []
    for pair in sorted(found ^ expected, key=sorted):
        if pair & ignore:
            continue
        i, j = sorted(pair)
        diff.append(f"{'+' if pair in found else '-'}{i}-{j}")
    return diff
