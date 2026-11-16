"""Hand-over of one step to autodE: start/end complexes and active bonds.

autodE's ``get_ts`` takes a bond rearrangement and refuses more forming than
breaking bonds. Additions form more bonds than they break (C-N, and each
proton lands on a new atom), so they are searched in reverse, from the
hemiaminal side: the same TS. Dehydrations run forward.

Each side is split into its separate molecules (autodE complexes are made of
molecules), with their atoms from our embedded complexes; ``order`` maps the
autodE atom index back to the complex index, so ts.xyz comes back in the
atom order of reactant.xyz / product.xyz.
"""

from __future__ import annotations

from rdkit import Chem

from predcc_dft.ts.mechanism import Step


def active_bonds(step: Step) -> tuple[set[frozenset], set[frozenset]]:
    """(forming, breaking) bonds of the forward step, complex indices."""
    forming = {frozenset(b) for b in step.forming} | {
        frozenset((t.h, t.dst)) for t in step.transfers
    }
    breaking = {frozenset(b) for b in step.breaking} | {
        frozenset((t.h, t.src)) for t in step.transfers
    }
    return forming, breaking


def xyz_atoms(xyz: str) -> list[tuple[str, float, float, float]]:
    """Atoms of one xyz block (text)."""
    lines = xyz.splitlines()
    n = int(lines[0].split()[0])
    out = []
    for line in lines[2 : 2 + n]:
        el, x, y, z = line.split()[:4]
        out.append((el, float(x), float(y), float(z)))
    return out


def _side(mol: Chem.Mol, xyz: str) -> dict:
    atoms = xyz_atoms(xyz)
    frags = []
    order: list[int] = []
    for f in Chem.GetMolFrags(mol):
        frags.append(
            {
                "charge": sum(
                    mol.GetAtomWithIdx(i).GetFormalCharge() for i in f
                ),
                "smiles": Chem.MolFragmentToSmiles(mol, atomsToUse=list(f)),
                "complex_atoms": list(f),
                "atoms": [list(atoms[i]) for i in f],
            }
        )
        order += list(f)
    return {"fragments": frags, "order": order}


def task(step: Step, reactant_xyz: str, product_xyz: str) -> dict:
    """Direction, both sides and the bond rearrangement for autodE."""
    forming, breaking = active_bonds(step)
    reverse = len(forming) > len(breaking)
    if reverse:
        start = _side(step.product, product_xyz)
        end = _side(step.reactant, reactant_xyz)
        forming, breaking = breaking, forming
    else:
        start = _side(step.reactant, reactant_xyz)
        end = _side(step.product, product_xyz)
    pos = {atom: k for k, atom in enumerate(start["order"])}

    def to_autode(bonds: set[frozenset]) -> list[list[int]]:
        return sorted(sorted(pos[i] for i in b) for b in bonds)

    return {
        "direction": "reverse" if reverse else "forward",
        "start": start,
        "end": end,
        "forming": to_autode(forming),
        "breaking": to_autode(breaking),
        "bond_indices": "autodE order: atoms of start fragments in turn",
    }
