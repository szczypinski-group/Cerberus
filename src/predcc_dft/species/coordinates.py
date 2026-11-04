"""Step 3: initial 3D coordinates with ETKDG (v4, random seed 67 by default).

One seed geometry per stereoisomer; acylhydrazones get a syn and an anti seed
about the amide C(O)-N bond. Every seed keeps the shared atom order (heavy
atoms in canonical order, hydrogens appended), so ensembles can be pooled.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolTransforms

from predcc_dft.chemistry import geometry, identifiers
from predcc_dft.protocol.config import Config


def _params(
    method: str, seed: int, random_coords: bool
) -> rdDistGeom.EmbedParameters:
    try:
        params = getattr(rdDistGeom, method)()
    except AttributeError as err:
        raise RuntimeError(
            f"RDKit has no {method}; upgrade RDKit (see pyproject.toml)"
        ) from err
    params.randomSeed = seed
    params.useRandomCoords = random_coords
    return params


def _optimise(molh: Chem.Mol, force_field: str) -> str:
    if force_field.upper().startswith(
        "MMFF"
    ) and AllChem.MMFFHasAllMoleculeParams(molh):
        AllChem.MMFFOptimizeMolecule(
            molh, mmffVariant=force_field, maxIters=2000
        )
        return force_field
    AllChem.UFFOptimizeMolecule(molh, maxIters=2000)
    return "UFF"


def _stereo_ok(molh: Chem.Mol, expected_smiles: str) -> bool:
    check = Chem.Mol(molh)
    Chem.AssignStereochemistryFrom3D(check)
    return identifiers.canonical(check) == expected_smiles


def embed(
    mol: Chem.Mol, cfg: Config, rotamer: str | None = None
) -> tuple[Chem.Mol, dict]:
    """Embed one seed geometry. Returns (mol with Hs + conformer, metadata)."""
    ccfg = cfg["coordinates"]
    expected = identifiers.canonical(mol)
    molh = Chem.AddHs(mol)
    seed0 = int(ccfg["random_seed"])
    for attempt in range(int(ccfg["max_embed_attempts"])):
        seed = seed0 + attempt
        params = _params(ccfg["method"], seed, random_coords=attempt > 0)
        trial = Chem.Mol(molh)
        if rdDistGeom.EmbedMolecule(trial, params) != 0:
            continue
        if rotamer is not None:
            atoms = geometry.amide_rotamer_atoms(trial)
            if atoms is not None:
                rdMolTransforms.SetDihedralDeg(
                    trial.GetConformer(),
                    *atoms,
                    0.0 if rotamer == "syn" else 180.0,
                )
        ff = _optimise(trial, ccfg["force_field"])
        if not _stereo_ok(trial, expected):
            continue
        meta = {
            "method": ccfg["method"],
            "random_seed": seed,
            "attempt": attempt + 1,
            "force_field": ff,
            "stereo_preserved": True,
        }
        if rotamer is not None:
            atoms = geometry.amide_rotamer_atoms(trial)
            assert atoms is not None  # rotamer seeds are acylhydrazones only
            dih = rdMolTransforms.GetDihedralDeg(trial.GetConformer(), *atoms)
            meta.update(
                rotamer_requested=rotamer,
                rotamer_after_ff=geometry.rotamer_label(dih),
                amide_dihedral_deg=round(dih, 1),
            )
        return trial, meta
    raise RuntimeError(
        f"embedding failed after {ccfg['max_embed_attempts']} attempts "
        f"for {expected}"
    )


def xyz_block(molh: Chem.Mol, comment: str) -> str:
    """Xyz text of the molecule's conformer."""
    conf = molh.GetConformer()
    lines = [str(molh.GetNumAtoms()), comment]
    for atom in molh.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():<2s} {p.x:16.10f} {p.y:16.10f} {p.z:16.10f}"
        )
    return "\n".join(lines) + "\n"
