"""Step 7: structure sanity check, then collapse isomers into one geometry.

For every converged, imaginary-free conformer from step 6:
  1. connectivity from the 3D geometry (rdDetermineBonds) must equal the
     intended molecule (catches hemiaminals, proton transfers, ring closures,
     dissociation);
  2. E/Z is read back from 3D and must be one of the molecule's isomers
     (a different tautomer or stereo outcome fails);
  3. acylhydrazones are labelled syn/anti from the amide O=C-N-N dihedral.
The passing conformer with the lowest G = E + (G - E_el) becomes the
molecule's single geometry, molecules/<key>/best.xyz; the populations of each
isomer/rotamer are kept.
"""

from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolTransforms
from rdkit.Geometry import Point3D

from predcc_dft import (
    SCHEMA_VERSION,
    chem,
    conformers,
    coordinates,
    frequencies,
    orca,
)
from predcc_dft.io_utils import (
    Atoms,
    Config,
    format_xyz,
    now,
    read_xyz_frames,
    write_text,
    write_toml,
)

KB_HARTREE = 3.166811563e-6  # Boltzmann constant in Eh/K


def _reference(cfg: Config, rec: dict) -> Chem.Mol:
    first = rec["isomers"][0]["seeds"][0]["sdf"]
    return Chem.MolFromMolFile(str(cfg.root / first), removeHs=False)


def _bonds(mol: Chem.Mol) -> set[tuple[int, int]]:
    """Bonded atom pairs."""
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol.GetBonds()
    }


def check_geometry(
    cfg: Config, rec: dict, ref: Chem.Mol, atoms: Atoms
) -> dict:
    """Return {ok, reason, inchikey, stereo, rotamer, label}."""
    if [a.GetSymbol() for a in ref.GetAtoms()] != [el for el, *_ in atoms]:
        return {
            "ok": False,
            "reason": "atom order/elements differ from the seed",
        }
    raw = Chem.MolFromXYZBlock(format_xyz(atoms))
    rdDetermineBonds.DetermineConnectivity(
        raw, covFactor=cfg["sanity"]["covalent_factor"]
    )
    found, expected = _bonds(raw), _bonds(ref)
    if found != expected:
        formed = sorted(found - expected)
        broken = sorted(expected - found)
        return {
            "ok": False,
            "reason": (
                f"connectivity changed: formed {formed}, broken {broken}"
            ),
        }

    mol = Chem.Mol(ref)
    conf = mol.GetConformer()
    for i, (_, x, y, z) in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    Chem.AssignStereochemistryFrom3D(mol)
    smiles = chem.canonical(mol)
    match = next(
        (iso for iso in rec["isomers"] if iso["smiles"] == smiles), None
    )
    if match is None:
        return {
            "ok": False,
            "reason": f"unexpected stereo/tautomer from 3D: {smiles}",
        }
    out = {
        "ok": True,
        "reason": "",
        "inchikey": match["inchikey"],
        "stereo": match["stereo"],
        "smiles": smiles,
        "rotamer": "",
    }
    rot_atoms = coordinates.amide_rotamer_atoms(mol)
    if rec.get("linkage") == "acylhydrazone" and rot_atoms is not None:
        dih = rdMolTransforms.GetDihedralDeg(conf, *rot_atoms)
        out.update(
            rotamer=coordinates.rotamer_label(dih),
            amide_dihedral_deg=round(dih, 1),
        )
    out["label"] = out["stereo"] + (
        f"_{out['rotamer']}" if out["rotamer"] else ""
    )
    return out


def run(cfg: Config) -> dict:
    """Check every finished molecule and write structure.toml."""
    temp = cfg["frequencies"]["temperature_k"]
    summary = {
        "collapsed": 0,
        "waiting": 0,
        "no_valid_conformer": 0,
        "still_running": 0,
    }
    for rec in conformers.molecules(cfg):
        results = frequencies.selected_results(cfg, rec)
        if not results:
            summary["waiting"] += 1
            continue
        statuses = [r["record"]["calculation"]["status"] for r in results]
        max_attempts = int(cfg["frequencies"]["max_attempts"])
        retry_due = any(
            r["record"]["calculation"]["status"] in ("failed", "rejected")
            and r["record"]["calculation"]["attempt"] < max_attempts
            for r in results
        )
        # Collapse only once every conformer is final: a retry still to come
        # could change best.xyz after the single points ran on the old one.
        if "pending" in statuses or retry_due:
            summary["still_running"] += 1
            continue
        ref = _reference(cfg, rec)
        rows = []
        for r in results:
            calc = r["record"]["calculation"]
            row = {
                "conformer": r["name"],
                "seed": r["seed"],
                "attempt": calc["attempt"],
                "freq_status": calc["status"],
                "freq_reason": calc["reason"],
                "directory": r["dir"].relative_to(cfg.root).as_posix(),
            }
            if calc["status"] == "completed":
                atoms = read_xyz_frames(r["dir"] / "final.xyz")[-1][1]
                check = check_geometry(cfg, rec, ref, atoms)
                e = r["record"]["energy"]["electronic_hartree"]
                gc = r["record"]["thermo"]["g_corr_hartree"]
                row.update(
                    {k: v for k, v in check.items() if k != "ok"},
                    sanity="passed" if check["ok"] else "failed",
                    e_hartree=e,
                    g_corr_hartree=gc,
                    g_hartree=(e + gc)
                    if (e is not None and gc is not None)
                    else None,
                    _atoms=atoms,
                )
            else:
                row["sanity"] = "not checked"
            rows.append(row)

        passed = [
            r
            for r in rows
            if r.get("sanity") == "passed" and r.get("g_hartree") is not None
        ]
        mdir = conformers.mol_dir(cfg, rec)
        out = {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "compound": rec["compound"],
            "temperature_k": temp,
        }
        if not passed:
            out["status"] = "failed"
            out["reason"] = "no conformer passed frequencies and sanity checks"
            summary["no_valid_conformer"] += 1
        else:
            g0 = min(r["g_hartree"] for r in passed)
            weights = {
                r["conformer"]: math.exp(
                    -(r["g_hartree"] - g0) / (KB_HARTREE * temp)
                )
                for r in passed
            }
            total = sum(weights.values())
            populations: dict[str, float] = {}
            for r in passed:
                r["rel_g_kcal_mol"] = round(
                    (r["g_hartree"] - g0) * orca.HARTREE_TO_KCAL, 3
                )
                r["population"] = round(weights[r["conformer"]] / total, 4)
                populations[r["label"]] = round(
                    populations.get(r["label"], 0.0) + r["population"], 4
                )
            best = min(passed, key=lambda r: r["g_hartree"])
            # SOP 2.7/5.1 step 11: G of the Boltzmann ensemble over
            # conformers, E/Z isomers and rotamers, relative to the best one:
            # G_ens - G_best = -RT ln(sum_i exp(-(G_i - G_best)/RT)) <= 0.
            ensemble_kcal = (
                -KB_HARTREE * temp * math.log(total) * orca.HARTREE_TO_KCAL
            )
            write_text(
                mdir / "best.xyz",
                format_xyz(
                    best["_atoms"],
                    f"{best['inchikey']} {best['label']} {best['conformer']} "
                    f"G={best['g_hartree']:.8f}",
                ),
            )
            out.update(
                status="collapsed",
                best={
                    "conformer": best["conformer"],
                    "label": best["label"],
                    "inchikey": best["inchikey"],
                    "stereo": best["stereo"],
                    "rotamer": best["rotamer"],
                    "smiles": best["smiles"],
                    "geometry": (mdir / "best.xyz")
                    .relative_to(cfg.root)
                    .as_posix(),
                    "e_hartree": best["e_hartree"],
                    "g_corr_hartree": best["g_corr_hartree"],
                    "g_hartree": best["g_hartree"],
                    "level": "r2SCAN-3c/CPCM(acetonitrile)",
                },
                populations_by_isomer=populations,
                g_ensemble_correction_kcal_mol=ensemble_kcal,
                n_conformers_in_ensemble=len(passed),
            )
            summary["collapsed"] += 1
        out["conformer"] = [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
        ]
        write_toml(mdir / "structure.toml", out)
    return summary
