"""molecules/<key>/trunk.toml: everything the trunk knows about one molecule.

One file per molecule, rebuilt every round after step 7, so a parser never
has to walk the trunk folders: identity, the status of steps 3-7, every file
path (seeds, CREST/CENSO results, each Opt+Freq attempt with its input,
output, record and final geometry, structure.toml, best.xyz) and the chosen
geometry with its energies. The per-calculation record.toml files and
structure.toml stay the source of truth; this is the searchable view.
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.protocol import project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml, write_toml
from predcc_dft.structures import conformers

FILE = "trunk.toml"


def path(cfg: Config, rec: dict) -> Path:
    """molecules/<key>/trunk.toml."""
    return project.mol_dir(cfg, rec) / FILE


def _rel(cfg: Config, p: Path) -> str:
    return p.relative_to(cfg.root).as_posix()


def _step5(cfg: Config, rec: dict) -> str:
    cdir = project.mol_dir(cfg, rec) / "conformers"
    if conformers.step5_done(cfg, rec):
        return "done"
    if (cdir / "FAILED").exists():
        return "failed"
    if conformers.engine(cfg) == "orca" and conformers.crest_done(cfg, rec):
        return "crest done"
    return "written" if (cdir / "run.sh").exists() else "not started"


def _attempts(cfg: Config, mdir: Path) -> list[dict]:
    rows = []
    for adir in sorted((mdir / "frequencies").glob("*/attempt*")):
        if not adir.is_dir():
            continue
        rec_file = adir / "record.toml"
        record = read_toml(rec_file) if rec_file.exists() else {}
        calc = record.get("calculation", {})
        rows.append(
            {
                "conformer": adir.parent.name,
                "attempt": adir.name,
                "status": calc.get("status", "pending"),
                "reason": calc.get("reason", ""),
                "electronic_hartree": record.get("energy", {}).get(
                    "electronic_hartree"
                ),
                "g_corr_hartree": record.get("thermo", {}).get(
                    "g_corr_hartree"
                ),
                "directory": _rel(cfg, adir),
                "input": _rel(cfg, adir / "input.inp"),
                "output": _rel(cfg, adir / "output.out"),
                "record": _rel(cfg, rec_file),
                "final_xyz": _rel(cfg, adir / "final.xyz"),
            }
        )
    return rows


def write(cfg: Config, rec: dict) -> dict:
    """Rewrite one molecule's trunk.toml and return it."""
    mdir = project.mol_dir(cfg, rec)
    struct_file = mdir / "structure.toml"
    struct = read_toml(struct_file) if struct_file.exists() else {}
    seeds = [
        {
            "isomer": iso["inchikey"],
            "label": s["label"],
            "xyz": s["xyz"],
            "sdf": s.get("sdf", ""),
        }
        for iso in rec["isomers"]
        for s in iso["seeds"]
    ]
    conf_dir = mdir / "conformers"
    attempts = _attempts(cfg, mdir)
    data: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "module": "trunk",
        **project.identity(rec),
        "smiles": rec.get("smiles", ""),
        "inchikeys": [i["inchikey"] for i in rec["isomers"]],
        "charge": rec.get("charge", 0),
        "multiplicity": rec.get("multiplicity", 1),
        "status": {
            "seeds (step 3)": len(seeds),
            "conformers (steps 4-5)": _step5(cfg, rec),
            "frequencies (step 6)": {
                s: sum(a["status"] == s for a in attempts)
                for s in sorted({a["status"] for a in attempts})
            },
            "structure (step 7)": struct.get("status", "not yet"),
        },
        "files": {
            "trunk_toml": _rel(cfg, path(cfg, rec)),
            "molecule_toml": _rel(cfg, mdir / "molecule.toml"),
            "conformers_dir": _rel(cfg, conf_dir),
            "crest": sorted(
                _rel(cfg, p)
                for p in conf_dir.glob("crest_*/crest_conformers.xyz")
            ),
            "censo": sorted(
                _rel(cfg, p) for p in conf_dir.glob("censo/*_OPTIMIZATION.*")
            ),
            "structure_toml": _rel(cfg, struct_file),
            "best_xyz": _rel(cfg, mdir / "best.xyz"),
        },
        "seeds": seeds,
        "frequencies": attempts,
    }
    if struct.get("status") == "collapsed":
        data["best"] = struct["best"]
        data["g_ensemble_correction_kcal_mol"] = struct.get(
            "g_ensemble_correction_kcal_mol", 0.0
        )
    write_toml(path(cfg, rec), data)
    return data


def run(cfg: Config) -> int:
    """Rewrite trunk.toml of every molecule; returns how many."""
    if not cfg.molecules_file.exists():
        return 0
    recs = project.molecules(cfg)
    for rec in recs:
        write(cfg, rec)
    return len(recs)
