"""production/molecules/<key>/production.toml: one file per molecule.

Everything the production branch has for one molecule, rebuilt every round:
identity and, per production version, the status, every job's paths
(input, output, record), the locked method and solvation model, E, dG_solv
and G. Versions computed under earlier locks are kept. result.toml and the
record.toml files stay the source of truth.
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.protocol import project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml, write_toml

BRANCH = "production"
FILE = "production.toml"


def path(cfg: Config, rec: dict) -> Path:
    """production/molecules/<key>/production.toml."""
    return project.branch_mol_dir(cfg, BRANCH, rec) / FILE


def _rel(cfg: Config, p: Path) -> str:
    return p.relative_to(cfg.root).as_posix()


def _version(cfg: Config, vdir: Path) -> dict:
    res_file = vdir / "result.toml"
    res = read_toml(res_file) if res_file.exists() else {}
    jobs = {}
    for inp in sorted(vdir.glob("*/input.inp")):
        jdir = inp.parent
        rec_file = jdir / "record.toml"
        record = read_toml(rec_file) if rec_file.exists() else {}
        jobs[jdir.name] = {
            "status": record.get("calculation", {}).get("status", "pending"),
            "electronic_hartree": record.get("energy", {}).get(
                "electronic_hartree"
            ),
            "input": _rel(cfg, jdir / "input.inp"),
            "output": _rel(cfg, jdir / "output.out"),
            "record": _rel(cfg, rec_file),
        }
    out: dict = {
        "status": res.get("status", "pending"),
        "result_toml": _rel(cfg, res_file),
        "jobs": jobs,
    }
    for key in ("method", "solvation_model", "g_hartree", "geometry_sha256"):
        if key in res:
            out[key] = res[key]
    if "energy" in res:
        out["electronic_hartree"] = res["energy"].get("electronic_hartree")
    if "solvation" in res:
        out["dg_solv_kcal_mol"] = res["solvation"].get("dg_solv_kcal_mol")
    return out


def write(cfg: Config, rec: dict) -> dict | None:
    """Rewrite one molecule's production.toml (None if nothing ran yet)."""
    pdir = project.branch_mol_dir(cfg, BRANCH, rec)
    versions = sorted(p for p in pdir.glob("v*") if p.is_dir())
    if not versions:
        return None
    data: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "module": BRANCH,
        **project.identity(rec, project.structure(cfg, rec)),
        "files": {
            "production_toml": _rel(cfg, path(cfg, rec)),
            "trunk_best_xyz": _rel(
                cfg, project.mol_dir(cfg, rec) / "best.xyz"
            ),
        },
        "versions": {v.name: _version(cfg, v) for v in versions},
    }
    write_toml(path(cfg, rec), data)
    return data


def run(cfg: Config) -> int:
    """Rewrite production.toml of every selected molecule."""
    return sum(
        write(cfg, rec) is not None for rec in project.selected(cfg, BRANCH)
    )
