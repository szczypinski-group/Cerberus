"""benchmark/molecules/<key>/benchmark.toml: one file per molecule.

Everything the benchmark branch has for one molecule, rebuilt every round:
identity, status, every file path (each reference, solvation and method job
with its input, output and record), the DLPNO-CCSD(T)/CBS reference, the
solvation free energies and every method's energy. reference.toml,
methods/summary.toml and the record.toml files stay the source of truth.
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.benchmark import grid, reference
from predcc_dft.protocol import project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml, write_toml

BRANCH = "benchmark"
FILE = "benchmark.toml"


def path(cfg: Config, rec: dict) -> Path:
    """benchmark/molecules/<key>/benchmark.toml."""
    return project.branch_mol_dir(cfg, BRANCH, rec) / FILE


def _job(cfg: Config, jdir: Path) -> dict:
    rec_file = jdir / "record.toml"
    record = read_toml(rec_file) if rec_file.exists() else {}
    calc = record.get("calculation", {})
    rel = cfg.root
    return {
        "status": calc.get("status", "pending"),
        "electronic_hartree": record.get("energy", {}).get(
            "electronic_hartree"
        ),
        "directory": jdir.relative_to(rel).as_posix(),
        "input": (jdir / "input.inp").relative_to(rel).as_posix(),
        "output": (jdir / "output.out").relative_to(rel).as_posix(),
        "record": rec_file.relative_to(rel).as_posix(),
    }


def write(cfg: Config, rec: dict) -> dict | None:
    """Rewrite one molecule's benchmark.toml (None if nothing ran yet)."""
    bdir = project.branch_mol_dir(cfg, BRANCH, rec)
    if not bdir.exists():
        return None
    struct = project.structure(cfg, rec)
    ref = reference.data(cfg, rec) or {}
    rdir, mdir = bdir / "reference", bdir / "methods"
    jobs = {
        j.parent.name: _job(cfg, j.parent)
        for j in sorted(rdir.glob("*/input.inp"))
    }
    methods = {
        j.parent.name: _job(cfg, j.parent)
        | {
            k: grid.method_meta(cfg, j.parent.name).get(k, "")
            for k in ("functional", "basis")
        }
        for j in sorted(mdir.glob("*/input.inp"))
    }
    n_methods = len(grid.methods(cfg))
    done = sum(m["status"] == "completed" for m in methods.values())
    data: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "module": BRANCH,
        **project.identity(rec, struct),
        "status": {
            "reference": "done" if "reference" in ref else "not yet",
            "methods": f"{done}/{n_methods}",
        },
        "files": {
            "benchmark_toml": path(cfg, rec).relative_to(cfg.root).as_posix(),
            "reference_toml": (rdir / "reference.toml")
            .relative_to(cfg.root)
            .as_posix(),
            "methods_summary_toml": (mdir / "summary.toml")
            .relative_to(cfg.root)
            .as_posix(),
            "trunk_best_xyz": (project.mol_dir(cfg, rec) / "best.xyz")
            .relative_to(cfg.root)
            .as_posix(),
        },
        "reference_jobs": jobs,
        "methods": methods,
    }
    for key in ("geometry_sha256", "reference", "solvation", "thermo"):
        if key in ref:
            data[key] = ref[key]
    write_toml(path(cfg, rec), data)
    return data


def run(cfg: Config) -> int:
    """Rewrite benchmark.toml of every selected molecule; returns how many."""
    return sum(
        write(cfg, rec) is not None for rec in project.selected(cfg, BRANCH)
    )
