"""Benchmark step 9: every method of the functional x basis-set grid.

One gas-phase single point per molecule and method on the trunk best.xyz:

    benchmark/molecules/<key>/methods/<method>/   input.inp ... record.toml
    benchmark/molecules/<key>/methods/summary.toml   all methods in one file
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends.slurm import write_array
from predcc_dft.energies import single_point
from predcc_dft.protocol import constants, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml, write_toml
from predcc_dft.protocol.methods import (
    functional_methods,
    level_methods,
    method_inputs,
)

STAGE = "benchmark"
BRANCH = "benchmark"


def methods(cfg: Config) -> dict[str, dict]:
    """Methods to benchmark.

    [benchmark] levels, functionals and methods select a subset (``predcc
    benchmark --level/--functional`` set them); all empty = every method.
    """
    table = cfg.methods.get("methods", {})
    b = cfg["benchmark"]
    wanted = list(b.get("methods") or [])
    for level in b.get("levels") or []:
        wanted += [
            m for m in level_methods(cfg.methods, level) if m not in wanted
        ]
    for functional in b.get("functionals") or []:
        wanted += [
            m
            for m in functional_methods(cfg.methods, functional)
            if m not in wanted
        ]
    wanted = wanted or list(table)
    unknown = [m for m in wanted if m not in table]
    if unknown:
        raise KeyError(f"methods not in methods.toml: {', '.join(unknown)}")
    return {m: table[m] for m in wanted}


def method_meta(cfg: Config, label: str) -> dict:
    """Grouping metadata of a method label (or of the DLPNO reference)."""
    if label == constants.REFERENCE_LABEL:
        return {
            "functional": "DLPNO-CCSD(T)",
            "basis": "CBS(X/Y)",
            "family": "reference",
            "zeta": "",
            "baseline": False,
            "level": "reference",
            "functional_label": "",
            "basis_label": "",
        }
    m = cfg.methods.get("methods", {}).get(label, {})
    return {
        **{
            k: m.get(k, "")
            for k in ("functional", "basis", "family", "zeta", "baseline")
        },
        "level": ";".join(m.get("levels", [])),
        "functional_label": m.get("functional_label", ""),
        "basis_label": m.get("basis_label", ""),
    }


def _dir(cfg: Config, rec: dict) -> Path:
    return project.branch_mol_dir(cfg, BRANCH, rec) / "methods"


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write one ORCA single point per selected molecule x method to run."""
    tasks, waiting = [], 0
    labels = list(methods(cfg))
    for rec in project.selected(cfg, BRANCH):
        if project.structure(cfg, rec) is None:
            waiting += 1
            continue
        geometry = (project.mol_dir(cfg, rec) / "best.xyz").read_text(
            encoding="utf-8"
        )
        for label in labels:
            jdir = _dir(cfg, rec) / label
            if not single_point.needs_run(
                jdir, single_point.NEEDS_SP, geometry, force
            ):
                continue
            keywords, blocks = method_inputs(cfg.methods, label)
            single_point.write(
                cfg, STAGE, jdir, keywords, blocks, geometry, rec, label
            )
            tasks.append(jdir)
    script = write_array(cfg, STAGE, tasks, cfg.benchmark_dir / "jobs")
    return {
        "tasks": len(tasks),
        "methods": len(labels),
        "waiting_for_trunk": waiting,
        "script": str(script) if script else None,
    }


def collect(cfg: Config) -> dict:
    """record.toml per finished single point, summary.toml per molecule.

    Covers every method folder on disk, not only the current selection, so
    results of an earlier `--level` run are never left uncollected.
    """
    counts = {"completed": 0, "failed": 0, "stale": 0, "pending": 0}
    for rec in project.selected(cfg, BRANCH):
        struct = project.structure(cfg, rec)
        if struct is None:
            continue
        geometry = (project.mol_dir(cfg, rec) / "best.xyz").read_text(
            encoding="utf-8"
        )
        rows: dict[str, dict] = {}
        for inp in sorted(_dir(cfg, rec).glob("*/input.inp")):
            jdir, label = inp.parent, inp.parent.name
            status, res = single_point.collect(
                jdir,
                single_point.calculation(rec, "sp", label, "gas", BRANCH),
                single_point.NEEDS_SP,
                geometry,
            )
            counts[status] += 1
            rows[label] = _summary_row(cfg, label, status, jdir, res)
        if rows:
            _write_summary(cfg, rec, struct, geometry, rows)
    return counts


def _summary_row(
    cfg: Config, label: str, status: str, jdir: Path, res: dict
) -> dict:
    """One method's table in summary.toml."""
    m = cfg.methods.get("methods", {}).get(label, {})
    rec_file = jdir / "record.toml"
    reason = (
        read_toml(rec_file)["calculation"].get("reason", "")
        if rec_file.exists()
        else ""
    )
    row = {
        "status": status,
        "reason": reason,
        "functional": m.get("functional", ""),
        "basis": m.get("basis", ""),
        "functional_label": m.get("functional_label", ""),
        "basis_label": m.get("basis_label", ""),
        "levels": m.get("levels", []),
    }
    if status == "completed":  # numbers only from valid, current results
        row["electronic_hartree"] = res["final_energy"]
        for key in ("homo_ev", "lumo_ev"):
            if key in res:
                row[key] = res[key]
    return row


def _write_summary(
    cfg: Config, rec: dict, struct: dict, geometry: str, rows: dict
) -> None:
    """methods/summary.toml: every method of one molecule in one file.

    A convenience view rebuilt each round; the per-method record.toml files
    stay the source of truth (SOP 2.1).
    """
    ref_file = (
        project.branch_mol_dir(cfg, BRANCH, rec)
        / "reference"
        / "reference.toml"
    )
    ref = read_toml(ref_file) if ref_file.exists() else {}
    write_toml(
        _dir(cfg, rec) / "summary.toml",
        {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "branch": BRANCH,
            **project.identity(rec, struct),
            "geometry_sha256": records.geometry_sha256(geometry),
            "units": {"energy": "hartree", "orbitals": "eV"},
            "counts": {
                s: sum(r["status"] == s for r in rows.values())
                for s in ("completed", "failed", "stale", "pending")
            },
            "reference": {
                "label": constants.REFERENCE_LABEL,
                "e_cbs_hartree": ref.get("reference", {}).get("e_cbs_hartree"),
            },
            "methods": rows,
        },
    )


def energies(cfg: Config, rec: dict) -> dict[str, float]:
    """Completed, current-geometry method energies (Eh) of one molecule."""
    out = {}
    for path in sorted(_dir(cfg, rec).glob("*/record.toml")):
        data = read_toml(path)
        if data["calculation"]["status"] == "completed":
            out[path.parent.name] = data["energy"]["electronic_hartree"]
    return out
