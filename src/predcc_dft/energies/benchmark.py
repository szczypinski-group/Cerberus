"""Step 9: benchmark single points of the method matrix (SOP 5.1 step 9, 6.1).

Every method in methods.toml is run as a gas-phase single point on each
molecule's best r2SCAN-3c/CPCM geometry (step 7). Descriptors that come free
with the output (HOMO/LUMO) are stored in the record.

molecules/<key>/benchmark/<method>/{input.inp, geom.xyz, run.sh, job.slurm,
                                    output.out, record.toml}
molecules/<key>/benchmark/summary.toml   every method of the molecule at once
"""

from __future__ import annotations

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends import orca
from predcc_dft.backends.slurm import orca_run_script, write_array
from predcc_dft.protocol import constants, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, write_toml
from predcc_dft.protocol.methods import (
    functional_methods,
    level_methods,
    method_inputs,
)


def methods(cfg: Config) -> dict[str, dict]:
    """Methods to benchmark.

    [benchmark] levels, functionals and methods select a subset (``predcc
    level`` / ``predcc functional`` set them); all empty = every method.
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


def _calculation(rec: dict, label: str) -> dict:
    return {
        "id": records.calc_id(rec["compound"], "best", "sp", label, "gas"),
        "compound": rec["compound"],
        "conformer": "best",
        "step": "sp",
        "method": label,
        "solvation": "gas",
        "geometry_from": "best.xyz (step 7)",
    }


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write one ORCA single point per molecule x method still to run."""
    st = cfg["slurm"]["stage"]["benchmark"]
    tasks, waiting = [], 0
    labels = list(methods(cfg))
    for rec in project.molecules(cfg):
        if project.structure(cfg, rec) is None:
            waiting += 1
            continue
        mdir = project.mol_dir(cfg, rec)
        geometry = (mdir / "best.xyz").read_text(encoding="utf-8")
        for label in labels:
            jdir = mdir / "benchmark" / label
            res = orca.parse_output(jdir / "output.out")
            status, _ = records.single_point_status(
                jdir, res, ("final_energy",), geometry
            )
            if status == "completed" and not force:
                continue
            if status == "stale":  # new geometry: a fresh submission budget
                (jdir / ".submissions").unlink(missing_ok=True)
            keywords, blocks = method_inputs(cfg.methods, label)
            orca.write_input(
                jdir,
                keywords,
                st["cpus"],
                int(st["mem_per_cpu_mb"] * 0.75),
                geometry,
                rec["charge"],
                rec["multiplicity"],
                blocks,
            )
            orca_run_script(jdir, f"{rec['compound']} {label}", cfg)
            tasks.append(jdir)
    script = write_array(cfg, "benchmark", tasks)
    return {
        "tasks": len(tasks),
        "methods": len(labels),
        "waiting_for_step7": waiting,
        "script": str(script) if script else None,
    }


def collect(cfg: Config) -> dict:
    """Write record.toml for every finished benchmark single point.

    Covers every method folder on disk, not only the current selection, so
    results of an earlier `predcc level` run are never left uncollected.
    """
    counts = {"completed": 0, "failed": 0, "stale": 0, "pending": 0}
    for rec in project.molecules(cfg):
        mdir = project.mol_dir(cfg, rec)
        best = mdir / "best.xyz"
        struct = project.structure(cfg, rec)
        if struct is None or not best.exists():
            continue
        geometry = best.read_text(encoding="utf-8")
        rows: dict[str, dict] = {}
        for inp in sorted((mdir / "benchmark").glob("*/input.inp")):
            jdir, label = inp.parent, inp.parent.name
            res = orca.parse_output(jdir / "output.out")
            status, reason = records.single_point_status(
                jdir, res, ("final_energy",), geometry
            )
            counts[status] += 1
            if status != "pending":
                records.write(
                    jdir,
                    {
                        **_calculation(rec, label),
                        "status": status,
                        "reason": reason,
                    },
                    res,
                )
            rows[label] = _summary_row(cfg, label, status, reason, res)
        if rows:
            _write_summary(cfg, rec, struct, geometry, rows)
    return counts


def _summary_row(
    cfg: Config, label: str, status: str, reason: str, res: dict
) -> dict:
    """One method's line in benchmark/summary.toml."""
    m = cfg.methods.get("methods", {}).get(label, {})
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
    """benchmark/summary.toml: every method of one molecule in one file.

    A convenience view rebuilt from the outputs each round; the per-method
    record.toml files stay the source of truth (SOP 2.1).
    """
    ref = project.reference_data(cfg, rec) or {}
    write_toml(
        project.mol_dir(cfg, rec) / "benchmark" / "summary.toml",
        {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "compound": rec["compound"],
            "name": rec["name"],
            "role": rec["role"],
            "isomer": struct["best"]["label"],
            "inchikey": struct["best"]["inchikey"],
            "geometry": struct["best"]["geometry"],
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
