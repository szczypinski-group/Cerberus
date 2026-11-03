"""Per-calculation record.toml (SOP 2.5): the source of truth for every result.

Every ORCA task folder gets one record.toml once its job has finished, with the
calculation ID, status, provenance and parsed energies.
derived/calculations.csv is rebuilt from these records only.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from predcc_dft import SCHEMA_VERSION
from predcc_dft.io_utils import (
    Config,
    now,
    pipeline_commit,
    read_toml,
    write_toml,
)


def calc_id(
    compound: str, conformer: str, step: str, method: str, solvation: str
) -> str:
    """Calculation ID: <key>.<conformer>.<step>.<method>.<solvation>."""
    return f"{compound}.{conformer}.{step}.{method}.{solvation}"


def geometry_sha256(text: str) -> str:
    """Hash of an xyz geometry (provenance of every single point)."""
    return hashlib.sha256(text.encode()).hexdigest()


def on_geometry(task_dir: Path, geometry: str) -> bool:
    """True if the task folder was written for exactly this geometry."""
    geom = task_dir / "geom.xyz"
    return geom.exists() and geom.read_text(encoding="utf-8") == geometry


def single_point_status(
    task_dir: Path, parsed: dict, needs: tuple[str, ...], geometry: str
) -> tuple[str, str]:
    """Status of a single point on the current best geometry.

    A finished output computed on an older best.xyz is "stale": it is never
    used, and the job is rewritten and rerun.
    """
    status, reason = status_from(parsed, needs)
    if status != "pending" and not on_geometry(task_dir, geometry):
        return "stale", "best.xyz changed since this job was written"
    return status, reason


def job_info(task_dir: Path) -> dict:
    """SLURM job ID, cores and wall time written by run.sh, if present."""
    f = task_dir / "job_info.toml"
    if not f.exists():
        return {}
    try:
        return read_toml(f)
    except (ValueError, OSError):
        return {}


def write(
    task_dir: Path,
    calculation: dict,
    parsed: dict,
    extra: dict | None = None,
) -> dict:
    """Write task_dir/record.toml and return it.

    calculation: id, compound, conformer, step, method, solvation, attempt,
    status, reason, geometry_from. parsed: orca.parse_output() result.
    """
    job = job_info(task_dir)
    geom = task_dir / "geom.xyz"
    if geom.exists():
        calculation = {
            **calculation,
            "geometry_sha256": geometry_sha256(
                geom.read_text(encoding="utf-8")
            ),
        }
    if "attempt" not in calculation:
        from predcc_dft.slurm import submissions

        calculation = {
            **calculation,
            "attempt": max(1, submissions(task_dir)),
        }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "calculation": {
            "rotamer": "",
            "attempt": 1,
            "reason": "",
            **calculation,
        },
        "software": {
            "orca": parsed.get("orca_version", ""),
            "pipeline": pipeline_commit(),
        },
        "job": {
            "slurm_job_id": str(job.get("slurm_job_id", "")),
            "cores": job.get("cores"),
            "wall_time_s": job.get("wall_time_s"),
            "host": job.get("host", ""),
        },
        "evaluated": now(),
    }
    energy = {
        k: parsed[src]
        for k, src in (
            ("electronic_hartree", "final_energy"),
            ("scf_hartree", "e_ref"),
            ("correlation_hartree", "e_corr"),
            ("dg_solv_cosmors_hartree", "dg_solv_cosmors"),
        )
        if src in parsed
    }
    if energy:
        record["energy"] = energy
    orbitals = {k: parsed[k] for k in ("homo_ev", "lumo_ev") if k in parsed}
    if orbitals:
        record["descriptors"] = orbitals
    if "t1" in parsed:
        record["diagnostics"] = {"t1": parsed["t1"]}
    if extra:
        for key, val in extra.items():
            if isinstance(val, dict):
                record.setdefault(key, {}).update(val)
            else:
                record[key] = val
    write_toml(task_dir / "record.toml", record)
    return record


def status_from(parsed: dict, needs: tuple[str, ...] = ()) -> tuple[str, str]:
    """Status and reason of a single-point job from its parsed output."""
    if not parsed.get("exists"):
        return "pending", "no output yet"
    if not parsed.get("terminated_normally"):
        return "failed", "ORCA did not terminate normally"
    missing = [k for k in needs if k not in parsed]
    if missing:
        return "failed", f"missing in output: {', '.join(missing)}"
    return "completed", ""


CSV_COLUMNS = [
    "id",
    "compound",
    "conformer",
    "step",
    "method",
    "solvation",
    "attempt",
    "status",
    "reason",
    "geometry_from",
    "electronic_hartree",
    "g_corr_hartree",
    "n_imaginary",
    "t1",
    "orca",
    "pipeline",
    "slurm_job_id",
    "cores",
    "wall_time_s",
    "directory",
]


def collect_csv(cfg: Config) -> int:
    """Rebuild derived/calculations.csv from every record.toml."""
    rows = []
    for path in sorted(cfg.molecules_dir.glob("**/record.toml")):
        rec = read_toml(path)
        calc = rec.get("calculation", {})
        rows.append(
            {
                **{k: calc.get(k, "") for k in CSV_COLUMNS[:10]},
                "electronic_hartree": rec.get("energy", {}).get(
                    "electronic_hartree", ""
                ),
                "g_corr_hartree": rec.get("thermo", {}).get(
                    "g_corr_hartree", ""
                ),
                "n_imaginary": rec.get("thermo", {}).get("n_imaginary", ""),
                "t1": rec.get("diagnostics", {}).get("t1", ""),
                "orca": rec.get("software", {}).get("orca", ""),
                "pipeline": rec.get("software", {}).get("pipeline", ""),
                **{
                    k: rec.get("job", {}).get(k, "")
                    for k in ("slurm_job_id", "cores", "wall_time_s")
                },
                "directory": path.parent.relative_to(cfg.root).as_posix(),
            }
        )
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.derived_dir / "calculations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
