"""One ORCA single point on a trunk geometry: write, check, collect.

Shared by both branches: the benchmark's DLPNO reference, solvation and
method grid, and production's locked method and solvation model. A task
folder holds input.inp, geom.xyz (a copy of best.xyz), run.sh, job.slurm,
output.out and, once finished, record.toml (SOP 2.5).
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft.backends import orca
from predcc_dft.backends.slurm import orca_run_script
from predcc_dft.protocol import records
from predcc_dft.protocol.config import Config

NEEDS_SP = ("final_energy",)
NEEDS_DLPNO = ("e_ref", "e_corr")


def needs_run(
    task_dir: Path, needs: tuple[str, ...], geometry: str, force: bool
) -> bool:
    """True unless a finished output on exactly this geometry exists."""
    res = orca.parse_output(task_dir / "output.out")
    status, _ = records.single_point_status(task_dir, res, needs, geometry)
    if status == "completed" and not force:
        return False
    if status == "stale":  # new geometry: a fresh submission budget
        (task_dir / ".submissions").unlink(missing_ok=True)
    return True


def write(
    cfg: Config,
    stage: str,
    task_dir: Path,
    keywords: str,
    blocks: str,
    geometry: str,
    rec: dict,
    label: str,
) -> None:
    """input.inp + geom.xyz + run.sh with the stage's %pal/%maxcore."""
    st = cfg["slurm"]["stage"][stage]
    orca.write_input(
        task_dir,
        keywords,
        st["cpus"],
        int(st["mem_per_cpu_mb"] * 0.75),
        geometry,
        rec["charge"],
        rec["multiplicity"],
        blocks,
    )
    orca_run_script(task_dir, f"{rec['compound']} {label}", cfg)


def collect(
    task_dir: Path,
    calculation: dict,
    needs: tuple[str, ...],
    geometry: str,
    extra: dict | None = None,
) -> tuple[str, dict]:
    """Status and parsed numbers of a task; writes its record.toml.

    Numbers are returned only for a completed task on the current geometry;
    failed, stale and pending tasks return an empty dict.
    """
    if not (task_dir / "input.inp").exists():
        return "pending", {}
    res = orca.parse_output(task_dir / "output.out")
    status, reason = records.single_point_status(
        task_dir, res, needs, geometry
    )
    if status != "pending":
        records.write(
            task_dir,
            {**calculation, "status": status, "reason": reason},
            res,
            extra,
        )
    return status, (res if status == "completed" else {})


def calculation(
    rec: dict, step: str, method: str, solvation: str, branch: str
) -> dict:
    """The [calculation] table of a single point on best.xyz."""
    return {
        "id": records.calc_id(
            rec["compound"], "best", step, method, solvation
        ),
        "compound": rec["compound"],
        "conformer": "best",
        "step": step,
        "method": method,
        "solvation": solvation,
        "branch": branch,
        "geometry_from": "best.xyz (trunk step 7)",
    }
