"""Task folders of the TS module: ORCA jobs and autodE searches.

An ORCA task folder holds input.inp, geom.xyz, run.sh, job.slurm and, once
run, output.out (+ final.xyz). An autodE task folder holds task.toml, run.sh,
job.slurm and, once run, autode.log, autode_result.toml, ts.xyz and work/
(every ORCA/xTB file autodE made). A task is (re)submitted until it is done,
at most ``[pipeline] max_submissions`` times; a failed optimisation restarts
from its last geometry.

ORCA kinds and what "done" means:

    opt      geometry optimisation (+ Freq) converged
    freq     frequency job finished with a frequency block
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from predcc_dft.backends import orca
from predcc_dft.backends.slurm import orca_run_script
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml, write_text, write_toml
from predcc_dft.ts.autode_job import KEYWORD_SETS

AUTODE_RUN = """#!/bin/bash
# autodE TS search: {label}
set -uo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
START=$(date +%s)
cd "$HERE"
{python} -m predcc_dft.ts.autode_job "$HERE" > "$HERE/autode.log" 2>&1
STATUS=$?
cat > "$HERE/job_info.toml" <<EOF
slurm_job_id = "${{SLURM_ARRAY_JOB_ID:-${{SLURM_JOB_ID:-}}}}\
${{SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}}"
cores = ${{SLURM_NTASKS:-1}}
wall_time_s = $(( $(date +%s) - START ))
exit_code = $STATUS
host = "$(hostname)"
EOF
echo "$STATUS" > "$HERE/exit_code"
exit "$STATUS"
"""


def maxcore(cfg: Config, stage: str) -> int:
    """%maxcore (MB per core) from the stage's SLURM memory, 75 %."""
    return int(cfg["slurm"]["stage"][stage]["mem_per_cpu_mb"] * 0.75)


def write(  # noqa: PLR0913
    cfg: Config,
    stage: str,
    task_dir: Path,
    keywords: str,
    blocks: str,
    geometry: str,
    charge: int,
    label: str,
) -> None:
    """input.inp + geom.xyz + run.sh of one task."""
    orca.write_input(
        task_dir,
        keywords,
        cfg["slurm"]["stage"][stage]["cpus"],
        maxcore(cfg, stage),
        geometry,
        charge,
        1,
        blocks,
    )
    orca_run_script(task_dir, label, cfg)


def write_autode(cfg: Config, task_dir: Path, label: str) -> None:
    """Complete autode/task.toml with the settings; write run.sh.

    The search itself (complexes, active bonds) was written at set-up; the
    programs, keywords, cores and memory come from [ts.autode] and the
    ts_autode SLURM stage each time the task is written.
    """
    a = cfg["ts"]["autode"]
    stage = cfg["slurm"]["stage"]["ts_autode"]
    task = read_toml(task_dir / "task.toml")
    task |= {
        "n_cores": int(stage["cpus"]),
        "max_core_mb": maxcore(cfg, "ts_autode"),
        "hcode": a["hcode"],
        "lcode": a["lcode"],
        "solvent": a["solvent"],
        "min_imag_freq_cm1": float(a["min_imag_freq_cm1"]),
        "orca_keywords": {k: a["keywords"][k] for k in KEYWORD_SETS},
        "orca_blocks": dict(a.get("blocks", {})),
    }
    write_toml(task_dir / "task.toml", task)
    python = a.get("python") or sys.executable
    write_text(
        task_dir / "run.sh",
        AUTODE_RUN.format(label=label, python=shlex.quote(python)),
        executable=True,
    )


def parsed(task_dir: Path) -> dict:
    """orca.parse_output of the task's output.out."""
    return orca.parse_output(task_dir / "output.out")


def done(task_dir: Path, kind: str) -> bool:
    """True once the task has what the next stage needs."""
    res = parsed(task_dir)
    if not res.get("terminated_normally"):
        return False
    if kind == "freq":
        return bool(res.get("frequencies_cm1"))
    if kind == "opt":
        return (
            bool(res.get("opt_converged"))
            and (task_dir / "final.xyz").exists()
        )
    raise ValueError(f"unknown task kind {kind!r}")


def start_geometry(task_dir: Path, initial: str) -> str:
    """Restart an unfinished optimisation from its last geometry."""
    final = task_dir / "final.xyz"
    if final.exists() and (task_dir / "output.out").exists():
        return final.read_text(encoding="utf-8")
    return initial


def ts_check(res: dict, threshold: float) -> tuple[bool, str]:
    """Exactly one imaginary mode, stronger than -threshold cm^-1."""
    freqs = res.get("frequencies_cm1")
    if not freqs:
        return False, "no frequencies in output"
    imag = [f for f in freqs if f < 0]
    if len(imag) != 1:
        return False, f"{len(imag)} imaginary modes ({imag})"
    if imag[0] > -threshold:
        return False, f"imaginary mode {imag[0]:.1f} cm-1 weaker than " + (
            f"-{threshold:g}"
        )
    return True, ""


def minimum_check(res: dict, tolerance: float) -> tuple[bool, str]:
    """No imaginary mode below -tolerance cm^-1."""
    freqs = res.get("frequencies_cm1")
    if freqs is None:
        return False, "no frequencies in output"
    imag = [f for f in freqs if f < -tolerance]
    return (not imag), (f"imaginary modes {imag}" if imag else "")
