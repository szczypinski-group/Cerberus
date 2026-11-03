"""SLURM job arrays and submission.

Each stage gets one array script, jobs/<stage>/<stage>.slurm, with the task
folders written into the script itself (SLURM copies the script at submission,
so rewriting it later can never shift a queued task onto the wrong folder).
Every task folder also has its own run.sh and job.slurm, so a single
calculation can be run or submitted alone.
"""

from __future__ import annotations

import getpass
import logging
import re
import shutil
import subprocess
from pathlib import Path

from predcc_dft.io_utils import Config, read_toml, write_text, write_toml

log = logging.getLogger(__name__)

ORCA_RUN = """#!/bin/bash
# ORCA task: {label}
set -uo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
ORCA_BIN="${{ORCA_BIN:-$(command -v orca)}}"
SCRATCH_ROOT="{scratch_root}"
START=$(date +%s)
if [ -n "$SCRATCH_ROOT" ]; then
    WORK="$SCRATCH_ROOT/$(basename "$HERE")_${{SLURM_JOB_ID:-local}}_\
${{SLURM_ARRAY_TASK_ID:-0}}"
    mkdir -p "$WORK" && cp "$HERE/input.inp" "$HERE/geom.xyz" "$WORK/"
else
    WORK="$HERE"
fi
cd "$WORK"
"$ORCA_BIN" input.inp > "$HERE/output.out" 2> "$HERE/error.log"
STATUS=$?
[ -f "$WORK/input.xyz" ] && cp "$WORK/input.xyz" "$HERE/final.xyz"
[ -f "$WORK/input.hess" ] && cp "$WORK/input.hess" "$HERE/input.hess"
rm -f "$WORK"/*.tmp "$WORK"/*.tmp.* 2>/dev/null
[ "$WORK" != "$HERE" ] && rm -rf "$WORK"
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


def orca_run_script(directory: Path, label: str, cfg: Config) -> None:
    """Write run.sh for one ORCA task folder."""
    write_text(
        directory / "run.sh",
        ORCA_RUN.format(
            label=label, scratch_root=cfg["slurm"].get("scratch_root", "")
        ),
        executable=True,
    )


# Stages whose programs run as MPI ranks (ORCA) rather than threads.
MPI_STAGES = {
    "dft_opt",
    "frequencies",
    "reference",
    "benchmark",
    "orca_input",
}


def _header(
    cfg: Config,
    stage: str,
    job_name: str,
    output: str,
    array: str | None = None,
) -> list[str]:
    """SBATCH header plus module loads and setup lines."""
    s, st = cfg["slurm"], cfg["slurm"]["stage"][stage]
    lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}"]
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines.append("#SBATCH --nodes=1")
    if stage in MPI_STAGES:  # ORCA: one MPI rank per core
        lines += [
            f"#SBATCH --ntasks={st['cpus']}",
            "#SBATCH --cpus-per-task=1",
        ]
    else:  # CREST/CENSO/Python: threads
        lines += [
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={st['cpus']}",
        ]
    lines += [
        f"#SBATCH --mem-per-cpu={st['mem_per_cpu_mb']}M",
        f"#SBATCH --time={st['time']}",
        f"#SBATCH --output={output}",
    ]
    if s.get("account"):
        lines.append(f"#SBATCH --account={s['account']}")
    if s.get("partition"):
        lines.append(f"#SBATCH --partition={s['partition']}")
    lines += ["", "set -uo pipefail"]
    lines += [f"module load {m}" for m in s.get("modules", [])]
    lines += list(s.get("setup_lines", []))
    return lines


def write_single(cfg: Config, stage: str, task_dir: Path, label: str) -> Path:
    """Standalone batch file for one task: sbatch <task_dir>/job.slurm."""
    name = f"{stage}_{label}".replace(" ", "_")[:60]
    lines = _header(cfg, stage, name, f"{task_dir}/slurm-%j.out")
    lines += [
        "",
        f'cd "{task_dir}"',
        'echo "$(hostname): $PWD"',
        "bash run.sh",
        "",
    ]
    script = task_dir / "job.slurm"
    write_text(script, "\n".join(lines))
    return script


def submissions(task_dir: Path) -> int:
    """How often `predcc run --submit` has submitted this task folder."""
    f = task_dir / ".submissions"
    return int(f.read_text(encoding="utf-8").strip() or 0) if f.exists() else 0


def write_array(cfg: Config, stage: str, task_dirs: list[Path]) -> Path | None:
    """Write jobs/<stage>/manifest.txt and jobs/<stage>/<stage>.slurm.

    Task folders already submitted ``[pipeline] max_submissions`` times are
    left out and listed in jobs/<stage>/given_up.txt for inspection.
    """
    stage_dir = cfg.jobs_dir / stage
    (stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    limit = int(cfg["pipeline"]["max_submissions"])
    given_up = [d for d in task_dirs if submissions(d) >= limit]
    task_dirs = [d for d in task_dirs if submissions(d) < limit]
    (stage_dir / "given_up.txt").write_text(
        "".join(f"{d.relative_to(cfg.root).as_posix()}\n" for d in given_up),
        encoding="utf-8",
        newline="\n",
    )
    manifest = stage_dir / "manifest.txt"
    script = stage_dir / f"{stage}.slurm"
    rel = [d.relative_to(cfg.root).as_posix() for d in task_dirs]
    manifest.write_text(
        "".join(f"{r}\n" for r in rel), encoding="utf-8", newline="\n"
    )
    if not task_dirs:
        script.unlink(missing_ok=True)
        return None
    for d, r in zip(task_dirs, rel, strict=True):
        write_single(cfg, stage, d, r.replace("/", "_"))

    array = f"1-{len(task_dirs)}%{cfg['slurm']['array_limit']}"
    lines = _header(cfg, stage, stage, f"{stage_dir}/logs/%A_%a.out", array)
    lines += ["", f'cd "{cfg.root}"', "TASKS=("]
    lines += [f'  "{r}"' for r in rel]
    lines += [
        ")",
        'TASK_DIR="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"',
        'echo "task $SLURM_ARRAY_TASK_ID: $TASK_DIR on $(hostname)"',
        'bash "$TASK_DIR/run.sh"',
        "",
    ]
    write_text(script, "\n".join(lines))
    return script


# ------------------------------------------------------------------ submission


def available() -> bool:
    """True if sbatch and squeue are on PATH."""
    return bool(shutil.which("sbatch") and shutil.which("squeue"))


def sbatch(script: Path, dependency: list[str] | None = None) -> str:
    """Submit a batch script; return the job ID."""
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency=afterany:{':'.join(dependency)}")
    cmd.append(str(script))
    out = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"sbatch {script} failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def active_jobs() -> set[str]:
    """Job IDs (array parents included) of the user's queued/running jobs."""
    if not shutil.which("squeue"):
        return set()
    out = subprocess.run(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%F %i"],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    ids: set[str] = set()
    for line in out.stdout.split():
        ids.update(p for p in re.split(r"[_\[\]]", line) if p.isdigit())
    return ids


def _ledger(cfg: Config) -> Path:
    return cfg.jobs_dir / "submitted.toml"


def running_stages(cfg: Config) -> dict[str, str]:
    """Stages whose last submitted array is still queued or running."""
    path = _ledger(cfg)
    if not path.exists():
        return {}
    active = active_jobs()
    stages = read_toml(path).get("stage", {})
    return {
        s: v["job_id"]
        for s, v in stages.items()
        if v.get("job_id") and v["job_id"] in active
    }


def record_submission(
    cfg: Config, stage: str, job_id: str, task_dirs: list[Path]
) -> None:
    """Remember the stage's job ID and count each task's submissions."""
    from predcc_dft.io_utils import now

    path = _ledger(cfg)
    data = read_toml(path) if path.exists() else {}
    data.setdefault("stage", {})[stage] = {
        "job_id": job_id,
        "tasks": len(task_dirs),
        "submitted": now(),
    }
    write_toml(path, data)
    for d in task_dirs:
        (d / ".submissions").write_text(
            f"{submissions(d) + 1}\n", encoding="utf-8"
        )


def manifest_tasks(cfg: Config, stage: str) -> list[Path]:
    """Task folders listed in jobs/<stage>/manifest.txt."""
    manifest = cfg.jobs_dir / stage / "manifest.txt"
    if not manifest.exists():
        return []
    return [
        cfg.root / line
        for line in manifest.read_text(encoding="utf-8").split()
        if line
    ]
