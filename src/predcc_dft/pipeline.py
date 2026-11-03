"""One command for the whole benchmark: ``predcc run [--submit]``.

Each call ("round") does everything that can be done with the results on disk:

  1. steps 1-3 (prepare) if the input lists or their settings changed;
  2. local steps: structure check and collapse (7), parse finished single
     points (8-9), records -> derived tables, ranking, production candidate
     (10-11, SOP 2.7/8);
  3. writes the job arrays of every stage that has work ready
     (conformers, dft_opt, frequencies incl. retries, reference, benchmark).

With ``--submit`` the arrays are submitted and a small driver job is queued
behind them (``--dependency=afterany``) that runs the next round, so the whole
benchmark advances by itself until nothing is left to run. Without SLURM (or
without --submit) the round just prints what to submit.

Safety: nothing is written while a stage of this project is still in the queue
(partial outputs would look like failures); a task folder is submitted at most
``[pipeline] max_submissions`` times; the chain stops after ``max_rounds``.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from predcc_dft import (
    analysis,
    benchmark,
    conformers,
    dft_opt,
    frequencies,
    prepare,
    records,
    reference,
    sanity,
    slurm,
    status,
)
from predcc_dft.io_utils import Config, read_toml, write_text, write_toml

log = logging.getLogger(__name__)

STAGES = ("conformers", "dft_opt", "frequencies", "reference", "benchmark")


def _writers(cfg: Config) -> dict[str, Callable[..., dict]]:
    """Job writer per stage (dft_opt only on the ORCA route)."""
    writers: dict[str, Callable[..., dict]] = {
        "conformers": conformers.write_jobs,
        "dft_opt": dft_opt.write_jobs,
        "frequencies": frequencies.write_jobs,
        "reference": reference.write_jobs,
        "benchmark": benchmark.write_jobs,
    }
    if conformers.engine(cfg) != "orca":
        writers.pop("dft_opt")
    return writers


def local_steps(cfg: Config) -> dict:
    """Steps 7 and 10-11 plus the derived tables; cheap, run every round."""
    return {
        "check (step 7)": sanity.run(cfg),
        "reference (step 8)": reference.collect(cfg),
        "benchmark (step 9)": benchmark.collect(cfg),
        "calculations.csv rows": records.collect_csv(cfg),
        "analysis": analysis.run(cfg),
    }


def _driver_script(cfg: Config, next_round: int, config: Path | None) -> Path:
    """Write jobs/driver/driver.slurm, which runs the next round."""
    stage_dir = cfg.jobs_dir / "driver"
    (stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "predcc_dft", "--root", str(cfg.root)]
    if config is not None:
        cmd += ["--config", str(config.resolve())]
    cmd += ["run", "--submit", "--round", str(next_round)]
    for level in cfg["benchmark"].get("levels") or []:
        cmd += ["--level", level]
    for functional in cfg["benchmark"].get("functionals") or []:
        cmd += ["--functional", functional]
    lines = slurm._header(  # noqa: SLF001 - shared SBATCH header
        cfg,
        "driver",
        "predcc_driver",
        f"{stage_dir}/logs/round{next_round}_%j.out",
    )
    lines += ["", f'cd "{cfg.root}"', shlex.join(cmd), ""]
    script = stage_dir / "driver.slurm"
    write_text(script, "\n".join(lines))
    return script


def _cancel_other_drivers(cfg: Config) -> None:
    """Cancel a queued driver job that is not this one."""
    path = cfg.jobs_dir / "submitted.toml"
    if not path.exists():
        return
    driver = read_toml(path).get("driver", {}).get("job_id", "")
    me = os.environ.get("SLURM_JOB_ID", "")
    if driver and driver != me and driver in slurm.active_jobs():
        subprocess.run(["scancel", driver], check=False)


def _ensure_driver(
    cfg: Config, running: dict[str, str], round_no: int, config: Path | None
) -> str:
    """Keep the chain alive behind stages someone else submitted."""
    path = cfg.jobs_dir / "submitted.toml"
    driver = read_toml(path).get("driver", {}).get("job_id", "")
    me = os.environ.get("SLURM_JOB_ID", "")
    if driver and driver != me and driver in slurm.active_jobs():
        return f"job {driver} already queued"
    script = _driver_script(cfg, round_no + 1, config)
    job_id = slurm.sbatch(script, dependency=list(running.values()))
    _record_driver(cfg, job_id, round_no + 1)
    return f"job {job_id} queued behind the running stages"


def _record_driver(cfg: Config, job_id: str, round_no: int) -> None:
    """Remember the driver job in jobs/submitted.toml."""
    path = cfg.jobs_dir / "submitted.toml"
    data = read_toml(path) if path.exists() else {}
    data["driver"] = {"job_id": job_id, "round": round_no}
    write_toml(path, data)


def _submit(
    cfg: Config,
    written: dict[str, dict],
    round_no: int,
    config: Path | None,
) -> dict:
    """Submit every written array, then the driver behind them."""
    jobs = {}
    for stage, res in written.items():
        if not res.get("script"):
            continue
        tasks = slurm.manifest_tasks(cfg, stage)
        job_id = slurm.sbatch(Path(res["script"]))
        slurm.record_submission(cfg, stage, job_id, tasks)
        jobs[stage] = job_id
        log.info("submitted %s: job %s (%d tasks)", stage, job_id, len(tasks))
    out: dict = {"submitted": jobs or "nothing"}
    if not jobs:
        out["driver"] = "not needed: no jobs left to run"
        return out
    max_rounds = int(cfg["pipeline"]["max_rounds"])
    if round_no >= max_rounds:
        out["driver"] = f"not queued: max_rounds = {max_rounds} reached"
        return out
    _cancel_other_drivers(cfg)
    script = _driver_script(cfg, round_no + 1, config)
    driver_id = slurm.sbatch(script, dependency=list(jobs.values()))
    _record_driver(cfg, driver_id, round_no + 1)
    out["driver"] = f"job {driver_id} runs round {round_no + 1} afterwards"
    return out


def advance(
    cfg: Config,
    submit: bool = False,
    round_no: int = 1,
    config: Path | None = None,
    force_prepare: bool = False,
) -> dict:
    """Run one round. Returns a report dict (printed by the CLI)."""
    cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.jobs_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _advance(cfg, submit, round_no, config, force_prepare)


def _advance(
    cfg: Config,
    submit: bool,
    round_no: int,
    config: Path | None,
    force_prepare: bool,
) -> dict:
    """Body of advance(), run under the project lock."""
    report: dict = {"round": round_no}
    running = slurm.running_stages(cfg)
    if running:
        report["busy"] = (
            "still in the queue: "
            + ", ".join(f"{s} (job {j})" for s, j in running.items())
            + "; nothing written. The driver job continues automatically"
            " after them (or rerun this command later)."
        )
        if submit and round_no < int(cfg["pipeline"]["max_rounds"]):
            report["driver"] = _ensure_driver(cfg, running, round_no, config)
        report["status"] = status.summary(cfg)
        return report

    if force_prepare or not prepare.up_to_date(cfg):
        report["prepare (steps 1-3)"] = prepare.run(cfg)
    report.update(local_steps(cfg))

    written = {stage: fn(cfg) for stage, fn in _writers(cfg).items()}
    report["jobs written"] = {
        s: f"{r['tasks']} tasks"
        + (f" -> {r['script']}" if r["script"] else "")
        for s, r in written.items()
    }
    given_up = {
        s: len((cfg.jobs_dir / s / "given_up.txt").read_text().split())
        for s in written
        if (cfg.jobs_dir / s / "given_up.txt").exists()
    }
    given_up = {s: n for s, n in given_up.items() if n}
    if given_up:
        report["given up (see jobs/<stage>/given_up.txt)"] = given_up

    if submit:
        if not slurm.available():
            report["submit"] = "sbatch/squeue not found: submit by hand"
        else:
            report.update(_submit(cfg, written, round_no, config))
    else:
        scripts = [r["script"] for r in written.values() if r["script"]]
        report["next"] = (
            "sbatch " + " ; sbatch ".join(scripts) + "   (or use --submit)"
            if scripts
            else "nothing to submit"
        )
    report["status"] = status.summary(cfg)
    return report


def cancel(cfg: Config) -> list[str]:
    """Cancel every queued job of this project (jobs/submitted.toml)."""
    path = cfg.jobs_dir / "submitted.toml"
    if not path.exists():
        return []
    data = read_toml(path)
    ids = [v["job_id"] for v in data.get("stage", {}).values()]
    ids.append(data.get("driver", {}).get("job_id", ""))
    active = slurm.active_jobs()
    ids = [i for i in ids if i and i in active]
    if ids:
        subprocess.run(["scancel", *ids], check=False)
    return ids
