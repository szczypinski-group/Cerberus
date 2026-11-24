"""One command per chain: ``predcc run | benchmark | production | ts``.

A study has one shared TRUNK (steps 1-7: inputs -> best.xyz) and two BRANCHES
that start from the same best.xyz. Each chain runs the trunk plus, optionally,
one branch:

    run          trunk only
    benchmark    trunk + benchmark branch (method grid, DLPNO, ranking)
    production   trunk + production branch (locked method, many molecules)
    ts           trunk + TS module (transition states, rates, kinetics)

Each call ("round") does everything that can be done with the results on disk:

  1. steps 1-3 (prepare) if the input lists or their settings changed;
  2. local steps: structure check and collapse (7); the branch's collect and
     analysis (records -> derived tables);
  3. writes the job arrays of every stage that has work ready.

With ``--submit`` the arrays are submitted and a small driver job is queued
behind them (``--dependency=afterany``) that runs the next round of the same
chain, so it advances by itself until nothing is left to run. A benchmark
chain and a production chain can run side by side: each waits only for the
trunk and its own stages, and has its own driver.

Safety: nothing is written while a stage this chain depends on is still in the
queue (partial outputs would look like failures); a task folder is submitted
at most ``[pipeline] max_submissions`` times; a chain stops after
``max_rounds``.

``predcc trunk --submit`` is the single-job alternative for the trunk: one
SLURM job (``[slurm.stage.trunk]``) runs the same rounds itself, executing
each round's task folders one after another inside its own allocation
instead of submitting arrays (``run_trunk_here``). Other chains wait for it
like for any trunk stage.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import IO

from predcc_dft.backends import slurm
from predcc_dft.benchmark import report as benchmark_report
from predcc_dft.production import report as production_report
from predcc_dft.protocol import lock, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml, write_text, write_toml
from predcc_dft.species import prepare
from predcc_dft.structures import (
    conformers,
    dft_opt,
    frequencies,
    sanity,
    summary,
)
from predcc_dft.ts import report as ts_report
from predcc_dft.ts import stage as ts_stage
from predcc_dft.workflow import index, status

log = logging.getLogger(__name__)

CHAINS = ("trunk", "benchmark", "production", "ts")
TRUNK_STAGES = ("conformers", "dft_opt", "frequencies")
TRUNK_JOB = "trunk"  # `predcc trunk`: the whole trunk in one SLURM job
BRANCH_STAGES = {
    "benchmark": ("reference", "benchmark"),
    "production": ("production",),
    "ts": ts_stage.STAGES,
}
COMMAND = {
    "trunk": "run",
    "benchmark": "benchmark",
    "production": "production",
    "ts": "ts",
}


def _trunk_writers(cfg: Config) -> dict[str, Callable[..., dict]]:
    """Job writer per trunk stage (dft_opt only on the ORCA route)."""
    writers: dict[str, Callable[..., dict]] = {
        "conformers": conformers.write_jobs,
        "dft_opt": dft_opt.write_jobs,
        "frequencies": frequencies.write_jobs,
    }
    if conformers.engine(cfg) != "orca":
        writers.pop("dft_opt")
    return writers


def _branch(chain: str) -> ModuleType | None:
    """The branch report module (local_steps, writers) of a chain."""
    return {
        "benchmark": benchmark_report,
        "production": production_report,
        "ts": ts_report,
    }.get(chain)


def stage_dir(cfg: Config, stage: str) -> Path:
    """Folder of a stage's job array: jobs/<stage> or <branch>/jobs/<stage>."""
    for branch, stages in BRANCH_STAGES.items():
        if stage in stages:
            return cfg.branch_dir(branch) / "jobs" / stage
    return cfg.jobs_dir / stage


def trunk_local_steps(cfg: Config) -> dict:
    """Step 7 and the trunk's calculations.csv; cheap, run every round."""
    return {
        "check (step 7)": sanity.run(cfg),
        "trunk.toml files": summary.run(cfg),
        "trunk calculations.csv rows": records.collect_csv(cfg),
    }


def local_steps(cfg: Config, chain: str = "trunk") -> dict:
    """Trunk local steps, then the branch's collect and analysis."""
    out = trunk_local_steps(cfg)
    branch = _branch(chain)
    if branch is not None:
        out.update(branch.local_steps(cfg))
    return out


def _predcc_command(
    cfg: Config, config: Path | None, args: list[str]
) -> list[str]:
    """`python -m predcc_dft --root <study> [--config ...] <args>`."""
    cmd = [sys.executable, "-m", "predcc_dft", "--root", str(cfg.root)]
    if config is not None:
        cmd += ["--config", str(config.resolve())]
    return cmd + args


def _driver_script(
    cfg: Config, chain: str, next_round: int, config: Path | None
) -> Path:
    """Write jobs/driver/<chain>.slurm, which runs the chain's next round."""
    drv_dir = cfg.jobs_dir / "driver"
    (drv_dir / "logs").mkdir(parents=True, exist_ok=True)
    cmd = _predcc_command(
        cfg, config, [COMMAND[chain], "--submit", "--round", str(next_round)]
    )
    if chain == "benchmark":
        for level in cfg["benchmark"].get("levels") or []:
            cmd += ["--level", level]
        for functional in cfg["benchmark"].get("functionals") or []:
            cmd += ["--functional", functional]
    lines = slurm.header(
        cfg,
        "driver",
        f"predcc_{chain}",
        f"{drv_dir}/logs/{chain}_round{next_round}_%j.out",
    )
    lines += ["", f'cd "{cfg.root}"', shlex.join(cmd), ""]
    script = drv_dir / f"{chain}.slurm"
    write_text(script, "\n".join(lines))
    return script


def _drivers(cfg: Config) -> dict:
    """Chain -> {job_id, round} of the last driver jobs."""
    path = cfg.jobs_dir / "submitted.toml"
    drivers = read_toml(path).get("driver", {}) if path.exists() else {}
    return {} if "job_id" in drivers else drivers  # old single-driver layout


def _record_driver(
    cfg: Config, chain: str, job_id: str, round_no: int
) -> None:
    """Remember the chain's driver job in jobs/submitted.toml."""
    path = cfg.jobs_dir / "submitted.toml"
    data = read_toml(path) if path.exists() else {}
    drivers = _drivers(cfg)
    drivers[chain] = {"job_id": job_id, "round": round_no}
    data["driver"] = drivers
    write_toml(path, data)


def _other_driver(cfg: Config, chain: str) -> str:
    """Queued driver job of this chain that is not the running one."""
    driver = _drivers(cfg).get(chain, {}).get("job_id", "")
    me = os.environ.get("SLURM_JOB_ID", "")
    if driver and driver != me and driver in slurm.active_jobs():
        return driver
    return ""


def _ensure_driver(
    cfg: Config,
    chain: str,
    running: dict[str, str],
    round_no: int,
    config: Path | None,
) -> str:
    """Keep the chain alive behind stages someone else submitted."""
    queued = _other_driver(cfg, chain)
    if queued:
        return f"job {queued} already queued"
    script = _driver_script(cfg, chain, round_no + 1, config)
    job_id = slurm.sbatch(script, dependency=list(running.values()))
    _record_driver(cfg, chain, job_id, round_no + 1)
    return f"job {job_id} queued behind the running stages"


def _submit(
    cfg: Config,
    chain: str,
    written: dict[str, dict],
    round_no: int,
    config: Path | None,
) -> dict:
    """Submit every written array, then the chain's driver behind them."""
    jobs = {}
    for stage, res in written.items():
        if not res.get("script"):
            continue
        tasks = slurm.manifest_tasks(cfg, Path(res["script"]).parent)
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
    queued = _other_driver(cfg, chain)
    if queued:
        subprocess.run(["scancel", queued], check=False)
    script = _driver_script(cfg, chain, round_no + 1, config)
    driver_id = slurm.sbatch(script, dependency=list(jobs.values()))
    _record_driver(cfg, chain, driver_id, round_no + 1)
    out["driver"] = f"job {driver_id} runs round {round_no + 1} afterwards"
    return out


def _lock(fh: IO[str]) -> None:
    """Exclusive lock on an open file (POSIX flock; msvcrt on Windows)."""
    if sys.platform == "win32":  # local dry runs on Windows
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(fh, fcntl.LOCK_EX)


def advance(
    cfg: Config,
    submit: bool = False,
    round_no: int = 1,
    config: Path | None = None,
    force_prepare: bool = False,
    chain: str = "trunk",
) -> dict:
    """Run one round of a chain. Returns a report dict (printed by the CLI)."""
    if chain not in CHAINS:
        raise ValueError(f"chain must be one of {CHAINS}, not {chain!r}")
    if chain == "production":
        lock.read(cfg)  # fail early, with instructions, if there is no lock
    cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.jobs_dir / ".lock").open("w", encoding="utf-8") as fh:
        _lock(fh)
        return _advance(cfg, chain, submit, round_no, config, force_prepare)


def _advance(
    cfg: Config,
    chain: str,
    submit: bool,
    round_no: int,
    config: Path | None,
    force_prepare: bool,
) -> dict:
    """Body of advance(), run under the project lock."""
    report: dict = {"round": round_no, "chain": chain}
    relevant = {TRUNK_JOB, *TRUNK_STAGES, *BRANCH_STAGES.get(chain, ())}
    me = os.environ.get("SLURM_JOB_ID", "")  # the trunk job checks itself
    running = {
        s: j
        for s, j in slurm.running_stages(cfg).items()
        if s in relevant and j != me
    }
    if running:
        report["busy"] = (
            "still in the queue: "
            + ", ".join(f"{s} (job {j})" for s, j in running.items())
            + "; nothing written. The driver job continues automatically"
            " after them (or rerun this command later)."
        )
        if submit and round_no < int(cfg["pipeline"]["max_rounds"]):
            report["driver"] = _ensure_driver(
                cfg, chain, running, round_no, config
            )
        report["status"] = status.summary(cfg)
        return report

    if force_prepare or not prepare.up_to_date(cfg):
        report["prepare (steps 1-3)"] = prepare.run(cfg)
    report.update(local_steps(cfg, chain))
    report["index/<molecule>.toml files"] = index.run(cfg)

    writers = dict(_trunk_writers(cfg))
    branch = _branch(chain)
    if branch is not None:
        writers.update(branch.writers())
    written = {stage: fn(cfg) for stage, fn in writers.items()}
    report["jobs written"] = {
        s: f"{r['tasks']} tasks"
        + (f" -> {r['script']}" if r["script"] else "")
        for s, r in written.items()
    }
    given_up = {}
    for s in written:
        f = stage_dir(cfg, s) / "given_up.txt"
        n = len(f.read_text(encoding="utf-8").split()) if f.exists() else 0
        if n:
            given_up[s] = n
    if given_up:
        report["given up (see <stage>/given_up.txt)"] = given_up

    if submit:
        if not slurm.available():
            report["submit"] = "sbatch/squeue not found: submit by hand"
        else:
            report.update(_submit(cfg, chain, written, round_no, config))
    else:
        scripts = [r["script"] for r in written.values() if r["script"]]
        report["next"] = (
            "sbatch " + " ; sbatch ".join(scripts) + "   (or use --submit)"
            if scripts
            else "nothing to submit"
        )
    report["status"] = status.summary(cfg)
    return report


# ------------------------------------------------------ the trunk in one job


def trunk_script(
    cfg: Config, config: Path | None = None, force_prepare: bool = False
) -> Path:
    """Write jobs/trunk/trunk.slurm: one job that runs the whole trunk."""
    job_dir = cfg.jobs_dir / TRUNK_JOB
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    args = ["trunk", "--here"] + (["--force-prepare"] if force_prepare else [])
    lines = slurm.header(
        cfg, TRUNK_JOB, "predcc_trunk", f"{job_dir}/logs/trunk_%j.out"
    )
    lines += [
        "",
        f'cd "{cfg.root}"',
        shlex.join(_predcc_command(cfg, config, args)),
        "",
    ]
    script = job_dir / "trunk.slurm"
    write_text(script, "\n".join(lines))
    return script


def trunk_job(
    cfg: Config,
    submit: bool = False,
    config: Path | None = None,
    force_prepare: bool = False,
) -> dict:
    """`predcc trunk [--submit]`: write (and submit) the single trunk job."""
    report: dict = {}
    running = {
        s: j
        for s, j in slurm.running_stages(cfg).items()
        if s in (TRUNK_JOB, *TRUNK_STAGES)
    }
    if running:
        report["busy"] = (
            "already in the queue: "
            + ", ".join(f"{s} (job {j})" for s, j in running.items())
            + "; nothing submitted."
        )
        return report
    st = cfg["slurm"]["stage"][TRUNK_JOB]
    cpus, per = int(st["cpus"]), int(st.get("cores_per_task", st["cpus"]))
    report["layout"] = (
        f"{cpus} cores, {max(1, cpus // per)} calculations at once, "
        f"{min(per, cpus)} cores each"
    )
    if cpus % per:
        report["warning"] = (
            f"[slurm.stage.trunk] cpus = {cpus} is not a multiple of "
            f"cores_per_task = {per}: {cpus % per} cores stay idle"
        )
    script = trunk_script(cfg, config, force_prepare)
    report["script"] = str(script)
    if not submit:
        report["next"] = f"sbatch {script}   (or use --submit)"
    elif not slurm.available():
        report["submit"] = "sbatch/squeue not found: submit by hand"
    else:
        job_id = slurm.sbatch(script)
        slurm.record_submission(cfg, TRUNK_JOB, job_id, [])
        report["submitted"] = (
            f"job {job_id}: steps 1-7 for every molecule, task after task"
        )
    return report


def core_slots(cfg: Config) -> list[list[int]]:
    """Split this job's cores into one group per simultaneous calculation.

    ``[slurm.stage.trunk] cpus`` cores (at most those this process may use)
    in groups of ``cores_per_task``: 16 cores, 4 per task -> 4 at once.
    """
    st = cfg["slurm"]["stage"][TRUNK_JOB]
    per = max(1, int(st.get("cores_per_task", st["cpus"])))
    try:
        mine = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        mine = list(range(os.cpu_count() or 1))
    usable = mine[: max(1, int(st["cpus"]))]
    if len(usable) < per:
        return [usable]
    return [
        usable[i : i + per] for i in range(0, len(usable) // per * per, per)
    ]


@contextlib.contextmanager
def _stage_cores(cfg: Config, n: int) -> Iterator[None]:
    """Write every trunk task for n cores (ORCA %pal, CREST -T) meanwhile."""
    stages = cfg["slurm"]["stage"]
    saved = {s: stages[s]["cpus"] for s in TRUNK_STAGES if s in stages}
    try:
        for s in saved:
            stages[s]["cpus"] = n
        yield
    finally:
        for s, v in saved.items():
            stages[s]["cpus"] = v


_PAL = re.compile(r"%pal nprocs \d+ end")


def _run_task(cfg: Config, stage: str, task: Path, cores: list[int]) -> int:
    """Run one task folder's run.sh on these cores; return its exit code."""
    n = len(cores)
    inp = task / "input.inp"
    if inp.exists():  # written earlier for another core count
        text = inp.read_text(encoding="utf-8")
        fixed = _PAL.sub(f"%pal nprocs {n} end", text)
        if fixed != text:
            write_text(inp, fixed)
    env = dict(os.environ)
    env["PREDCC_CORES"] = str(n)  # what job_info.toml / record.toml report
    env["OMP_NUM_THREADS"] = "1" if stage in slurm.MPI_STAGES else str(n)
    if stage not in slurm.MPI_STAGES:  # CREST/CENSO read their threads here
        env["SLURM_CPUS_PER_TASK"] = str(n)
    # Several mpiruns share the allocation: let the ranks inherit the
    # taskset mask instead of all binding to the first cores.
    env["OMPI_MCA_hwloc_base_binding_policy"] = "none"
    env["PRTE_MCA_hwloc_default_binding_policy"] = "none"  # Open MPI 5
    cmd = ["bash", str(task / "run.sh")]
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", ",".join(map(str, cores)), *cmd]
    with (task / "trunk_job.out").open("w", encoding="utf-8") as fh:
        done = subprocess.run(
            cmd,
            cwd=task,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return done.returncode


def _run_parallel(
    cfg: Config, round_no: int, tasks: dict[str, list[Path]]
) -> dict:
    """Run every task of a round, one per free core group at a time."""
    slots = core_slots(cfg)
    free: queue.Queue[list[int]] = queue.Queue()
    for group in slots:
        free.put(group)
    jobs = [(s, d) for s, dirs in tasks.items() for d in dirs]
    log.info(
        "round %d: %d tasks, %d at a time, %d cores each",
        round_no,
        len(jobs),
        len(slots),
        len(slots[0]),
    )

    def one(stage: str, task: Path) -> int:
        cores = free.get()
        try:
            log.info(
                "start %s on cores %s: %s",
                stage,
                ",".join(map(str, cores)),
                task.relative_to(cfg.root).as_posix(),
            )
            return _run_task(cfg, stage, task, cores)
        finally:
            free.put(cores)

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        codes = list(pool.map(lambda j: one(*j), jobs))
    failed: dict[str, int] = dict.fromkeys(tasks, 0)
    for (stage, _), code in zip(jobs, codes, strict=True):
        failed[stage] += code != 0
    return {
        s: f"{len(dirs)} tasks, {failed[s]} failed"
        for s, dirs in tasks.items()
    }


def run_trunk_here(cfg: Config, force_prepare: bool = False) -> dict:
    """Run trunk rounds until nothing is left, executing tasks in-process.

    Each round is ``advance(chain="trunk")`` without submission, with every
    task written for ``[slurm.stage.trunk] cores_per_task`` cores. The task
    folders are then run here (inside the trunk job's allocation), as many
    at once as there are core groups (``core_slots``). Submission counts and
    ``max_rounds`` apply exactly as for the array route.
    """
    max_rounds = int(cfg["pipeline"]["max_rounds"])
    per_task = len(core_slots(cfg)[0])
    rounds: dict[str, dict] = {}
    rep: dict = {}
    with _stage_cores(cfg, per_task):
        for round_no in range(1, max_rounds + 1):
            rep = advance(
                cfg,
                round_no=round_no,
                force_prepare=force_prepare and round_no == 1,
                chain="trunk",
            )
            if "busy" in rep:
                return {"stopped": rep["busy"], "rounds": rounds or "none"}
            tasks = {
                s: slurm.manifest_tasks(cfg, stage_dir(cfg, s))
                for s in _trunk_writers(cfg)
            }
            tasks = {s: t for s, t in tasks.items() if t}
            if not tasks:
                break
            for dirs in tasks.values():
                slurm.count_submissions(dirs)
            rounds[f"round {round_no}"] = _run_parallel(cfg, round_no, tasks)
        else:
            return {
                "stopped": f"max_rounds = {max_rounds} reached",
                "rounds": rounds,
                "status": status.summary(cfg),
            }
    out: dict = {"finished": "nothing left to run", "rounds": rounds or "none"}
    key = "given up (see <stage>/given_up.txt)"
    if key in rep:
        out[key] = rep[key]
    out["status"] = rep["status"]
    return out


def cancel(cfg: Config) -> list[str]:
    """Cancel every queued job of this study (jobs/submitted.toml)."""
    path = cfg.jobs_dir / "submitted.toml"
    if not path.exists():
        return []
    data = read_toml(path)
    ids = [v["job_id"] for v in data.get("stage", {}).values()]
    drivers = data.get("driver", {})
    if "job_id" in drivers:  # old single-driver layout
        ids.append(drivers["job_id"])
    else:
        ids += [d.get("job_id", "") for d in drivers.values()]
    active = slurm.active_jobs()
    ids = [i for i in ids if i and i in active]
    if ids:
        subprocess.run(["scancel", *ids], check=False)
    return ids
