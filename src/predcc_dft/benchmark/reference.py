"""Benchmark steps 8 and 10: DLPNO-CCSD(T)/CBS reference and solvation.

Every solvation model of benchmark/methods.toml, on the trunk best.xyz.

    benchmark/molecules/<key>/reference/
        ref_x/ ref_y/                 DLPNO-CCSD(T), cardinal numbers X, Y
        solv_gas/ solv_<label>/       solvation level, gas and each model
        reference.toml                identity, trunk G terms, dG_solv per
                                      model, CBS reference energy, T1
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends.slurm import write_array
from predcc_dft.energies import reference as dlpno
from predcc_dft.energies import single_point, solvation, thermo
from predcc_dft.protocol import project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, pipeline_commit, read_toml, write_toml
from predcc_dft.protocol.methods import solvation_models

STAGE = "reference"
BRANCH = "benchmark"


def job_inputs(cfg: Config) -> dict[str, tuple[str, str]]:
    """Task name -> (keywords, blocks): DLPNO jobs + every solvation job."""
    return {
        **dlpno.job_inputs(cfg),
        **solvation.job_inputs(
            cfg["energies"]["solvation_keywords"],
            solvation_models(cfg.methods),
        ),
    }


def _calculation(rec: dict, job: str) -> dict:
    if job.startswith("ref_"):
        return single_point.calculation(
            rec, "sp", f"dlpno_ccsdt_{job[-1]}", "gas", BRANCH
        )
    return single_point.calculation(
        rec, "solv", "r2scan3c", job.removeprefix("solv_"), BRANCH
    )


def _needs(job: str) -> tuple[str, ...]:
    return (
        single_point.NEEDS_DLPNO
        if job.startswith("ref_")
        else single_point.NEEDS_SP
    )


def _dir(cfg: Config, rec: dict) -> Path:
    return project.branch_mol_dir(cfg, BRANCH, rec) / "reference"


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write every unfinished reference/solvation job of the selection."""
    tasks, waiting = [], 0
    for rec in project.selected(cfg, BRANCH):
        if project.structure(cfg, rec) is None:
            waiting += 1
            continue
        geometry = (project.mol_dir(cfg, rec) / "best.xyz").read_text(
            encoding="utf-8"
        )
        for job, (keywords, blocks) in job_inputs(cfg).items():
            jdir = _dir(cfg, rec) / job
            if not single_point.needs_run(jdir, _needs(job), geometry, force):
                continue
            single_point.write(
                cfg, STAGE, jdir, keywords, blocks, geometry, rec, job
            )
            tasks.append(jdir)
    script = write_array(cfg, STAGE, tasks, cfg.benchmark_dir / "jobs")
    return {
        "tasks": len(tasks),
        "waiting_for_trunk": waiting,
        "script": str(script) if script else None,
    }


def collect(cfg: Config) -> dict:
    """record.toml per finished job and reference.toml per molecule."""
    models = solvation_models(cfg.methods)
    n_done, incomplete = 0, 0
    for rec in project.selected(cfg, BRANCH):
        struct = project.structure(cfg, rec)
        if struct is None:
            incomplete += 1
            continue
        geometry = (project.mol_dir(cfg, rec) / "best.xyz").read_text(
            encoding="utf-8"
        )
        res, state = {}, {}
        for job in job_inputs(cfg):
            state[job], res[job] = single_point.collect(
                _dir(cfg, rec) / job,
                _calculation(rec, job),
                _needs(job),
                geometry,
            )
        ref_ok = state["ref_x"] == state["ref_y"] == "completed"
        solv_data = {}
        for label, model in models.items():
            dg = solvation.dg_solv(label, model, res)
            if dg is not None:
                solv_data[label] = {"dg_solv_kcal_mol": dg}
        if not ref_ok and not solv_data:
            incomplete += 1
            continue
        data = {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "pipeline": pipeline_commit(),
            "branch": BRANCH,
            **project.identity(rec, struct),
            "geometry_sha256": records.geometry_sha256(geometry),
            "thermo": thermo.species_terms(cfg, struct),
            "solvation": {
                "level": cfg["energies"]["solvation_keywords"],
                "e_gas_hartree": res["solv_gas"].get("final_energy"),
                **solv_data,
            },
        }
        if ref_ok:
            data["reference"] = dlpno.summarize(
                cfg, res["ref_x"], res["ref_y"]
            )
        write_toml(_dir(cfg, rec) / "reference.toml", data)
        n_done += ref_ok
    return {"species_with_reference": n_done, "incomplete": incomplete}


def data(cfg: Config, rec: dict) -> dict | None:
    """reference.toml of one molecule, if collected."""
    path = _dir(cfg, rec) / "reference.toml"
    return read_toml(path) if path.exists() else None
