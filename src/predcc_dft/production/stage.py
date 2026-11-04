"""Production: the locked method and solvation model on the trunk best.xyz.

Per molecule and production version (production/production.toml):

    production/molecules/<key>/v<version>/
        energy/        locked functional/basis set, gas phase   record.toml
        solv_gas/      solvation level, gas phase               record.toml
        solv_<model>/  solvation level, locked model            record.toml
        result.toml    identity, lock, E, G terms, dG_solv, G
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends.slurm import write_array
from predcc_dft.energies import single_point, solvation, thermo
from predcc_dft.protocol import constants, lock, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, pipeline_commit, read_toml, write_toml

STAGE = "production"
BRANCH = "production"


def _model(the_lock: dict) -> tuple[str, dict]:
    s = the_lock["solvation"]
    return s["label"], {
        "orca_keywords": s.get("orca_keywords", ""),
        "orca_blocks": s.get("orca_blocks", ""),
        "parse": s.get("parse", ""),
    }


def tasks(the_lock: dict) -> dict[str, tuple[str, str]]:
    """Task name -> (keywords, blocks) of one production version."""
    e = the_lock["energy"]
    label, model = _model(the_lock)
    return {
        "energy": (e["orca_keywords"], e.get("orca_blocks", "")),
        **solvation.job_inputs(
            the_lock["solvation"]["level_keywords"], {label: model}
        ),
    }


def version_dir(cfg: Config, the_lock: dict, rec: dict) -> Path:
    """production/molecules/<key>/v<version>/."""
    return project.branch_mol_dir(cfg, BRANCH, rec) / lock.version_dir(
        the_lock
    )


def _calculation(the_lock: dict, rec: dict, task: str) -> dict:
    if task == "energy":
        return single_point.calculation(
            rec, "sp", the_lock["energy"]["method"], "gas", BRANCH
        )
    return single_point.calculation(
        rec, "solv", "r2scan3c", task.removeprefix("solv_"), BRANCH
    )


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write every unfinished production job of the selection."""
    the_lock = lock.read(cfg)
    jobs = tasks(the_lock)
    out, waiting = [], 0
    for rec in project.selected(cfg, BRANCH):
        if project.structure(cfg, rec) is None:
            waiting += 1
            continue
        geometry = (project.mol_dir(cfg, rec) / "best.xyz").read_text(
            encoding="utf-8"
        )
        for task, (keywords, blocks) in jobs.items():
            jdir = version_dir(cfg, the_lock, rec) / task
            if not single_point.needs_run(
                jdir, single_point.NEEDS_SP, geometry, force
            ):
                continue
            single_point.write(
                cfg, STAGE, jdir, keywords, blocks, geometry, rec, task
            )
            out.append(jdir)
    script = write_array(cfg, STAGE, out, cfg.production_dir / "jobs")
    return {
        "tasks": len(out),
        "production_version": the_lock["production_version"],
        "waiting_for_trunk": waiting,
        "script": str(script) if script else None,
    }


def collect(cfg: Config) -> dict:
    """record.toml per finished job and result.toml per molecule."""
    the_lock = lock.read(cfg)
    version = the_lock["production_version"]
    label, model = _model(the_lock)
    extra = {"calculation": {"production_version": version}}
    counts = {"complete": 0, "incomplete": 0}
    for rec in project.selected(cfg, BRANCH):
        struct = project.structure(cfg, rec)
        if struct is None:
            counts["incomplete"] += 1
            continue
        geometry = (project.mol_dir(cfg, rec) / "best.xyz").read_text(
            encoding="utf-8"
        )
        vdir = version_dir(cfg, the_lock, rec)
        res, state = {}, {}
        for task in tasks(the_lock):
            state[task], res[task] = single_point.collect(
                vdir / task,
                _calculation(the_lock, rec, task),
                single_point.NEEDS_SP,
                geometry,
                extra,
            )
        if not any(s != "pending" for s in state.values()):
            counts["incomplete"] += 1
            continue
        terms = thermo.species_terms(cfg, struct)
        e = res["energy"].get("final_energy")
        dg = solvation.dg_solv(label, model, res)
        done = e is not None and dg is not None
        data: dict = {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "pipeline": pipeline_commit(),
            "branch": BRANCH,
            "production_version": version,
            **project.identity(rec, struct),
            "geometry_sha256": records.geometry_sha256(geometry),
            "status": "complete" if done else "incomplete",
            "tasks": state,
            "method": the_lock["energy"]["method"],
            "solvation_model": label,
            "thermo": terms,
            "energy": {"electronic_hartree": e},
            "solvation": {"dg_solv_kcal_mol": dg},
        }
        if done:
            ens = (
                terms["g_ensemble_correction_kcal_mol"]
                if cfg["analysis"].get("ensemble_correction", True)
                else 0.0
            )
            data["g_hartree"] = (
                e
                + terms["g_corr_hartree"]
                + (dg + terms["standard_state_kcal_mol"] + ens)
                / constants.HARTREE_TO_KCAL
            )
        write_toml(vdir / "result.toml", data)
        counts["complete" if done else "incomplete"] += 1
    return {"production_version": version, **counts}


def result(cfg: Config, the_lock: dict, rec: dict) -> dict | None:
    """result.toml of one molecule for this production version."""
    path = version_dir(cfg, the_lock, rec) / "result.toml"
    return read_toml(path) if path.exists() else None
