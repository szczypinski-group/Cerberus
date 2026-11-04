"""SOP 5.1 step 8: DLPNO-CCSD(T)/CBS reference on each best geometry.

Per molecule (molecules/<key>/reference/):
  ref_x, ref_y      DLPNO-CCSD(T) with cardinal numbers X, Y (gas phase);
                    two-point CBS extrapolation in Python
  solv_gas          solvation level ([reference] solvation_keywords), gas phase
  solv_<label>      same level in every [solvation.<label>] of methods.toml;
                    dG_solv = E(model) - E(gas)
Each finished task gets a record.toml; reference.toml collects the species.

The thermal correction (step 6) and dG_solv stored here are reused unchanged
for every benchmarked functional/basis pair; only the electronic energy
changes (SOP 5.1 steps 6, 10, 11).
"""

from __future__ import annotations

import math

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends import orca
from predcc_dft.backends.slurm import orca_run_script, write_array
from predcc_dft.energies import solvation
from predcc_dft.protocol import constants, methods, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, pipeline_commit, write_toml


def job_inputs(cfg: Config) -> dict[str, tuple[str, str]]:
    """Task name -> (keywords, blocks): DLPNO reference + solvation jobs."""
    r = cfg["reference"]
    return {
        "ref_x": (r["keywords_x"], ""),
        "ref_y": (r["keywords_y"], ""),
        **solvation.job_inputs(cfg),
    }


def _calculation(cfg: Config, rec: dict, job: str) -> dict:
    if job.startswith("ref_"):
        method, solv = f"dlpno_ccsdt_{job[-1]}", "gas"
        step = "sp"
    else:
        method, solv = "r2scan3c", job.removeprefix("solv_")
        step = "solv"
    return {
        "id": records.calc_id(rec["compound"], "best", step, method, solv),
        "compound": rec["compound"],
        "conformer": "best",
        "step": step,
        "method": method,
        "solvation": solv,
        "geometry_from": "best.xyz (step 7)",
    }


def _needs(job: str) -> tuple[str, ...]:
    return ("e_ref", "e_corr") if job.startswith("ref_") else ("final_energy",)


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write ORCA inputs for every unfinished reference/solvation job."""
    st = cfg["slurm"]["stage"]["reference"]
    tasks, waiting = [], 0
    for rec in project.molecules(cfg):
        if project.structure(cfg, rec) is None:
            waiting += 1
            continue
        mdir = project.mol_dir(cfg, rec)
        geometry = (mdir / "best.xyz").read_text(encoding="utf-8")
        for job, (keywords, blocks) in job_inputs(cfg).items():
            jdir = mdir / "reference" / job
            res = orca.parse_output(jdir / "output.out")
            status, _ = records.single_point_status(
                jdir, res, _needs(job), geometry
            )
            if status == "completed" and not force:
                continue
            if status == "stale":  # new geometry: a fresh submission budget
                (jdir / ".submissions").unlink(missing_ok=True)
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
            orca_run_script(jdir, f"{rec['compound']} {job}", cfg)
            tasks.append(jdir)
    script = write_array(cfg, "reference", tasks)
    return {
        "tasks": len(tasks),
        "waiting_for_step7": waiting,
        "script": str(script) if script else None,
    }


def cbs(
    cfg: Config, scf_x: float, corr_x: float, scf_y: float, corr_y: float
) -> tuple[float, float]:
    """Two-point CBS extrapolation of SCF and correlation energies."""
    r = cfg["reference"]
    x, y = r["cardinal_x"], r["cardinal_y"]
    a, b = r["scf_alpha"], r["corr_beta"]
    # E(n) = E_CBS + A exp(-a sqrt n)  ->  solve the two equations for E_CBS
    ex, ey = math.exp(a * math.sqrt(x)), math.exp(a * math.sqrt(y))
    scf = (scf_x * ex - scf_y * ey) / (ex - ey)
    corr = (x**b * corr_x - y**b * corr_y) / (x**b - y**b)
    return scf, corr


def standard_state_kcal(temperature: float) -> float:
    """1 atm -> 1 mol/L: RT ln(RT/p) with RT/p in L/mol."""
    return (
        constants.R_KCAL * temperature * math.log(0.0820573661 * temperature)
    )


def collect(cfg: Config) -> dict:
    """Write record.toml per finished job and reference.toml per molecule."""
    r = cfg["reference"]
    temp = r["temperature_k"]
    n_done, missing = 0, []
    for rec in project.molecules(cfg):
        struct = project.structure(cfg, rec)
        if struct is None:
            missing.append(rec["compound"])
            continue
        mdir = project.mol_dir(cfg, rec)
        geometry = (mdir / "best.xyz").read_text(encoding="utf-8")
        res, state = {}, {}
        for job in job_inputs(cfg):
            jdir = mdir / "reference" / job
            res[job] = orca.parse_output(jdir / "output.out")
            status, reason = records.single_point_status(
                jdir, res[job], _needs(job), geometry
            )
            state[job] = status
            if status != "pending":
                records.write(
                    jdir,
                    {
                        **_calculation(cfg, rec, job),
                        "status": status,
                        "reason": reason,
                    },
                    res[job],
                )
            if status != "completed":
                res[job] = {}  # never use failed or stale numbers
        ref_ok = all(state[j] == "completed" for j in ("ref_x", "ref_y"))
        solv_data = {}
        for label in methods.solvation_models(cfg.methods):
            dg = solvation.dg_solv(cfg, label, res)
            if dg is not None:
                solv_data[label] = {"dg_solv_kcal_mol": dg}
        if not ref_ok and not solv_data:
            missing.append(rec["compound"])
            continue
        data = {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "pipeline": pipeline_commit(),
            "compound": rec["compound"],
            "name": rec["name"],
            "role": rec["role"],
            "geometry": struct["best"]["geometry"],
            "isomer": struct["best"]["label"],
            "inchikey": struct["best"]["inchikey"],
            "geometry_sha256": records.geometry_sha256(geometry),
            "g_ensemble_correction_kcal_mol": struct.get(
                "g_ensemble_correction_kcal_mol", 0.0
            ),
            "temperature_k": temp,
            "thermo": {
                "g_corr_hartree": struct["best"]["g_corr_hartree"],
                "source": "step 6 r2SCAN-3c/CPCM freq (reused)",
            },
            "standard_state_kcal_mol": (
                standard_state_kcal(temp)
                if r.get("standard_state_correction", True)
                else 0.0
            ),
            "solvation": {
                "level": r["solvation_keywords"],
                "e_gas_hartree": res["solv_gas"].get("final_energy"),
                **solv_data,
            },
        }
        if ref_ok:
            needed = [
                res["ref_x"]["e_ref"],
                res["ref_x"]["e_corr"],
                res["ref_y"]["e_ref"],
                res["ref_y"]["e_corr"],
            ]
            scf_cbs, corr_cbs = cbs(cfg, *needed)
            data["reference"] = {
                "label": constants.REFERENCE_LABEL,
                "scf_x_hartree": needed[0],
                "corr_x_hartree": needed[1],
                "scf_y_hartree": needed[2],
                "corr_y_hartree": needed[3],
                "scf_cbs_hartree": scf_cbs,
                "corr_cbs_hartree": corr_cbs,
                "e_cbs_hartree": scf_cbs + corr_cbs,
                "t1_x": res["ref_x"].get("t1"),
                "t1_y": res["ref_y"].get("t1"),
            }
        write_toml(mdir / "reference" / "reference.toml", data)
        n_done += ref_ok
    return {"species_with_reference": n_done, "incomplete": len(missing)}
