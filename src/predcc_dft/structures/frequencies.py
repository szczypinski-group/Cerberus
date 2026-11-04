"""Step 6: ORCA TightOpt + Freq at r2SCAN-3c/CPCM(acetonitrile).

Runs for every step-5 conformer within the energy window. A conformer passes
only with a normally terminated, converged optimisation and no imaginary
frequency. Failures get a second attempt (VeryTightOpt, exact Hessian) from
the last geometry. The thermal correction of the conformer finally selected
in step 7 is reused for every benchmarked functional/basis pair.

molecules/<key>/frequencies/<CONF>/attempt<k>/
    input.inp, geom.xyz, run.sh, output.out, final.xyz, record.toml
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends import orca
from predcc_dft.backends.slurm import orca_run_script, write_array
from predcc_dft.protocol import constants, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import format_xyz, now, read_toml, write_toml
from predcc_dft.structures import conformers


def _maxcore(cfg: Config, stage: str) -> int:
    return int(cfg["slurm"]["stage"][stage]["mem_per_cpu_mb"] * 0.75)


def _attempts(conf_dir: Path) -> list[Path]:
    return sorted(
        conf_dir.glob("attempt*"),
        key=lambda p: int(p.name.replace("attempt", "")),
    )


def evaluate(
    cfg: Config,
    attempt_dir: Path,
    compound: str,
    conf_name: str,
    write: bool = True,
) -> dict:
    """Parse one attempt and write its record.toml. Returns the record."""
    f = cfg["frequencies"]
    res = orca.parse_output(attempt_dir / "output.out")
    status, reason = "completed", ""
    if not res.get("exists"):
        status, reason = "pending", "no output yet"
    elif not res.get("terminated_normally"):
        status, reason = "failed", "ORCA did not terminate normally"
    elif not res.get("opt_converged"):
        status, reason = "failed", "optimisation not converged"
    elif res.get("n_imaginary") is None:
        status, reason = "failed", "no frequencies found"
    elif res["lowest_freq_cm1"] is not None and res["lowest_freq_cm1"] < -abs(
        f["imaginary_tolerance_cm1"]
    ):
        status, reason = (
            "rejected",
            f"{res['n_imaginary']} imaginary frequency(ies), "
            f"lowest {res['lowest_freq_cm1']:.1f} cm-1",
        )
    elif not (attempt_dir / "final.xyz").exists():
        status, reason = "failed", "final geometry missing"

    calculation = {
        "id": records.calc_id(
            compound, conf_name, "optfreq", "r2scan3c", "cpcm_mecn"
        ),
        "compound": compound,
        "conformer": conf_name,
        "step": "optfreq",
        "method": "r2scan3c",
        "solvation": "cpcm_mecn",
        "attempt": int(attempt_dir.name.replace("attempt", "")),
        "status": status,
        "reason": reason,
        "geometry_from": (
            f"step5:{cfg['conformers'].get('dft_engine', 'censo')}"
            if attempt_dir.name == "attempt1"
            else "previous attempt"
        ),
    }
    extra = {}
    if status in ("completed", "rejected"):
        extra["thermo"] = {
            "temperature_k": res.get("temperature"),
            "n_imaginary": res.get("n_imaginary"),
            "lowest_freq_cm1": res.get("lowest_freq_cm1"),
            "zpe_hartree": res.get("zpe"),
            "g_corr_hartree": res.get("g_minus_e"),
            "enthalpy_hartree": res.get("enthalpy"),
            "gibbs_hartree": res.get("gibbs"),
        }
    if status == "pending" or not write:
        return {"calculation": calculation, **extra}
    return records.write(attempt_dir, calculation, res, extra)


def latest(
    cfg: Config, conf_dir: Path, compound: str, write: bool = True
) -> tuple[Path, dict] | None:
    """Last attempt folder of a conformer and its (re-)evaluated record."""
    att = _attempts(conf_dir)
    if not att:
        return None
    return att[-1], evaluate(cfg, att[-1], compound, conf_dir.name, write)


def _write_attempt(
    cfg: Config, attempt_dir: Path, geometry: str, retry: bool, label: str
) -> None:
    f, st = cfg["frequencies"], cfg["slurm"]["stage"]["frequencies"]
    blocks = f"%freq\n  Temp {f['temperature_k']}\nend"
    if retry and f.get("retry_blocks"):
        blocks += "\n" + f["retry_blocks"]
    orca.write_input(
        attempt_dir,
        f["retry_keywords"] if retry else f["keywords"],
        st["cpus"],
        _maxcore(cfg, "frequencies"),
        geometry,
        blocks=blocks,
    )
    orca_run_script(attempt_dir, label, cfg)


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write Opt+Freq tasks (and retries) for every step-5 result."""
    f = cfg["frequencies"]
    tasks, waiting, retried, exhausted, no_conformers = [], 0, 0, 0, 0
    for rec in project.molecules(cfg):
        result = conformers.collect(cfg, rec)
        if result is None:
            waiting += 1
            continue
        confs = result["conformers"]
        if not confs:
            no_conformers += 1
            continue
        g0 = confs[0]["gtot"]
        selected = [
            c
            for c in confs
            if (c["gtot"] - g0) * constants.HARTREE_TO_KCAL
            <= f["window_kcal_mol"]
        ][: f["max_conformers"]]
        fdir = project.mol_dir(cfg, rec) / "frequencies"
        for conf in selected:
            cdir = fdir / conf["name"]
            label = f"{rec['compound']} {conf['name']}"
            state = latest(cfg, cdir, rec["compound"])
            if state is None or force:
                a1 = cdir / "attempt1"
                if force or not (a1 / "output.out").exists():
                    _write_attempt(
                        cfg, a1, format_xyz(conf["atoms"], label), False, label
                    )
                    tasks.append(a1)
                continue
            adir, record = state
            status = record["calculation"]["status"]
            if status == "pending":
                tasks.append(adir)  # written but never ran: resubmit
            elif status in ("failed", "rejected"):
                n = record["calculation"]["attempt"]
                if n >= f["max_attempts"]:
                    exhausted += 1
                    continue
                geom_src = (
                    adir / "final.xyz"
                    if (adir / "final.xyz").exists()
                    else adir / "geom.xyz"
                )
                nxt = cdir / f"attempt{n + 1}"
                _write_attempt(
                    cfg, nxt, geom_src.read_text(encoding="utf-8"), True, label
                )
                tasks.append(nxt)
                retried += 1
        write_toml(
            fdir / "selection.toml",
            {
                "schema_version": SCHEMA_VERSION,
                "generated": now(),
                "window_kcal_mol": f["window_kcal_mol"],
                "max_conformers": f["max_conformers"],
                "conformer": [
                    {
                        "name": c["name"],
                        "seed": c["seed"],
                        "step5_rel_kcal_mol": round(
                            (c["gtot"] - g0) * constants.HARTREE_TO_KCAL, 3
                        ),
                    }
                    for c in selected
                ],
            },
        )
    script = write_array(cfg, "frequencies", tasks)
    return {
        "tasks": len(tasks),
        "retries": retried,
        "exhausted": exhausted,
        "waiting_for_step5": waiting,
        "step5_no_converged_conformer": no_conformers,
        "script": str(script) if script else None,
    }


def selected_results(cfg: Config, rec: dict, write: bool = True) -> list[dict]:
    """Latest attempt of every selected conformer: [{name, dir, record}]."""
    fdir = project.mol_dir(cfg, rec) / "frequencies"
    sel = fdir / "selection.toml"
    if not sel.exists():
        return []
    out = []
    for c in read_toml(sel).get("conformer", []):
        state = latest(cfg, fdir / c["name"], rec["compound"], write)
        if state is not None:
            out.append(
                {
                    "name": c["name"],
                    "seed": c["seed"],
                    "dir": state[0],
                    "record": state[1],
                }
            )
    return out
