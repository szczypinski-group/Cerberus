"""One-off ORCA input + run.sh + job.slurm for any step, from any geometry.

Uses exactly the keywords, blocks, %pal and %maxcore of the pipeline stage, so
a pilot calculation is identical to what the full run will do.

    predcc orca-input --step freq --molecule RISHSDLMXPLIHT
    predcc orca-input --step ref_x --xyz my.xyz --out pilot/ref_x
    predcc orca-input --step sp_wb97mv_def2tzvp --molecule RISHSDLMXPLIHT
    predcc orca-input --step custom --keywords "r2SCAN-3c SP" --xyz my.xyz
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import benchmark, orca, reference
from predcc_dft.io_utils import Config, format_xyz, read_toml, read_xyz_frames
from predcc_dft.slurm import orca_run_script, write_single

FIXED_STEPS = {
    "dft_opt": ("dft_opt", "step 5 ORCA route: r2SCAN-3c/CPCM optimisation"),
    "freq": ("frequencies", "step 6: TightOpt + Freq"),
    "freq_retry": (
        "frequencies",
        "step 6 retry: VeryTightOpt + Freq, exact Hessian",
    ),
    "custom": ("orca_input", "your own keywords (--keywords, --blocks)"),
}


def steps(cfg: Config) -> dict[str, tuple[str, str]]:
    """Step name -> (SLURM resource stage, description)."""
    out = dict(FIXED_STEPS)
    for job in reference.job_inputs(cfg):
        out[job] = ("reference", "step 8/10: reference or solvation SP")
    for label in cfg.methods.get("methods", {}):
        out[f"sp_{label}"] = ("benchmark", f"step 9: {label} single point")
    return out


def keywords_for(
    cfg: Config,
    step: str,
    keywords: str | None = None,
    blocks: str | None = None,
) -> tuple[str, str]:
    """Keywords and blocks of a step, with optional overrides."""
    ref_jobs = reference.job_inputs(cfg)
    if step == "dft_opt":
        kw, bl = cfg["dft_opt"]["keywords"], cfg["dft_opt"].get("blocks", "")
    elif step in ("freq", "freq_retry"):
        f = cfg["frequencies"]
        kw = f["retry_keywords"] if step == "freq_retry" else f["keywords"]
        bl = f"%freq\n  Temp {f['temperature_k']}\nend"
        if step == "freq_retry" and f.get("retry_blocks"):
            bl += "\n" + f["retry_blocks"]
    elif step in ref_jobs:
        kw, bl = ref_jobs[step]
    elif step.startswith("sp_") and step[3:] in cfg.methods.get("methods", {}):
        kw, bl = benchmark.inputs_for(cfg, step[3:])
    elif step == "custom":
        if not keywords:
            raise ValueError("--step custom needs --keywords")
        kw, bl = keywords, blocks or ""
    else:
        raise ValueError(
            f"unknown step {step}; choose from {', '.join(steps(cfg))}"
        )
    if keywords and step != "custom":
        kw = keywords
    if blocks is not None and step != "custom":
        bl = blocks
    return kw, bl


def _molecule(cfg: Config, key: str) -> tuple[dict, Path]:
    data = read_toml(cfg.molecules_file)
    for role in ("aldehyde", "amine", "imine", "water"):
        for rec in data.get(role, []):
            if rec["compound"] == key:
                mdir = cfg.root / rec["directory"]
                geom = mdir / "best.xyz"
                if not geom.exists():
                    geom = cfg.root / rec["isomers"][0]["seeds"][0]["xyz"]
                return rec, geom
    raise KeyError(f"compound {key} not in molecules.toml")


def make(
    cfg: Config,
    step: str,
    out: Path | None = None,
    xyz: Path | None = None,
    molecule: str | None = None,
    frame: int = -1,
    charge: int | None = None,
    multiplicity: int | None = None,
    keywords: str | None = None,
    blocks: str | None = None,
) -> dict:
    """Write input.inp, geom.xyz, run.sh and job.slurm for one calculation."""
    all_steps = steps(cfg)
    if step not in all_steps:
        raise ValueError(
            f"unknown step {step}; choose from {', '.join(all_steps)}"
        )
    stage = all_steps[step][0]
    if molecule:
        rec, geom_path = _molecule(cfg, molecule)
        charge = rec["charge"] if charge is None else charge
        multiplicity = (
            rec["multiplicity"] if multiplicity is None else multiplicity
        )
        out = out or (cfg.root / rec["directory"] / "manual" / step)
    elif xyz:
        geom_path = xyz
        out = out or (xyz.parent / f"{xyz.stem}_{step}")
    else:
        raise ValueError("give --xyz or --molecule")
    charge = 0 if charge is None else charge
    multiplicity = 1 if multiplicity is None else multiplicity

    comment, atoms = read_xyz_frames(geom_path)[frame]
    kw, bl = keywords_for(cfg, step, keywords, blocks)
    st = cfg["slurm"]["stage"][stage]
    out = out if out.is_absolute() else (Path.cwd() / out)
    orca.write_input(
        out,
        kw,
        st["cpus"],
        int(st["mem_per_cpu_mb"] * 0.75),
        format_xyz(atoms, comment or geom_path.name),
        charge,
        multiplicity,
        bl,
    )
    name = molecule or geom_path.stem
    orca_run_script(out, f"{name} {step}", cfg)
    job = write_single(cfg, stage, out, f"{name}_{step}")
    return {
        "directory": str(out),
        "input": str(out / "input.inp"),
        "slurm": str(job),
        "keywords": kw,
        "charge": charge,
        "multiplicity": multiplicity,
        "resources": (
            f"{st['cpus']} cores, {st['mem_per_cpu_mb']} MB/core, {st['time']}"
        ),
    }
