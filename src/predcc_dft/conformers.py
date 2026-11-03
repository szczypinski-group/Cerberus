"""Steps 4-5: CREST per seed, then CENSO on the pooled ensemble.

CREST (GFN2-xTB, ALPB acetonitrile) runs on every seed; CENSO then optimises
and re-ranks the pooled ensemble of all E/Z isomers and rotamers of a molecule
at r2SCAN-3c/CPCM. One SLURM array task per molecule.

Output per molecule: molecules/<key>/conformers/
    crest_<label>/crest_conformers.xyz   CREST ensemble per seed
    ensemble_input.xyz                   concatenation, in seed order
    ensemble_index.toml                  input frame -> seed
    censo/2_OPTIMIZATION.xyz|.json       DFT-optimised, ranked ensemble
"""

from __future__ import annotations

import json
from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.io_utils import (
    Config,
    now,
    read_toml,
    read_xyz_frames,
    write_text,
    write_toml,
)
from predcc_dft.slurm import write_array

CENSO_RESULT = "2_OPTIMIZATION"


def molecules(cfg: Config) -> list[dict]:
    """All molecule records from molecules.toml, in a stable order."""
    data = read_toml(cfg.molecules_file)
    out = []
    for role in ("aldehyde", "amine", "imine", "water"):
        out += data.get(role, [])
    return out


def mol_dir(cfg: Config, rec: dict) -> Path:
    """Folder of one molecule."""
    return cfg.root / rec["directory"]


def censo_done(cfg: Config, rec: dict) -> bool:
    """True once CENSO wrote its optimisation results."""
    d = mol_dir(cfg, rec) / "conformers" / "censo"
    return (d / f"{CENSO_RESULT}.json").exists() and (
        d / f"{CENSO_RESULT}.xyz"
    ).exists()


def _run_script(
    cfg: Config, rec: dict, labels: list[str], n_atoms: int
) -> str:
    c = cfg["conformers"]
    rc = cfg.data_file(c["censo_rc"]).resolve()
    parts = " ".join(f"--{p}" for p in c["censo_parts"])
    skip_crest = n_atoms <= int(c["skip_crest_max_atoms"])
    lines = [
        "#!/bin/bash",
        f"# Steps 4-5 for {rec['name']} ({rec['compound']})",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$HERE"',
        'NPROC="${SLURM_CPUS_PER_TASK:-1}"',
        "rm -f ensemble_input.xyz DONE FAILED",
        "trap 'touch FAILED' ERR",
    ]
    for label in labels:
        seed = f"../seeds/{label}.xyz"
        lines.append(f"mkdir -p crest_{label}")
        if skip_crest:
            lines.append(
                f"cp {seed} crest_{label}/crest_conformers.xyz"
                "  # too small for CREST"
            )
        else:
            lines.append(
                f"(cd crest_{label} && [ -s crest_conformers.xyz ] || "
                f"crest ../{seed} --gfn2 --alpb {c['solvent']} "
                f"--ewin {c['crest_ewin_kcal_mol']} "
                f"--chrg {rec['charge']} --uhf {rec['multiplicity'] - 1} "
                '-T "$NPROC" > crest.out 2>&1)'
            )
        lines.append(
            f"cat crest_{label}/crest_conformers.xyz >> ensemble_input.xyz"
        )
    if engine(cfg) == "censo":
        lines += [
            "mkdir -p censo && cp ensemble_input.xyz censo/",
            "cd censo",
            f"censo -i ensemble_input.xyz --inprc {rc} "
            f"--charge {rec['charge']} "
            f"--unpaired {rec['multiplicity'] - 1} --solvent {c['solvent']} "
            f'--maxcores "$NPROC" {parts} > censo.out 2>&1',
            "cd ..",
        ]
    else:
        lines.append("# step 5 runs as ORCA jobs (stage dft_opt)")
    lines += ["touch DONE", ""]
    return "\n".join(lines)


def engine(cfg: Config) -> str:
    """Step 5 engine: "censo" or "orca"."""
    eng = cfg["conformers"].get("dft_engine", "censo").lower()
    if eng not in ("censo", "orca"):
        raise ValueError(
            f"conformers.dft_engine must be 'censo' or 'orca', not {eng!r}"
        )
    return eng


def crest_done(cfg: Config, rec: dict) -> bool:
    """True once CREST finished for every seed."""
    cdir = mol_dir(cfg, rec) / "conformers"
    labels = [s["label"] for iso in rec["isomers"] for s in iso["seeds"]]
    return (cdir / "DONE").exists() and all(
        (cdir / f"crest_{label}" / "crest_conformers.xyz").exists()
        for label in labels
    )


def step5_done(cfg: Config, rec: dict) -> bool:
    """True once step 5 finished with the configured engine."""
    if engine(cfg) == "censo":
        return censo_done(cfg, rec)
    from predcc_dft import dft_opt  # local import: dft_opt imports this module

    return dft_opt.all_done(cfg, rec)


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write the CREST (+ CENSO) task of every unfinished molecule."""
    tasks = []
    for rec in molecules(cfg):
        done = (
            censo_done(cfg, rec)
            if engine(cfg) == "censo"
            else crest_done(cfg, rec)
        )
        if done and not force:
            continue
        labels = [s["label"] for iso in rec["isomers"] for s in iso["seeds"]]
        seed_xyz = mol_dir(cfg, rec) / "seeds" / f"{labels[0]}.xyz"
        n_atoms = len(read_xyz_frames(seed_xyz)[0][1])
        cdir = mol_dir(cfg, rec) / "conformers"
        write_text(
            cdir / "run.sh",
            _run_script(cfg, rec, labels, n_atoms),
            executable=True,
        )
        tasks.append(cdir)
    script = write_array(cfg, "conformers", tasks)
    return {
        "tasks": len(tasks),
        "engine": engine(cfg),
        "script": str(script) if script else None,
    }


def seed_index(cfg: Config, rec: dict) -> dict[int, str]:
    """Pooled-ensemble frame number (1-based, CONF<n>) -> seed label."""
    cdir = mol_dir(cfg, rec) / "conformers"
    seed_of: dict[int, str] = {}
    i = 1
    for label in [s["label"] for iso in rec["isomers"] for s in iso["seeds"]]:
        f = cdir / f"crest_{label}" / "crest_conformers.xyz"
        for _ in range(len(read_xyz_frames(f)) if f.exists() else 0):
            seed_of[i] = label
            i += 1
    return seed_of


def collect(cfg: Config, rec: dict) -> dict | None:
    """Step 5 results for step 6, sorted by gtot.

    Returns {"conformers": [{name, gtot, energy, atoms, seed}]}.

    gtot is CENSO's G (censo engine) or the ORCA optimised energy (orca).
    """
    if engine(cfg) == "orca":
        from predcc_dft import dft_opt

        return dft_opt.collect(cfg, rec)
    cdir = mol_dir(cfg, rec) / "conformers"
    if not censo_done(cfg, rec):
        return None
    data = json.loads(
        (cdir / "censo" / f"{CENSO_RESULT}.json").read_text(encoding="utf-8")
    )
    per_conf = data.get("data", data)
    frames = {
        comment.split()[0]: atoms
        for comment, atoms in read_xyz_frames(
            cdir / "censo" / f"{CENSO_RESULT}.xyz"
        )
    }

    seed_of = seed_index(cfg, rec)
    write_toml(
        cdir / "ensemble_index.toml",
        {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "frames": [
                {"conformer": f"CONF{k}", "seed": v}
                for k, v in seed_of.items()
            ],
        },
    )

    confs = []
    for name, vals in per_conf.items():
        if name not in frames or vals.get("gtot") is None:
            continue
        num = (
            int(name.replace("CONF", "")) if name.startswith("CONF") else None
        )
        confs.append(
            {
                "name": name,
                "gtot": vals["gtot"],
                "energy": vals.get("energy"),
                "atoms": frames[name],
                "seed": seed_of.get(num, "") if num is not None else "",
            }
        )
    return {"conformers": sorted(confs, key=lambda c: c["gtot"])}
