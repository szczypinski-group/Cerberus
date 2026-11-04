"""Step 5, ORCA route (conformers.dft_engine = "orca").

Instead of CENSO, every pooled CREST conformer within the xTB window (up to
max_conformers) is optimised with a plain ORCA input at
r2SCAN-3c/CPCM(acetonitrile), then re-ranked by the DFT energy. Each conformer
gets input.inp, geom.xyz, run.sh and a standalone job.slurm; all are also
listed in jobs/dft_opt/dft_opt.slurm.

molecules/<key>/conformers/dft/CONF<n>/{input.inp, geom.xyz, run.sh, job.slurm,
                                        output.out, final.xyz}
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends import orca
from predcc_dft.backends.slurm import orca_run_script, write_array
from predcc_dft.protocol import constants, project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import (
    format_xyz,
    now,
    read_toml,
    read_xyz_frames,
    write_toml,
)
from predcc_dft.structures import conformers


def _pooled(cfg: Config, rec: dict) -> list[dict]:
    """Pooled CREST frames in seed order: [{name, atoms, xtb_energy, seed}]."""
    cdir = project.mol_dir(cfg, rec) / "conformers"
    frames, n = [], 1
    for label in [s["label"] for iso in rec["isomers"] for s in iso["seeds"]]:
        f = cdir / f"crest_{label}" / "crest_conformers.xyz"
        for comment, atoms in read_xyz_frames(f) if f.exists() else []:
            try:
                energy = float(comment.split()[0])
            except (ValueError, IndexError):
                energy = None
            frames.append(
                {
                    "name": f"CONF{n}",
                    "atoms": atoms,
                    "xtb_energy": energy,
                    "seed": label,
                }
            )
            n += 1
    return frames


def _selection(cfg: Config, rec: dict) -> list[dict]:
    d = cfg["dft_opt"]
    frames = _pooled(cfg, rec)
    energies = [f["xtb_energy"] for f in frames if f["xtb_energy"] is not None]
    if energies:
        e0 = min(energies)
        frames = [
            f
            for f in frames
            if f["xtb_energy"] is None
            or (f["xtb_energy"] - e0) * constants.HARTREE_TO_KCAL
            <= d["window_kcal_mol"]
        ]
        frames.sort(
            key=lambda f: (f["xtb_energy"] is None, f["xtb_energy"] or 0.0)
        )
    return frames[: d["max_conformers"]]


def _dir(cfg: Config, rec: dict) -> Path:
    return project.mol_dir(cfg, rec) / "conformers" / "dft"


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write ORCA optimisations of the selected CREST conformers."""
    d, st = cfg["dft_opt"], cfg["slurm"]["stage"]["dft_opt"]
    tasks, waiting = [], 0
    for rec in project.molecules(cfg):
        if not conformers.crest_done(cfg, rec):
            waiting += 1
            continue
        selected = _selection(cfg, rec)
        ddir = _dir(cfg, rec)
        for frame in selected:
            jdir = ddir / frame["name"]
            res = orca.parse_output(jdir / "output.out")
            if res.get("terminated_normally") and not force:
                continue
            label = f"{rec['compound']} {frame['name']} {frame['seed']}"
            orca.write_input(
                jdir,
                d["keywords"],
                st["cpus"],
                int(st["mem_per_cpu_mb"] * 0.75),
                format_xyz(frame["atoms"], label),
                rec["charge"],
                rec["multiplicity"],
                d.get("blocks", ""),
            )
            orca_run_script(jdir, label, cfg)
            tasks.append(jdir)
        write_toml(
            ddir / "selection.toml",
            {
                "schema_version": SCHEMA_VERSION,
                "generated": now(),
                "window_kcal_mol": d["window_kcal_mol"],
                "max_conformers": d["max_conformers"],
                "conformer": [
                    {
                        "name": f["name"],
                        "seed": f["seed"],
                        "xtb_energy": f["xtb_energy"],
                    }
                    for f in selected
                ],
            },
        )
    script = write_array(cfg, "dft_opt", tasks)
    return {
        "tasks": len(tasks),
        "waiting_for_crest": waiting,
        "script": str(script) if script else None,
    }


def all_done(cfg: Config, rec: dict) -> bool:
    """True once every selected optimisation has an output."""
    sel = _dir(cfg, rec) / "selection.toml"
    if not sel.exists():
        return False
    names = [c["name"] for c in read_toml(sel).get("conformer", [])]
    return bool(names) and all(
        (_dir(cfg, rec) / n / "output.out").exists() for n in names
    )


def collect(cfg: Config, rec: dict) -> dict | None:
    """Optimised conformers within rank_window, sorted by DFT energy (gtot)."""
    if not all_done(cfg, rec):
        return None
    ddir = _dir(cfg, rec)
    confs = []
    for c in read_toml(ddir / "selection.toml")["conformer"]:
        jdir = ddir / c["name"]
        res = orca.parse_output(jdir / "output.out")
        if not (
            res.get("terminated_normally")
            and res.get("opt_converged")
            and "final_energy" in res
            and (jdir / "final.xyz").exists()
        ):
            continue
        confs.append(
            {
                "name": c["name"],
                "gtot": res["final_energy"],
                "energy": res["final_energy"],
                "atoms": read_xyz_frames(jdir / "final.xyz")[-1][1],
                "seed": c["seed"],
            }
        )
    if not confs:
        return {"conformers": []}
    confs.sort(key=lambda c: c["gtot"])
    e0 = confs[0]["gtot"]
    window = cfg["dft_opt"]["rank_window_kcal_mol"]
    return {
        "conformers": [
            c
            for c in confs
            if (c["gtot"] - e0) * constants.HARTREE_TO_KCAL <= window
        ]
    }
