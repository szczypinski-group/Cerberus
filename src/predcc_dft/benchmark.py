"""Step 9: benchmark single points of the method matrix (SOP 5.1 step 9, 6.1).

Every method in methods.toml is run as a gas-phase single point on each
molecule's best r2SCAN-3c/CPCM geometry (step 7). Descriptors that come free
with the output (HOMO/LUMO) are stored in the record.

molecules/<key>/benchmark/<method>/{input.inp, geom.xyz, run.sh, job.slurm,
                                    output.out, record.toml}
"""

from __future__ import annotations

from predcc_dft import (
    conformers,
    method_matrix,
    orca,
    records,
    reference,
)
from predcc_dft.io_utils import Config
from predcc_dft.slurm import orca_run_script, write_array


def methods(cfg: Config) -> dict[str, dict]:
    """Methods to benchmark.

    [benchmark] levels, functionals and methods select a subset (``predcc
    level`` / ``predcc functional`` set them); all empty = every method.
    """
    table = cfg.methods.get("methods", {})
    b = cfg["benchmark"]
    wanted = list(b.get("methods") or [])
    for level in b.get("levels") or []:
        wanted += [
            m
            for m in method_matrix.level_methods(cfg.methods, level)
            if m not in wanted
        ]
    for functional in b.get("functionals") or []:
        wanted += [
            m
            for m in method_matrix.functional_methods(cfg.methods, functional)
            if m not in wanted
        ]
    wanted = wanted or list(table)
    unknown = [m for m in wanted if m not in table]
    if unknown:
        raise KeyError(f"methods not in methods.toml: {', '.join(unknown)}")
    return {m: table[m] for m in wanted}


def inputs_for(cfg: Config, label: str) -> tuple[str, str]:
    """ORCA keywords and blocks of one benchmark method."""
    m = cfg.methods["methods"][label]
    keywords = f"{m['orca_keywords']} {cfg.methods.get('common_keywords', '')}"
    blocks = "\n".join(
        b.strip()
        for b in (
            m.get("orca_blocks", ""),
            cfg.methods.get("common_blocks", ""),
        )
        if b and b.strip()
    )
    return keywords.strip(), blocks


def _calculation(rec: dict, label: str) -> dict:
    return {
        "id": records.calc_id(rec["compound"], "best", "sp", label, "gas"),
        "compound": rec["compound"],
        "conformer": "best",
        "step": "sp",
        "method": label,
        "solvation": "gas",
        "geometry_from": "best.xyz (step 7)",
    }


def write_jobs(cfg: Config, force: bool = False) -> dict:
    """Write one ORCA single point per molecule x method still to run."""
    st = cfg["slurm"]["stage"]["benchmark"]
    tasks, waiting = [], 0
    labels = list(methods(cfg))
    for rec in conformers.molecules(cfg):
        if reference.structure(cfg, rec) is None:
            waiting += 1
            continue
        mdir = conformers.mol_dir(cfg, rec)
        geometry = (mdir / "best.xyz").read_text()
        for label in labels:
            jdir = mdir / "benchmark" / label
            res = orca.parse_output(jdir / "output.out")
            status, _ = records.single_point_status(
                jdir, res, ("final_energy",), geometry
            )
            if status == "completed" and not force:
                continue
            if status == "stale":  # new geometry: a fresh submission budget
                (jdir / ".submissions").unlink(missing_ok=True)
            keywords, blocks = inputs_for(cfg, label)
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
            orca_run_script(jdir, f"{rec['compound']} {label}", cfg)
            tasks.append(jdir)
    script = write_array(cfg, "benchmark", tasks)
    return {
        "tasks": len(tasks),
        "methods": len(labels),
        "waiting_for_step7": waiting,
        "script": str(script) if script else None,
    }


def collect(cfg: Config) -> dict:
    """Write record.toml for every finished benchmark single point.

    Covers every method folder on disk, not only the current selection, so
    results of an earlier `predcc level` run are never left uncollected.
    """
    counts = {"completed": 0, "failed": 0, "stale": 0, "pending": 0}
    for rec in conformers.molecules(cfg):
        mdir = conformers.mol_dir(cfg, rec)
        best = mdir / "best.xyz"
        if reference.structure(cfg, rec) is None or not best.exists():
            continue
        geometry = best.read_text()
        for inp in sorted((mdir / "benchmark").glob("*/input.inp")):
            jdir, label = inp.parent, inp.parent.name
            res = orca.parse_output(jdir / "output.out")
            status, reason = records.single_point_status(
                jdir, res, ("final_energy",), geometry
            )
            counts[status] += 1
            if status != "pending":
                records.write(
                    jdir,
                    {
                        **_calculation(rec, label),
                        "status": status,
                        "reason": reason,
                    },
                    res,
                )
    return counts


def energies(cfg: Config, rec: dict) -> dict[str, float]:
    """Completed benchmark electronic energies (Eh) of one molecule."""
    from predcc_dft.io_utils import read_toml

    out = {}
    mdir = conformers.mol_dir(cfg, rec)
    for label in cfg.methods.get("methods", {}):
        path = mdir / "benchmark" / label / "record.toml"
        if not path.exists():
            continue
        data = read_toml(path)
        if data["calculation"]["status"] == "completed":
            out[label] = data["energy"]["electronic_hartree"]
    return out
