"""Per-molecule progress: trunk steps 3-7 and both branches.

Written to derived/setup_status.csv. A branch column shows "-" for molecules
outside that branch's selection (<branch>/molecules.txt).
"""

from __future__ import annotations

import csv

from predcc_dft.benchmark import grid
from predcc_dft.benchmark import reference as bench_reference
from predcc_dft.production import stage as production_stage
from predcc_dft.protocol import lock, project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml
from predcc_dft.structures import conformers, frequencies

COLUMNS = [
    "compound",
    "role",
    "name",
    "seeds",
    "steps4_5",
    "freq",
    "structure",
    "best",
    "bench_reference",
    "bench_methods",
    "production",
]


def _step5(cfg: Config, rec: dict) -> str:
    cdir = project.mol_dir(cfg, rec) / "conformers"
    if conformers.step5_done(cfg, rec):
        return "done"
    if (cdir / "FAILED").exists():
        return "failed"
    if conformers.engine(cfg) == "orca" and conformers.crest_done(cfg, rec):
        return "crest done"
    if (cdir / "run.sh").exists():
        return "written"
    return "-"


def _lock_or_none(cfg: Config) -> dict | None:
    try:
        return lock.read(cfg)
    except lock.LockError:
        return None


def run(cfg: Config) -> list[dict]:
    """One row per trunk molecule; also writes derived/setup_status.csv."""
    if not cfg.molecules_file.exists():
        return []
    n_methods = len(grid.methods(cfg))
    in_bench = {r["compound"] for r in project.selected(cfg, "benchmark")}
    in_prod = {r["compound"] for r in project.selected(cfg, "production")}
    the_lock = _lock_or_none(cfg)
    rows = []
    for rec in project.molecules(cfg):
        mdir = project.mol_dir(cfg, rec)
        counts: dict[str, int] = {}
        for r in frequencies.selected_results(cfg, rec, write=False):
            s = r["record"]["calculation"]["status"]
            counts[s] = counts.get(s, 0) + 1
        struct_path = mdir / "structure.toml"
        struct = read_toml(struct_path) if struct_path.exists() else {}
        key = rec["compound"]
        if key in in_bench:
            ref = bench_reference.data(cfg, rec) or {}
            bench_ref = "done" if "reference" in ref else "-"
            bench_methods = f"{len(grid.energies(cfg, rec))}/{n_methods}"
        else:
            bench_ref = bench_methods = "-"
        if key not in in_prod:
            prod = "-"
        elif the_lock is None:
            prod = "no lock"
        else:
            res = production_stage.result(cfg, the_lock, rec) or {}
            prod = (
                f"v{the_lock['production_version']} "
                f"{res.get('status', 'pending')}"
            )
        rows.append(
            {
                "compound": key,
                "role": rec["role"],
                "name": rec["name"],
                "seeds": sum(len(i["seeds"]) for i in rec["isomers"]),
                "steps4_5": _step5(cfg, rec),
                "freq": ", ".join(
                    f"{k} {v}" for k, v in sorted(counts.items())
                )
                or "-",
                "structure": struct.get("status", "-"),
                "best": struct.get("best", {}).get("label", ""),
                "bench_reference": bench_ref,
                "bench_methods": bench_methods,
                "production": prod,
            }
        )
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.derived_dir / "setup_status.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def summary(cfg: Config) -> dict:
    """Counts of molecules past each step, trunk and branches."""
    rows = run(cfg)
    n_methods = len(grid.methods(cfg))
    bench = [r for r in rows if r["bench_methods"] != "-"]
    prod = [r for r in rows if r["production"] != "-"]
    return {
        "molecules": len(rows),
        "conformers done (4-5)": sum(r["steps4_5"] == "done" for r in rows),
        "best geometry (6-7)": sum(
            r["structure"] == "collapsed" for r in rows
        ),
        "structure failed (7)": sum(r["structure"] == "failed" for r in rows),
        "benchmark": {
            "molecules": len(bench),
            "DLPNO reference": sum(
                r["bench_reference"] == "done" for r in bench
            ),
            "all methods": sum(
                r["bench_methods"] == f"{n_methods}/{n_methods}" for r in bench
            ),
        },
        "production": {
            "molecules": len(prod),
            "complete": sum(
                r["production"].endswith(" complete") for r in prod
            ),
        },
    }


def table(rows: list[dict]) -> str:
    """Plain-text table of status rows."""
    cols = [c for c in COLUMNS if c != "name"]
    if not rows:
        return "no molecules yet"
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    lines += [
        "  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows
    ]
    return "\n".join(lines)
