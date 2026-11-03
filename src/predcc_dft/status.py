"""Per-molecule progress through steps 3-9 (derived/setup_status.csv)."""

from __future__ import annotations

import csv

from predcc_dft import benchmark, conformers, frequencies
from predcc_dft.io_utils import Config, read_toml

COLUMNS = [
    "compound",
    "role",
    "name",
    "seeds",
    "steps4_5",
    "freq",
    "structure",
    "best",
    "reference",
    "benchmark",
]


def _step5(cfg: Config, rec: dict) -> str:
    cdir = conformers.mol_dir(cfg, rec) / "conformers"
    if conformers.step5_done(cfg, rec):
        return "done"
    if (cdir / "FAILED").exists():
        return "failed"
    if conformers.engine(cfg) == "orca" and conformers.crest_done(cfg, rec):
        return "crest done"
    if (cdir / "run.sh").exists():
        return "written"
    return "-"


def run(cfg: Config) -> list[dict]:
    """One row per molecule; also writes derived/setup_status.csv."""
    if not cfg.molecules_file.exists():
        return []
    n_methods = len(benchmark.methods(cfg))
    rows = []
    for rec in conformers.molecules(cfg):
        mdir = conformers.mol_dir(cfg, rec)
        counts: dict[str, int] = {}
        for r in frequencies.selected_results(cfg, rec, write=False):
            s = r["record"]["calculation"]["status"]
            counts[s] = counts.get(s, 0) + 1
        struct_path = mdir / "structure.toml"
        struct = read_toml(struct_path) if struct_path.exists() else {}
        ref = mdir / "reference" / "reference.toml"
        ref_data = read_toml(ref) if ref.exists() else {}
        rows.append(
            {
                "compound": rec["compound"],
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
                "reference": "done" if "reference" in ref_data else "-",
                "benchmark": (
                    f"{len(benchmark.energies(cfg, rec))}/{n_methods}"
                ),
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
    """Counts of molecules past each step."""
    rows = run(cfg)
    n_methods = len(benchmark.methods(cfg))
    return {
        "molecules": len(rows),
        "conformers done (4-5)": sum(r["steps4_5"] == "done" for r in rows),
        "best geometry (6-7)": sum(
            r["structure"] == "collapsed" for r in rows
        ),
        "structure failed (7)": sum(r["structure"] == "failed" for r in rows),
        "DLPNO reference (8)": sum(r["reference"] == "done" for r in rows),
        "all benchmark SPs (9)": sum(
            r["benchmark"] == f"{n_methods}/{n_methods}" for r in rows
        ),
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
