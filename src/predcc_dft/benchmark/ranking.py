"""SOP 8: error metrics per method, linkage class and level of theory."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from predcc_dft.analysis import assembly
from predcc_dft.protocol.config import Config


def _stats(values: list[float], n_boot: int, rng: random.Random) -> dict:
    """Error metrics and a bootstrap 95 % CI of the MAE."""
    absval = [abs(v) for v in values]
    n = len(values)
    mae = statistics.fmean(absval)
    boots = (
        sorted(
            statistics.fmean(rng.choices(absval, k=n)) for _ in range(n_boot)
        )
        if n > 1 and n_boot
        else []
    )
    return {
        "n": n,
        "mae_kcal_mol": mae,
        "rmse_kcal_mol": math.sqrt(statistics.fmean(v * v for v in values)),
        "mse_kcal_mol": statistics.fmean(values),
        "sd_kcal_mol": statistics.stdev(values) if n > 1 else 0.0,
        "max_abs_kcal_mol": max(absval),
        "mae_ci_low": boots[int(0.025 * len(boots))] if boots else None,
        "mae_ci_high": boots[int(0.975 * len(boots)) - 1] if boots else None,
    }


def ranking(cfg: Config, err_rows: list[dict]) -> list[dict]:
    """Metrics per reference x solvation x method x linkage class (and all)."""
    temperature = cfg["energies"]["temperature_k"]
    a = cfg["analysis"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in err_rows:
        if row["t1_flag"]:  # SOP 7: DLPNO unreliable; reported, not ranked
            continue
        for linkage in ["all", *row["linkages"].split(";")]:
            groups[
                (row["reference"], row["solvation"], row["method"], linkage)
            ].append(row)
    out = []
    for (ref, solv, method, linkage), rows in sorted(groups.items()):
        rng = random.Random(a.get("bootstrap_seed", 67))
        st = _stats(
            [r["error_kcal_mol"] for r in rows],
            int(a.get("bootstrap_samples", 2000)),
            rng,
        )
        ddg = [
            r["error_ddg_kcal_mol"]
            for r in rows
            if r["error_ddg_kcal_mol"] is not None
        ]
        out.append(
            {
                "reference": ref,
                "solvation": solv,
                "method": method,
                **{k: rows[0][k] for k in assembly.META_COLUMNS},
                "linkage": linkage,
                **st,
                "mae_log_k": st["mae_kcal_mol"]
                / assembly.rt_ln10(temperature),
                "mae_ddg_kcal_mol": (
                    statistics.fmean(abs(v) for v in ddg) if ddg else None
                ),
                "mae_gain_vs_baseline": None,
            }
        )
    baselines = {
        (r["reference"], r["solvation"], r["linkage"]): r["mae_kcal_mol"]
        for r in out
        if r["baseline"] is True
    }
    for r in out:
        base = baselines.get((r["reference"], r["solvation"], r["linkage"]))
        if base is not None:
            r["mae_gain_vs_baseline"] = base - r["mae_kcal_mol"]
    out.sort(
        key=lambda r: (
            r["reference"],
            r["solvation"],
            r["linkage"],
            r["mae_kcal_mol"],
        )
    )
    return out


def mark_complete(rank: list[dict]) -> None:
    """Flag rows whose method covers every reaction of its group.

    Methods that failed on some molecules are ranked on fewer reactions;
    compare MAEs fairly only between rows with complete = True.
    """
    most: dict[tuple, int] = defaultdict(int)
    for r in rank:
        key = (r["reference"], r["solvation"], r["linkage"])
        most[key] = max(most[key], r["n"])
    for r in rank:
        r["complete"] = (
            r["n"] == most[(r["reference"], r["solvation"], r["linkage"])]
        )


def level_summary(rank: list[dict]) -> list[dict]:
    """Per level x reference x solvation x linkage: best, worst, mean MAE."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rank:
        for level in str(r.get("level", "")).split(";"):
            if level and level != "reference":
                groups[
                    (level, r["reference"], r["solvation"], r["linkage"])
                ].append(r)
    out = []
    for (level, ref, solv, linkage), rows in sorted(groups.items()):
        best = min(rows, key=lambda r: r["mae_kcal_mol"])
        worst = max(rows, key=lambda r: r["mae_kcal_mol"])
        out.append(
            {
                "level": level,
                "reference": ref,
                "solvation": solv,
                "linkage": linkage,
                "n_methods": len(rows),
                "best_method": best["method"],
                "best_mae_kcal_mol": best["mae_kcal_mol"],
                "best_sd_kcal_mol": best["sd_kcal_mol"],
                "worst_method": worst["method"],
                "worst_mae_kcal_mol": worst["mae_kcal_mol"],
                "mean_mae_kcal_mol": statistics.fmean(
                    r["mae_kcal_mol"] for r in rows
                ),
                "all_complete": all(r["complete"] for r in rows),
            }
        )
    return out
