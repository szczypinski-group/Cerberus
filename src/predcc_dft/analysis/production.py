"""SOP 8.1, 2.8: draft of the production lock from the ranking."""

from __future__ import annotations

import statistics
from collections import defaultdict

from predcc_dft import SCHEMA_VERSION
from predcc_dft.protocol import constants, methods
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, pipeline_commit


def candidate(
    cfg: Config, rank: list[dict], rxn_rows: list[dict]
) -> dict | None:
    """Production-lock draft (SOP 2.8) from the ranking against DLPNO."""
    ref_label = cfg["analysis"].get("reference", constants.REFERENCE_LABEL)
    vs_ref = [
        r
        for r in rank
        if r["reference"] == ref_label and r["solvation"] == "gas"
    ]
    overall = [
        r for r in vs_ref if r["linkage"] == "all" and r.get("complete", True)
    ]
    if not overall:
        return None
    best = min(overall, key=lambda r: (r["mae_kcal_mol"], r["sd_kcal_mol"]))
    by_linkage = {
        r["linkage"]: round(r["mae_kcal_mol"], 3)
        for r in vs_ref
        if r["method"] == best["method"] and r["linkage"] != "all"
    }
    vs_exp = [
        r
        for r in rank
        if r["reference"] == "experiment"
        and r["linkage"] == "all"
        and r["method"] == best["method"]
    ]
    solvation = (
        min(vs_exp, key=lambda r: r["mae_kcal_mol"])["solvation"]
        if vs_exp
        else cfg["analysis"].get("solvation", "smd_mecn")
    )
    # SOP 6.2: compare the solvation-model spread with the top-method spread.
    ref_dg = defaultdict(list)
    for r in rxn_rows:
        if r["method"] == ref_label and r["solvation"] != "gas":
            ref_dg[r["reaction"]].append(r["dg_kcal_mol"])
    solv_spread = [max(v) - min(v) for v in ref_dg.values() if len(v) > 1]
    top = sorted(overall, key=lambda r: r["mae_kcal_mol"])[:3]
    temperature = cfg["reference"]["temperature_k"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate - review, then copy to config/production.toml",
        "production_version": "1.0",
        "selected": now().date(),
        "benchmark_commit": pipeline_commit(),
        "geometry": {"method": "r2scan3c", "solvation": "cpcm_mecn"},
        "energy": {
            "method": best["method"],
            "solvation": solvation,
            "orca_keywords": methods.method_inputs(
                cfg.methods, best["method"]
            )[0]
            if best["method"] in cfg.methods.get("methods", {})
            else "",
        },
        "thermo": {
            "method": "r2scan3c",
            "entropy": "ORCA default (qRRHO)",
            "temperature_k": temperature,
        },
        "conformers": {
            "crest_window_kcal_mol": cfg["conformers"]["crest_ewin_kcal_mol"],
            "dft_window_kcal_mol": cfg["frequencies"]["window_kcal_mol"],
        },
        "selection": {
            "reference": ref_label,
            "n_reactions": best["n"],
            "mae_kcal_mol": round(best["mae_kcal_mol"], 3),
            "sd_kcal_mol": round(best["sd_kcal_mol"], 3),
            "mae_by_linkage": by_linkage,
            "runner_up": [r["method"] for r in top[1:]],
            "experimental_comparison": bool(vs_exp),
        },
        "diagnostics": {
            "mean_solvation_model_spread_kcal_mol": (
                round(statistics.fmean(solv_spread), 3)
                if solv_spread
                else None
            ),
            "top3_method_mae_spread_kcal_mol": round(
                top[-1]["mae_kcal_mol"] - top[0]["mae_kcal_mol"], 3
            ),
        },
        "ts_validation": {"checked": False, "n_ts": 0},
    }
