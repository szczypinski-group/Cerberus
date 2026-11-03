"""Steps 10-11, SOP 2.7 and 8: assemble dG, compare, rank, propose a lock.

Reads only record.toml / reference.toml / structure.toml files, reactions.toml
and experimental/*.toml, and regenerates (derived/, tidy long format):

  species_energies.csv    compound x method x solvation: E, G_corr, dG_solv, G
  reaction_energies.csv   reaction x method x solvation: dE, dG, ddG, log K
  errors.csv              ... x reference (DLPNO-CCSD(T)/CBS, experiment)
  method_ranking.csv      MAE, RMSE, MSE, SD, max, bootstrap 95 % CI (MAE),
                          per linkage class and overall, gain over the baseline
  production_candidate.toml   SOP 2.8 lock draft; review, then copy to
                          config/production.toml (8.1)

G(species) = E(method, gas, best geometry) + G_corr(r2SCAN-3c, step 6)
             + dG_solv(model) + RT ln(24.46)       (solvated models only)
"""

from __future__ import annotations

import csv
import logging
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from predcc_dft import SCHEMA_VERSION, benchmark, conformers, orca, reference
from predcc_dft.io_utils import (
    Config,
    now,
    pipeline_commit,
    read_toml,
    write_toml,
)

log = logging.getLogger(__name__)

META = ("functional", "basis", "family", "zeta", "baseline")


def _rt_ln10(temperature: float) -> float:
    return reference.R_KCAL * temperature * math.log(10)


def _method_meta(cfg: Config, label: str) -> dict:
    """Plotting/grouping metadata of a method label."""
    if label == reference.REFERENCE_LABEL:
        return {
            "functional": "DLPNO-CCSD(T)",
            "basis": "CBS(X/Y)",
            "family": "reference",
            "zeta": "",
            "baseline": False,
            "level": "reference",
            "functional_label": "",
            "basis_label": "",
        }
    m = cfg.methods.get("methods", {}).get(label, {})
    return {
        **{k: m.get(k, "") for k in META},
        "level": ";".join(m.get("levels", [])),
        "functional_label": m.get("functional_label", ""),
        "basis_label": m.get("basis_label", ""),
    }


# ------------------------------------------------------------------ species


def species_energies(cfg: Config) -> dict[tuple[str, str, str], dict]:
    """(compound, method, solvation) -> energy components."""
    out = {}
    for rec in conformers.molecules(cfg):
        data = reference.species_data(cfg, rec)
        if data is None:
            continue
        electronic = dict(benchmark.energies(cfg, rec))
        if "reference" in data:
            electronic[reference.REFERENCE_LABEL] = data["reference"][
                "e_cbs_hartree"
            ]
        g_corr = data["thermo"]["g_corr_hartree"]
        ss = data["standard_state_kcal_mol"]
        g_ens = (
            data.get("g_ensemble_correction_kcal_mol", 0.0)
            if cfg["analysis"].get("ensemble_correction", True)
            else 0.0
        )
        t1s = [
            data.get("reference", {}).get(k)
            for k in ("t1_x", "t1_y")
            if data.get("reference", {}).get(k) is not None
        ]
        solv = {"gas": 0.0}
        for label in reference.solvation_models(cfg):
            entry = data["solvation"].get(label)
            if isinstance(entry, dict):
                solv[label] = entry["dg_solv_kcal_mol"]
        for method, e in electronic.items():
            for model, dg_solv in solv.items():
                ss_m = ss if model != "gas" else 0.0
                out[(rec["compound"], method, model)] = {
                    "compound": rec["compound"],
                    "name": rec["name"],
                    "role": rec["role"],
                    "isomer": data["isomer"],
                    "method": method,
                    **_method_meta(cfg, method),
                    "solvation": model,
                    "e_hartree": e,
                    "g_corr_hartree": g_corr,
                    "dg_solv_kcal_mol": dg_solv,
                    "standard_state_kcal_mol": ss_m,
                    "g_ensemble_correction_kcal_mol": g_ens,
                    "g_hartree": e
                    + g_corr
                    + (dg_solv + ss_m + g_ens) / orca.HARTREE_TO_KCAL,
                    "t1": max(t1s) if t1s else None,
                }
    return out


# ------------------------------------------------------------------ reactions


def _stoich(rxn: dict) -> list[tuple[str, int]]:
    """(compound, signed coefficient) pairs; products positive."""
    return [(c["compound"], -c["coefficient"]) for c in rxn["reactants"]] + [
        (c["compound"], c["coefficient"]) for c in rxn["products"]
    ]


def _parents(cfg: Config, reactions: dict) -> dict[str, str]:
    """Linkage -> parent condensation key (for ddG, SOP 3)."""
    mols = {r["compound"]: r for r in conformers.molecules(cfg)}
    by_names = {}
    for rxn in reactions.get("condensation", []):
        ald, nuc = (c["compound"] for c in rxn["reactants"])
        by_names[f"{mols[ald]['name']} + {mols[nuc]['name']}"] = rxn["key"]
    parents = {}
    for linkage, spec in cfg["analysis"].get("parents", {}).items():
        key = spec if spec.startswith("CND-") else by_names.get(spec)
        if key is None:
            log.warning("parent %r for %s not found", spec, linkage)
        else:
            parents[linkage] = key
    order = {r["compound"]: i for i, r in enumerate(conformers.molecules(cfg))}
    for rxn in sorted(
        reactions.get("condensation", []),
        key=lambda r: tuple(order[c["compound"]] for c in r["reactants"]),
    ):
        for linkage in rxn["linkages"]:
            parents.setdefault(linkage, rxn["key"])
    return parents


def reaction_energies(
    cfg: Config, species: dict, reactions: dict
) -> list[dict]:
    """dE, dG, ddG and log K for every reaction x method x solvation."""
    temperature = cfg["reference"]["temperature_k"]
    methods = sorted({k[1] for k in species})
    models = sorted({k[2] for k in species})
    parents = _parents(cfg, reactions)
    t1_limit = float(cfg["analysis"].get("t1_threshold", 0.02))
    rows: dict[tuple[str, str, str], dict] = {}
    for rtype in ("condensation", "exchange"):
        for rxn in reactions.get(rtype, []):
            comps = _stoich(rxn)
            for method in methods:
                for model in models:
                    keys = [(c, method, model) for c, _ in comps]
                    if any(k not in species for k in keys):
                        continue
                    de = sum(
                        n * species[(c, method, model)]["e_hartree"]
                        for c, n in comps
                    )
                    dg = sum(
                        n * species[(c, method, model)]["g_hartree"]
                        for c, n in comps
                    )
                    de *= orca.HARTREE_TO_KCAL
                    dg *= orca.HARTREE_TO_KCAL
                    t1s = [
                        species[k]["t1"]
                        for k in keys
                        if species[k]["t1"] is not None
                    ]
                    max_t1 = max(t1s) if t1s else None
                    rows[(rxn["key"], method, model)] = {
                        "reaction": rxn["key"],
                        "type": rtype,
                        "linkages": ";".join(rxn["linkages"]),
                        "method": method,
                        **_method_meta(cfg, method),
                        "solvation": model,
                        "de_kcal_mol": de,
                        "dg_kcal_mol": dg,
                        "ddg_kcal_mol": None,
                        "log_k": -dg / _rt_ln10(temperature),
                        "parent": "",
                        "max_t1": max_t1,
                        "t1_flag": max_t1 is not None and max_t1 > t1_limit,
                    }
    for row in rows.values():
        if row["type"] != "condensation":
            continue
        parent = parents.get(row["linkages"].split(";")[0], "")
        prow = rows.get((parent, row["method"], row["solvation"]))
        if prow is not None:
            row["parent"] = parent
            row["ddg_kcal_mol"] = row["dg_kcal_mol"] - prow["dg_kcal_mol"]
    return list(rows.values())


# ------------------------------------------------------------------ experiment


def experimental(cfg: Config, reactions: dict) -> dict[str, dict]:
    """Reaction key -> {dg_kcal_mol, log_k, uncertainty} usable for comparison.

    Only dimensionless K in MeCN with water as a solute are comparable to the
    computed 1 M standard-state dG (SOP 4.4, 7); others are skipped with a log.
    """
    known = {
        r["key"]
        for t in ("condensation", "exchange")
        for r in reactions.get(t, [])
    }
    out: dict[str, dict] = {}
    if not cfg.experimental_dir.exists():
        return out
    for path in sorted(cfg.experimental_dir.glob("*.toml")):
        if path.name.startswith("_"):
            continue
        data = read_toml(path)
        key = data.get("reaction", path.stem)
        if key not in known:
            log.warning(
                "%s: reaction %s not in reactions.toml", path.name, key
            )
            continue
        if data.get("k_units", "dimensionless") != "dimensionless" or (
            key.startswith("CND-") and not data.get("water_included", True)
        ):
            log.warning(
                "%s: K not dimensionless with water; skipped", path.name
            )
            continue
        temperature = data.get("temperature_k", 298.15)
        out[key] = {
            "log_k": data["log_k"],
            "log_k_uncertainty": data.get("log_k_uncertainty"),
            "dg_kcal_mol": -_rt_ln10(temperature) * data["log_k"],
            "temperature_k": temperature,
        }
    return out


# ------------------------------------------------------------------ errors


def errors(cfg: Config, rxn_rows: list[dict], exp: dict) -> list[dict]:
    """Signed errors against the DLPNO reference and against experiment."""
    temperature = cfg["reference"]["temperature_k"]
    index = {(r["reaction"], r["method"], r["solvation"]): r for r in rxn_rows}
    ref_label = cfg["analysis"].get("reference", reference.REFERENCE_LABEL)
    out = []
    for row in rxn_rows:
        refs = []
        # SOP 8: gas-phase dE vs DLPNO ranks functionals (thermal and
        # solvation terms are identical and cancel); solution dG vs experiment.
        if row["method"] != ref_label and row["solvation"] == "gas":
            ref = index.get((row["reaction"], ref_label, row["solvation"]))
            if ref is not None:
                refs.append(
                    (ref_label, ref["dg_kcal_mol"], ref["ddg_kcal_mol"])
                )
        if row["reaction"] in exp and row["solvation"] != "gas":
            refs.append(
                ("experiment", exp[row["reaction"]]["dg_kcal_mol"], None)
            )
        for name, value, ref_ddg in refs:
            err = row["dg_kcal_mol"] - value
            # SOP 8 error decomposition (experiment rows): electronic part =
            # dE(method) - dE(DLPNO); the rest = dG(DLPNO, same thermal and
            # solvation terms) - experiment, i.e. thermal + solvation + ref.
            electronic = rest = None
            if name == "experiment":
                dl = index.get((row["reaction"], ref_label, row["solvation"]))
                if dl is not None:
                    electronic = row["de_kcal_mol"] - dl["de_kcal_mol"]
                    rest = dl["dg_kcal_mol"] - value
            out.append(
                {
                    **{k: row[k] for k in row if k != "parent"},
                    "reference": name,
                    "ref_value": value,
                    "error_kcal_mol": err,
                    "error_log_k": -err / _rt_ln10(temperature),
                    "error_electronic_kcal_mol": electronic,
                    "error_thermal_solvation_kcal_mol": rest,
                    "error_ddg_kcal_mol": (
                        row["ddg_kcal_mol"] - ref_ddg
                        if row["ddg_kcal_mol"] is not None
                        and ref_ddg is not None
                        else None
                    ),
                }
            )
    return out


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
    temperature = cfg["reference"]["temperature_k"]
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
                **_method_meta(cfg, method),
                "linkage": linkage,
                **st,
                "mae_log_k": st["mae_kcal_mol"] / _rt_ln10(temperature),
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


# ------------------------------------------------------------------ outputs


def _fmt(key: str, v: object) -> object:
    """CSV cell: full precision for hartree energies, 6 decimals otherwise."""
    if isinstance(v, float):
        return f"{v:.10f}" if key.endswith("_hartree") else f"{v:.6f}"
    return "" if v is None else v


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows (dicts with equal keys) as CSV."""
    with path.open("w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: _fmt(k, v) for k, v in r.items()} for r in rows)


def _candidate(
    cfg: Config, rank: list[dict], rxn_rows: list[dict]
) -> dict | None:
    """Production-lock draft (SOP 2.8) from the ranking against DLPNO."""
    ref_label = cfg["analysis"].get("reference", reference.REFERENCE_LABEL)
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
            "orca_keywords": benchmark.inputs_for(cfg, best["method"])[0]
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


def _mark_complete(rank: list[dict]) -> None:
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


def errors_wide(cfg: Config, err_rows: list[dict]) -> list[dict]:
    """Gas-phase errors vs DLPNO: one row per reaction, one column per method.

    Spreadsheet-friendly view of errors.csv (kcal/mol, method - reference).
    """
    ref_label = cfg["analysis"].get("reference", reference.REFERENCE_LABEL)
    methods: list[str] = []
    rows: dict[str, dict] = {}
    for e in err_rows:
        if e["reference"] != ref_label or e["solvation"] != "gas":
            continue
        if e["method"] not in methods:
            methods.append(e["method"])
        row = rows.setdefault(
            e["reaction"],
            {
                "reaction": e["reaction"],
                "type": e["type"],
                "linkages": e["linkages"],
                "ref_dg_kcal_mol": e["ref_value"],
            },
        )
        row[e["method"]] = e["error_kcal_mol"]
    return [{**r, **{m: r.get(m) for m in methods}} for r in rows.values()]


def run(cfg: Config) -> dict:
    """Rebuild every derived table from the records."""
    reactions = read_toml(cfg.reactions_file)
    species = species_energies(cfg)
    rxn_rows = reaction_energies(cfg, species, reactions)
    exp = experimental(cfg, reactions)
    err_rows = errors(cfg, rxn_rows, exp)
    rank = ranking(cfg, err_rows)
    _mark_complete(rank)
    d = cfg.derived_dir
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "species_energies.csv", list(species.values()))
    _write_csv(d / "reaction_energies.csv", rxn_rows)
    _write_csv(d / "errors.csv", err_rows)
    _write_csv(d / "method_ranking.csv", rank)
    _write_csv(d / "level_summary.csv", level_summary(rank))
    _write_csv(d / "errors_wide.csv", errors_wide(cfg, err_rows))
    cand = _candidate(cfg, rank, rxn_rows)
    if cand is not None:
        write_toml(d / "production_candidate.toml", cand)
    return {
        "species_rows": len(species),
        "reaction_rows": len(rxn_rows),
        "experimental_reactions": len(exp),
        "error_rows": len(err_rows),
        "best_vs_reference": (
            f"{cand['energy']['method']} "
            f"(MAE {cand['selection']['mae_kcal_mol']} kcal/mol)"
            if cand
            else "-"
        ),
    }
