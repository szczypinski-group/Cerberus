"""SOP 5.1 step 11: assemble species G and reaction dE, dG, ddG, log K.

G(species) = E(method, gas, best geometry) + G_corr(r2SCAN-3c, step 6)
             + dG_solv(model) + RT ln(24.46)   (solvated models only)
             + G_ens - G_best                  (conformer/isomer ensemble)
"""

from __future__ import annotations

import logging
import math

from predcc_dft.protocol import constants, methods, project
from predcc_dft.protocol.config import Config

log = logging.getLogger(__name__)

META = ("functional", "basis", "family", "zeta", "baseline")


def rt_ln10(temperature: float) -> float:
    """RT ln 10 in kcal/mol: dG per unit of log K."""
    return constants.R_KCAL * temperature * math.log(10)


def method_meta(cfg: Config, label: str) -> dict:
    """Plotting/grouping metadata of a method label."""
    if label == constants.REFERENCE_LABEL:
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


def species_energies(cfg: Config) -> dict[tuple[str, str, str], dict]:
    """(compound, method, solvation) -> energy components."""
    out = {}
    for rec in project.molecules(cfg):
        data = project.reference_data(cfg, rec)
        if data is None:
            continue
        electronic = dict(project.benchmark_energies(cfg, rec))
        if "reference" in data:
            electronic[constants.REFERENCE_LABEL] = data["reference"][
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
        for label in methods.solvation_models(cfg.methods):
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
                    **method_meta(cfg, method),
                    "solvation": model,
                    "e_hartree": e,
                    "g_corr_hartree": g_corr,
                    "dg_solv_kcal_mol": dg_solv,
                    "standard_state_kcal_mol": ss_m,
                    "g_ensemble_correction_kcal_mol": g_ens,
                    "g_hartree": e
                    + g_corr
                    + (dg_solv + ss_m + g_ens) / constants.HARTREE_TO_KCAL,
                    "t1": max(t1s) if t1s else None,
                }
    return out


def _stoich(rxn: dict) -> list[tuple[str, int]]:
    """(compound, signed coefficient) pairs; products positive."""
    return [(c["compound"], -c["coefficient"]) for c in rxn["reactants"]] + [
        (c["compound"], c["coefficient"]) for c in rxn["products"]
    ]


def _parents(cfg: Config, reactions: dict) -> dict[str, str]:
    """Linkage -> parent condensation key (for ddG, SOP 3)."""
    mols = {r["compound"]: r for r in project.molecules(cfg)}
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
    order = {r["compound"]: i for i, r in enumerate(project.molecules(cfg))}
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
    method_labels = sorted({k[1] for k in species})
    models = sorted({k[2] for k in species})
    parents = _parents(cfg, reactions)
    t1_limit = float(cfg["analysis"].get("t1_threshold", 0.02))
    rows: dict[tuple[str, str, str], dict] = {}
    for rtype in ("condensation", "exchange"):
        for rxn in reactions.get(rtype, []):
            comps = _stoich(rxn)
            for method in method_labels:
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
                    de *= constants.HARTREE_TO_KCAL
                    dg *= constants.HARTREE_TO_KCAL
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
                        **method_meta(cfg, method),
                        "solvation": model,
                        "de_kcal_mol": de,
                        "dg_kcal_mol": dg,
                        "ddg_kcal_mol": None,
                        "log_k": -dg / rt_ln10(temperature),
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
