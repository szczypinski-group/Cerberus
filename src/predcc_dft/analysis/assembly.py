"""SOP 5.1 step 11: species G and reaction dE, dG, ddG, log K.

G(species) = E(method, gas, best geometry) + G_corr(r2SCAN-3c, step 6)
             + dG_solv(model) + RT ln(24.46)   (solvated models only)
             + G_ens - G_best                  (conformer/isomer ensemble)
"""

from __future__ import annotations

import logging
import math

from predcc_dft.protocol import constants, project
from predcc_dft.protocol.config import Config

log = logging.getLogger(__name__)

# Method metadata copied into every species / reaction / error row.
META_COLUMNS = (
    "functional",
    "basis",
    "family",
    "zeta",
    "baseline",
    "level",
    "functional_label",
    "basis_label",
)


def rt_ln10(temperature: float) -> float:
    """RT ln 10 in kcal/mol: dG per unit of log K."""
    return constants.R_KCAL * temperature * math.log(10)


def species_row(
    rec: dict,
    isomer: str,
    method: str,
    meta: dict,
    model: str,
    e_hartree: float,
    terms: dict,
    dg_solv_kcal: float,
    t1: float | None = None,
    ensemble: bool = True,
) -> dict:
    """One species_energies.csv row: every G term in its own column.

    terms: energies.thermo.species_terms() of the molecule. The standard
    state applies to solvated models only (gas-phase rows are 1 atm).
    """
    ss = terms["standard_state_kcal_mol"] if model != "gas" else 0.0
    g_ens = terms["g_ensemble_correction_kcal_mol"] if ensemble else 0.0
    g_corr = terms["g_corr_hartree"]
    return {
        "compound": rec["compound"],
        "name": rec["name"],
        "role": rec["role"],
        "isomer": isomer,
        "method": method,
        **{k: meta.get(k, "") for k in META_COLUMNS},
        "solvation": model,
        "e_hartree": e_hartree,
        "g_corr_hartree": g_corr,
        "dg_solv_kcal_mol": dg_solv_kcal,
        "standard_state_kcal_mol": ss,
        "g_ensemble_correction_kcal_mol": g_ens,
        "g_hartree": e_hartree
        + g_corr
        + (dg_solv_kcal + ss + g_ens) / constants.HARTREE_TO_KCAL,
        "t1": t1,
    }


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
    temperature = cfg["energies"]["temperature_k"]
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
                        **{k: species[keys[0]][k] for k in META_COLUMNS},
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
