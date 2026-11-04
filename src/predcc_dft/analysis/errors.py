"""SOP 4.4, 8: experimental K and signed errors against every reference."""

from __future__ import annotations

import logging

from predcc_dft.analysis import assembly
from predcc_dft.protocol import constants
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml

log = logging.getLogger(__name__)


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
            "dg_kcal_mol": -assembly.rt_ln10(temperature) * data["log_k"],
            "temperature_k": temperature,
        }
    return out


def errors(cfg: Config, rxn_rows: list[dict], exp: dict) -> list[dict]:
    """Signed errors against the DLPNO reference and against experiment."""
    temperature = cfg["reference"]["temperature_k"]
    index = {(r["reaction"], r["method"], r["solvation"]): r for r in rxn_rows}
    ref_label = cfg["analysis"].get("reference", constants.REFERENCE_LABEL)
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
                    "error_log_k": -err / assembly.rt_ln10(temperature),
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


def errors_wide(cfg: Config, err_rows: list[dict]) -> list[dict]:
    """Gas-phase errors vs DLPNO: one row per reaction, one column per method.

    Spreadsheet-friendly view of errors.csv (kcal/mol, method - reference).
    """
    ref_label = cfg["analysis"].get("reference", constants.REFERENCE_LABEL)
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
