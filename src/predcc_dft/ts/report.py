"""TS module analysis: barriers, rate constants and concentrations over time.

Rebuilt every round into ts/derived/<tag>/ (<tag>: energy protocol, see
predcc_dft.ts.stage):

  species.csv               every minimum: E, G terms, G
  barriers.csv              reaction x step x shuttle: state, imaginary mode,
                            dG_act, dG, k_f, k_r, share of the step's flux
  summary.csv               reaction: K, equilibrium and 24 h concentrations,
                            yield, t50/t95 with the +-1 kcal/mol band
  kinetics/<reaction>.csv   concentrations over time (central estimate)

and, in ts/reactions/<key>/reaction.toml, [kinetics.<tag>]: the summary,
every channel and the path of the concentration table of that reaction.
"""

from __future__ import annotations

import math

from predcc_dft.analysis import tables
from predcc_dft.protocol import constants, project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml
from predcc_dft.ts import kinetics, stage

H2K = constants.HARTREE_TO_KCAL


def _molecule(cfg: Config, label: str) -> dict:
    path = stage.molecule_file(cfg, label)
    return read_toml(path) if path.exists() else {}


def _g(cfg: Config, tag: str, label: str) -> dict:
    """Energies of one molecule under one protocol (empty if none yet)."""
    return _molecule(cfg, label).get("energies", {}).get(tag, {})


def _row(row: dict) -> dict:
    """CSV row: floats to 6 significant digits, hartree at full precision.

    Concentrations span many decades, so fixed decimals would hide them.
    """
    return {
        k: (
            f"{v:.6g}"
            if isinstance(v, float) and not k.endswith("_hartree")
            else v
        )
        for k, v in row.items()
    }


def _salt_dg(cfg: Config, g: dict[str, float]) -> float | None:
    override = cfg["ts"]["kinetics"].get("salt_dg_kcal_mol", "computed")
    if not isinstance(override, str):
        return float(override)
    if all(s in g for s in ("S", "T", "N")):
        return (g["S"] - g["T"] - g["N"]) * H2K
    return None


def channels(
    cfg: Config, tag: str, rxn: dict
) -> tuple[list[kinetics.Channel], list[dict], dict[str, float]]:
    """Usable kinetic channels, a barriers row per channel, species G."""
    g: dict[str, float] = {}
    for s, label in stage.symbols(rxn).items():
        res = _g(cfg, tag, label)
        if "g_hartree" in res:
            g[s] = res["g_hartree"]
    out, rows = [], []
    for (step, sh), cdir in stage.channel_dirs(cfg, rxn).items():
        path = stage.channel_file(cdir)
        ch = read_toml(path) if path.exists() else {}
        mech = ch.get("mechanism", {})
        res = ch.get("energies", {}).get(tag, {})
        row: dict = {
            "reaction": rxn["key"],
            "aldehyde": rxn["recs"]["aldehyde"]["name"],
            "amine": rxn["recs"]["amine"]["name"],
            "step": step,
            "shuttle": sh,
            "state": ch.get("state", "no_complex"),
            "reason": ch.get("reason", ""),
            "imaginary_cm1": (
                ch.get("freq", {}).get("imaginary_cm1") or [None]
            )[0],
            "g_ts_hartree": res.get("g_hartree"),
            "dg_act_kcal_mol": None,
            "dg_kcal_mol": None,
            "molecularity": None,
            "used": False,
            "channel_toml": stage.rel(cfg, path),
        }
        reac: list[str] = mech.get("kinetic_reactants") or []
        prod: list[str] = mech.get("kinetic_products") or []
        if reac and "g_hartree" in res and all(s in g for s in reac + prod):
            g_r = sum(g[s] for s in reac)
            row["dg_act_kcal_mol"] = (res["g_hartree"] - g_r) * H2K
            row["dg_kcal_mol"] = (sum(g[s] for s in prod) - g_r) * H2K
            row["molecularity"] = len(reac)
            if row["state"] == "verified":
                row["used"] = True
                out.append(
                    kinetics.Channel(
                        f"{step}/{sh}",
                        step,
                        reac,
                        prod,
                        row["dg_act_kcal_mol"],
                        row["dg_kcal_mol"],
                    )
                )
        rows.append(row)
    return out, rows, g


def _conditions(cfg: Config) -> dict:
    c = cfg["ts"]["conditions"]
    ald = float(c["aldehyde_mol_l"])
    return {
        "temperature_k": float(c["temperature_k"]),
        "t_end_s": float(c["time_h"]) * 3600.0,
        "c0": {
            "A": ald,
            "N": float(c["amine_mol_l"]),
            "T": ald * float(c["tfa_mol_percent"]) / 100.0,
            "W": float(c["water_mol_l"]),
        },
    }


def simulate(cfg: Config, tag: str, rxn: dict) -> dict:
    """Kinetics of one reaction: summary row, time series, channel rows."""
    chs, rows, g = channels(cfg, tag, rxn)
    cond = _conditions(cfg)
    kin = cfg["ts"]["kinetics"]
    cap = float(kin["diffusion_limit_m1_s1"])
    summary: dict = {
        "reaction": rxn["key"],
        "aldehyde": rxn["recs"]["aldehyde"]["name"],
        "amine": rxn["recs"]["amine"]["name"],
        "imine": rxn["recs"]["imine"]["name"],
        "energy_protocol": tag,
        "temperature_k": cond["temperature_k"],
        "aldehyde0_mol_l": cond["c0"]["A"],
        "amine0_mol_l": cond["c0"]["N"],
        "tfa0_mol_l": cond["c0"]["T"],
        "water0_mol_l": cond["c0"]["W"],
        "time_h": cond["t_end_s"] / 3600.0,
        "status": "incomplete",
        "notes": "",
    }
    notes = []
    if all(s in g for s in "ANIW"):
        dg = (g["I"] + g["W"] - g["A"] - g["N"]) * H2K
        summary["dg_condensation_kcal_mol"] = dg
        summary["log_k_condensation"] = 0.0 - dg / (
            math.log(10) * constants.R_KCAL * cond["temperature_k"]
        )
    if cond["c0"]["T"] > 0:
        dg_salt = _salt_dg(cfg, g)
        summary["dg_salt_kcal_mol"] = dg_salt
        if dg_salt is None:
            chs = [c for c in chs if "S" not in c.reactants]
            notes.append("no salt G: TFA kept as free acid")
        else:
            chs.append(kinetics.acid_base_channel(dg_salt, cap))
    steps = {c.step for c in chs}
    missing = [s for s in ("addition", "dehydration") if s not in steps]
    if missing:
        summary["notes"] = "; ".join(
            [*notes, f"no usable TS for {', '.join(missing)}"]
        )
        return {"summary": summary, "rows": rows, "series": []}
    delta = float(kin["barrier_shift_kcal_mol"])
    runs = {}
    try:
        for name, shift in (("", 0.0), ("_fast", -delta), ("_slow", delta)):
            runs[name] = kinetics.simulate(
                kinetics.shifted(chs, shift),
                cond["c0"],
                cond["temperature_k"],
                cond["t_end_s"],
                int(kin["points"]),
                cap,
            )
    except kinetics.IntegrationError as exc:
        summary["status"] = "integration failed"
        summary["notes"] = "; ".join([*notes, str(exc)])
        return {"summary": summary, "rows": rows, "series": []}
    limiting = min(cond["c0"]["A"], cond["c0"]["N"])
    central = runs[""]
    eq = central["equilibrium"]
    summary["status"] = "complete"
    summary |= {f"{s}_eq_mol_l": eq[s] for s in kinetics.SPECIES}
    summary["imine_yield_eq"] = eq["I"] / limiting
    for name, run in runs.items():
        i_end = float(run["conc"]["I"][-1])
        summary[f"imine_end_mol_l{name}"] = i_end
        summary[f"imine_yield_end{name}"] = i_end / limiting
        summary[f"t50_h{name}"] = (
            run["t50_s"] / 3600.0 if run["t50_s"] is not None else None
        )
        summary[f"t95_h{name}"] = (
            run["t95_s"] / 3600.0 if run["t95_s"] is not None else None
        )
    if eq["A"] > 0 and eq["N"] > 0:
        q = eq["I"] * eq["W"] / (eq["A"] * eq["N"])
        summary["log_q_eq"] = math.log10(q) if q > 0 else None
    if eq.get("T", 0) + eq.get("S", 0) > 0:
        summary["tfa_as_salt_eq"] = eq["S"] / (eq["S"] + eq["T"])
    for step in ("addition", "dehydration"):
        shares = {
            c.name: central["flux_share"].get(c.name, 0.0)
            for c in chs
            if c.step == step
        }
        summary[f"main_{step}_channel"] = max(
            shares, key=lambda name: shares[name]
        )
    summary["channels_used"] = ";".join(c.name for c in chs)
    summary["notes"] = "; ".join(notes)
    by_name = {c.name: c for c in chs}
    for row in rows:
        name = f"{row['step']}/{row['shuttle']}"
        if name in by_name:
            kf, kr = central["rate_constants"][name]
            order = by_name[name].order
            row["k_f"] = kf
            row["k_r"] = kr
            row["k_f_units"] = f"M^{1 - order} s^-1" if order > 1 else "s^-1"
            row["flux_share"] = central["flux_share"].get(name, 0.0)
    series = []
    for k, t in enumerate(central["t_s"]):
        point = {"t_s": float(t), "t_h": float(t) / 3600.0}
        point |= {
            f"{s}_mol_l": float(central["conc"][s][k])
            for s in kinetics.SPECIES
        }
        point["imine_yield"] = point["I_mol_l"] / limiting
        series.append(point)
    return {"summary": summary, "rows": rows, "series": series}


def _species_rows(cfg: Config, tag: str, rxns: list[dict]) -> list[dict]:
    rows = []
    for label in stage.species(cfg, rxns):
        mol = _molecule(cfg, label)
        res = _g(cfg, tag, label)
        rows.append(
            {
                "label": label,
                "symbol": mol.get("symbol", ""),
                "role": mol.get("role", ""),
                "kind": mol.get("kind", ""),
                "name": mol.get("name", ""),
                "smiles": mol.get("smiles", ""),
                "status": mol.get("status", "pending"),
                "electronic_hartree": res.get("electronic_hartree"),
                "g_corr_hartree": res.get("g_corr_hartree"),
                "dg_solv_kcal_mol": res.get("dg_solv_kcal_mol"),
                "standard_state_kcal_mol": res.get("standard_state_kcal_mol"),
                "g_hartree": res.get("g_hartree"),
                "ts_toml": stage.rel(cfg, stage.molecule_file(cfg, label)),
            }
        )
    return rows


COLUMNS_EXTRA = ("k_f", "k_r", "k_f_units", "flux_share")


def run(cfg: Config) -> dict:
    """Rebuild ts/derived/<tag>/ and [kinetics.<tag>] of every reaction."""
    tag = stage.energy_spec(cfg)["tag"]
    rxns = stage.reactions(cfg)
    project.write_index(cfg, stage.BRANCH, project.selected(cfg, stage.BRANCH))
    d = cfg.ts_dir / "derived" / tag
    (d / "kinetics").mkdir(parents=True, exist_ok=True)
    summaries, barrier_rows = [], []
    for rxn in rxns:
        res = simulate(cfg, tag, rxn)
        summaries.append(res["summary"])
        for row in res["rows"]:
            barrier_rows.append(row | {k: row.get(k) for k in COLUMNS_EXTRA})
        series = d / "kinetics" / f"{rxn['key']}.csv"
        if res["series"]:
            tables.write_csv(series, [_row(p) for p in res["series"]])
        path = stage.reaction_file(cfg, rxn["key"])
        data = read_toml(path) if path.exists() else {}
        data.setdefault("kinetics", {})[tag] = {
            "generated": now(),
            "concentrations_csv": (
                stage.rel(cfg, series) if res["series"] else ""
            ),
            "summary": {
                k: v for k, v in res["summary"].items() if v is not None
            },
            "channels": [
                {k: v for k, v in r.items() if v is not None}
                for r in res["rows"]
            ],
        }
        stage.update_toml(path, {"kinetics": data["kinetics"]})
    keys = list(dict.fromkeys(k for s in summaries for k in s))
    tables.write_csv(
        d / "summary.csv",
        [_row({k: s.get(k) for k in keys}) for s in summaries],
    )
    tables.write_csv(
        d / "barriers.csv",
        [_row(r) for r in barrier_rows],
    )
    tables.write_csv(d / "species.csv", _species_rows(cfg, tag, rxns))
    return {
        "energy protocol": tag,
        "reactions": len(rxns),
        "kinetics complete": sum(s["status"] == "complete" for s in summaries),
    }


def local_steps(cfg: Config) -> dict:
    """Set up new channels, collect finished jobs, rebuild derived/."""
    return {
        "ts set-up": stage.setup(cfg),
        "ts collect": stage.collect(cfg),
        "ts analysis": run(cfg),
    }


def writers() -> dict:
    """SLURM stage -> job writer of the TS module."""
    return stage.writers()
