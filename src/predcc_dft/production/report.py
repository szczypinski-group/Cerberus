"""Production analysis: G per species, dG and log K per reaction.

Rebuilt every round into production/derived/v<version>/, one folder per
production version so results of different locks are never mixed:

  species_energies.csv    compound x {gas, locked model}: E, G terms, G
  reaction_energies.csv   reaction x {gas, locked model}: dE, dG, ddG, log K
  errors.csv              vs experiment, if production/experimental/ has data
  calculations.csv        every record.toml of this version

Same columns as the benchmark tables, so one analysis script reads both.
"""

from __future__ import annotations

from predcc_dft.analysis import assembly, errors, tables
from predcc_dft.production import stage
from predcc_dft.protocol import lock, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml

BRANCH = "production"


def _meta(the_lock: dict) -> dict:
    e = the_lock["energy"]
    return {
        "functional": e.get("functional", ""),
        "basis": e.get("basis", ""),
        "family": e.get("family", ""),
        "zeta": e.get("zeta", ""),
        "baseline": False,
        "level": f"production v{the_lock['production_version']}",
        "functional_label": "",
        "basis_label": "",
    }


def species_energies(cfg: Config, the_lock: dict) -> dict:
    """(compound, method, solvation) -> species row of completed results."""
    ensemble = cfg["analysis"].get("ensemble_correction", True)
    meta = _meta(the_lock)
    out = {}
    for rec in project.selected(cfg, BRANCH):
        data = stage.result(cfg, the_lock, rec)
        if data is None or data["status"] != "complete":
            continue
        e = data["energy"]["electronic_hartree"]
        for model, dg in (
            ("gas", 0.0),
            (data["solvation_model"], data["solvation"]["dg_solv_kcal_mol"]),
        ):
            out[(rec["compound"], data["method"], model)] = (
                assembly.species_row(
                    rec,
                    data["isomer"],
                    data["method"],
                    meta,
                    model,
                    e,
                    data["thermo"],
                    dg,
                    None,
                    ensemble,
                )
            )
    return out


def run(cfg: Config) -> dict:
    """Rebuild production/derived/v<version>/ from this version's records."""
    the_lock = lock.read(cfg)
    vname = lock.version_dir(the_lock)
    recs = project.selected(cfg, BRANCH)
    project.write_index(cfg, BRANCH, recs)
    reactions = read_toml(cfg.reactions_file)
    species = species_energies(cfg, the_lock)
    rxn_rows = assembly.reaction_energies(cfg, species, reactions)
    d = cfg.production_dir / "derived" / vname
    d.mkdir(parents=True, exist_ok=True)
    tables.write_csv(d / "species_energies.csv", list(species.values()))
    tables.write_csv(d / "reaction_energies.csv", rxn_rows)
    exp = errors.experimental(cfg.production_dir / "experimental", reactions)
    err_rows = errors.errors(cfg, rxn_rows, exp, None) if exp else []
    if err_rows:
        tables.write_csv(d / "errors.csv", err_rows)
    n_calc = records.collect_csv(
        cfg,
        cfg.production_dir / "molecules",
        d,
        the_lock["production_version"],
    )
    return {
        "production_version": the_lock["production_version"],
        "molecules": len(recs),
        "species_rows": len(species),
        "reaction_rows": len(rxn_rows),
        "experimental_reactions": len(exp),
        "calculations": n_calc,
    }


def local_steps(cfg: Config) -> dict:
    """Collect every finished production job, then rebuild derived/."""
    return {
        "production (single points)": stage.collect(cfg),
        "analysis": run(cfg),
    }


def writers() -> dict:
    """SLURM stage -> job writer of the production branch."""
    return {stage.STAGE: stage.write_jobs}
