"""Benchmark analysis (SOP 2.7, 8, 8.1): tables, ranking, production draft.

Rebuilt every round from the branch's records into benchmark/derived/:

  species_energies.csv    compound x method x solvation: E, G terms, G
  reaction_energies.csv   reaction x method x solvation: dE, dG, ddG, log K
  errors.csv              ... x reference (DLPNO-CCSD(T)/CBS, experiment)
  errors_wide.csv         reaction x method, gas-phase errors vs DLPNO
  method_ranking.csv      MAE, SD, ... per method, linkage class, level
  level_summary.csv       best / worst / mean MAE per level of theory
  calculations.csv        every benchmark record.toml
  production_candidate.toml   input for `predcc promote`
"""

from __future__ import annotations

from predcc_dft.analysis import assembly, errors, tables
from predcc_dft.benchmark import candidate, grid, ranking, reference
from predcc_dft.protocol import constants, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml, write_toml
from predcc_dft.protocol.methods import solvation_models

BRANCH = "benchmark"


def species_energies(cfg: Config) -> dict[tuple[str, str, str], dict]:
    """(compound, method, solvation) -> species row, DLPNO included."""
    ensemble = cfg["analysis"].get("ensemble_correction", True)
    out = {}
    for rec in project.selected(cfg, BRANCH):
        data = reference.data(cfg, rec)
        if data is None:
            continue
        electronic = dict(grid.energies(cfg, rec))
        ref = data.get("reference", {})
        if "e_cbs_hartree" in ref:
            electronic[constants.REFERENCE_LABEL] = ref["e_cbs_hartree"]
        t1s = [ref[k] for k in ("t1_x", "t1_y") if ref.get(k) is not None]
        solv = {"gas": 0.0}
        for label in solvation_models(cfg.methods):
            entry = data["solvation"].get(label)
            if isinstance(entry, dict):
                solv[label] = entry["dg_solv_kcal_mol"]
        for method, e in electronic.items():
            meta = grid.method_meta(cfg, method)
            for model, dg in solv.items():
                out[(rec["compound"], method, model)] = assembly.species_row(
                    rec,
                    data["isomer"],
                    method,
                    meta,
                    model,
                    e,
                    data["thermo"],
                    dg,
                    max(t1s) if t1s else None,
                    ensemble,
                )
    return out


def run(cfg: Config) -> dict:
    """Rebuild benchmark/derived/ from the benchmark records."""
    recs = project.selected(cfg, BRANCH)
    project.write_index(cfg, BRANCH, recs)
    reactions = read_toml(cfg.reactions_file)
    species = species_energies(cfg)
    rxn_rows = assembly.reaction_energies(cfg, species, reactions)
    exp = errors.experimental(cfg.benchmark_dir / "experimental", reactions)
    ref_label = cfg["analysis"].get("reference", constants.REFERENCE_LABEL)
    err_rows = errors.errors(cfg, rxn_rows, exp, ref_label)
    rank = ranking.ranking(cfg, err_rows)
    ranking.mark_complete(rank)
    d = cfg.benchmark_dir / "derived"
    d.mkdir(parents=True, exist_ok=True)
    tables.write_csv(d / "species_energies.csv", list(species.values()))
    tables.write_csv(d / "reaction_energies.csv", rxn_rows)
    tables.write_csv(d / "errors.csv", err_rows)
    tables.write_csv(d / "method_ranking.csv", rank)
    tables.write_csv(d / "level_summary.csv", ranking.level_summary(rank))
    tables.write_csv(
        d / "errors_wide.csv", errors.errors_wide(ref_label, err_rows)
    )
    n_calc = records.collect_csv(cfg, cfg.benchmark_dir / "molecules", d)
    cand = candidate.candidate(cfg, rank, rxn_rows)
    if cand is not None:
        write_toml(d / "production_candidate.toml", cand)
    return {
        "molecules": len(recs),
        "species_rows": len(species),
        "reaction_rows": len(rxn_rows),
        "experimental_reactions": len(exp),
        "calculations": n_calc,
        "best_vs_reference": (
            f"{cand['energy']['method']} "
            f"(MAE {cand['selection']['mae_kcal_mol']} kcal/mol)"
            if cand
            else "-"
        ),
    }


def local_steps(cfg: Config) -> dict:
    """Collect every finished benchmark job, then rebuild derived/."""
    return {
        "reference (steps 8, 10)": reference.collect(cfg),
        "methods (step 9)": grid.collect(cfg),
        "analysis": run(cfg),
    }


def writers() -> dict:
    """SLURM stage -> job writer of the benchmark branch."""
    return {
        reference.STAGE: reference.write_jobs,
        grid.STAGE: grid.write_jobs,
    }
