"""SOP 5.1 step 11, 2.7, 8 and 8.1: from records to tables and a lock draft.

  assembly     species G and reaction dG/ddG/log K
  errors       experimental K; errors vs DLPNO-CCSD(T)/CBS and experiment
  ranking      MAE, SD, ... per method, linkage class and level
  production   production-lock draft (production_candidate.toml)


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
from pathlib import Path

from predcc_dft.analysis import assembly, errors, production, ranking
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml, write_toml


def _fmt(key: str, v: object) -> object:
    """CSV cell: full precision for hartree energies, 6 decimals otherwise."""
    if isinstance(v, float):
        return f"{v:.10f}" if key.endswith("_hartree") else f"{v:.6f}"
    return "" if v is None else v


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows (dicts with equal keys) as CSV."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: _fmt(k, v) for k, v in r.items()} for r in rows)


def run(cfg: Config) -> dict:
    """Rebuild every derived table from the records."""
    reactions = read_toml(cfg.reactions_file)
    species = assembly.species_energies(cfg)
    rxn_rows = assembly.reaction_energies(cfg, species, reactions)
    exp = errors.experimental(cfg, reactions)
    err_rows = errors.errors(cfg, rxn_rows, exp)
    rank = ranking.ranking(cfg, err_rows)
    ranking.mark_complete(rank)
    d = cfg.derived_dir
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "species_energies.csv", list(species.values()))
    _write_csv(d / "reaction_energies.csv", rxn_rows)
    _write_csv(d / "errors.csv", err_rows)
    _write_csv(d / "method_ranking.csv", rank)
    _write_csv(d / "level_summary.csv", ranking.level_summary(rank))
    _write_csv(d / "errors_wide.csv", errors.errors_wide(cfg, err_rows))
    cand = production.candidate(cfg, rank, rxn_rows)
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
