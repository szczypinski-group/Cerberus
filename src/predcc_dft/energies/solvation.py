"""SOP 5.1 step 10, 6.2: solvation free energies on each best geometry.

Per molecule (molecules/<key>/reference/):
  solv_gas          solvation level ([reference] solvation_keywords), gas phase
  solv_<label>      same level in every [solvation.<label>] of methods.toml
dG_solv = E(model) - E(gas), or the value ORCA prints (openCOSMO-RS). The jobs
run in the "reference" SLURM stage, next to the DLPNO-CCSD(T) reference.
"""

from __future__ import annotations

from predcc_dft.protocol import constants, methods
from predcc_dft.protocol.config import Config


def job_inputs(cfg: Config) -> dict[str, tuple[str, str]]:
    """Task name -> (keywords, blocks) for every solvation job."""
    level = cfg["reference"]["solvation_keywords"]
    jobs = {"solv_gas": (level, "")}
    for label, model in methods.solvation_models(cfg.methods).items():
        jobs[f"solv_{label}"] = (
            f"{level} {model.get('orca_keywords', '')}".strip(),
            model.get("orca_blocks", ""),
        )
    return jobs


def dg_solv(cfg: Config, label: str, res: dict) -> float | None:
    """dG_solv (kcal/mol) of one model, from its job and the gas job."""
    model = methods.solvation_models(cfg.methods)[label]
    if model.get("parse") == "cosmors":
        val = res[f"solv_{label}"].get("dg_solv_cosmors")
        return None if val is None else val * constants.HARTREE_TO_KCAL
    e_solv = res[f"solv_{label}"].get("final_energy")
    e_gas = res["solv_gas"].get("final_energy")
    if e_solv is None or e_gas is None:
        return None
    return (e_solv - e_gas) * constants.HARTREE_TO_KCAL
