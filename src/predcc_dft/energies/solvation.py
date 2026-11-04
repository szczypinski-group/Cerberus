"""SOP 5.1 step 10, 6.2: solvation free energies on a trunk best geometry.

Two kinds of job per molecule, at the [energies] solvation_keywords level:
  solv_gas          gas phase
  solv_<label>      one per solvation model: every [solvation.<label>] of
                    benchmark/methods.toml, or the locked model in production
dG_solv = E(model) - E(gas), or the value ORCA prints (openCOSMO-RS).
"""

from __future__ import annotations

from predcc_dft.protocol import constants


def job_inputs(
    level: str, models: dict[str, dict]
) -> dict[str, tuple[str, str]]:
    """Task name -> (keywords, blocks) for the gas job and every model."""
    jobs = {"solv_gas": (level, "")}
    for label, model in models.items():
        jobs[f"solv_{label}"] = (
            f"{level} {model.get('orca_keywords', '')}".strip(),
            model.get("orca_blocks", ""),
        )
    return jobs


def dg_solv(label: str, model: dict, res: dict) -> float | None:
    """dG_solv (kcal/mol) of one model from parsed outputs {task: parsed}."""
    if model.get("parse") == "cosmors":
        val = res.get(f"solv_{label}", {}).get("dg_solv_cosmors")
        return None if val is None else val * constants.HARTREE_TO_KCAL
    e_solv = res.get(f"solv_{label}", {}).get("final_energy")
    e_gas = res.get("solv_gas", {}).get("final_energy")
    if e_solv is None or e_gas is None:
        return None
    return (e_solv - e_gas) * constants.HARTREE_TO_KCAL
