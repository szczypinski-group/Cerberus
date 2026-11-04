"""Free-energy terms taken from the trunk, identical for every method.

G(species) = E(method) + G_corr + dG_solv + standard state + G_ens - G_best,
where G_corr (r2SCAN-3c thermal correction) and the ensemble correction come
from the trunk's structure.toml (steps 6-7) and are reused by both branches.
"""

from __future__ import annotations

import math

from predcc_dft.protocol import constants
from predcc_dft.protocol.config import Config


def standard_state_kcal(temperature: float) -> float:
    """1 atm -> 1 mol/L: RT ln(RT/p) with RT/p in L/mol."""
    return (
        constants.R_KCAL * temperature * math.log(0.0820573661 * temperature)
    )


def species_terms(cfg: Config, struct: dict) -> dict:
    """Method-independent terms of one species from its structure.toml."""
    e = cfg["energies"]
    temperature = e["temperature_k"]
    return {
        "temperature_k": temperature,
        "g_corr_hartree": struct["best"]["g_corr_hartree"],
        "g_corr_source": "trunk step 6 r2SCAN-3c/CPCM freq (reused)",
        "standard_state_kcal_mol": (
            standard_state_kcal(temperature)
            if e.get("standard_state_correction", True)
            else 0.0
        ),
        "g_ensemble_correction_kcal_mol": struct.get(
            "g_ensemble_correction_kcal_mol", 0.0
        ),
    }
