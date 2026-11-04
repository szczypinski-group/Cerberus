"""SOP 5.1 step 8: DLPNO-CCSD(T)/CBS reference energies.

Two gas-phase jobs per molecule, ref_x and ref_y (cardinal numbers X and Y,
[reference] in settings.toml), combined by a two-point CBS extrapolation.
Run by the benchmark branch only.
"""

from __future__ import annotations

import math

from predcc_dft.protocol import constants
from predcc_dft.protocol.config import Config


def job_inputs(cfg: Config) -> dict[str, tuple[str, str]]:
    """Task name -> (keywords, blocks) of the two DLPNO-CCSD(T) jobs."""
    r = cfg["reference"]
    return {"ref_x": (r["keywords_x"], ""), "ref_y": (r["keywords_y"], "")}


def cbs(
    cfg: Config, scf_x: float, corr_x: float, scf_y: float, corr_y: float
) -> tuple[float, float]:
    """Two-point CBS extrapolation of SCF and correlation energies."""
    r = cfg["reference"]
    x, y = r["cardinal_x"], r["cardinal_y"]
    a, b = r["scf_alpha"], r["corr_beta"]
    # E(n) = E_CBS + A exp(-a sqrt n)  ->  solve the two equations for E_CBS
    ex, ey = math.exp(a * math.sqrt(x)), math.exp(a * math.sqrt(y))
    scf = (scf_x * ex - scf_y * ey) / (ex - ey)
    corr = (x**b * corr_x - y**b * corr_y) / (x**b - y**b)
    return scf, corr


def summarize(cfg: Config, res_x: dict, res_y: dict) -> dict:
    """[reference] table of reference.toml from two completed DLPNO jobs."""
    scf_cbs, corr_cbs = cbs(
        cfg, res_x["e_ref"], res_x["e_corr"], res_y["e_ref"], res_y["e_corr"]
    )
    return {
        "label": constants.REFERENCE_LABEL,
        "scf_x_hartree": res_x["e_ref"],
        "corr_x_hartree": res_x["e_corr"],
        "scf_y_hartree": res_y["e_ref"],
        "corr_y_hartree": res_y["e_corr"],
        "scf_cbs_hartree": scf_cbs,
        "corr_cbs_hartree": corr_cbs,
        "e_cbs_hartree": scf_cbs + corr_cbs,
        "t1_x": res_x.get("t1"),
        "t1_y": res_y.get("t1"),
    }
