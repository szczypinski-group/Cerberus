"""ORCA input writing and output parsing."""

from __future__ import annotations

import re
from pathlib import Path

HARTREE_TO_KCAL = 627.509474

_FLOAT = r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)"
_PATTERNS = {
    "final_energy": re.compile(rf"FINAL SINGLE POINT ENERGY\s+{_FLOAT}"),
    "g_minus_e": re.compile(rf"G-E\(el\)\s+\.\.\.\s+{_FLOAT}\s+Eh"),
    "zpe": re.compile(rf"Zero point energy\s+\.\.\.\s+{_FLOAT}\s+Eh"),
    "enthalpy": re.compile(rf"Total enthalpy\s+\.\.\.\s+{_FLOAT}\s+Eh"),
    "gibbs": re.compile(rf"Final Gibbs free energy\s+\.\.\.\s+{_FLOAT}\s+Eh"),
    "temperature": re.compile(rf"Temperature\s+\.\.\.\s+{_FLOAT}\s+K"),
    "e_ref": re.compile(rf"^\s*E\(0\)\s+\.\.\.\s+{_FLOAT}", re.M),
    "e_corr": re.compile(rf"Final correlation energy\s+\.\.\.\s+{_FLOAT}"),
    "e_ccsd_t": re.compile(rf"E\(CCSD\(T\)\)\s+\.\.\.\s+{_FLOAT}"),
    "t1": re.compile(rf"T1 diagnostic\s+\.\.\.\s+{_FLOAT}"),
    # openCOSMO-RS (ORCA 6): "Free energy of solvation (dGsolv) : x Eh"
    "dg_solv_cosmors": re.compile(
        rf"Free energy of solvation \(dGsolv\)\s*[:.]+\s*{_FLOAT}\s*Eh"
    ),
}
_ORBITALS = re.compile(
    r"^\s*\d+\s+(\d\.\d+)\s+-?\d+\.\d+\s+(-?\d+\.\d+)\s*$", re.M
)
_FREQ = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", re.M)
_VERSION = re.compile(r"Program Version\s+(\S+)")


def write_input(
    directory: Path,
    keywords: str,
    nprocs: int,
    maxcore_mb: int,
    geometry: str,
    charge: int = 0,
    multiplicity: int = 1,
    blocks: str = "",
) -> Path:
    """Write input.inp + geom.xyz into directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "geom.xyz").write_text(geometry, newline="\n")
    text = [
        f"! {keywords}",
        f"%pal nprocs {nprocs} end",
        f"%maxcore {maxcore_mb}",
    ]
    if blocks.strip():
        text.append(blocks.strip())
    text.append(f"* xyzfile {charge} {multiplicity} geom.xyz")
    path = directory / "input.inp"
    path.write_text("\n".join(text) + "\n", newline="\n")
    return path


def _last(pattern: re.Pattern, text: str) -> float | None:
    hits = pattern.findall(text)
    return float(hits[-1]) if hits else None


def _frontier(text: str) -> dict:
    """HOMO and LUMO (eV) from the last ORBITAL ENERGIES block."""
    start = text.rfind("ORBITAL ENERGIES")
    if start < 0:
        return {}
    end = text.find("MULLIKEN", start)
    rows = _ORBITALS.findall(text[start : end if end > 0 else None])
    occ = [float(ev) for o, ev in rows if float(o) > 0.0]
    virt = [float(ev) for o, ev in rows if float(o) == 0.0]
    out = {}
    if occ:
        out["homo_ev"] = occ[-1]
    if virt:
        out["lumo_ev"] = virt[0]
    return out


def parse_output(path: Path) -> dict:
    """Parse an ORCA output. Missing quantities are absent from the dict."""
    if not path.exists():
        return {"exists": False}
    text = path.read_text(errors="replace")
    out: dict = {
        "exists": True,
        "terminated_normally": "ORCA TERMINATED NORMALLY" in text,
        "opt_converged": "THE OPTIMIZATION HAS CONVERGED" in text,
    }
    version = _VERSION.search(text)
    if version:
        out["orca_version"] = version.group(1)
    for key, pattern in _PATTERNS.items():
        val = _last(pattern, text)
        if val is not None:
            out[key] = val
    out.update(_frontier(text))
    # Frequencies: take the last VIBRATIONAL FREQUENCIES block.
    block_start = text.rfind("VIBRATIONAL FREQUENCIES")
    if block_start >= 0:
        block_end = text.find("NORMAL MODES", block_start)
        block = text[block_start : block_end if block_end > 0 else None]
        freqs = [float(f) for f in _FREQ.findall(block)]
        nonzero = [f for f in freqs if abs(f) > 1e-6]
        out["frequencies_cm1"] = nonzero
        out["n_imaginary"] = sum(1 for f in nonzero if f < 0)
        out["lowest_freq_cm1"] = min(nonzero) if nonzero else None
    return out
