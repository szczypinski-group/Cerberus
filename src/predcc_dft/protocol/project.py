"""Project registry and hand-over files between SOP levels (SOP 2.4).

The levels never import each other: a later level reads what an earlier one
wrote, through the readers here.

    molecules.toml                      species/    -> every later level
    molecules/<key>/structure.toml      structures/ -> energies/, analysis/
    molecules/<key>/reference/reference.toml
    molecules/<key>/benchmark/<method>/record.toml  energies/ -> analysis/
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml

ROLES = ("aldehyde", "amine", "imine", "water")


def molecules(cfg: Config) -> list[dict]:
    """All molecule records from molecules.toml, in a stable order."""
    data = read_toml(cfg.molecules_file)
    out = []
    for role in ROLES:
        out += data.get(role, [])
    return out


def mol_dir(cfg: Config, rec: dict) -> Path:
    """Folder of one molecule."""
    return cfg.root / rec["directory"]


def structure(cfg: Config, rec: dict) -> dict | None:
    """structure.toml of a collapsed molecule (step 7), else None."""
    path = mol_dir(cfg, rec) / "structure.toml"
    if not path.exists():
        return None
    data = read_toml(path)
    return data if data.get("status") == "collapsed" else None


def reference_data(cfg: Config, rec: dict) -> dict | None:
    """reference.toml of one molecule (steps 8 and 10), if collected."""
    path = mol_dir(cfg, rec) / "reference" / "reference.toml"
    return read_toml(path) if path.exists() else None


def benchmark_energies(cfg: Config, rec: dict) -> dict[str, float]:
    """Completed benchmark electronic energies (Eh) of a molecule (step 9)."""
    out = {}
    mdir = mol_dir(cfg, rec)
    for label in cfg.methods.get("methods", {}):
        path = mdir / "benchmark" / label / "record.toml"
        if not path.exists():
            continue
        data = read_toml(path)
        if data["calculation"]["status"] == "completed":
            out[label] = data["energy"]["electronic_hartree"]
    return out
