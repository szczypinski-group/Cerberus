"""production/production.toml: the locked production protocol (SOP 2.8).

The only link between the two branches. It is written by `predcc promote`
from benchmark/methods.toml (and, by default, the benchmark's
production_candidate.toml) and is self-contained: the production branch reads
the ORCA keyword lines from here, never from the benchmark.

A version is frozen: rewriting an existing version with different content is
refused, so results of different protocols are never mixed (SOP 12.1). A new
choice of method means a new production_version, stored beside the old one.
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, pipeline_commit, read_toml, write_toml
from predcc_dft.protocol.methods import method_inputs, solvation_models

META = ("functional", "basis", "family", "zeta", "dispersion")
FROZEN_KEYS = ("production_version", "energy", "solvation", "thermo")


class LockError(ValueError):
    """Missing, incomplete or conflicting production lock."""


def path(cfg: Config) -> Path:
    """production/production.toml."""
    return cfg.production_dir / "production.toml"


def build(
    cfg: Config,
    method: str,
    solvation: str,
    version: str,
    evidence: dict | None = None,
) -> dict:
    """Lock for one method + solvation model of benchmark/methods.toml."""
    table = cfg.methods.get("methods", {})
    if method not in table:
        raise LockError(f"method {method!r} not in methods.toml")
    models = solvation_models(cfg.methods)
    if solvation not in models:
        raise LockError(
            f"solvation {solvation!r} not in methods.toml "
            f"(defined: {', '.join(models)})"
        )
    keywords, blocks = method_inputs(cfg.methods, method)
    m, s = table[method], models[solvation]
    e = cfg["energies"]
    return {
        "schema_version": SCHEMA_VERSION,
        "production_version": str(version),
        "selected": now().date(),
        "pipeline": pipeline_commit(),
        "source": "benchmark candidate" if evidence else "chosen by hand",
        "geometry": {
            "method": "r2scan3c",
            "solvation": "cpcm_mecn",
            "source": "trunk best.xyz (steps 1-7)",
        },
        "energy": {
            "method": method,
            **{k: m.get(k, "") for k in META},
            "orca_keywords": keywords,
            "orca_blocks": blocks,
        },
        "solvation": {
            "label": solvation,
            "model": s.get("model", ""),
            "solvent": s.get("solvent", ""),
            "level_keywords": e["solvation_keywords"],
            "orca_keywords": s.get("orca_keywords", ""),
            "orca_blocks": s.get("orca_blocks", ""),
            "parse": s.get("parse", ""),
        },
        "thermo": {
            "method": "r2scan3c",
            "temperature_k": e["temperature_k"],
            "standard_state_correction": e.get(
                "standard_state_correction", True
            ),
        },
        "selection": evidence or {},
    }


def read(cfg: Config) -> dict:
    """The lock, checked for the fields the production branch needs."""
    p = path(cfg)
    if not p.exists():
        raise LockError(
            f"{p} not found: run `predcc promote` (benchmark candidate) or "
            "`predcc promote --method <label> --solvation <label>`"
        )
    lock = read_toml(p)
    missing = [
        f"{section}.{key}"
        for section, keys in (
            ("energy", ("method", "orca_keywords")),
            ("solvation", ("label", "level_keywords", "orca_keywords")),
            ("thermo", ("temperature_k",)),
        )
        for key in keys
        if key not in lock.get(section, {})
    ]
    if "production_version" not in lock:
        missing.append("production_version")
    if missing:
        raise LockError(f"{p}: missing {', '.join(missing)}")
    return lock


def version_dir(lock: dict) -> str:
    """Per-molecule folder of one production version: v<version>."""
    return f"v{lock['production_version']}"


def write(cfg: Config, lock: dict, force: bool = False) -> Path:
    """Write the lock; refuse to change an existing version's protocol."""
    p = path(cfg)
    archive = cfg.production_dir / "locks" / f"{version_dir(lock)}.toml"
    for old_path in (p, archive):
        if not old_path.exists() or force:
            continue
        old = read_toml(old_path)
        if old.get("production_version") != lock["production_version"]:
            continue
        changed = [k for k in FROZEN_KEYS if old.get(k) != lock.get(k)]
        if changed:
            raise LockError(
                f"production_version {lock['production_version']} is "
                f"already locked with a different {', '.join(changed)}; "
                "use a new --version"
            )
    write_toml(p, lock)
    write_toml(archive, lock)  # every version kept, for provenance
    return p
