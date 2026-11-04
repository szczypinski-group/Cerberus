"""Configuration: packaged defaults <- config/settings.toml <- --config.

A project directory (any folder, default: the current one) holds

    config/settings.toml   optional overrides of the packaged defaults
    config/methods.toml    optional replacement of the packaged method matrix
    inputs/                aldehyde and amine lists
    experimental/          measured equilibria (SOP 2.5), optional

and everything the pipeline writes (molecules/, jobs/, derived/, ...).
"""

from __future__ import annotations

import copy
from pathlib import Path

from predcc_dft.protocol.io import PACKAGE_DIR, read_toml
from predcc_dft.protocol.methods import expand

DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_SETTINGS = DATA_DIR / "settings.toml"
DEFAULT_METHODS = DATA_DIR / "methods.toml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


class Config(dict):
    """Configuration dictionary plus resolved project paths."""

    def __init__(self, data: dict, root: Path, methods: dict) -> None:
        """Store settings, the project root and the method matrix."""
        super().__init__(data)
        self.root = root.resolve()
        self.methods = methods

    def path(self, rel: str) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.root / p

    def data_file(self, name: str) -> Path:
        """Project config/<name> if present, else the packaged default."""
        p = Path(name).expanduser()
        if p.is_absolute():
            return p
        local = self.root / "config" / p
        return local if local.exists() else DATA_DIR / p

    @property
    def molecules_dir(self) -> Path:
        """One folder per molecule (compound key)."""
        return self.path(self["project"]["molecules_dir"])

    @property
    def jobs_dir(self) -> Path:
        """SLURM scripts, manifests and logs per stage."""
        return self.path(self["project"]["jobs_dir"])

    @property
    def derived_dir(self) -> Path:
        """Generated tables; safe to delete and rebuild."""
        return self.path(self["project"]["derived_dir"])

    @property
    def experimental_dir(self) -> Path:
        """Measured equilibria, one TOML per reaction key (SOP 2.5)."""
        return self.path(
            self["project"].get("experimental_dir", "experimental")
        )

    @property
    def molecules_file(self) -> Path:
        """Every molecule of the project."""
        return self.root / "molecules.toml"

    @property
    def reactions_file(self) -> Path:
        """Condensation and exchange reactions."""
        return self.root / "reactions.toml"


def load_config(
    user_config: Path | None = None, root: Path | None = None
) -> Config:
    """Packaged defaults <- config/settings.toml <- user_config."""
    root = (root or Path.cwd()).resolve()
    data = read_toml(DEFAULT_SETTINGS)
    project_settings = root / "config" / "settings.toml"
    if project_settings.exists():
        data = _merge(data, read_toml(project_settings))
    if user_config is not None:
        data = _merge(data, read_toml(user_config))
    methods_file = root / "config" / "methods.toml"
    methods = read_toml(
        methods_file if methods_file.exists() else DEFAULT_METHODS
    )
    return Config(data, root, expand(methods))
