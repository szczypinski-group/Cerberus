"""File helpers: TOML read/atomic write, xyz files, configuration loading.

A project directory (any folder, default: the current one) holds

    config/settings.toml   optional overrides of the packaged defaults
    config/methods.toml    optional replacement of the packaged method matrix
    inputs/                aldehyde and amine lists
    experimental/          measured equilibria (SOP 2.5), optional

and everything the pipeline writes (molecules/, jobs/, derived/, ...).
"""

from __future__ import annotations

import copy
import datetime as _dt
import functools
import os
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_SETTINGS = DATA_DIR / "settings.toml"
DEFAULT_METHODS = DATA_DIR / "methods.toml"

Atoms = list[tuple[str, float, float, float]]


def read_toml(path: Path) -> dict:
    """Read a TOML file."""
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _clean(obj: Any) -> Any:  # noqa: ANN401
    """Drop None values (TOML has no null) recursively."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj if v is not None]
    return obj


def write_toml(path: Path, data: dict) -> None:
    """Write TOML atomically: temp file in the same folder, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(_clean(data), fh, multiline_strings=True)
        Path(tmp).replace(path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def write_text(path: Path, text: str, executable: bool = False) -> None:
    """Write a text file, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, newline="\n")  # LF: scripts run on Linux
    if executable:
        path.chmod(0o755)


def now() -> _dt.datetime:
    """Current UTC time, whole seconds."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0)


@functools.cache
def pipeline_commit() -> str:
    """Git commit of the installed pipeline code (SOP 2.6), or the version."""
    from predcc_dft import __version__

    try:
        out = subprocess.run(
            ["git", "-C", str(PACKAGE_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(PACKAGE_DIR), "status", "--porcelain", "."],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return f"{out.stdout.strip()}{suffix}"
    except (OSError, subprocess.SubprocessError):
        return f"v{__version__}"


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
    from predcc_dft.method_matrix import expand

    return Config(data, root, expand(methods))


# ------------------------------------------------------------------ xyz files


def read_xyz_frames(path: Path) -> list[tuple[str, Atoms]]:
    """Read a (multi-frame) xyz file -> [(comment, [(el, x, y, z)])]."""
    lines = path.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        atoms = []
        for line in lines[i + 2 : i + 2 + n]:
            parts = line.split()
            atoms.append(
                (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
            )
        frames.append((comment.strip(), atoms))
        i += 2 + n
    return frames


def format_xyz(atoms: Atoms, comment: str = "") -> str:
    """Format atoms as an xyz block."""
    lines = [str(len(atoms)), comment]
    lines += [
        f"{el:<2s} {x:16.10f} {y:16.10f} {z:16.10f}" for el, x, y, z in atoms
    ]
    return "\n".join(lines) + "\n"
