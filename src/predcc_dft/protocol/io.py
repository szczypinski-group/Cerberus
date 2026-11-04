"""File I/O of the data protocol (SOP 2.1): TOML, text and xyz files.

TOML files are written atomically (temp file + rename), so a half-written
record never exists; every text file is UTF-8 with LF line endings, so a
project prepared on Windows runs unchanged on the Linux cluster.
"""

from __future__ import annotations

import datetime as _dt
import functools
import os
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from predcc_dft import __version__

PACKAGE_DIR = Path(__file__).resolve().parent.parent

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
    path.write_text(
        text, encoding="utf-8", newline="\n"
    )  # LF: scripts run on Linux
    if executable:
        path.chmod(0o755)


def now() -> _dt.datetime:
    """Current UTC time, whole seconds."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0)


@functools.cache
def pipeline_commit() -> str:
    """Git commit of the installed pipeline code (SOP 2.6), or the version."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PACKAGE_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            encoding="utf-8",
            timeout=10,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(PACKAGE_DIR), "status", "--porcelain", "."],
            capture_output=True,
            encoding="utf-8",
            timeout=10,
            check=True,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return f"{out.stdout.strip()}{suffix}"
    except (OSError, subprocess.SubprocessError):
        return f"v{__version__}"


def read_xyz_frames(path: Path) -> list[tuple[str, Atoms]]:
    """Read a (multi-frame) xyz file -> [(comment, [(el, x, y, z)])]."""
    lines = path.read_text(encoding="utf-8").splitlines()
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
