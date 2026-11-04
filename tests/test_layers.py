"""Architecture test: imports only go downward, and the branches are insular.

    workflow  ->  benchmark | production        (the two branches)
              ->  energies | analysis           (shared by both branches)
              ->  species | structures          (the trunk, steps 1-7)
              ->  backends | chemistry | protocol

The trunk levels never import each other or a branch; a later level reads
the files an earlier one wrote (through predcc_dft.protocol.project). The
benchmark and production branches never import each other.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "predcc_dft"

FOUNDATION = {"protocol", "chemistry", "backends"}
ALLOWED = {
    "protocol": {"protocol"},
    "chemistry": {"chemistry"},
    "backends": {"backends", "protocol"},
    "species": {"species"} | FOUNDATION,
    "structures": {"structures"} | FOUNDATION,
    "energies": {"energies"} | FOUNDATION,
    "analysis": {"analysis", "protocol"},
    "benchmark": {"benchmark", "energies", "analysis"} | FOUNDATION,
    "production": {"production", "energies", "analysis"} | FOUNDATION,
    "workflow": {
        "workflow",
        "species",
        "structures",
        "energies",
        "analysis",
        "benchmark",
        "production",
    }
    | FOUNDATION,
}


def _layer(module: str) -> str | None:
    """Layer of a module inside predcc_dft (None = the package root)."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[1] in ALLOWED else None


def _imports(path: Path) -> set[str]:
    """Every predcc_dft module imported anywhere in a file."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("predcc_dft"):
                found |= {f"{node.module}.{a.name}" for a in node.names}
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {
                a.name for a in node.names if a.name.startswith("predcc_dft")
            }
    return found


def _modules() -> list[Path]:
    return sorted(
        p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts
    )


@pytest.mark.parametrize(
    "path", _modules(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_imports_follow_layers(path: Path) -> None:
    """No module imports a layer it may not depend on."""
    rel = path.relative_to(PACKAGE)
    if len(rel.parts) == 1:  # predcc_dft/__init__.py, __main__.py
        return
    layer = rel.parts[0]
    assert layer in ALLOWED, f"{rel}: unknown layer {layer!r}"
    for module in _imports(path):
        target = _layer(module)
        if target is None:  # predcc_dft root: SCHEMA_VERSION, __version__
            continue
        assert target in ALLOWED[layer], (
            f"{rel} imports {module}: {layer} may only use "
            f"{sorted(ALLOWED[layer])}"
        )


def test_every_module_is_in_a_layer() -> None:
    """New code goes into a layer, not the package root."""
    root = {p.name for p in PACKAGE.glob("*.py")}
    assert root == {"__init__.py", "__main__.py"}
