"""One TOML per molecule in every module, and index/<key>.toml linking them.

Runs the mocked trunk, then the benchmark, production and TS chains, and
checks that each module wrote its own file per molecule (trunk.toml,
benchmark.toml, production.toml, ts.toml) with paths that exist, and that
index/<key>.toml points to all of them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from test_branches import finish_production, trunk_done
from test_pipeline import finish_single_points

from predcc_dft.protocol import lock, project
from predcc_dft.protocol.config import load_config
from predcc_dft.workflow import pipeline
from predcc_dft.workflow.cli import main


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _paths(data: dict) -> list[str]:
    """Every value of a [files] table (str or list of str)."""
    out: list[str] = []
    for v in data.get("files", {}).values():
        out += v if isinstance(v, list) else [v]
    return out


def test_module_files_and_index(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Each module's file per molecule, and the root index of each molecule."""
    root, cfg = trunk_done(tmp_path)
    pipeline.advance(cfg, chain="benchmark")
    finish_single_points(root, cfg)
    pipeline.advance(cfg, chain="benchmark")
    lock.write(cfg, lock.build(cfg, "r2scan3c", "smd_mecn", "1.0"))
    pipeline.advance(cfg, chain="production")
    finish_production(root)
    pipeline.advance(cfg, chain="production")
    (root / "ts" / "molecules.txt").write_text("benzaldehyde\naniline\n")
    with (root / "config" / "settings.toml").open("a") as fh:
        fh.write('\n[ts]\nshuttles = ["none"]\n')
    cfg = load_config(root=root)
    rep = pipeline.advance(cfg, chain="ts")
    assert rep["index/<molecule>.toml files"] > 0

    recs = project.molecules(cfg)
    ald = next(r for r in recs if r["name"] == "benzaldehyde")
    key = ald["compound"]
    trunk = _toml(root / "molecules" / key / "trunk.toml")
    bench = _toml(root / "benchmark" / "molecules" / key / "benchmark.toml")
    prod = _toml(root / "production" / "molecules" / key / "production.toml")
    ts = _toml(root / "ts" / "species" / key / "ts.toml")
    for data, module in (
        (trunk, "trunk"),
        (bench, "benchmark"),
        (prod, "production"),
        (ts, "ts"),
    ):
        assert data["module"] == module
        assert data["compound"] == key
        assert data["name"] == "benzaldehyde"
        assert data["role"] == "aldehyde"
    for p in _paths(trunk):  # trunk: every path it lists exists
        assert (root / p).exists(), p
    assert trunk["status"]["structure (step 7)"] == "collapsed"
    assert trunk["best"]["g_corr_hartree"] == pytest.approx(0.1)
    for a in trunk["frequencies"]:
        assert (root / a["output"]).exists()
    assert bench["status"]["reference"] == "done"
    assert bench["reference"]["e_cbs_hartree"] < 0
    for job in [*bench["reference_jobs"].values(), *bench["methods"].values()]:
        assert job["status"] == "completed"
        assert (root / job["record"]).exists()
    v1 = prod["versions"]["v1.0"]
    assert v1["status"] == "complete" and "g_hartree" in v1
    for job in v1["jobs"].values():
        assert (root / job["output"]).exists()
    assert ts["in_reactions"][0]["side"] == "reactant"

    idx = _toml(root / "index" / f"{key}.toml")
    assert idx["name"] == "benzaldehyde"
    mods = idx["modules"]
    assert set(mods) == {"trunk", "benchmark", "production", "ts"}
    for name, entry in mods.items():
        assert entry["present"], name
        assert (root / entry["file"]).exists(), name
    assert mods["trunk"]["status"] == "collapsed"
    assert mods["benchmark"]["reference"] == "done"
    assert mods["production"]["versions"] == {"v1.0": "complete"}
    assert mods["ts"]["selected"]
    assert mods["ts"]["reactions"][0]["side"] == "reactant"
    hemi = next((root / "index").glob("hemiaminal_*.toml"))
    assert set(_toml(hemi)["modules"]) == {"ts"}  # TS-only molecule

    # trunk molecules outside the TS selection: no ts.toml, still indexed
    benzyl = next(r for r in recs if r["name"] == "benzylamine")
    other = _toml(root / "index" / f"{benzyl['compound']}.toml")
    assert not other["modules"]["ts"]["present"]
    assert not other["modules"]["ts"]["selected"]

    capsys.readouterr()
    for query in ("benzaldehyde", "O=Cc1ccccc1", key):
        assert main(["--root", str(root), "show", query]) == 0
        out = capsys.readouterr().out
        assert f"# index/{key}.toml" in out
        assert 'name = "benzaldehyde"' in out
    assert main(["--root", str(root), "show", "no-such-molecule"]) == 1
