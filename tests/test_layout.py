"""Study layout: branch data kept wherever the settings put it.

A study can keep its branches under one data folder, with measured data in a
folder of its own name, e.g.

    project/
      config/ inputs/ molecules/ jobs/ derived/      (trunk, study root)
      data/benchmark/{molecules, nmr, jobs, derived}
      data/production/{molecules, nmr, jobs, derived}
      data/ts/

`predcc init --data-dir data --experimental-dir nmr` writes that layout into
config/settings.toml ([project]); every module then reads and writes there.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from test_pipeline import (
    FAST,
    finish_freq,
    mock_crest_censo,
    to_txt,
)

from predcc_dft.analysis import errors
from predcc_dft.protocol import lock, project
from predcc_dft.protocol.config import load_config
from predcc_dft.protocol.io import read_toml
from predcc_dft.workflow import pipeline
from predcc_dft.workflow.cli import cmd_init, main

ALDEHYDES = "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n"
AMINES = "name,smiles\naniline,Nc1ccccc1\n"
BRANCH_FILES = (
    "data/benchmark/methods.toml",
    "data/benchmark/molecules.txt",
    "data/benchmark/nmr/_example.toml",
    "data/production/molecules.txt",
    "data/production/nmr/_example.toml",
    "data/ts/molecules.txt",
)


def _files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    }


def test_init_data_layout(tmp_path: Path) -> None:
    """--data-dir / --experimental-dir: branches under data/, nmr/ folders."""
    root = tmp_path / "project"
    assert (
        main(
            [
                "init",
                str(root),
                "--data-dir",
                "data",
                "--experimental-dir",
                "nmr",
            ]
        )
        == 0
    )
    files = _files(root)
    assert set(BRANCH_FILES) <= files
    assert not any(
        f.split("/")[0] in ("benchmark", "production", "ts") for f in files
    )
    proj = tomllib.loads((root / "config" / "settings.toml").read_text())[
        "project"
    ]
    assert proj == {
        "benchmark_dir": "data/benchmark",
        "production_dir": "data/production",
        "ts_dir": "data/ts",
        "experimental_dir": "nmr",
    }
    cfg = load_config(root=root)
    assert cfg.benchmark_dir == root / "data" / "benchmark"
    assert cfg.experimental_dir("production") == (
        root / "data" / "production" / "nmr"
    )


def test_existing_settings_layout_is_followed(tmp_path: Path) -> None:
    """A settings.toml written first decides where init puts the branches."""
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.toml").write_text(
        '[project]\nbenchmark_dir = "data/benchmark"\n'
        'production_dir = "data/production"\nts_dir = "data/ts"\n'
        'experimental_dir = "nmr"\n'
    )
    assert cmd_init(root, example=False) == 0
    assert set(BRANCH_FILES) <= _files(root)


def test_default_layout_unchanged(tmp_path: Path) -> None:
    """Without the options, init keeps benchmark/ ... at the study root."""
    root = tmp_path / "project"
    assert cmd_init(root, example=False) == 0
    files = _files(root)
    assert "benchmark/experimental/_example.toml" in files
    assert "production/molecules.txt" in files
    assert not any(f.startswith("data/") for f in files)


def test_branches_run_in_data_layout(tmp_path: Path) -> None:
    """Benchmark and production read and write only under data/."""
    root = tmp_path / "project"
    cmd_init(root, example=False, data_dir="data", experimental_dir="nmr")
    (root / "inputs" / "aldehydes.txt").write_text(to_txt(ALDEHYDES))
    (root / "inputs" / "amines.txt").write_text(to_txt(AMINES))
    with (root / "config" / "settings.toml").open("a") as fh:
        fh.write(FAST)
    cfg = load_config(root=root)
    pipeline.advance(cfg)
    mock_crest_censo(root, cfg)
    pipeline.advance(cfg)
    for a in sorted(root.glob("molecules/*/frequencies/*/attempt1")):
        finish_freq(a, [25.0, 50.0])
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["check (step 7)"]["collapsed"] == len(project.molecules(cfg))
    assert rep["jobs written"]["reference"].split()[0] != "0"
    key = next(
        r["compound"]
        for r in project.molecules(cfg)
        if r["name"] == "benzaldehyde"
    )
    bmol = root / "data" / "benchmark" / "molecules" / key
    assert (bmol / "reference" / "ref_x" / "input.inp").exists()
    assert (root / "data" / "benchmark" / "jobs" / "reference").is_dir()
    assert (root / "data" / "benchmark" / "derived").is_dir()

    lock.write(cfg, lock.build(cfg, "r2scan3c", "smd_mecn", "1.0"))
    assert (root / "data" / "production" / "production.toml").exists()
    pipeline.advance(cfg, chain="production")
    pmol = root / "data" / "production" / "molecules" / key / "v1.0"
    assert (pmol / "energy" / "input.inp").exists()

    top = {p.name for p in root.iterdir()}
    assert not {"benchmark", "production", "ts"} & top

    # Measured data: *.toml in nmr/ are read, raw data beside them ignored
    rxn = read_toml(cfg.reactions_file)["condensation"][0]["key"]
    nmr = cfg.experimental_dir("benchmark")
    (nmr / f"{rxn}.toml").write_text(
        f'reaction = "{rxn}"\nlog_k = 1.2\nlog_k_uncertainty = 0.1\n'
        'k_units = "dimensionless"\nwater_included = true\n'
    )
    raw = nmr / "raw" / "2026-11-18_benzaldehyde_aniline" / "1"
    raw.mkdir(parents=True)
    (raw / "fid").write_bytes(b"\x00\x01")
    (raw / "acqus").write_text("##TITLE= raw spectrum\n")
    exp = errors.experimental(nmr, read_toml(cfg.reactions_file))
    assert list(exp) == [rxn]
    assert exp[rxn]["log_k"] == 1.2
