"""The two branches: production, promote, subsets, insularity, labels.

Uses the mocked trunk of test_pipeline (CREST/CENSO/ORCA outputs written by
hand), then drives the benchmark and production chains like the SLURM driver.
"""

from __future__ import annotations

import csv
import hashlib
import tomllib
from pathlib import Path

import pytest
from test_pipeline import (
    finish_freq,
    finish_single_points,
    make_project,
    mock_crest_censo,
    noise,
    orca_sp_output,
)

from predcc_dft.protocol import lock, project
from predcc_dft.protocol.config import load_config
from predcc_dft.protocol.io import read_xyz_frames
from predcc_dft.workflow import pipeline
from predcc_dft.workflow.cli import main

ALDEHYDES = (
    "name,smiles\nbenzaldehyde,O=Cc1ccccc1\nsalicylaldehyde,O=Cc1ccccc1O\n"
)
AMINES = "name,smiles\naniline,Nc1ccccc1\nbenzylamine,NCc1ccccc1\n"


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def trunk_done(tmp_path: Path) -> tuple[Path, object]:
    """Project whose trunk has finished: every molecule has best.xyz."""
    root = make_project(tmp_path, ALDEHYDES, AMINES)
    cfg = load_config(root=root)
    pipeline.advance(cfg)
    mock_crest_censo(root, cfg)
    pipeline.advance(cfg)
    for a in sorted(root.glob("molecules/*/frequencies/*/attempt1")):
        finish_freq(a, [25.0, 50.0])
    rep = pipeline.advance(cfg)
    assert rep["check (step 7)"]["collapsed"] == len(project.molecules(cfg))
    return root, cfg


def finish_production(root: Path, version: str = "1.0") -> int:
    """Fake every production output of one version; returns the count."""
    n = 0
    for vdir in sorted(root.glob(f"production/molecules/*/v{version}")):
        key = vdir.parent.name
        n_atoms = len(
            read_xyz_frames(root / "molecules" / key / "best.xyz")[0][1]
        )
        e = -40.0 * n_atoms + noise(key, 0.01)
        for task, val in (
            ("energy", e - 0.02 * n_atoms),
            ("solv_gas", e),
            ("solv_smd_mecn", e - 0.006),
        ):
            (vdir / task / "output.out").write_text(orca_sp_output(val))
            n += 1
    return n


def snapshot(folder: Path) -> dict[str, str]:
    """Relative path -> SHA-256 of every file below a folder."""
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# ------------------------------------------------------------------ production


def test_production_end_to_end(tmp_path: Path) -> None:  # noqa: PLR0915
    """Lock -> one method + solvation per molecule -> G, dG, log K tables."""
    root, cfg = trunk_done(tmp_path)
    n_mol = len(project.molecules(cfg))

    # No lock yet: clear instructions, nothing written
    with pytest.raises(lock.LockError, match="predcc promote"):
        pipeline.advance(cfg, chain="production")
    assert main(["--root", str(root), "production"]) == 2

    # Lock by hand from methods.toml
    assert (
        main(
            [
                "--root",
                str(root),
                "promote",
                "--method",
                "wb97mv_def2tzvp",
                "--solvation",
                "smd_mecn",
            ]
        )
        == 0
    )
    the_lock = _toml(root / "production" / "production.toml")
    assert the_lock["production_version"] == "1.0"
    assert the_lock["energy"]["orca_keywords"].startswith(
        "wB97M-V def2-TZVP TightSCF DefGrid3"
    )
    assert the_lock["solvation"]["label"] == "smd_mecn"
    assert (root / "production" / "locks" / "v1.0.toml").exists()

    # Round 1: three jobs per molecule, nothing else
    cfg = load_config(root=root)
    rep = pipeline.advance(cfg, chain="production")
    assert rep["jobs written"]["production"].startswith(f"{3 * n_mol} tasks")
    assert "reference" not in rep["jobs written"]  # no DLPNO in production
    assert not (root / "benchmark" / "molecules").exists()
    imine = next(r for r in project.molecules(cfg) if r["role"] == "imine")
    vdir = root / "production" / "molecules" / imine["compound"] / "v1.0"
    assert sorted(p.name for p in vdir.iterdir()) == [
        "energy",
        "solv_gas",
        "solv_smd_mecn",
    ]
    energy_inp = (vdir / "energy" / "input.inp").read_text(encoding="utf-8")
    assert "! wB97M-V def2-TZVP TightSCF DefGrid3" in energy_inp
    assert "SMDsolvent" in (vdir / "solv_smd_mecn" / "input.inp").read_text(
        encoding="utf-8"
    )
    # Same geometry as the trunk (and therefore as the benchmark)
    best = root / imine["directory"] / "best.xyz"
    assert (vdir / "energy" / "geom.xyz").read_text(
        encoding="utf-8"
    ) == best.read_text(encoding="utf-8")
    assert (root / "production" / "jobs" / "production").is_dir()

    # Round 2: results -> result.toml and derived/v1.0/
    assert finish_production(root) == 3 * n_mol
    rep = pipeline.advance(cfg, chain="production")
    assert rep["production (single points)"]["complete"] == n_mol
    assert rep["jobs written"]["production"].startswith("0 tasks")
    assert rep["status"]["production"]["complete"] == n_mol
    res = _toml(vdir / "result.toml")
    assert res["status"] == "complete" and res["production_version"] == "1.0"
    assert {res["compound"], res["role"], res["name"]} == {
        imine["compound"],
        "imine",
        imine["name"],
    }
    assert res["inchikey"].startswith(imine["compound"])
    assert res["g_hartree"] < res["energy"]["electronic_hartree"] + 1
    rec = _toml(vdir / "energy" / "record.toml")
    assert rec["calculation"]["branch"] == "production"
    assert rec["calculation"]["production_version"] == "1.0"
    assert rec["calculation"]["method"] == "wb97mv_def2tzvp"

    d = root / "production" / "derived" / "v1.0"
    species = list(csv.DictReader((d / "species_energies.csv").open()))
    assert len(species) == 2 * n_mol  # gas + the locked model
    assert {s["solvation"] for s in species} == {"gas", "smd_mecn"}
    rxn = list(csv.DictReader((d / "reaction_energies.csv").open()))
    assert rxn and all(r["log_k"] for r in rxn)
    calcs = list(csv.DictReader((d / "calculations.csv").open()))
    assert len(calcs) == 3 * n_mol
    assert {c["production_version"] for c in calcs} == {"1.0"}
    # Same columns as the benchmark tables
    with (d / "species_energies.csv").open() as fh:
        header = fh.readline().strip().split(",")
    assert header[:5] == ["compound", "name", "role", "isomer", "method"]
    assert not (d / "errors.csv").exists()  # no experiment supplied

    # A locked version is frozen
    with pytest.raises(lock.LockError, match="already locked"):
        lock.write(
            cfg, lock.build(cfg, "r2scan3c", "smd_mecn", "1.0"), force=False
        )
    # A new version lives beside the old one; v1.0 is untouched
    v10 = snapshot(root / "production" / "molecules")
    assert (
        main(
            [
                "--root",
                str(root),
                "promote",
                "--method",
                "r2scan3c",
                "--solvation",
                "cpcm_mecn",
                "--version",
                "1.1",
            ]
        )
        == 0
    )
    cfg = load_config(root=root)
    rep = pipeline.advance(cfg, chain="production")
    assert rep["jobs written"]["production"].startswith(f"{3 * n_mol} tasks")
    assert (vdir.parent / "v1.1" / "solv_cpcm_mecn").is_dir()
    after = snapshot(root / "production" / "molecules")
    assert {k: v for k, v in after.items() if k in v10} == v10
    assert sorted(
        p.name for p in (root / "production" / "locks").iterdir()
    ) == [
        "v1.0.toml",
        "v1.1.toml",
    ]


def test_promote_from_benchmark_candidate(tmp_path: Path) -> None:
    """`predcc promote` with no options takes the benchmark's candidate."""
    root, cfg = trunk_done(tmp_path)
    assert main(["--root", str(root), "promote"]) == 2  # no candidate yet
    pipeline.advance(cfg, chain="benchmark")
    finish_single_points(root, cfg)
    pipeline.advance(cfg, chain="benchmark")
    cand = _toml(root / "benchmark" / "derived" / "production_candidate.toml")
    assert main(["--root", str(root), "promote"]) == 0
    the_lock = _toml(root / "production" / "production.toml")
    assert the_lock["energy"]["method"] == cand["energy"]["method"]
    assert the_lock["solvation"]["label"] == cand["energy"]["solvation"]
    assert the_lock["source"] == "benchmark candidate"
    assert the_lock["selection"]["mae_kcal_mol"] == pytest.approx(
        cand["selection"]["mae_kcal_mol"]
    )


# ------------------------------------------------------------------ subsets


def test_benchmark_subset(tmp_path: Path) -> None:
    """benchmark/molecules.txt limits the benchmark, not the trunk."""
    root, cfg = trunk_done(tmp_path)
    mols = project.molecules(cfg)
    (root / "benchmark" / "molecules.txt").write_text(
        "# benchmark set\nbenzaldehyde\nNc1ccccc1\n"  # name + SMILES
    )
    chosen = project.selected(cfg, "benchmark")
    names = sorted(r["name"] for r in chosen)
    assert names == [
        "aniline",
        "benzaldehyde",
        "benzaldehyde + aniline (imine)",
        "water",
    ]
    assert len(project.selected(cfg, "production")) == len(mols)

    rep = pipeline.advance(cfg, chain="benchmark")
    n_ref = 3 + 2  # ref_x, ref_y, solv_gas + two solvation models
    assert rep["jobs written"]["reference"].startswith(f"{n_ref * 4} tasks")
    assert rep["jobs written"]["benchmark"].startswith(f"{3 * 4} tasks")
    assert sorted(
        p.name for p in (root / "benchmark" / "molecules").iterdir()
    ) == sorted(r["compound"] for r in chosen)
    index = _toml(root / "benchmark" / "molecules.toml")
    assert index["counts"] == {
        "aldehyde": 1,
        "amine": 1,
        "imine": 1,
        "water": 1,
    }
    assert index["selection"] == "benchmark/molecules.txt"
    status = pipeline.status.run(cfg)
    outside = [
        r
        for r in status
        if r["compound"] not in {c["compound"] for c in chosen}
    ]
    assert outside and all(r["bench_methods"] == "-" for r in outside)

    # An imine key alone brings its aldehyde and amine
    imine = next(
        r for r in mols if r["name"] == "salicylaldehyde + benzylamine (imine)"
    )
    (root / "benchmark" / "molecules.txt").write_text(imine["compound"] + "\n")
    names = sorted(r["name"] for r in project.selected(cfg, "benchmark"))
    assert names == [
        "benzylamine",
        "salicylaldehyde",
        "salicylaldehyde + benzylamine (imine)",
        "water",
    ]

    # A typo is an error, not a silently smaller benchmark
    (root / "benchmark" / "molecules.txt").write_text("benzaldehide\n")
    with pytest.raises(project.SelectionError, match="benzaldehide"):
        project.selected(cfg, "benchmark")
    assert main(["--root", str(root), "benchmark"]) == 2


# ------------------------------------------------------------------ insularity


def test_branches_are_insular(tmp_path: Path) -> None:
    """Each branch writes only inside its own folder; the trunk is shared."""
    root, cfg = trunk_done(tmp_path)
    trunk_files = set(snapshot(root / "molecules"))

    pipeline.advance(cfg, chain="benchmark")
    finish_single_points(root, cfg)
    pipeline.advance(cfg, chain="benchmark")
    assert not (root / "production" / "molecules").exists()
    bench = snapshot(root / "benchmark")

    lock.write(cfg, lock.build(cfg, "r2scan3c", "smd_mecn", "1.0"))
    pipeline.advance(cfg, chain="production")
    finish_production(root)
    pipeline.advance(cfg, chain="production")

    assert snapshot(root / "benchmark") == bench  # production touched nothing
    assert set(snapshot(root / "molecules")) == trunk_files  # no branch files
    for key_dir in (root / "molecules").iterdir():
        assert not {"methods", "reference", "benchmark", "v1.0"} & {
            p.name for p in key_dir.iterdir()
        }
    # One geometry for both branches
    imine = next(r for r in project.molecules(cfg) if r["role"] == "imine")
    key = imine["compound"]
    shas = {
        _toml(
            root / "benchmark" / "molecules" / key / "methods" / "summary.toml"
        )["geometry_sha256"],
        _toml(
            root
            / "benchmark"
            / "molecules"
            / key
            / "reference"
            / "reference.toml"
        )["geometry_sha256"],
        _toml(
            root / "production" / "molecules" / key / "v1.0" / "result.toml"
        )["geometry_sha256"],
    }
    assert len(shas) == 1


def test_labels_identical_everywhere(tmp_path: Path) -> None:
    """Every per-molecule file carries the same identity as molecules.toml."""
    root, cfg = trunk_done(tmp_path)
    pipeline.advance(cfg, chain="benchmark")
    finish_single_points(root, cfg)
    pipeline.advance(cfg, chain="benchmark")
    lock.write(cfg, lock.build(cfg, "r2scan3c", "smd_mecn", "1.0"))
    pipeline.advance(cfg, chain="production")
    finish_production(root)
    pipeline.advance(cfg, chain="production")

    for rec in project.molecules(cfg):
        struct = _toml(root / rec["directory"] / "structure.toml")
        want = {
            "compound": rec["compound"],
            "name": rec["name"],
            "role": rec["role"],
            "isomer": struct["best"]["label"],
            "inchikey": struct["best"]["inchikey"],
        }
        key = rec["compound"]
        files = [
            root
            / "benchmark"
            / "molecules"
            / key
            / "reference"
            / "reference.toml",
            root
            / "benchmark"
            / "molecules"
            / key
            / "methods"
            / "summary.toml",
            root / "production" / "molecules" / key / "v1.0" / "result.toml",
        ]
        for f in files:
            data = _toml(f)
            assert {k: data[k] for k in want} == want, f
    for branch in ("benchmark", "production"):
        index = _toml(root / branch / "molecules.toml")
        assert set(index) >= {"aldehyde", "amine", "imine", "water"}
        assert index["branch"] == branch
