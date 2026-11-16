"""TS module: mechanisms, complexes, kinetics and the `predcc ts` chain.

The chain test fakes every autodE result and ORCA output (minima, TS Freq)
with energies chosen to give known barriers, then checks the barriers, the
one-TOML-per-molecule/TS files, the rate tables and the kinetics. One test
runs the real autodE (if installed) against stand-in ORCA/xTB programs.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tomllib
from pathlib import Path

import pytest
from test_branches import snapshot, trunk_done
from test_pipeline import orca_freq_output, orca_sp_output

from predcc_dft.energies import thermo
from predcc_dft.protocol import constants
from predcc_dft.protocol.config import load_config
from predcc_dft.protocol.io import format_xyz, read_xyz_frames, write_toml
from predcc_dft.ts import autode_job, engine, geometry, kinetics, mechanism
from predcc_dft.ts import stage as ts_stage
from predcc_dft.ts import tasks as ts_tasks
from predcc_dft.workflow import pipeline
from predcc_dft.workflow.cli import main

H2K = constants.HARTREE_TO_KCAL
STD = thermo.standard_state_kcal(298.15) / H2K
G_CORR = 0.1
BARRIERS = {  # kcal/mol from the separated reactants (1 M)
    ("addition", "none"): 24.0,
    ("addition", "tfa"): 14.0,
    ("addition", "tfa_salt"): 16.0,
    ("dehydration", "none"): 30.0,
    ("dehydration", "tfa"): 15.0,
    ("dehydration", "tfa_salt"): 17.0,
}
SHUTTLES = '["none", "tfa", "tfa_salt"]'


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ chemistry


@pytest.mark.parametrize("shuttle", mechanism.SHUTTLES)
def test_mechanism_steps(shuttle: str) -> None:
    """Every shuttle gives the right products and comes out unchanged."""
    ald, amine = "O=Cc1ccccc1", "NCc1ccccc1"
    hemi = mechanism.hemiaminal_smiles(ald, amine)
    assert hemi == "OC(NCc1ccccc1)c1ccccc1"
    add = mechanism.addition(ald, amine, shuttle)
    deh = mechanism.dehydration(hemi, amine, shuttle)
    extra = mechanism.shuttle_species(shuttle, amine)
    assert add.charge == deh.charge == 0
    assert hemi in add.fragments("product")
    assert "C(=NCc1ccccc1)c1ccccc1" in deh.fragments("product")
    assert "O" in deh.fragments("product")
    out = {
        "none": [],
        "water": ["O"],
        "amine": ["NCc1ccccc1"],
        "tfa": ["O=C(O)C(F)(F)F"],
        "tfa_salt": ["NCc1ccccc1", "O=C(O)C(F)(F)F"],  # leaves as N + T
    }[shuttle]
    for smi in out:
        assert smi in add.fragments("product")
    if shuttle != "tfa_salt":
        assert sorted(extra) == sorted(out)
    reac, prod = mechanism.kinetic_species("addition", shuttle)
    assert reac[:2] == ["A", "N"] and prod[0] == "H"


@pytest.mark.parametrize("shuttle", ["none", "water", "tfa_salt"])
def test_complexes_have_the_right_bonds(shuttle: str) -> None:
    """Embedded NEB ends: same atoms, reactant and product connectivity."""
    step = mechanism.addition("O=Cc1ccccc1", "Nc1ccccc1", shuttle)
    reactant, product = geometry.complexes(step)
    assert [a[0] for a in read_xyz_frames_text(reactant)] == [
        a[0] for a in read_xyz_frames_text(product)
    ]
    for text, mol in ((reactant, step.reactant), (product, step.product)):
        found = geometry.bonds_from_xyz(text, 0, 1.3)
        assert geometry.matches(found, geometry.graph_bonds(mol), set()) == []


def read_xyz_frames_text(text: str) -> list:
    """Atoms of an xyz text."""
    return [ln.split() for ln in text.splitlines()[2:] if ln.strip()]


# ------------------------------------------------------------------ kinetics


def _network(salt: bool = True) -> list[kinetics.Channel]:
    chs = [
        kinetics.Channel("add/none", "addition", ["A", "N"], ["H"], 22, 3),
        kinetics.Channel(
            "add/tfa", "addition", ["A", "N", "T"], ["H", "T"], 12, 3
        ),
        kinetics.Channel("deh/none", "dehydration", ["H"], ["I", "W"], 30, -5),
        kinetics.Channel(
            "deh/water", "dehydration", ["H", "W"], ["I", "W", "W"], 18, -5
        ),
        kinetics.Channel(
            "deh/tfa", "dehydration", ["H", "T"], ["I", "W", "T"], 14, -5
        ),
    ]
    if salt:
        chs.append(
            kinetics.Channel(
                "add/salt",
                "addition",
                ["A", "N", "S"],
                ["H", "N", "T"],
                14,
                3 + 2,
            )
        )
        chs.append(kinetics.acid_base_channel(-2.0, 1e9))
    return chs


def test_kinetics_conservation_and_equilibrium() -> None:
    """Mass balance, [W] = [I], and the long-time limit equals K."""
    c0 = {"A": 0.05, "N": 0.05, "T": 0.001}
    res = kinetics.simulate(_network(), c0, 298.15, 86400.0)
    c = res["conc"]
    assert max(abs(c["W"] - c["I"])) < 1e-12
    assert max(abs(c["A"] + c["H"] + c["I"] - 0.05)) < 1e-10
    assert max(abs(c["N"] + c["S"] + c["H"] + c["I"] - 0.05)) < 1e-10
    assert max(abs(c["T"] + c["S"] - 0.001)) < 1e-10
    eq = res["equilibrium"]
    q = eq["I"] * eq["W"] / (eq["A"] * eq["N"])
    k = math.exp(2.0 / (constants.R_KCAL * 298.15))  # dG = 3 - 5
    assert math.log10(q) == pytest.approx(math.log10(k), abs=1e-3)
    assert eq["S"] / 0.001 == pytest.approx(  # TFA + N <=> S
        1 - eq["T"] / 0.001, rel=1e-6
    )
    assert res["t50_s"] < res["t95_s"] < 86400.0
    assert res["flux_share"]["add/tfa"] > 0.9


def test_tfa_speeds_up_and_band() -> None:
    """2 mol% TFA is faster than none; a higher barrier is slower."""
    chs = _network(salt=False)
    with_acid = kinetics.simulate(
        chs, {"A": 0.05, "N": 0.05, "T": 0.001}, 298.15, 86400.0
    )
    without = kinetics.simulate(chs, {"A": 0.05, "N": 0.05}, 298.15, 86400.0)
    assert without["conc"]["I"][-1] < with_acid["conc"]["I"][-1]
    slow = kinetics.simulate(
        kinetics.shifted(chs, 1.0),
        {"A": 0.05, "N": 0.05, "T": 0.001},
        298.15,
        86400.0,
    )
    assert slow["t50_s"] > with_acid["t50_s"]
    k1 = kinetics.eyring(10.0, 298.15)
    assert k1 / kinetics.eyring(11.0, 298.15) == pytest.approx(
        math.exp(1 / (constants.R_KCAL * 298.15))
    )


# ------------------------------------------------------------------ chain


def _g_trunk(root: Path, key: str) -> float:
    best = _toml(root / "molecules" / key / "structure.toml")["best"]
    return best["e_hartree"] + best["g_corr_hartree"] + STD


def _write(task: Path, text: str, final: bool = False) -> None:
    (task / "output.out").write_text(text, encoding="utf-8")
    if final:
        (task / "final.xyz").write_text(
            (task / "geom.xyz").read_text(encoding="utf-8"), encoding="utf-8"
        )


def _e_for(g_target: float) -> float:
    """Electronic energy giving G = g_target with G_CORR and 1 M."""
    return g_target - G_CORR - STD


def fake_minima(root: Path, rxn: dict) -> dict[str, float]:
    """Hemiaminal +1, TFA, salt -3 kcal/mol (vs TFA + amine); returns G."""
    a = _g_trunk(root, rxn["recs"]["aldehyde"]["compound"])
    n = _g_trunk(root, rxn["recs"]["amine"]["compound"])
    g = {"A": a, "N": n, "T": -320.0}
    g["H"] = a + n + 1.0 / H2K
    g["S"] = g["T"] + n - 3.0 / H2K
    labels = {
        "H": rxn["hemiaminal"],
        "T": ts_stage.TFA_LABEL,
        "S": rxn["salt"],
    }
    for sym, label in labels.items():
        for i, task in enumerate(
            sorted((root / "ts" / "species" / label).glob("conf[0-9]*"))
        ):
            e = _e_for(g[sym]) + i * 0.5 / H2K  # conf1 lowest
            _write(task, orca_freq_output(e, [30.0, 60.0], G_CORR), True)
    return g


def fake_autode(root: Path, failed: str = "") -> int:
    """autodE: TS = midpoint of the two complexes (failed: no TS)."""
    n = 0
    for task in sorted(root.glob("ts/reactions/*/*/autode")):
        assert (task / "run.sh").exists()
        t = _toml(task / "task.toml")
        assert t["hcode"] == "orca" and t["lcode"] == "xtb"
        if task.parent.name == failed:
            write_toml(
                task / "autode_result.toml",
                {"status": "failed", "reason": "autodE found no TS"},
            )
            n += 1
            continue
        r = read_xyz_frames(task.parent / "reactant.xyz")[0][1]
        p = read_xyz_frames(task.parent / "product.xyz")[0][1]
        mid = [
            (a[0], *((a[k] + b[k]) / 2 for k in (1, 2, 3)))
            for a, b in zip(r, p, strict=True)
        ]
        (task / "ts.xyz").write_text(format_xyz(mid, "TS"), encoding="utf-8")
        write_toml(
            task / "autode_result.toml",
            {
                "status": "found",
                "reason": "",
                "energy_hartree": -1.0,
                "imaginary_cm1": [-400.0],
                "attempts": [{"method": "get_ts_adaptive_path", "result": ""}],
            },
        )
        n += 1
    return n


def fake_freq(root: Path, g: dict[str, float], bad: str = "") -> int:
    """ORCA Freq on the TS: one imaginary mode and the BARRIERS energies."""
    n = 0
    for task in sorted(root.glob("ts/reactions/*/*/freq")):
        mech = _toml(task.parent / "channel.toml")["mechanism"]
        info = _toml(task.parent / "channel.toml")
        g_r = sum(g[s] for s in mech["kinetic_reactants"])
        dg = BARRIERS[(info["step"], info["shuttle"])]
        freqs = [-60.0, -420.0] if task.parent.name == bad else [-420.0]
        _write(
            task,
            orca_freq_output(_e_for(g_r + dg / H2K), [*freqs, 40.0], G_CORR),
        )
        n += 1
    return n


def _jobs(rep: dict) -> dict[str, int]:
    return {
        s: int(v.split()[0])
        for s, v in rep["jobs written"].items()
        if s.startswith("ts_")
    }


def ts_project(tmp_path: Path) -> tuple[Path, object, dict]:
    """Finished trunk, TS selection benzaldehyde + benzylamine."""
    root, _ = trunk_done(tmp_path)
    (root / "ts" / "molecules.txt").write_text(
        "benzaldehyde\nbenzylamine\n", encoding="utf-8"
    )
    with (root / "config" / "settings.toml").open("a") as fh:
        fh.write(
            f"\n[ts]\nshuttles = {SHUTTLES}\n[ts.minima]\nconformers = 2\n"
        )
    cfg = load_config(root=root)
    (rxn,) = ts_stage.reactions(cfg)
    return root, cfg, rxn


# ------------------------------------------------------------------ autodE


@pytest.mark.parametrize("shuttle", mechanism.SHUTTLES)
def test_autode_handover(shuttle: str) -> None:
    """Additions run in reverse; never more forming than breaking bonds."""
    ald, amine = "O=Cc1ccccc1", "Nc1ccccc1"
    hemi = mechanism.hemiaminal_smiles(ald, amine)
    for step in (
        mechanism.addition(ald, amine, shuttle),
        mechanism.dehydration(hemi, amine, shuttle),
    ):
        reactant, product = geometry.complexes(step)
        t = engine.task(step, reactant, product)
        assert t["direction"] == (
            "reverse" if step.name == "addition" else "forward"
        )
        assert len(t["forming"]) < len(t["breaking"])
        n = step.reactant.GetNumAtoms()
        assert sorted(t["start"]["order"]) == list(range(n))
        assert sum(len(f["atoms"]) for f in t["end"]["fragments"]) == n
        assert {f["charge"] for f in t["start"]["fragments"]} <= {-1, 0, 1}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX stand-in programs")
def test_autode_job_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real autodE runs our search; failing programs: clean result."""
    pytest.importorskip("autode")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for prog in ("orca", "xtb"):
        (bin_dir / prog).write_text("#!/bin/bash\nexit 1\n")
        (bin_dir / prog).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("ORCA_BIN", raising=False)
    step = mechanism.dehydration(
        mechanism.hemiaminal_smiles("O=Cc1ccccc1", "NCc1ccccc1"),
        "NCc1ccccc1",
        "water",
    )
    reactant, product = geometry.complexes(step)
    task_dir = tmp_path / "autode"
    cfg = load_config(root=tmp_path)
    write_toml(
        task_dir / "task.toml",
        {"name": "t", **engine.task(step, reactant, product)},
    )
    ts_tasks.write_autode(cfg, task_dir, "test")
    monkeypatch.chdir(tmp_path)
    assert autode_job.main([str(task_dir)]) == 1
    res = _toml(task_dir / "autode_result.toml")
    assert res["status"] == "failed"
    methods = [a["method"] for a in res["attempts"]]
    assert "get_ts_adaptive_path" in methods  # the search really ran


# ------------------------------------------------------------------ chain


def test_ts_chain(tmp_path: Path) -> None:  # noqa: PLR0915
    """Minima + autodE -> Freq -> barriers -> kinetics, round by round."""
    root, cfg, rxn = ts_project(tmp_path)
    outside = {d: snapshot(root / d) for d in ("benchmark", "production")}
    trunk_files = set(snapshot(root / "molecules"))
    rep = pipeline.advance(cfg, chain="ts")
    assert rep["ts set-up"]["channels"] == 6
    assert rep["ts set-up"]["failed"] == 0
    n_confs = sum(  # hemiaminal, TFA, salt: up to 2 distinct conformers
        len(read_xyz_frames(f))
        for f in root.glob("ts/species/*/conformers.xyz")
    )
    assert 3 <= n_confs <= 6
    assert _jobs(rep) == {
        "ts_minima": n_confs,
        "ts_autode": 6,
        "ts_freq": 0,
        "ts_sp": 0,  # energy "auto" without a lock: opt-level energies
    }
    ch = root / "ts" / "reactions" / rxn["key"] / "addition_tfa"
    task = _toml(ch / "autode" / "task.toml")
    assert task["direction"] == "reverse"  # hemiaminal side -> A + N
    assert task["orca_keywords"]["opt_ts"].startswith("OptTS")
    assert "Calc_Hess true" in task["orca_blocks"]["opt_ts"]
    assert "predcc_dft.ts.autode_job" in (ch / "autode" / "run.sh").read_text()
    info = _toml(ch / "channel.toml")
    assert info["mechanism"]["kinetic_reactants"] == ["A", "N", "T"]
    assert len(info["mechanism"]["transfers"]) == 2

    g = fake_minima(root, rxn)
    assert fake_autode(root, failed="addition_none") == 6
    rep = pipeline.advance(cfg, chain="ts")
    assert _jobs(rep)["ts_freq"] == 5  # not the failed search
    assert _jobs(rep)["ts_autode"] == 0
    salt = _toml(root / "ts" / "species" / rxn["salt"] / "ts.toml")
    assert salt["best_conformer"] == "conf1"
    assert salt["role"] == "salt" and salt["symbol"] == "S"
    assert salt["energies"]["opt"]["g_hartree"] == pytest.approx(
        g["S"], abs=1e-9
    )
    for row in salt["conformers"]:  # every path, in the one file
        assert (root / row["output"]).exists()

    assert fake_freq(root, g, bad="dehydration_tfa_salt") == 5
    rep = pipeline.advance(cfg, chain="ts")
    assert sum(_jobs(rep).values()) == 0
    col = rep["ts collect"]
    assert col["channels verified"] == 4
    assert col["channels autode_failed"] == 1
    assert col["channels bad_ts"] == 1
    assert rep["ts analysis"]["kinetics complete"] == 1

    # One TOML per molecule: identity, side, every file, energies
    ald = _toml(
        root
        / "ts"
        / "species"
        / rxn["recs"]["aldehyde"]["compound"]
        / "ts.toml"
    )
    assert ald["in_reactions"][0]["side"] == "reactant"
    assert ald["name"] == "benzaldehyde"
    for key in ("geometry", "trunk_structure_toml"):
        assert (root / ald["files"][key]).exists()
    imine = _toml(
        root / "ts" / "species" / rxn["recs"]["imine"]["compound"] / "ts.toml"
    )
    assert imine["in_reactions"][0]["side"] == "product"
    # One TOML per TS
    info = _toml(ch / "channel.toml")
    assert info["state"] == "verified"
    assert info["freq"]["imaginary_cm1"] == [-420.0]
    for key in ("ts_xyz", "freq_output", "reactant_xyz", "autode_result"):
        assert (root / info["files"][key]).exists(), key
    assert "g_hartree" in info["energies"]["opt"]
    bad = _toml(ch.parent / "dehydration_tfa_salt" / "channel.toml")
    assert bad["state"] == "bad_ts"

    d = root / "ts" / "derived" / "opt"
    with (d / "barriers.csv").open(encoding="utf-8") as fh:
        rows = {(r["step"], r["shuttle"]): r for r in csv.DictReader(fh)}
    for key, dg in BARRIERS.items():
        if key in (("addition", "none"), ("dehydration", "tfa_salt")):
            continue
        assert float(rows[key]["dg_act_kcal_mol"]) == pytest.approx(
            dg, abs=1e-4
        )
    assert rows[("addition", "none")]["state"] == "autode_failed"
    assert rows[("dehydration", "tfa_salt")]["used"] == "False"
    assert float(rows[("addition", "tfa")]["dg_kcal_mol"]) == pytest.approx(
        1.0, abs=1e-4
    )
    k_tfa = float(rows[("addition", "tfa")]["k_f"])
    assert k_tfa == pytest.approx(kinetics.eyring(14.0, 298.15), rel=1e-4)
    assert rows[("addition", "tfa")]["k_f_units"] == "M^-2 s^-1"

    with (d / "summary.csv").open(encoding="utf-8") as fh:
        (summ,) = list(csv.DictReader(fh))
    assert summ["status"] == "complete"
    assert summ["main_addition_channel"] == "addition/tfa"
    assert summ["main_dehydration_channel"] == "dehydration/tfa"
    assert float(summ["tfa0_mol_l"]) == pytest.approx(0.001)
    assert float(summ["log_q_eq"]) == pytest.approx(
        float(summ["log_k_condensation"]), abs=1e-3
    )
    assert float(summ["dg_salt_kcal_mol"]) == pytest.approx(-3.0, abs=1e-4)
    assert (
        float(summ["t50_h_fast"])
        < float(summ["t50_h"])
        < float(summ["t50_h_slow"])
    )
    with (d / "kinetics" / f"{rxn['key']}.csv").open(encoding="utf-8") as fh:
        series = list(csv.DictReader(fh))
    assert float(series[-1]["t_h"]) == pytest.approx(24.0)
    for row in series:
        assert float(row["W_mol_l"]) == pytest.approx(
            float(row["I_mol_l"]), rel=1e-6, abs=1e-15
        )
    reaction = _toml(root / "ts" / "reactions" / rxn["key"] / "reaction.toml")
    assert reaction["kinetics"]["opt"]["summary"]["status"] == "complete"
    assert (root / reaction["kinetics"]["opt"]["concentrations_csv"]).exists()
    assert reaction["molecules"]["S"]["side"] == "shuttle"
    assert reaction["channels"]["addition_tfa"]["state"] == "verified"
    assert (root / "ts" / "molecules.toml").exists()

    for folder, before in outside.items():  # the TS module is insular
        assert snapshot(root / folder) == before, folder
    assert set(snapshot(root / "molecules")) == trunk_files  # no TS files
    rep = pipeline.advance(cfg, chain="ts")  # idempotent
    assert sum(_jobs(rep).values()) == 0
    assert main(["--root", str(root), "analyse", "ts"]) == 0


def _opt_energy(sp_task: Path) -> float:
    """E of the Opt/Freq job of the point a single point belongs to.

    Gives the locked protocol the same G as the "opt" one (dG_solv = 0), so
    the barriers stay those of BARRIERS.
    """
    point = sp_task.parent.parent.parent
    if (point / "freq").exists():  # a TS
        out = point / "freq" / "output.out"
    elif (point / "conformers.xyz").exists():  # optimised in the TS module
        best = _toml(point / "ts.toml")["best_conformer"]
        out = point / best / "output.out"
    else:  # trunk species
        mol = _toml(point / "ts.toml")
        return mol["trunk"]["e_hartree"]
    text = out.read_text(encoding="utf-8")
    return float(text.split("FINAL SINGLE POINT ENERGY")[-1].split()[0])


def test_ts_with_lock_single_points(tmp_path: Path) -> None:
    """Energy = lock: locked method + solvation single points per point."""
    root, cfg, rxn = ts_project(tmp_path)
    assert (
        main(
            [
                "--root",
                str(root),
                "promote",
                "--method",
                "r2scan3c",
                "--solvation",
                "smd_mecn",
            ]
        )
        == 0
    )
    cfg = load_config(root=root)
    pipeline.advance(cfg, chain="ts")
    g = fake_minima(root, rxn)
    fake_autode(root)
    rep = pipeline.advance(cfg, chain="ts")
    # trunk A, N, I, W + hemiaminal, TFA, salt: energy, gas, smd each
    assert _jobs(rep)["ts_sp"] == 7 * 3
    fake_freq(root, g)
    rep = pipeline.advance(cfg, chain="ts")
    assert _jobs(rep)["ts_sp"] == (7 + 6) * 3  # + the six TSs
    for task in root.glob("ts/**/sp/v1.0/*"):
        (task / "output.out").write_text(orca_sp_output(_opt_energy(task)))
    rep = pipeline.advance(cfg, chain="ts")
    assert _jobs(rep)["ts_sp"] == 0
    tfa = _toml(ts_stage.molecule_file(cfg, ts_stage.TFA_LABEL))
    res = tfa["energies"]["v1.0"]
    assert res["method"] == "r2scan3c"
    assert "g_hartree" in res
    for sp in res["single_points"].values():
        assert (root / sp["output"]).exists()
    rec = next(root.glob("ts/reactions/*/addition_tfa/sp/v1.0/energy"))
    assert _toml(rec / "record.toml")["calculation"]["branch"] == "ts"
    reaction = _toml(root / "ts" / "reactions" / rxn["key"] / "reaction.toml")
    summ = reaction["kinetics"]["v1.0"]["summary"]
    assert summ["status"] == "complete"
    assert summ["main_addition_channel"] == "addition/tfa"


def test_cli_ts_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """`predcc ts` prints the round report and writes the job arrays."""
    root, _, _ = ts_project(tmp_path)
    assert main(["--root", str(root), "ts"]) == 0
    out = capsys.readouterr().out
    assert "PreDCC ts round 1" in out
    script = root / "ts" / "jobs" / "ts_autode" / "ts_autode.slurm"
    assert script.exists()
    assert "--ntasks=16" in script.read_text()
