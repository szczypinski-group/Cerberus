"""End-to-end tests of `predcc run` with mocked CREST/CENSO/ORCA outputs.

Each test builds a project in a temporary folder and drives it round by round
the way the SLURM driver job would, writing fake program outputs in between.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import zlib
from pathlib import Path

import pytest

from predcc_dft.benchmark import report as benchmark_report
from predcc_dft.energies import reference, thermo
from predcc_dft.protocol import methods, project
from predcc_dft.protocol.config import load_config
from predcc_dft.protocol.io import format_xyz, read_xyz_frames
from predcc_dft.structures import dft_opt, frequencies
from predcc_dft.workflow import orca_inputs, pipeline
from predcc_dft.workflow.cli import cmd_init, main

ALDEHYDES = (
    "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n"
    "terephthalaldehyde,O=Cc1ccc(C=O)cc1\nacetophenone,CC(=O)c1ccccc1\n"
)
NUCLEOPHILES = (
    "name,smiles\naniline,Nc1ccccc1\nhydrazine,NN\n"
    "benzhydrazide,NNC(=O)c1ccccc1\nanilinium chloride,[NH3+]c1ccccc1.[Cl-]\n"
)
FAST = """
[products]
linkages = ["imine", "oxime", "hydrazone", "acylhydrazone", "azine"]
[pipeline]
max_rounds = 30
max_submissions = 2
[analysis]
bootstrap_samples = 200
[benchmark]
methods = ["b3lyp_631pgdp", "r2scan3c", "wb97mv_def2tzvp"]
"""


# ------------------------------------------------------------------ fakes


def fake_energy(n_atoms: int, salt: float = 0.0) -> float:
    """Deterministic fake electronic energy."""
    return -40.0 * n_atoms - salt


def noise(text: str, scale: float) -> float:
    """Deterministic pseudo-random number in [-scale, scale]."""
    return scale * ((zlib.crc32(text.encode()) % 2001) / 1000.0 - 1.0)


def orca_freq_output(energy: float, freqs: list[float], g_corr: float) -> str:
    """Minimal ORCA Opt+Freq output."""
    lines = [
        "Program Version 6.0.1",
        "          ***        THE OPTIMIZATION HAS CONVERGED     ***",
        f"FINAL SINGLE POINT ENERGY      {energy:.10f}",
        "VIBRATIONAL FREQUENCIES",
        "",
    ]
    lines += [f"   {i}:         0.00 cm**-1" for i in range(6)]
    for i, f in enumerate(freqs, start=6):
        lines.append(f"   {i}:     {f:8.2f} cm**-1")
    lines += [
        "NORMAL MODES",
        "Temperature         ...   298.15 K",
        "Zero point energy                ...      0.10000000 Eh",
        f"G-E(el)                           ...      {g_corr:.8f} Eh",
        f"Final Gibbs free energy         ...    {energy + g_corr:.8f} Eh",
        "****ORCA TERMINATED NORMALLY****",
    ]
    return "\n".join(lines) + "\n"


def orca_dlpno_output(scf: float, corr: float, t1: float = 0.01) -> str:
    """Minimal DLPNO-CCSD(T) output."""
    return (
        "\n".join(
            [
                "Program Version 6.0.1",
                f"E(0)                                       ...   {scf:.10f}",
                f"T1 diagnostic                ...      {t1:.6f}",
                f"Final correlation energy     ...     {corr:.10f}",
                f"E(CCSD(T))                   ...   {scf + corr:.10f}",
                f"FINAL SINGLE POINT ENERGY      {scf + corr:.10f}",
                "****ORCA TERMINATED NORMALLY****",
            ]
        )
        + "\n"
    )


def orca_sp_output(energy: float) -> str:
    """Minimal single-point output with an orbital-energy block."""
    return (
        "Program Version 6.0.1\n"
        "ORBITAL ENERGIES\n----------------\n"
        "  NO   OCC          E(Eh)            E(eV)\n"
        "   0   2.0000      -0.250000        -6.8028\n"
        "   1   0.0000       0.010000         0.2721\n"
        "MULLIKEN POPULATION ANALYSIS\n"
        f"FINAL SINGLE POINT ENERGY      {energy:.10f}\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )


def orca_opt_output(energy: float) -> str:
    """Minimal ORCA optimisation output."""
    return (
        "          ***        THE OPTIMIZATION HAS CONVERGED     ***\n"
        f"FINAL SINGLE POINT ENERGY      {energy:.10f}\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )


def bench(root: Path, directory: str | Path) -> Path:
    """Benchmark-branch folder of a trunk molecule folder (same key)."""
    return (
        root
        / "benchmark"
        / Path(directory).relative_to(
            root if Path(directory).is_absolute() else Path()
        )
    )


def to_txt(table: str) -> str:
    """'name,smiles' rows (compact test notation) -> 'SMILES name' lines."""
    lines = table.strip().splitlines()[1:]
    return "".join(
        f"{smi} {name}\n" for name, smi in (ln.split(",", 1) for ln in lines)
    )


def make_project(tmp_path: Path, aldehydes: str, nucleophiles: str) -> Path:
    """Initialised project with the given input lists."""
    root = tmp_path / "project"
    cmd_init(root, example=False)
    (root / "inputs" / "aldehydes.txt").write_text(to_txt(aldehydes))
    (root / "inputs" / "amines.txt").write_text(to_txt(nucleophiles))
    with (root / "config" / "settings.toml").open("a") as fh:
        fh.write(FAST)
    return root


def mock_crest_censo(root: Path, cfg: object) -> None:
    """CREST: the seed itself; CENSO: every pooled frame, increasing G."""
    for rec in project.molecules(cfg):
        mdir = root / rec["directory"]
        frames, n = [], 1
        for iso in rec["isomers"]:
            for seed in iso["seeds"]:
                atoms = read_xyz_frames(
                    mdir / "seeds" / f"{seed['label']}.xyz"
                )[0][1]
                crest = mdir / "conformers" / f"crest_{seed['label']}"
                crest.mkdir(parents=True, exist_ok=True)
                (crest / "crest_conformers.xyz").write_text(
                    format_xyz(atoms, "crest")
                )
                frames.append((f"CONF{n}", atoms))
                n += 1
        censo = mdir / "conformers" / "censo"
        censo.mkdir(exist_ok=True)
        (censo / "2_OPTIMIZATION.xyz").write_text(
            "".join(format_xyz(a, name) for name, a in frames)
        )
        data = {
            name: {
                "energy": fake_energy(len(a)),
                "gtot": fake_energy(len(a)) + 0.001 * k,
            }
            for k, (name, a) in enumerate(frames)
        }
        (censo / "2_OPTIMIZATION.json").write_text(
            json.dumps({"part_name": "optimization", "data": data})
        )


def finish_freq(task: Path, freqs: list[float]) -> None:
    """Fake a finished Opt+Freq attempt."""
    atoms = read_xyz_frames(task / "geom.xyz")[0][1]
    (task / "output.out").write_text(
        orca_freq_output(fake_energy(len(atoms)), freqs, 0.1)
    )
    (task / "final.xyz").write_text(
        (task / "geom.xyz").read_text(encoding="utf-8")
    )
    (task / "job_info.toml").write_text(
        'slurm_job_id = "42_1"\ncores = 8\nwall_time_s = 100\n'
        'exit_code = 0\nhost = "node1"\n'
    )


def finish_single_points(root: Path, cfg: object) -> None:
    """Fake every reference, solvation and benchmark single point."""
    offsets = {
        "b3lyp_631pgdp": 0.004,
        "r2scan3c": 0.0015,
        "wb97mv_def2tzvp": 0.0,
    }
    for rec in project.molecules(cfg):
        mdir = root / rec["directory"]
        n = len(read_xyz_frames(mdir / "best.xyz")[0][1])
        e = fake_energy(n) + noise(rec["compound"], 0.01)
        ref = bench(root, rec["directory"]) / "reference"
        if not ref.exists():  # not in this benchmark's selection
            continue
        (ref / "ref_x" / "output.out").write_text(
            orca_dlpno_output(e, -0.02 * n)
        )
        (ref / "ref_y" / "output.out").write_text(
            orca_dlpno_output(e - 0.001, -0.021 * n)
        )
        (ref / "solv_gas" / "output.out").write_text(orca_sp_output(e))
        (ref / "solv_smd_mecn" / "output.out").write_text(
            orca_sp_output(e - 0.005 - noise(rec["compound"] + "s", 0.002))
        )
        (ref / "solv_cpcm_mecn" / "output.out").write_text(
            orca_sp_output(e - 0.004)
        )
        methods_dir = bench(root, rec["directory"]) / "methods"
        for inp in sorted(methods_dir.glob("*/input.inp")):
            label = inp.parent.name
            off = offsets.get(label, 0.001 + abs(noise(label, 0.001)))
            err = off * n + noise(rec["compound"] + label, off * 3)
            (inp.parent / "output.out").write_text(
                orca_sp_output(e - 0.021 * n + err)
            )


# ------------------------------------------------------------------ tests


def test_full_pipeline(tmp_path: Path) -> None:  # noqa: PLR0915
    """Steps 1-11 through repeated `predcc run` rounds."""
    root = make_project(tmp_path, ALDEHYDES, NUCLEOPHILES)
    cfg = load_config(root=root)

    # Round 1: steps 1-3, conformer jobs
    rep = pipeline.advance(cfg, chain="benchmark")
    assert "prepare (steps 1-3)" in rep
    mols = tomllib.loads((root / "molecules.toml").read_text(encoding="utf-8"))
    assert set(mols) >= {"aldehyde", "amine", "imine", "water"}
    rejected = {
        r["name"]: r["reason"]
        for r in tomllib.loads(
            (root / "rejected.toml").read_text(encoding="utf-8")
        )["rejected"]
    }
    assert "acetophenone" in rejected
    assert "duplicate" in rejected["anilinium chloride"]
    tere = next(
        c
        for c in mols["imine"]
        if c["name"].startswith("terephthalaldehyde + aniline")
    )
    assert tere["multi_site"] and len(tere["origins"]) == 2
    acyl = next(c for c in mols["imine"] if c["linkage"] == "acylhydrazone")
    labels = {s["label"] for i in acyl["isomers"] for s in i["seeds"]}
    assert labels >= {"E_syn", "E_anti", "Z_syn", "Z_anti"}
    n_mol = len(project.molecules(cfg))
    assert rep["jobs written"]["conformers"].startswith(f"{n_mol} tasks")
    script = (root / "jobs" / "conformers" / "conformers.slurm").read_text(
        encoding="utf-8"
    )
    assert "TASKS=(" in script and "#SBATCH --array=1-" in script
    run_sh = (root / acyl["directory"] / "conformers" / "run.sh").read_text(
        encoding="utf-8"
    )
    assert "--alpb acetonitrile --ewin 3.0" in run_sh
    assert str(root / "config" / "censo2rc") in run_sh

    # Unchanged inputs: no second prepare; conformer jobs rewritten as before
    rep = pipeline.advance(cfg, chain="benchmark")
    assert "prepare (steps 1-3)" not in rep

    # Round 2: CENSO done -> frequency jobs
    mock_crest_censo(root, cfg)
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["jobs written"]["conformers"].startswith("0 tasks")
    attempts = sorted(root.glob("molecules/*/frequencies/*/attempt1"))
    assert attempts and rep["jobs written"]["frequencies"].startswith(
        f"{len(attempts)} tasks"
    )
    inp = (attempts[0] / "input.inp").read_text(encoding="utf-8")
    assert "! r2SCAN-3c TightOpt Freq CPCM(acetonitrile) TightSCF" in inp
    job = (attempts[0] / "job.slurm").read_text(encoding="utf-8")
    assert "--ntasks=8" in job and "--array" not in job

    # Round 3: one imaginary mode -> retry only that conformer
    bad = attempts[0]
    for a in attempts:
        finish_freq(a, [-35.0, 50.0] if a == bad else [25.0, 50.0])
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["jobs written"]["frequencies"].startswith("1 tasks")
    retry = bad.parent / "attempt2"
    assert "VeryTightOpt" in (retry / "input.inp").read_text(encoding="utf-8")
    rec1 = tomllib.loads((bad / "record.toml").read_text(encoding="utf-8"))
    assert rec1["calculation"]["status"] == "rejected"
    assert rec1["job"]["slurm_job_id"] == "42_1"

    # Round 4: retry done; break one geometry -> collapse, then SP jobs
    finish_freq(retry, [20.0, 300.0])
    victim_dir = root / acyl["directory"] / "frequencies" / "CONF1"
    victim = sorted(victim_dir.glob("attempt*"))[-1] / "final.xyz"
    atoms = read_xyz_frames(victim)[0][1]
    el, x, y, z = atoms[0]
    atoms[0] = (el, x + 3.0, y + 3.0, z + 3.0)
    victim.write_text(format_xyz(atoms, "broken"))
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["check (step 7)"]["collapsed"] == n_mol
    st = tomllib.loads(
        (root / acyl["directory"] / "structure.toml").read_text(
            encoding="utf-8"
        )
    )
    conf1 = next(c for c in st["conformer"] if c["conformer"] == "CONF1")
    assert conf1["sanity"] == "failed" and "connectivity" in conf1["reason"]
    assert abs(sum(st["populations_by_isomer"].values()) - 1.0) < 1e-3
    n_ref_jobs = 3 + len(methods.solvation_models(cfg.methods))
    assert rep["jobs written"]["reference"].startswith(
        f"{n_ref_jobs * n_mol} tasks"
    )
    assert rep["jobs written"]["benchmark"].startswith(f"{3 * n_mol} tasks")
    ref_dir = bench(root, acyl["directory"]) / "reference"
    assert "cc-pVQZ" in (ref_dir / "ref_y" / "input.inp").read_text(
        encoding="utf-8"
    )
    smd = (ref_dir / "solv_smd_mecn" / "input.inp").read_text(encoding="utf-8")
    assert 'SMDsolvent "acetonitrile"' in smd
    methods_dir = bench(root, acyl["directory"]) / "methods"
    sp = (methods_dir / "wb97mv_def2tzvp" / "input.inp").read_text(
        encoding="utf-8"
    )
    assert "! wB97M-V def2-TZVP TightSCF DefGrid3" in sp
    assert "STABPerform true" in sp

    # Round 5: every single point done -> derived tables and candidate
    finish_single_points(root, cfg)
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["jobs written"]["reference"].startswith("0 tasks")
    assert rep["jobs written"]["benchmark"].startswith("0 tasks")
    assert rep["reference (steps 8, 10)"]["species_with_reference"] == n_mol
    assert rep["status"]["benchmark"]["all methods"] == n_mol
    rec = tomllib.loads(
        (methods_dir / "r2scan3c" / "record.toml").read_text(encoding="utf-8")
    )
    assert rec["calculation"]["id"].endswith(".best.sp.r2scan3c.gas")
    assert rec["descriptors"]["homo_ev"] == pytest.approx(-6.8028)
    summary = tomllib.loads(
        (methods_dir / "summary.toml").read_text(encoding="utf-8")
    )
    assert summary["role"] == "imine" and summary["counts"]["completed"] == 3
    assert set(summary["methods"]) == {
        "b3lyp_631pgdp",
        "r2scan3c",
        "wb97mv_def2tzvp",
    }
    wb = summary["methods"]["wb97mv_def2tzvp"]
    assert wb["electronic_hartree"] == pytest.approx(
        tomllib.loads(
            (methods_dir / "wb97mv_def2tzvp" / "record.toml").read_text(
                encoding="utf-8"
            )
        )["energy"]["electronic_hartree"]
    )
    assert (
        wb["functional_label"] == "wb97mv" and wb["basis_label"] == "def2tzvp"
    )
    assert "range_separated" in wb["levels"]
    assert summary["reference"]["e_cbs_hartree"] < 0
    assert summary["geometry_sha256"] == rec["calculation"]["geometry_sha256"]
    derived = root / "benchmark" / "derived"
    for name in (
        "calculations.csv",
        "species_energies.csv",
        "reaction_energies.csv",
        "errors.csv",
        "method_ranking.csv",
        "production_candidate.toml",
    ):
        assert (derived / name).exists(), name
    cand = tomllib.loads(
        (derived / "production_candidate.toml").read_text(encoding="utf-8")
    )
    assert cand["energy"]["method"] == "wb97mv_def2tzvp"
    assert cand["selection"]["mae_kcal_mol"] < 1.0
    assert set(cand["selection"]["mae_by_linkage"]) >= {"imine", "hydrazone"}
    rank = (
        (derived / "method_ranking.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert any(",b3lyp_631pgdp," in r and ",all," in r for r in rank)
    rxn = (derived / "reaction_energies.csv").read_text(encoding="utf-8")
    assert "dlpno_ccsdt_cbs" in rxn and "TRN-" in rxn

    # Experimental K for one condensation -> experiment rows
    reactions = tomllib.loads(
        (root / "reactions.toml").read_text(encoding="utf-8")
    )
    key = reactions["condensation"][0]["key"]
    (root / "benchmark" / "experimental" / f"{key}.toml").write_text(
        f'schema_version = 1\nreaction = "{key}"\nlog_k = 1.5\n'
        'k_units = "dimensionless"\nwater_included = true\n'
        "temperature_k = 298.15\n"
    )
    res = benchmark_report.run(cfg)
    assert res["experimental_reactions"] == 1
    assert ",experiment," in (derived / "errors.csv").read_text(
        encoding="utf-8"
    )

    # Nothing left to do: the chain would stop here
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["next"] == "nothing to submit"


@pytest.mark.skipif(
    sys.platform == "win32", reason="fake sbatch/squeue are bash scripts"
)
def test_submit_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--submit: arrays + dependent driver; give up after max_submissions."""
    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n",
        "name,smiles\naniline,Nc1ccccc1\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    (bin_dir / "sbatch").write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> {log}\n'
        f"echo $(( 1000 + $(wc -l < {log}) ))\n"
    )
    (bin_dir / "squeue").write_text("#!/bin/bash\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cfg = load_config(root=root)

    rep = pipeline.advance(cfg, submit=True)
    assert rep["submitted"] == {"conformers": "1001"}
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[1].startswith("--parsable --dependency=afterany:1001")
    assert calls[1].endswith("trunk.slurm")
    driver = (root / "jobs" / "driver" / "trunk.slurm").read_text(
        encoding="utf-8"
    )
    assert (
        "-m predcc_dft --root" in driver and "run --submit --round 2" in driver
    )
    task = next(root.glob("molecules/*/conformers"))
    assert (task / ".submissions").read_text(encoding="utf-8").strip() == "1"

    # The jobs "failed": resubmitted once more, then given up
    pipeline.advance(cfg, submit=True, round_no=2)
    rep = pipeline.advance(cfg, submit=True, round_no=3)
    assert rep["submitted"] == "nothing"
    assert rep["given up (see <stage>/given_up.txt)"]["conformers"] > 0


@pytest.mark.skipif(
    sys.platform == "win32", reason="fake sbatch/squeue are bash scripts"
)
def test_busy_stage_blocks_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a stage is queued nothing is rewritten."""
    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n",
        "name,smiles\naniline,Nc1ccccc1\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho 555\n")
    (bin_dir / "squeue").write_text("#!/bin/bash\necho '555 555_[1-4%50]'\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cfg = load_config(root=root)
    pipeline.advance(cfg, submit=True)
    rep = pipeline.advance(cfg, submit=True, round_no=2)
    assert "busy" in rep and "conformers (job 555)" in rep["busy"]


def test_orca_route(tmp_path: Path) -> None:
    """Step 5 with dft_engine = "orca": CREST, then ORCA optimisations."""
    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n",
        "name,smiles\nbenzhydrazide,NNC(=O)c1ccccc1\n",
    )
    with (root / "config" / "settings.toml").open("a") as fh:
        fh.write(
            '[conformers]\ndft_engine = "orca"\n'
            "[dft_opt]\nmax_conformers = 3\n"
        )
    cfg = load_config(root=root)
    pipeline.advance(cfg)
    acyl = next(
        r
        for r in project.molecules(cfg)
        if r.get("linkage") == "acylhydrazone"
    )
    run_sh = (root / acyl["directory"] / "conformers" / "run.sh").read_text(
        encoding="utf-8"
    )
    assert "crest ../../seeds/E_syn.xyz" in run_sh and "censo -i" not in run_sh

    for rec in project.molecules(cfg):
        mdir = root / rec["directory"] / "conformers"
        for iso in rec["isomers"]:
            for seed in iso["seeds"]:
                atoms = read_xyz_frames(
                    root / rec["directory"] / "seeds" / f"{seed['label']}.xyz"
                )[0][1]
                crest = mdir / f"crest_{seed['label']}"
                crest.mkdir(parents=True)
                (crest / "crest_conformers.xyz").write_text(
                    format_xyz(atoms, "  -50.00000000")
                    + format_xyz(atoms, "  -49.90000000")
                )
        (mdir / "DONE").touch()
    rep = pipeline.advance(cfg)
    acyl_dft = sorted(
        (root / acyl["directory"] / "conformers" / "dft").glob("CONF*")
    )
    assert len(acyl_dft) == 3
    assert "! r2SCAN-3c Opt CPCM" in (acyl_dft[0] / "input.inp").read_text(
        encoding="utf-8"
    )
    assert rep["jobs written"]["frequencies"].startswith("0 tasks")

    for d in root.glob("molecules/*/conformers/dft/CONF*"):
        n = len(read_xyz_frames(d / "geom.xyz")[0][1])
        (d / "output.out").write_text(
            orca_opt_output(fake_energy(n, 0.001 * int(d.name[4:])))
        )
        (d / "final.xyz").write_text(
            (d / "geom.xyz").read_text(encoding="utf-8")
        )
    assert dft_opt.write_jobs(cfg)["tasks"] == 0
    res = frequencies.write_jobs(cfg)
    assert res["tasks"] > 0 and res["waiting_for_step5"] == 0


def test_orca_input(tmp_path: Path) -> None:
    """One-off inputs for every ORCA step, from a molecule and from an xyz."""
    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n",
        "name,smiles\naniline,Nc1ccccc1\n",
    )
    cfg = load_config(root=root)
    pipeline.advance(cfg)
    key = next(r for r in project.molecules(cfg) if r["role"] == "imine")[
        "compound"
    ]
    expect = {
        "dft_opt": "Opt CPCM",
        "freq": "TightOpt Freq",
        "freq_retry": "Calc_Hess",
        "ref_x": "cc-pVTZ",
        "ref_y": "cc-pVQZ",
        "solv_gas": "r2SCAN-3c TightSCF",
        "solv_smd_mecn": "SMDsolvent",
        "sp_b3lyp_631pgdp": "B3LYP 6-31+G(d,p)",
    }
    for step, text in expect.items():
        res = orca_inputs.make(cfg, step, molecule=key)
        inp = Path(res["input"]).read_text(encoding="utf-8")
        assert text in inp, (step, inp)
        assert "* xyzfile 0 1 geom.xyz" in inp
        assert Path(res["slurm"]).exists()
    xyz = root / "my.xyz"
    xyz.write_text(
        (root / "molecules" / key / "seeds" / "E.xyz").read_text(
            encoding="utf-8"
        )
    )
    res = orca_inputs.make(
        cfg, "custom", xyz=xyz, keywords="wB97M-V def2-TZVP SP", charge=1
    )
    inp = Path(res["input"]).read_text(encoding="utf-8")
    assert "! wB97M-V def2-TZVP SP" in inp and "* xyzfile 1 1 geom.xyz" in inp


def test_cli_and_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """CLI entry points and the CBS / standard-state helpers."""
    root = tmp_path / "p"
    assert main(["init", str(root), "--example"]) == 0
    assert (root / "inputs" / "aldehydes.txt").read_text(
        encoding="utf-8"
    ).count("\n") > 3
    assert main(["--root", str(root), "status"]) == 1  # nothing prepared yet
    with (root / "inputs" / "amines.txt").open("a") as fh:
        fh.write("NN hydrazine\n")
    assert main(["--root", str(root), "prepare"]) == 0
    # Default: imines only (aldehyde + primary amine)
    mols = tomllib.loads((root / "molecules.toml").read_text(encoding="utf-8"))
    n_ald, n_am = len(mols["aldehyde"]), len(mols["amine"])
    assert len(mols["imine"]) == n_ald * n_am
    assert {m["linkage"] for m in mols["imine"]} == {"imine"}
    rejected = tomllib.loads(
        (root / "rejected.toml").read_text(encoding="utf-8")
    )
    assert any(
        r["name"] == "hydrazine" and "not enabled" in r["reason"]
        for r in rejected["rejected"]
    )
    assert main(["--root", str(root), "status", "--summary"]) == 0
    assert "molecules" in capsys.readouterr().out
    cfg = load_config(root=root)
    scf, corr = reference.cbs(cfg, -100.0, -1.0, -100.1, -1.1)
    assert scf < -100.1 and corr < -1.1
    assert thermo.standard_state_kcal(298.15) == pytest.approx(1.894, abs=2e-3)


@pytest.mark.skipif(
    sys.platform == "win32", reason="fake sbatch/squeue are bash scripts"
)
def test_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`predcc level`: one command runs functionals x basis sets."""
    from predcc_dft.benchmark import grid as benchmark

    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n",
        "name,smiles\naniline,Nc1ccccc1\n",
    )
    cfg = load_config(root=root)
    dh = methods.level_methods(cfg.methods, "double_hybrid")
    assert len(dh) == 8 and "revdsdpbep86d4_def2tzvp" in dh
    kw = cfg.methods["methods"]["revdsdpbep86d4_def2tzvp"]["orca_keywords"]
    assert kw == "revDSD-PBEP86-D4/2021 def2-TZVP def2-TZVP/C"
    hyb = cfg.methods["methods"]["b3lyp_def2tzvp"]["orca_keywords"]
    assert hyb == "B3LYP def2-TZVP"  # no aux basis for hybrids
    assert methods.level_methods(cfg.methods, "baseline") == ["b3lyp_631pgdp"]
    with pytest.raises(KeyError, match="unknown level"):
        methods.level_methods(cfg.methods, "tripple_hybrid")

    # Levels override nothing else; they pick the benchmark methods.
    cfg["benchmark"] = {"levels": ["double_hybrid"], "methods": []}
    assert list(benchmark.methods(cfg)) == dh

    # Through the CLI, with the level carried into the driver job.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho 900\n")
    (bin_dir / "squeue").write_text("#!/bin/bash\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert main(["--root", str(root), "levels"]) == 0
    assert (
        main(["--root", str(root), "level", "double_hybrid", "--submit"]) == 0
    )
    driver = (root / "jobs" / "driver" / "benchmark.slurm").read_text(
        encoding="utf-8"
    )
    assert "--level double_hybrid" in driver
    assert main(["--root", str(root), "level", "nonsense"]) != 0


def test_functional_level_insular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Functional and level runs end to end; nothing leaves the project."""
    import csv

    from predcc_dft.benchmark import grid as benchmark

    home, elsewhere = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(elsewhere)
    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n"
        "4-nitrobenzaldehyde,O=Cc1ccc(cc1)[N+](=O)[O-]\n",
        "name,smiles\naniline,Nc1ccccc1\nbenzylamine,NCc1ccccc1\n",
    )
    cfg = load_config(root=root)
    pipeline.advance(cfg, chain="benchmark")
    mock_crest_censo(root, cfg)
    pipeline.advance(cfg, chain="benchmark")
    for a in sorted(root.glob("molecules/*/frequencies/*/attempt1")):
        finish_freq(a, [25.0, 50.0])

    # One functional with all of its basis sets
    pwpb = methods.functional_methods(cfg.methods, "pwpb95d4")
    assert len(pwpb) == 4
    cfg["benchmark"] = {"functionals": ["pwpb95d4"], "methods": []}
    assert list(benchmark.methods(cfg)) == pwpb
    rep = pipeline.advance(cfg, chain="benchmark")
    n_mol = len(project.molecules(cfg))
    assert rep["jobs written"]["benchmark"].startswith(f"{4 * n_mol} tasks")
    written = {
        p.name
        for p in root.glob("benchmark/molecules/*/methods/*")
        if p.is_dir()
    }
    assert written == set(pwpb)

    # Then the whole double-hybrid level: only the 4 missing methods are new
    cfg["benchmark"] = {"levels": ["double_hybrid"], "methods": []}
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["jobs written"]["benchmark"].startswith(f"{8 * n_mol} tasks")
    finish_single_points(root, cfg)
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["jobs written"]["benchmark"].startswith("0 tasks")

    d = root / "benchmark" / "derived"
    rank = list(
        csv.DictReader((d / "method_ranking.csv").open(encoding="utf-8"))
    )
    dh = [
        r
        for r in rank
        if r["level"] == "double_hybrid"
        and r["linkage"] == "all"
        and r["solvation"] == "gas"
    ]
    assert {r["method"] for r in dh} == set(
        methods.level_methods(cfg.methods, "double_hybrid")
    )
    assert {r["functional_label"] for r in dh} == {
        "revdsdpbep86d4",
        "pwpb95d4",
    }
    assert {r["basis_label"] for r in dh} == {
        "def2svp",
        "def2tzvp",
        "def2qzvp",
        "madef2tzvp",
    }
    assert all(r["complete"] == "True" for r in dh)
    summary = list(
        csv.DictReader((d / "level_summary.csv").open(encoding="utf-8"))
    )
    row = next(
        r
        for r in summary
        if r["level"] == "double_hybrid"
        and r["linkage"] == "all"
        and r["solvation"] == "gas"
    )
    assert row["n_methods"] == "8" and row["best_method"] in pwpb + [
        m for m in methods.level_methods(cfg.methods, "double_hybrid")
    ]
    wide = list(csv.DictReader((d / "errors_wide.csv").open(encoding="utf-8")))
    assert wide and set(
        methods.level_methods(cfg.methods, "double_hybrid")
    ) <= set(wide[0])

    # Insular: nothing in HOME or the working directory, relative paths only
    assert not any(home.iterdir()) and not any(elsewhere.iterdir())
    for path in root.rglob("*.csv"):
        assert str(tmp_path) not in path.read_text(encoding="utf-8"), path
    for path in root.rglob("record.toml"):
        assert str(tmp_path) not in path.read_text(encoding="utf-8"), path


def test_data_integrity(tmp_path: Path) -> None:  # noqa: PLR0915
    """Retry-before-collapse, stale geometries, T1 flags, decomposition."""
    import csv

    root = make_project(
        tmp_path,
        "name,smiles\nbenzaldehyde,O=Cc1ccccc1\n",
        "name,smiles\naniline,Nc1ccccc1\nbenzylamine,NCc1ccccc1\n",
    )
    cfg = load_config(root=root)
    pipeline.advance(cfg, chain="benchmark")
    mock_crest_censo(root, cfg)
    pipeline.advance(cfg, chain="benchmark")
    attempts = sorted(root.glob("molecules/*/frequencies/*/attempt1"))
    bad = attempts[0]
    for a in attempts:
        finish_freq(a, [-40.0, 50.0] if a == bad else [25.0, 50.0])

    # A retry is due: that molecule must not collapse yet.
    rep = pipeline.advance(cfg, chain="benchmark")
    assert rep["check (step 7)"]["still_running"] == 1
    victim = bad.parents[2]
    assert not (victim / "best.xyz").exists()
    assert not (bench(root, victim) / "reference").exists()
    finish_freq(bad.parent / "attempt2", [20.0, 300.0])
    rep = pipeline.advance(cfg, chain="benchmark")
    n_mol = len(project.molecules(cfg))
    assert rep["check (step 7)"]["collapsed"] == n_mol
    st = tomllib.loads((victim / "structure.toml").read_text(encoding="utf-8"))
    assert st["g_ensemble_correction_kcal_mol"] <= 0.0

    # High T1 on one species' DLPNO output
    finish_single_points(root, cfg)
    aniline = next(r for r in project.molecules(cfg) if r["name"] == "aniline")
    ref_x = (
        bench(root, aniline["directory"])
        / "reference"
        / "ref_x"
        / "output.out"
    )
    ref_x.write_text(
        ref_x.read_text(encoding="utf-8").replace("0.010000", "0.045000")
    )
    pipeline.advance(cfg, chain="benchmark")
    d = root / "benchmark" / "derived"
    errs = list(csv.DictReader((d / "errors.csv").open(encoding="utf-8")))
    flagged = {e["reaction"] for e in errs if e["t1_flag"] == "True"}
    assert flagged, "aniline reactions should be T1-flagged"
    rank = list(
        csv.DictReader((d / "method_ranking.csv").open(encoding="utf-8"))
    )
    all_gas = [
        r
        for r in rank
        if r["linkage"] == "all"
        and r["solvation"] == "gas"
        and r["method"] == "r2scan3c"
    ]
    n_unflagged = len(
        {
            e["reaction"]
            for e in errs
            if e["t1_flag"] == "False" and e["solvation"] == "gas"
        }
    )
    assert int(all_gas[0]["n"]) == n_unflagged
    first = next(
        csv.DictReader((d / "species_energies.csv").open(encoding="utf-8"))
    )
    assert len(first["e_hartree"].split(".")[1]) == 10

    # Experiment rows carry the electronic / thermal+solvation split
    reactions = tomllib.loads(
        (root / "reactions.toml").read_text(encoding="utf-8")
    )
    key = reactions["condensation"][0]["key"]
    (root / "benchmark" / "experimental" / f"{key}.toml").write_text(
        f'reaction = "{key}"\nlog_k = 1.0\n'
    )
    pipeline.advance(cfg, chain="benchmark")
    exp_rows = [
        e
        for e in csv.DictReader((d / "errors.csv").open(encoding="utf-8"))
        if e["reference"] == "experiment" and e["method"] == "r2scan3c"
    ]
    assert exp_rows
    for e in exp_rows:
        total = float(e["error_kcal_mol"])
        parts = float(e["error_electronic_kcal_mol"]) + float(
            e["error_thermal_solvation_kcal_mol"]
        )
        assert total == pytest.approx(parts, abs=1e-5)

    # best.xyz changes (re-optimised conformer): old single points go stale
    best_conf = st["best"]["conformer"]
    final = (
        sorted((victim / "frequencies" / best_conf).glob("attempt*"))[-1]
        / "final.xyz"
    )
    atoms = read_xyz_frames(final)[0][1]
    atoms = [(el, x + 0.01, y, z) for el, x, y, z in atoms]
    final.write_text(format_xyz(atoms, "moved"))
    rep = pipeline.advance(cfg, chain="benchmark")
    vmethods = bench(root, victim) / "methods"
    n_bench = len(list(vmethods.glob("*/input.inp")))
    assert rep["jobs written"]["benchmark"].startswith(f"{n_bench} tasks")
    assert rep["jobs written"]["reference"].startswith("5 tasks")
    assert rep["methods (step 9)"]["stale"] == n_bench
    rec = tomllib.loads(
        (vmethods / "r2scan3c" / "record.toml").read_text(encoding="utf-8")
    )
    assert rec["calculation"]["status"] == "stale"
    summary = tomllib.loads(
        (vmethods / "summary.toml").read_text(encoding="utf-8")
    )
    assert summary["counts"]["stale"] == n_bench
    assert "electronic_hartree" not in summary["methods"]["r2scan3c"]
    victim_key = victim.name
    species = list(
        csv.DictReader((d / "species_energies.csv").open(encoding="utf-8"))
    )
    assert not any(
        s["compound"] == victim_key and s["method"] == "r2scan3c"
        for s in species
    )


def test_text_input(tmp_path: Path) -> None:
    """One SMILES per line; optional name; comments; '#' in a SMILES."""
    from predcc_dft.species.inputs import read_list

    f = tmp_path / "amines.txt"
    f.write_text(
        "# my nucleophiles\n"
        "\n"
        "Nc1ccccc1\n"
        "NCc1ccccc1 benzyl amine\n"
        "NCC#C propargylamine\n"
    )
    entries = read_list(f)
    assert [(e["smiles"], e["name"]) for e in entries] == [
        ("Nc1ccccc1", "Nc1ccccc1"),
        ("NCc1ccccc1", "benzyl amine"),
        ("NCC#C", "propargylamine"),
    ]
    assert entries[0]["source"] == "amines.txt:3"

    root = tmp_path / "p"
    assert main(["init", str(root), "--example"]) == 0
    text = (root / "inputs" / "aldehydes.txt").read_text(encoding="utf-8")
    assert "O=Cc1ccccc1 benzaldehyde" in text
    cfg = load_config(root=root)
    assert cfg["inputs"]["aldehydes"] == "inputs/aldehydes.txt"
    assert len(read_list(root / "inputs" / "aldehydes.txt")) >= 5


@pytest.mark.parametrize("os_name", ["posix", "nt"])
def test_init_settings_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, os_name: str
) -> None:
    """config/settings.toml from init is valid TOML on Linux and Windows."""
    from predcc_dft.workflow import cli

    monkeypatch.setattr(cli.os, "name", os_name)
    monkeypatch.setattr(cli.sys, "prefix", "C:\\Users\\me\\.venv")
    root = tmp_path / "p"
    assert cli.cmd_init(root, example=True) == 0
    cfg = load_config(root=root)
    lines = cfg["slurm"]["setup_lines"]
    if os_name == "nt":
        assert lines == []
    else:
        prefix = Path("C:\\Users\\me\\.venv").as_posix()
        assert lines == [f"source {prefix}/bin/activate"]
