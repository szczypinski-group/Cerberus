"""Measured log K from NMR (`predcc nmr`): Bruker spectra -> conversion -> K.

Synthetic Bruker 1D experiments (Lorentzian lines with known areas, written
the way TopSpin stores them: acqus + fid, pdata/1/procs + 1r) are read with
nmrglue, integrated and turned into K. Skipped without nmrglue.
"""

from __future__ import annotations

import csv
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
from test_layout import ALDEHYDES, AMINES
from test_pipeline import FAST, to_txt

from predcc_dft.analysis import errors
from predcc_dft.experiment import equilibrium, nmr
from predcc_dft.protocol import project
from predcc_dft.protocol.config import load_config
from predcc_dft.protocol.io import read_toml
from predcc_dft.workflow import pipeline
from predcc_dft.workflow.cli import cmd_init, main

pytest.importorskip("nmrglue")

SF, SW_PPM, CAR, OFFSET = 400.13, 20.0, 5.0, 15.0


def _jcamp(path: Path, params: dict) -> None:
    lines = [
        "##TITLE= Parameter file, TopSpin 4.1.4",
        "##JCAMPDX= 5.0",
        "##ORIGIN= Bruker BioSpin GmbH",
        "##OWNER= test",
    ]
    lines += [f"##${k}= {v}" for k, v in params.items()]
    path.write_text("\n".join([*lines, "##END="]) + "\n")


def write_experiment(
    exp: Path,
    peaks: list[tuple[float, float]],
    raw: bool = True,
    processed: bool = True,
    phase: float = 0.0,
    td: int = 65536,
) -> Path:
    """Bruker 1D experiment folder with Lorentzian lines (ppm, area)."""
    exp.mkdir(parents=True, exist_ok=True)
    sw_hz = SW_PPM * SF
    _jcamp(
        exp / "acqus",
        {
            "TD": td,
            "SW_h": sw_hz,
            "SW": SW_PPM,
            "SFO1": SF,
            "BF1": SF,
            "O1": CAR * SF,
            "BYTORDA": 0,
            "DTYPA": 0,
            "AQ_mod": 3,
            "DECIM": 1,
            "DSPFVS": 20,
            "GRPDLY": 0,
            "NUC1": "<1H>",
        },
    )
    if raw:
        n = td // 2
        t = np.arange(n) / sw_hz
        fid = np.zeros(n, complex)
        for p, a in peaks:
            fid += a * np.exp(2j * np.pi * (p - CAR) * SF * t - np.pi * t)
        fid *= np.exp(1j * phase) * 2e6 / sum(a for _, a in peaks)
        inter = np.empty(2 * n, dtype="<i4")
        inter[0::2], inter[1::2] = np.round(fid.real), np.round(fid.imag)
        inter.tofile(exp / "fid")
    if processed:
        si = 65536
        ppm = OFFSET - np.arange(si) * (SW_PPM / si)
        g = 0.5 / SF  # half width, 1 Hz lines
        spec = sum(a * (g / np.pi) / ((ppm - p) ** 2 + g**2) for p, a in peaks)
        pdata = exp / "pdata" / "1"
        pdata.mkdir(parents=True, exist_ok=True)
        _jcamp(
            pdata / "procs",
            {
                "SI": si,
                "SF": SF,
                "SW_p": SW_PPM * SF,
                "OFFSET": OFFSET,
                "BYTORDP": 0,
                "DTYPP": 0,
                "NC_proc": 0,
            },
        )
        np.round(spec / spec.max() * 2e8).astype("<i4").tofile(pdata / "1r")
    return exp


def peaks(conversion: float) -> list[tuple[float, float]]:
    """Aldehyde CHO at 10.0, imine CH=N at 8.45, plus other signals."""
    return [
        (10.0, 1 - conversion),
        (8.45, conversion),
        (7.4, 5.0),
        (3.8, 2.0),
    ]


# ------------------------------------------------------------------ reading


@pytest.mark.parametrize("source", ["processed", "raw"])
def test_conversion_from_spectrum(tmp_path: Path, source: str) -> None:
    """Integrals give the known conversion (raw FID: auto-phased)."""
    exp = write_experiment(tmp_path / "1", peaks(0.7), phase=0.7)
    spec = nmr.read(exp, source)
    assert spec.source == source
    assert spec.ppm[0] > spec.ppm[-1]  # decreasing, as plotted
    assert spec.ppm[np.argmax(spec.intensity)] == pytest.approx(7.4, abs=0.01)
    x = equilibrium.conversion(
        nmr.integrate(spec, 9.9, 10.1), nmr.integrate(spec, 8.35, 8.55), 1, 1
    )
    assert x == pytest.approx(0.7, abs=0.005)


def test_auto_prefers_processed(tmp_path: Path) -> None:
    """auto: pdata when present, else the raw FID; neither: an error."""
    both = write_experiment(tmp_path / "1", peaks(0.5))
    assert nmr.read(both).source == "processed"
    raw = write_experiment(tmp_path / "2", peaks(0.5), processed=False)
    assert nmr.read(raw).source == "raw"
    with pytest.raises(nmr.SpectrumError, match="no experiment folder"):
        nmr.read(tmp_path / "missing")
    with pytest.raises(nmr.SpectrumError, match="outside the spectrum"):
        nmr.integrate(nmr.read(both), 30.0, 31.0)


def test_equilibrium_k() -> None:
    """K from conversion and initial concentrations; water is produced."""
    k = equilibrium.equilibrium_k(0.5, 0.05, 0.05, 0.0)
    assert k == pytest.approx(1.0)  # 0.025^2 / 0.025^2
    k = equilibrium.equilibrium_k(0.8, 0.05, 0.10, 0.01)
    assert k == pytest.approx((0.04 * 0.05) / (0.01 * 0.06))
    for bad in (0.0, 1.0):
        with pytest.raises(equilibrium.SampleError, match="infinite"):
            equilibrium.equilibrium_k(bad, 0.05, 0.05, 0.0)
    with pytest.raises(equilibrium.SampleError, match="more amine"):
        equilibrium.equilibrium_k(0.9, 0.05, 0.04, 0.0)
    # two protons in the imine signal halve its weight
    assert equilibrium.conversion(1.0, 2.0, 1, 2) == pytest.approx(0.5)


def test_imine_lookup() -> None:
    """The imine's origins give the reaction; ambiguous or multi-site: no."""
    recs = [
        {"compound": "ALD", "role": "aldehyde", "name": "ald", "smiles": ""},
        {
            "compound": "DIALD",
            "role": "aldehyde",
            "name": "diald",
            "smiles": "",
        },
        {"compound": "AM", "role": "amine", "name": "am", "smiles": ""},
        {
            "compound": "IM",
            "role": "imine",
            "name": "ald + am (imine)",
            "smiles": "",
            "origins": [{"aldehyde": "ALD", "amine": "AM"}],
        },
        {
            "compound": "IM2",
            "role": "imine",
            "name": "diald + am (imine)",
            "smiles": "",
            "multi_site": True,
            "origins": [{"aldehyde": "DIALD", "amine": "AM"}],
        },
    ]

    def cnd(key: str, ald: str, imine: str) -> dict:
        return {
            "key": key,
            "reactants": [
                {"compound": ald, "role": "aldehyde"},
                {"compound": "AM", "role": "amine"},
            ],
            "products": [{"compound": imine, "role": "imine"}],
        }

    reactions = {
        "condensation": [
            cnd("CND-1", "ALD", "IM"),
            cnd("CND-2", "DIALD", "IM2"),
        ]
    }
    hit = equilibrium._reaction(recs, {"imine": "IM"}, reactions)
    assert hit["rxn"]["key"] == "CND-1"
    assert (hit["aldehyde"]["compound"], hit["amine"]["compound"]) == (
        "ALD",
        "AM",
    )
    hit = equilibrium._reaction(
        recs, {"aldehyde": "ald", "amine": "am"}, reactions
    )
    assert hit["imine"]["compound"] == "IM"  # found from the reaction
    with pytest.raises(equilibrium.SampleError, match="multi_site"):
        equilibrium._reaction(recs, {"imine": "IM2"}, reactions)
    with pytest.raises(equilibrium.SampleError, match="0 matching"):
        equilibrium._reaction(
            recs, {"imine": "IM", "aldehyde": "diald"}, reactions
        )
    with pytest.raises(equilibrium.SampleError, match="no imine"):
        equilibrium._reaction(recs, {"imine": "nothing"}, reactions)
    with pytest.raises(equilibrium.SampleError, match="name the imine"):
        equilibrium._reaction(recs, {"amine": "am"}, reactions)


# ------------------------------------------------------------------ command


def nmr_project(tmp_path: Path) -> tuple[Path, object]:
    """Study in the data/ layout with nmr/ folders, steps 1-3 done."""
    root = tmp_path / "project"
    cmd_init(root, example=False, data_dir="data", experimental_dir="nmr")
    (root / "inputs" / "aldehydes.txt").write_text(to_txt(ALDEHYDES))
    (root / "inputs" / "amines.txt").write_text(to_txt(AMINES))
    with (root / "config" / "settings.toml").open("a") as fh:
        fh.write(FAST)
    cfg = load_config(root=root)
    pipeline.advance(cfg)  # steps 1-3: molecules.toml, reactions.toml
    return root, cfg


def test_nmr_command(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """samples.toml -> <reaction key>.toml read by the branch analysis."""
    root, cfg = nmr_project(tmp_path)
    folder = root / "data" / "benchmark" / "nmr"
    assert "[[sample]]" in (folder / "samples.toml").read_text()  # template
    write_experiment(folder / "raw" / "bza_ani" / "10", peaks(0.6))
    write_experiment(
        folder / "raw" / "bza_ani" / "11", peaks(0.62), processed=False
    )
    with (folder / "samples.toml").open("a") as fh:
        fh.write(
            "\n[[sample]]\n"
            'name = "24 h, processed"\n'
            'data = "raw/bza_ani/10"\n'
            'imine = "benzaldehyde + aniline (imine)"\n'
            "aldehyde_mol_l = 0.050\n"
            "amine_mol_l = 0.050\n"
            "\n[[sample]]\n"
            'name = "24 h, replicate from the raw FID"\n'
            'data = "raw/bza_ani/11"\n'
            'imine = "C(=Nc1ccccc1)c1ccccc1"\n'
            "aldehyde_mol_l = 0.050\n"
            "amine_mol_l = 0.050\n"
            "imine_ppm = [8.35, 8.55]\n"
            "\n[[sample]]\n"
            'name = "missing spectrum"\n'
            'data = "raw/nothing/1"\n'
            'aldehyde = "benzaldehyde"\n'
            'amine = "aniline"\n'
            "aldehyde_mol_l = 0.050\n"
            "amine_mol_l = 0.050\n"
        )
    capsys.readouterr()
    assert main(["--root", str(root), "nmr", "benchmark"]) == 0
    out = capsys.readouterr().out
    assert "reactions written: 1" in out
    assert "missing spectrum" in out and "no experiment folder" in out

    rxns = read_toml(cfg.reactions_file)["condensation"]
    (key,) = [r["key"] for r in rxns]
    res = tomllib.loads((folder / f"{key}.toml").read_text())
    assert res["source"] == "predcc nmr"
    assert res["n_samples"] == 2
    imine = next(r for r in project.molecules(cfg) if r["role"] == "imine")
    assert res["imine"]["compound"] == imine["compound"]
    assert res["imine"]["name"] == "benzaldehyde + aniline (imine)"
    assert res["imine"]["linkage"] == "imine"
    assert res["aldehyde"]["name"] == "benzaldehyde"
    assert res["amine"]["name"] == "aniline"
    assert res["imine"]["index_toml"] == f"index/{imine['compound']}.toml"
    expected = [math.log10((x * x) / ((1 - x) * (1 - x))) for x in (0.6, 0.62)]
    assert res["log_k"] == pytest.approx(sum(expected) / 2, abs=0.02)
    assert res["log_k_uncertainty"] >= 0.05
    assert [s["spectrum"] for s in res["samples"]] == ["processed", "raw"]
    assert res["samples"][0]["conversion"] == pytest.approx(0.6, abs=0.005)

    idx = read_toml(root / res["imine"]["index_toml"])  # root file of imine
    m = idx["measured"]["benchmark"]
    assert m["log_k"] == pytest.approx(res["log_k"])
    assert m["reaction"] == key and m["n_samples"] == 2
    assert m["file"] == f"data/benchmark/nmr/{key}.toml"
    aldehyde = res["aldehyde"]["compound"]
    assert "measured" not in read_toml(root / "index" / f"{aldehyde}.toml")

    exp = errors.experimental(folder, read_toml(cfg.reactions_file))
    assert list(exp) == [key]  # samples.toml itself is not a measurement
    assert exp[key]["log_k"] == pytest.approx(res["log_k"])
    table = root / "data" / "benchmark" / "derived" / "nmr_samples.csv"
    with table.open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2

    # A hand-written log K file is never overwritten
    (folder / f"{key}.toml").write_text(
        f'reaction = "{key}"\nlog_k = 9.9\nk_units = "dimensionless"\n'
    )
    assert main(["--root", str(root), "nmr", "benchmark"]) == 0
    assert "not overwritten" in capsys.readouterr().out
    assert read_toml(folder / f"{key}.toml")["log_k"] == 9.9


def test_nmr_command_without_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A branch without samples.toml is reported, not an error."""
    root, _ = nmr_project(tmp_path)
    (root / "data" / "production" / "nmr" / "samples.toml").unlink()
    assert main(["--root", str(root), "nmr"]) == 0
    assert "samples file: none" in capsys.readouterr().out
