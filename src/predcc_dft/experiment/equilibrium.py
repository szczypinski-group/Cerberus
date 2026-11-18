"""Measured log K from NMR samples (SOP 4.4).

Each branch's measured-data folder (<branch>/<experimental_dir>/, e.g.
data/benchmark/nmr/) holds the Bruker experiment folders and samples.toml:

    [defaults]                      # any key can be overridden per sample
    aldehyde_ppm = [9.90, 10.10]    # aldehyde CHO signal
    imine_ppm = [8.30, 8.60]        # imine CH=N signal
    aldehyde_protons = 1
    imine_protons = 1
    ...

    [[sample]]
    name = "benzaldehyde + aniline, 24 h"
    data = "raw/benzaldehyde_aniline/10"   # experiment folder, relative
    imine = "C(=Nc1ccccc1)c1ccccc1"        # name, SMILES, key or InChIKey
    aldehyde_mol_l = 0.050
    amine_mol_l = 0.050

The imine is looked up in molecules.toml: its aldehyde and amine (origins)
give the reaction; naming aldehyde and amine instead works too. Imines that
form at one of several sites (multi_site) are refused.

Per sample: conversion x = I_imine/n_imine / (I_imine/n_imine +
I_aldehyde/n_aldehyde); [imine] = x A0, [aldehyde] = (1 - x) A0,
[amine] = N0 - x A0, [water] = W0 + x A0 (water is produced with the
imine); K = [imine][water] / ([aldehyde][amine]), dimensionless.

Per reaction (all samples of one aldehyde + amine): log K is the mean,
its uncertainty the larger of the replicate spread, the integration error
(integral_rel_error, propagated) and log_k_uncertainty_min. Written to
<reaction key>.toml beside samples.toml, the file the branch analyses read.
A <reaction key>.toml that `predcc nmr` did not write is never overwritten.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.experiment import nmr
from predcc_dft.protocol import project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml, write_toml

SAMPLES = "samples.toml"
SOURCE = "predcc nmr"
DEFAULTS = {
    "aldehyde_ppm": [9.90, 10.10],
    "imine_ppm": [8.30, 8.60],
    "aldehyde_protons": 1,
    "imine_protons": 1,
    "spectrum": "auto",
    "procno": 1,
    "line_broadening_hz": 0.3,
    "water_mol_l": 0.0,
    "temperature_k": 298.15,
    "solvent": "CD3CN",
    "integral_rel_error": 0.02,
    "log_k_uncertainty_min": 0.05,
}

TEMPLATE = """\
# NMR samples of this branch: `predcc nmr` integrates each spectrum and
# writes <reaction key>.toml (measured log K) beside this file.
# Put each Bruker experiment folder (fid, acqus, pdata/1/1r ...) anywhere
# below this folder and point `data` at it. Processed spectra (pdata) are
# used when present (phase and baseline them in TopSpin); otherwise the raw
# FID is processed automatically.

[defaults]                      # any of these can be set per sample too
aldehyde_ppm = [9.90, 10.10]    # aldehyde CHO signal
imine_ppm = [8.30, 8.60]        # imine CH=N signal
aldehyde_protons = 1            # protons in each integrated signal
imine_protons = 1
spectrum = "auto"               # auto | processed | raw
procno = 1                      # pdata/<procno>
line_broadening_hz = 0.3        # raw FIDs only
water_mol_l = 0.0               # water added at the start (not produced)
temperature_k = 298.15
solvent = "CD3CN"
integral_rel_error = 0.02       # relative error of each integral
log_k_uncertainty_min = 0.05

# [[sample]]
# name = "benzaldehyde + aniline, 24 h"
# data = "raw/benzaldehyde_aniline/10"
# imine = "benzaldehyde + aniline (imine)"  # name, SMILES, compound key or
#                               # InChIKey, as in molecules.toml; its
#                               # aldehyde and amine are taken from there
# aldehyde_mol_l = 0.050
# amine_mol_l = 0.050
"""


class SampleError(ValueError):
    """A sample cannot be turned into a K value."""


def samples_path(cfg: Config, branch: str) -> Path:
    """<branch>/<experimental_dir>/samples.toml."""
    return cfg.experimental_dir(branch) / SAMPLES


def conversion(i_ald: float, i_im: float, n_ald: float, n_im: float) -> float:
    """Fraction of the aldehyde present as imine, per proton."""
    ald, im = i_ald / n_ald, i_im / n_im
    if ald + im <= 0:
        raise SampleError("no signal in either range")
    return im / (ald + im)


def equilibrium_k(x: float, a0: float, n0: float, w0: float) -> float:
    """K = [imine][water] / ([aldehyde][amine]) at conversion x."""
    if not 0.0 < x < 1.0:
        raise SampleError(f"conversion {x:.4f}: K is 0 or infinite")
    imine, water = x * a0, w0 + x * a0
    ald, amine = (1 - x) * a0, n0 - x * a0
    if amine <= 0:
        raise SampleError(
            f"conversion {x:.4f} needs more amine than {n0} mol/L"
        )
    return imine * water / (ald * amine)


def _reaction(recs: list[dict], s: dict, reactions: dict) -> dict:
    """Reaction, imine, aldehyde and amine records of a sample.

    The sample names its imine (name, SMILES, compound key or InChIKey):
    the aldehyde and amine come from the imine's origins in
    molecules.toml. Naming aldehyde and amine works too, and narrows an
    imine with several origins.
    """
    by_key = {r["compound"]: r for r in recs}

    def pick(entry: str, role: str) -> dict:
        hits = [
            r
            for r in recs
            if r["role"] == role and project.matches(str(entry), r)
        ]
        if not hits:
            raise SampleError(f"no {role} {entry!r} in molecules.toml")
        return hits[0]

    imine = pick(s["imine"], "imine") if "imine" in s else None
    ald = pick(s["aldehyde"], "aldehyde") if "aldehyde" in s else None
    amine = pick(s["amine"], "amine") if "amine" in s else None
    if imine is not None:
        origins = [
            o
            for o in imine.get("origins", [])
            if (ald is None or o["aldehyde"] == ald["compound"])
            and (amine is None or o["amine"] == amine["compound"])
        ]
        if len(origins) != 1:
            raise SampleError(
                f"{imine['name']} has {len(origins)} matching aldehyde/amine "
                "origins: name aldehyde and amine in the sample"
            )
        ald = by_key[origins[0]["aldehyde"]]
        amine = by_key[origins[0]["amine"]]
    if ald is None or amine is None:
        raise SampleError("name the imine (or the aldehyde and the amine)")
    for rxn in reactions.get("condensation", []):
        roles = {p["role"]: p["compound"] for p in rxn["reactants"]}
        products = {p["role"]: p["compound"] for p in rxn["products"]}
        if (
            roles.get("aldehyde") != ald["compound"]
            or roles.get("amine") != amine["compound"]
            or (
                imine is not None
                and products.get("imine") != imine["compound"]
            )
        ):
            continue
        imine = imine or by_key.get(products.get("imine", ""))
        if imine is None:
            continue
        if imine.get("multi_site"):
            raise SampleError(
                f"{imine['name']} forms at one of several aldehyde/amine "
                "sites (multi_site in molecules.toml): the conversion "
                "assumes one aldehyde group per molecule"
            )
        return {"rxn": rxn, "aldehyde": ald, "amine": amine, "imine": imine}
    raise SampleError(
        f"no condensation of {ald['name']} + {amine['name']} in reactions.toml"
    )


def _identity(rec: dict, index: Path) -> dict:
    """Identity of a molecule as stored in molecules.toml."""
    out = {
        "compound": rec["compound"],
        "name": rec["name"],
        "smiles": rec.get("smiles", ""),
        "inchikeys": [i["inchikey"] for i in rec.get("isomers", [])],
        "directory": rec.get("directory", ""),
        "index_toml": index.as_posix(),
    }
    if rec.get("linkage"):
        out["linkage"] = rec["linkage"]
    return out


def evaluate(folder: Path, s: dict) -> dict:
    """Integrals, conversion and K (with its uncertainty) of one sample."""
    for key in ("name", "data", "aldehyde_mol_l", "amine_mol_l"):
        if key not in s:
            raise SampleError(f"missing {key}")
    spec = nmr.read(
        folder / s["data"],
        str(s["spectrum"]),
        int(s["procno"]),
        float(s["line_broadening_hz"]),
    )
    i_ald = nmr.integrate(spec, *s["aldehyde_ppm"])
    i_im = nmr.integrate(spec, *s["imine_ppm"])
    if i_ald < 0 or i_im < 0:
        raise SampleError(
            f"negative integral (aldehyde {i_ald:.4g}, imine {i_im:.4g}): "
            "check the phase/baseline or the ppm ranges"
        )
    n_ald, n_im = float(s["aldehyde_protons"]), float(s["imine_protons"])
    a0, n0 = float(s["aldehyde_mol_l"]), float(s["amine_mol_l"])
    w0 = float(s["water_mol_l"])
    x = conversion(i_ald, i_im, n_ald, n_im)
    k = equilibrium_k(x, a0, n0, w0)
    e = float(s["integral_rel_error"])
    ks = []
    for f_ald, f_im in ((1 + e, 1 - e), (1 - e, 1 + e)):
        xx = conversion(i_ald * f_ald, i_im * f_im, n_ald, n_im)
        ks.append(equilibrium_k(xx, a0, n0, w0))
    return {
        "name": s["name"],
        "data": s["data"],
        "spectrum": spec.source,
        "aldehyde_ppm": list(s["aldehyde_ppm"]),
        "imine_ppm": list(s["imine_ppm"]),
        "integral_aldehyde": i_ald,
        "integral_imine": i_im,
        "conversion": x,
        "aldehyde0_mol_l": a0,
        "amine0_mol_l": n0,
        "water0_mol_l": w0,
        "k": k,
        "log_k": math.log10(k),
        "log_k_integration_error": abs(math.log10(ks[1]) - math.log10(ks[0]))
        / 2,
    }


def _ours(path: Path) -> bool:
    return not path.exists() or read_toml(path).get("source") == SOURCE


def run(cfg: Config, branch: str) -> dict:
    """Evaluate every sample of a branch; write <reaction key>.toml files."""
    path = samples_path(cfg, branch)
    if not path.exists():
        return {"samples file": "none"}
    if not cfg.reactions_file.exists():
        raise SampleError("no reactions.toml yet: run `predcc prepare`")
    data = read_toml(path)
    defaults = DEFAULTS | data.get("defaults", {})
    recs = project.molecules(cfg)
    reactions = read_toml(cfg.reactions_file)
    folder = path.parent
    by_rxn: dict[str, dict] = {}
    rows, failed = [], {}
    for raw in data.get("sample", []):
        s = defaults | raw
        name = s.get("name", s.get("data", "?"))
        try:
            hit = _reaction(recs, s, reactions)
            res = evaluate(folder, s)
        except (SampleError, nmr.SpectrumError, KeyError) as exc:
            failed[name] = str(exc)
            continue
        key = hit["rxn"]["key"]
        entry = by_rxn.setdefault(key, {"hit": hit, "samples": [], "s": s})
        entry["samples"].append(res)
        res["imine"] = hit["imine"]["name"]
        rows.append(
            {
                "reaction": key,
                **{k: v for k, v in res.items() if not isinstance(v, list)},
            }
        )  # noqa: E501
    written, kept = [], []
    for key, entry in by_rxn.items():
        dest = folder / f"{key}.toml"
        if not _ours(dest):
            kept.append(dest.name)
            continue
        logs = [r["log_k"] for r in entry["samples"]]
        spread = statistics.stdev(logs) if len(logs) > 1 else 0.0
        integ = statistics.mean(
            r["log_k_integration_error"] for r in entry["samples"]
        )  # noqa: E501
        s, hit = entry["s"], entry["hit"]
        write_toml(
            dest,
            {
                "schema_version": SCHEMA_VERSION,
                "source": SOURCE,
                "generated": now(),
                "reaction": key,
                "log_k": statistics.mean(logs),
                "log_k_uncertainty": max(
                    spread, integ, float(s["log_k_uncertainty_min"])
                ),
                "n_samples": len(logs),
                "k_definition": "[imine][H2O]/([aldehyde][amine])",
                "k_units": "dimensionless",
                "water_included": True,
                "source_method": "nmr_equilibrium",
                "temperature_k": float(s["temperature_k"]),
                "solvent": s["solvent"],
                "imine": _identity(
                    hit["imine"],
                    Path("index") / f"{hit['imine']['compound']}.toml",
                ),
                "aldehyde": _identity(
                    hit["aldehyde"],
                    Path("index") / f"{hit['aldehyde']['compound']}.toml",
                ),
                "amine": _identity(
                    hit["amine"],
                    Path("index") / f"{hit['amine']['compound']}.toml",
                ),
                "samples": entry["samples"],
            },
        )
        written.append(dest.name)
    table = cfg.branch_dir(branch) / "derived" / "nmr_samples.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    out: dict = {
        "samples": len(data.get("sample", [])),
        "reactions written": len(written),
    }
    if kept:
        out["not overwritten (not written by predcc nmr)"] = kept
    if failed:
        out["failed"] = failed
    return out
