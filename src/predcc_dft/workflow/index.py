"""index/<label>.toml: the root file of every molecule, across all modules.

Each module keeps its own file per molecule, in its own folder:

    trunk        molecules/<key>/trunk.toml
    benchmark    benchmark/molecules/<key>/benchmark.toml
    production   production/molecules/<key>/production.toml
    ts           ts/species/<label>/ts.toml

and index/<label>.toml (label = compound key; hemiaminals, TFA and salts of
the TS module have their own labels) names the molecule once and points to
every module file, with a one-line status of each. It is rebuilt after every
round of every chain (and by `predcc analyse`); modules never write here.
`predcc show <name | SMILES | key | InChIKey>` prints the matching files.
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.benchmark import molecule as benchmark_molecule
from predcc_dft.production import molecule as production_molecule
from predcc_dft.protocol import project
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import now, read_toml, write_toml
from predcc_dft.structures import summary as trunk_summary
from predcc_dft.ts import stage as ts_stage

INDEX_DIR = "index"


def index_dir(cfg: Config) -> Path:
    """<study>/index/."""
    return cfg.root / INDEX_DIR


def _rel(cfg: Config, p: Path) -> str:
    return p.relative_to(cfg.root).as_posix()


def _read(p: Path) -> dict:
    return read_toml(p) if p.exists() else {}


def _selected(cfg: Config, branch: str) -> set[str]:
    try:
        return {r["compound"] for r in project.selected(cfg, branch)}
    except project.SelectionError:
        return set()


def _ts_entry(cfg: Config, label: str) -> dict:
    p = ts_stage.molecule_file(cfg, label)
    data = _read(p)
    entry: dict = {"file": _rel(cfg, p), "present": p.exists()}
    if data:
        entry["status"] = data.get("status", "")
        entry["reactions"] = [
            {
                **r,
                "reaction_toml": _rel(
                    cfg, ts_stage.reaction_file(cfg, r["reaction"])
                ),
            }
            for r in data.get("in_reactions", [])
        ]
        entry["energy_protocols"] = sorted(data.get("energies", {}))
    return entry


def _trunk_molecule(cfg: Config, rec: dict, sel: dict) -> dict:
    t = trunk_summary.path(cfg, rec)
    b = benchmark_molecule.path(cfg, rec)
    pr = production_molecule.path(cfg, rec)
    tdata, bdata, pdata = _read(t), _read(b), _read(pr)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "label": rec["compound"],
        **project.identity(rec),
        "smiles": rec.get("smiles", ""),
        "inchikeys": [i["inchikey"] for i in rec["isomers"]],
        "modules": {
            "trunk": {
                "file": _rel(cfg, t),
                "present": t.exists(),
                "status": tdata.get("status", {}).get(
                    "structure (step 7)", ""
                ),
                "best_xyz": tdata.get("files", {}).get("best_xyz", ""),
            },
            "benchmark": {
                "file": _rel(cfg, b),
                "present": b.exists(),
                "selected": rec["compound"] in sel["benchmark"],
                **bdata.get("status", {}),
            },
            "production": {
                "file": _rel(cfg, pr),
                "present": pr.exists(),
                "selected": rec["compound"] in sel["production"],
                "versions": {
                    v: d.get("status", "")
                    for v, d in pdata.get("versions", {}).items()
                },
            },
            "ts": {
                **_ts_entry(cfg, rec["compound"]),
                "selected": rec["compound"] in sel["ts"],
            },
        },
    }


def _ts_only(cfg: Config, label: str) -> dict:
    data = _read(ts_stage.molecule_file(cfg, label))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "label": label,
        "name": data.get("name", ""),
        "role": data.get("role", ""),
        "smiles": data.get("smiles", ""),
        "modules": {"ts": _ts_entry(cfg, label)},
    }


def run(cfg: Config) -> int:
    """Rewrite index/<label>.toml of every molecule; returns how many."""
    if not cfg.molecules_file.exists():
        return 0
    sel = {b: _selected(cfg, b) for b in ("benchmark", "production", "ts")}
    out: dict[str, dict] = {}
    for rec in project.molecules(cfg):
        out[rec["compound"]] = _trunk_molecule(cfg, rec, sel)
    for p in sorted((cfg.ts_dir / "species").glob("*/ts.toml")):
        if p.parent.name not in out:
            out[p.parent.name] = _ts_only(cfg, p.parent.name)
    for label, data in out.items():
        write_toml(index_dir(cfg) / f"{label}.toml", data)
    return len(out)


def find(cfg: Config, query: str) -> list[Path]:
    """Index files of the molecules that a query names."""
    hits = []
    for p in sorted(index_dir(cfg).glob("*.toml")):
        data = read_toml(p)
        rec = {
            "name": data.get("name", ""),
            "compound": data.get("compound", data.get("label", "")),
            "smiles": data.get("smiles", ""),
            "isomers": [{"inchikey": k} for k in data.get("inchikeys", [])],
        }
        if query.strip() == data.get("label") or project.matches(query, rec):
            hits.append(p)
    return hits
