"""Project registry, branch folders and trunk hand-over files (SOP 2.4).

The levels never import each other: a later level reads what an earlier one
wrote, through the readers here.

    molecules.toml                  species/ (trunk)    -> everything later
    molecules/<key>/structure.toml  structures/ (trunk) -> both branches
    molecules/<key>/best.xyz        the one geometry both branches use
    <branch>/molecules.txt          optional subset of a branch
    <branch>/molecules/<key>/       a branch's results, same compound key
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml

ROLES = ("aldehyde", "amine", "imine", "water")


def molecules(cfg: Config) -> list[dict]:
    """All molecule records from molecules.toml, in a stable order."""
    data = read_toml(cfg.molecules_file)
    out = []
    for role in ROLES:
        out += data.get(role, [])
    return out


def mol_dir(cfg: Config, rec: dict) -> Path:
    """Folder of one molecule."""
    return cfg.root / rec["directory"]


def branch_mol_dir(cfg: Config, branch: str, rec: dict) -> Path:
    """Folder of one molecule inside a branch: <branch>/molecules/<key>/."""
    return cfg.branch_dir(branch) / "molecules" / rec["compound"]


def identity(rec: dict, struct: dict | None = None) -> dict:
    """Molecule labels written into every per-molecule file of a branch."""
    out = {
        "compound": rec["compound"],
        "name": rec["name"],
        "role": rec["role"],
    }
    if struct is not None:
        out.update(
            isomer=struct["best"]["label"],
            inchikey=struct["best"]["inchikey"],
            geometry=struct["best"]["geometry"],
        )
    return out


class SelectionError(ValueError):
    """An entry of <branch>/molecules.txt matches no molecule."""


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _matches(entry: str, rec: dict) -> bool:
    if entry in (rec["name"], rec["compound"]) or any(
        entry == iso["inchikey"] for iso in rec.get("isomers", [])
    ):
        return True
    can = _canonical(entry)
    return can is not None and any(
        can == _canonical(s)
        for s in (rec.get("smiles"), rec.get("input_smiles"))
        if s
    )


def selection_file(cfg: Config, branch: str) -> Path:
    """<branch>/molecules.txt: the branch's subset of the trunk molecules."""
    return cfg.branch_dir(branch) / "molecules.txt"


def selected(cfg: Config, branch: str) -> list[dict]:
    """Trunk molecules this branch computes, in molecules.toml order.

    No molecules.txt (or no entries in it) means every molecule. Otherwise
    each line names an aldehyde, amine or imine (name, SMILES, compound key
    or InChIKey); an aldehyde/amine pair brings its imines, an imine brings
    its aldehyde and amine, and water is always included.
    """
    recs = molecules(cfg)
    f = selection_file(cfg, branch)
    entries = []
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    if not entries:
        return recs
    chosen: set[str] = set()
    unknown = []
    for entry in entries:
        hits = [r["compound"] for r in recs if _matches(entry, r)]
        if not hits:
            unknown.append(entry)
        chosen.update(hits)
    if unknown:
        raise SelectionError(
            f"{f}: no molecule matches {', '.join(unknown)} "
            "(see molecules.toml and rejected.toml)"
        )
    for r in recs:
        if r["role"] == "imine":
            parents = {
                k
                for o in r.get("origins", [])
                for k in (o["aldehyde"], o["amine"])
            }
            if r["compound"] in chosen:
                chosen |= parents
            elif any(
                o["aldehyde"] in chosen and o["amine"] in chosen
                for o in r.get("origins", [])
            ):
                chosen.add(r["compound"])
        elif r["role"] == "water":
            chosen.add(r["compound"])
    return [r for r in recs if r["compound"] in chosen]


def write_index(cfg: Config, branch: str, recs: list[dict]) -> Path:
    """<branch>/molecules.toml: the selected molecules, grouped by role."""
    from predcc_dft import SCHEMA_VERSION
    from predcc_dft.protocol.io import now, write_toml

    fields = ("compound", "name", "smiles", "directory")
    path = cfg.branch_dir(branch) / "molecules.toml"
    write_toml(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "branch": branch,
            "selection": (
                selection_file(cfg, branch).relative_to(cfg.root).as_posix()
                if selection_file(cfg, branch).exists()
                else "all trunk molecules"
            ),
            "counts": {
                role: sum(r["role"] == role for r in recs) for role in ROLES
            },
            **{
                role: [
                    {
                        **{k: r[k] for k in fields if k in r},
                        "inchikeys": [i["inchikey"] for i in r["isomers"]],
                        "branch_directory": branch_mol_dir(cfg, branch, r)
                        .relative_to(cfg.root)
                        .as_posix(),
                    }
                    for r in recs
                    if r["role"] == role
                ]
                for role in ROLES
                if any(r["role"] == role for r in recs)
            },
        },
    )
    return path


def structure(cfg: Config, rec: dict) -> dict | None:
    """structure.toml of a collapsed molecule (step 7), else None."""
    path = mol_dir(cfg, rec) / "structure.toml"
    if not path.exists():
        return None
    data = read_toml(path)
    return data if data.get("status") == "collapsed" else None
