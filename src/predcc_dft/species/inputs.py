"""Step 1: acquire compound lists and keep those within specification."""

from __future__ import annotations

import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from predcc_dft.chemistry import identifiers, smarts
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import read_toml


class InputError(ValueError):
    """Molecule rejected at step 1, with a reason."""


def standardise(
    smiles: str, strip_salts: bool = True, neutralise: bool = True
) -> tuple[Chem.Mol, list[str]]:
    """Parse, keep the largest fragment, neutralise, sanitise.

    Returns (mol, notes).
    """
    notes: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InputError("unparseable SMILES")
    if len(Chem.GetMolFrags(mol)) > 1:
        if not strip_salts:
            raise InputError("multiple fragments")
        chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        removed = Chem.MolToSmiles(mol)
        mol = chooser.choose(mol)
        notes.append(f"salt/fragments stripped from {removed}")
    if neutralise and Chem.GetFormalCharge(mol) != 0:
        before = Chem.MolToSmiles(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        if Chem.MolToSmiles(mol) != before:
            notes.append(f"neutralised from {before}")
    Chem.SanitizeMol(mol)
    return mol, notes


def read_list(path: Path) -> list[dict]:
    """Read an input list: .txt/.smi (one SMILES per line), .csv or .toml.

    Text files (the default): one molecule per line, ``SMILES [name]``; a
    line without a name is named by its SMILES; blank lines and lines
    starting with '#' are skipped.
    """
    if not path.exists():
        raise FileNotFoundError(f"input list not found: {path}")
    entries: list[dict] = []
    suffix = path.suffix.lower()
    if suffix == ".toml":
        for i, item in enumerate(read_toml(path).get("molecule", []), start=1):
            entries.append(
                {
                    "name": item.get("name", f"{path.stem}_{i}"),
                    "smiles": item["smiles"],
                    "source": f"{path.name}:{i}",
                }
            )
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = {f.lower().strip(): f for f in reader.fieldnames or []}
            if "smiles" not in fields:
                raise ValueError(
                    f"{path} needs a 'smiles' column (and optionally 'name')"
                )
            for i, row in enumerate(reader, start=2):
                smi = row[fields["smiles"]].strip()
                if not smi:
                    continue
                name = (
                    row[fields["name"]].strip()
                    if "name" in fields
                    else f"{path.stem}_{i}"
                )
                entries.append(
                    {"name": name, "smiles": smi, "source": f"{path.name}:{i}"}
                )
    else:  # .txt / .smi: one SMILES per line, optional name after a space
        for i, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line or line.startswith("#"):  # whole-line comments only:
                continue  # '#' inside a SMILES is a triple bond
            parts = line.split(maxsplit=1)
            entries.append(
                {
                    "name": parts[1].strip() if len(parts) > 1 else parts[0],
                    "smiles": parts[0],
                    "source": f"{path.name}:{i}",
                }
            )
    return entries


def _check_common(mol: Chem.Mol, spec: dict) -> None:
    props = identifiers.properties(mol)
    elements = {a.GetSymbol() for a in mol.GetAtoms()} | {"H"}
    bad = sorted(elements - set(spec["allowed_elements"]))
    if bad:
        raise InputError(f"disallowed elements: {', '.join(bad)}")
    if spec.get("neutral_only", True) and props["charge"] != 0:
        raise InputError(f"net charge {props['charge']}")
    if props["multiplicity"] != 1:
        raise InputError("radical / open shell")
    if props["heavy_atoms"] > spec["max_heavy_atoms"]:
        raise InputError(
            f"{props['heavy_atoms']} heavy atoms > {spec['max_heavy_atoms']}"
        )
    if props["rotatable_bonds"] > spec["max_rotatable_bonds"]:
        raise InputError(
            f"{props['rotatable_bonds']} rotatable bonds > "
            f"{spec['max_rotatable_bonds']}"
        )


def _check_exclusions(mol: Chem.Mol, exclusions: dict) -> None:
    for name, pattern in exclusions.items():
        patt = Chem.MolFromSmarts(pattern)
        if patt is None:
            raise ValueError(f"bad exclusion SMARTS {name}: {pattern}")
        if mol.HasSubstructMatch(patt):
            raise InputError(f"competing group: {name}")


def classify_nucleophile(mol: Chem.Mol) -> dict[str, list[int]]:
    """Reacting NH2 atom indices per nucleophile class (input classes only)."""
    found: dict[str, list[int]] = {}
    for cls, pattern in smarts.NUCLEOPHILE_CLASSES.items():
        if cls == "hydrazone":
            continue
        sites = smarts.match_sites(mol, pattern)
        if sites:
            found[cls] = sites
    return found


def screen(
    entries: list[dict], role: str, cfg: Config
) -> tuple[list[dict], list[dict]]:
    """Return (accepted, rejected); accepted gain mol, sites and classes."""
    spec = cfg["specification"]
    exclusions = spec.get(f"{role}_exclude", {})
    accepted, rejected = [], []
    for entry in entries:
        record = dict(entry, role=role)
        try:
            mol, notes = standardise(
                entry["smiles"],
                spec.get("strip_salts", True),
                spec.get("neutralise", True),
            )
            _check_common(mol, spec)
            if role == "aldehyde":
                sites = smarts.match_sites(mol, smarts.ALDEHYDE_SMARTS)
                if not sites:
                    raise InputError(
                        "no aldehyde group "
                        "(ketones and formaldehyde not supported)"
                    )
                classes = {"aldehyde": sites}
            else:
                found = classify_nucleophile(mol)
                if not found:
                    raise InputError("no reactive NH2 of a supported class")
                allowed = set(cfg["products"]["linkages"])
                classes = {
                    c: s
                    for c, s in found.items()
                    if smarts.CLASS_TO_LINKAGE[c] in allowed
                }
                if not classes:
                    raise InputError(
                        f"{', '.join(found)}: product class not enabled in "
                        "[products] linkages"
                    )
                sites = sorted({i for s in classes.values() for i in s})
            if len(sites) > spec["max_reactive_sites"]:
                raise InputError(
                    f"{len(sites)} reactive sites > "
                    f"{spec['max_reactive_sites']}"
                )
            _check_exclusions(mol, exclusions)
        except InputError as err:
            rejected.append({**record, "reason": str(err)})
            continue
        record.update(
            mol=mol,
            sites=sites,
            classes=classes,
            notes=notes,
            canonical_smiles=identifiers.canonical(mol),
        )
        accepted.append(record)
    return accepted, rejected
