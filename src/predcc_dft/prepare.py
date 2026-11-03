"""Steps 1-3 driver: lists -> specification -> products -> seed coordinates.

Writes, in the main project directory:
  molecules.toml      every molecule, grouped by type ([[aldehyde]],
                      [[amine]], [[imine]], [[water]]), with
                      isomers and seed files
  reactions.toml      condensation (CND-) and exchange (TRN-) reactions
  rejected.toml       inputs outside specification, with the reason
  molecules/<compound key>/molecule.toml + seeds/<label>.xyz|.sdf
"""

from __future__ import annotations

import hashlib
import logging

from rdkit import Chem

from predcc_dft import (
    SCHEMA_VERSION,
    __version__,
    chem,
    compounds,
    coordinates,
    products,
)
from predcc_dft.io_utils import (
    Config,
    now,
    pipeline_commit,
    read_toml,
    write_text,
    write_toml,
)

log = logging.getLogger(__name__)

ROLE_ORDER = ["aldehyde", "amine", "imine", "water"]


def _input_compound(entry: dict) -> dict:
    """Record of one input molecule, atoms in canonical order."""
    mol = products._canonical_order(entry["mol"])
    # Site indices refer to the original order; remap to the canonical one.
    ranks = list(
        Chem.CanonicalRankAtoms(
            entry["mol"], breakTies=True, includeChirality=False
        )
    )
    new_index = {old: rank for old, rank in enumerate(ranks)}
    remap = lambda idxs: sorted(new_index[i] for i in idxs)  # noqa: E731
    key = chem.inchikey(mol)
    record = {
        "compound": chem.compound_key(key),
        "role": entry["role"],
        "name": entry["name"],
        "source": entry["source"],
        "input_smiles": entry["smiles"],
        "smiles": chem.canonical(mol),
        "notes": entry["notes"],
        "sites": remap(entry["sites"]),
        "isomers": {
            chem.canonical(mol): {
                "mol": mol,
                "smiles": chem.canonical(mol),
                "inchikey": key,
                "inchi": chem.inchi(mol),
                "stereo": products._stereo_label(mol),
            }
        },
    }
    if entry["role"] == "amine":
        record["classes"] = {c: remap(s) for c, s in entry["classes"].items()}
        record["linkages"] = sorted(
            {chem.CLASS_TO_LINKAGE[c] for c in entry["classes"]}
        )
    return record


def _water() -> dict:
    mol = Chem.MolFromSmiles(chem.WATER_SMILES)
    key = chem.inchikey(mol)
    return {
        "compound": chem.compound_key(key),
        "role": "water",
        "name": "water",
        "smiles": "O",
        "sites": [],
        "notes": [],
        "isomers": {
            "O": {
                "mol": mol,
                "smiles": "O",
                "inchikey": key,
                "inchi": chem.inchi(mol),
                "stereo": "none",
            }
        },
    }


def _seeds_for(comp: dict, iso: dict, cfg: Config) -> list[str | None]:
    if (
        comp.get("linkage") == "acylhydrazone"
        and cfg["products"].get("acylhydrazone_rotamers", True)
        and coordinates.amide_rotamer_atoms(iso["mol"]) is not None
    ):
        return ["syn", "anti"]
    return [None]


def _write_molecule(comp: dict, cfg: Config, force: bool) -> dict:
    """Create molecules/<key>/ with seed geometries; return the record."""
    mdir = cfg.molecules_dir / comp["compound"]
    seeds_dir = mdir / "seeds"
    iso_records = []
    for iso in comp["isomers"].values():
        seed_records = []
        for rot in _seeds_for(comp, iso, cfg):
            label = iso["stereo"] if rot is None else f"{iso['stereo']}_{rot}"
            xyz_path, sdf_path = (
                seeds_dir / f"{label}.xyz",
                seeds_dir / f"{label}.sdf",
            )
            meta_path = seeds_dir / f"{label}.toml"
            if xyz_path.exists() and meta_path.exists() and not force:
                meta = read_toml(meta_path)
            else:
                molh, meta = coordinates.embed(iso["mol"], cfg, rot)
                write_text(
                    xyz_path,
                    coordinates.xyz_block(molh, f"{iso['inchikey']} {label}"),
                )
                write_text(sdf_path, Chem.MolToMolBlock(molh))
                meta = {
                    "label": label,
                    "inchikey": iso["inchikey"],
                    "smiles": iso["smiles"],
                    **meta,
                }
                write_toml(meta_path, meta)
            seed_records.append(
                {
                    "label": label,
                    "xyz": xyz_path.relative_to(cfg.root).as_posix(),
                    "sdf": sdf_path.relative_to(cfg.root).as_posix(),
                    "random_seed": meta["random_seed"],
                    "rotamer": meta.get("rotamer_after_ff", ""),
                }
            )
        iso_rec = {
            "inchikey": iso["inchikey"],
            "inchi": iso["inchi"],
            "smiles": iso["smiles"],
            "stereo": iso["stereo"],
            "seeds": seed_records,
        }
        if "cn_bond" in iso:
            iso_rec["cn_bond_atoms"] = iso["cn_bond"]
        if iso.get("key_collision"):
            iso_rec["key_collision"] = True
            iso_rec["inchikey_standard"] = iso["inchikey_standard"]
        iso_records.append(iso_rec)

    molh = Chem.AddHs(next(iter(comp["isomers"].values()))["mol"])
    record = {
        "compound": comp["compound"],
        "role": comp["role"],
        "name": comp["name"],
        "smiles": comp["smiles"],
        "directory": mdir.relative_to(cfg.root).as_posix(),
        **chem.properties(molh),
        **{
            k: comp[k]
            for k in (
                "linkage",
                "linkages",
                "classes",
                "source",
                "input_smiles",
            )
            if k in comp
        },
        "reactive_sites": comp.get("sites", []),
        "n_reactive_sites": len(comp.get("sites", [])),
        "notes": comp.get("notes", []),
    }
    if "origins" in comp:
        record["origins"] = comp["origins"]
        record["multi_site"] = any(o["multi_site"] for o in comp["origins"])
    record["isomers"] = iso_records

    write_toml(
        mdir / "molecule.toml",
        {
            "schema_version": SCHEMA_VERSION,
            "generated": now(),
            "pipeline_version": __version__,
            "molecule": record,
            "settings": {
                "coordinates": dict(cfg["coordinates"]),
                "solvent": cfg["conformers"]["solvent"],
            },
        },
    )
    return record


def _reactions(records: dict[str, dict], water_key: str, cfg: Config) -> dict:
    """Condensation (CND-) and exchange (TRN-) reactions (SOP 2.3)."""
    condensations: dict[str, dict] = {}
    exchanges: dict[str, dict] = {}
    groups: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for rec in records.values():
        if rec["role"] != "imine":
            continue
        for org in rec["origins"]:
            key, canon = chem.reaction_key(
                "CND",
                [org["aldehyde"], org["amine"]],
                [rec["compound"], water_key],
            )
            entry = condensations.setdefault(
                key,
                {
                    "key": key,
                    "type": "condensation",
                    "linkages": [rec["linkage"]],
                    "canonical": canon,
                    "reactants": [
                        {
                            "compound": org["aldehyde"],
                            "role": "aldehyde",
                            "coefficient": 1,
                        },
                        {
                            "compound": org["amine"],
                            "role": "amine",
                            "coefficient": 1,
                        },
                    ],
                    "products": [
                        {
                            "compound": rec["compound"],
                            "role": "imine",
                            "coefficient": 1,
                        },
                        {
                            "compound": water_key,
                            "role": "water",
                            "coefficient": 1,
                        },
                    ],
                    "site_tags": [],
                },
            )
            if org["site_tag"] not in entry["site_tags"]:
                entry["site_tags"].append(org["site_tag"])
            if rec["linkage"] != "azine":
                groups.setdefault(
                    (org["aldehyde"], org["aldehyde_site"]), set()
                ).add((rec["compound"], org["amine"]))

    if cfg["products"].get("exchange_reactions", True):
        for (ald, site), member_set in groups.items():
            members = sorted(member_set)
            for i, (pa, na) in enumerate(members):
                for pb, nb in members[i + 1 :]:
                    if na == nb or pa == pb:
                        continue
                    # Direction: reactant condensate key sorts first.
                    key, canon = chem.reaction_key("TRN", [pa, nb], [pb, na])
                    exchanges.setdefault(
                        key,
                        {
                            "key": key,
                            "type": "exchange",
                            "canonical": canon,
                            "linkages": sorted(
                                {
                                    records[pa]["linkage"],
                                    records[pb]["linkage"],
                                }
                            ),
                            "aldehyde": ald,
                            "aldehyde_site": site,
                            "reactants": [
                                {
                                    "compound": pa,
                                    "role": "imine",
                                    "coefficient": 1,
                                },
                                {
                                    "compound": nb,
                                    "role": "amine",
                                    "coefficient": 1,
                                },
                            ],
                            "products": [
                                {
                                    "compound": pb,
                                    "role": "imine",
                                    "coefficient": 1,
                                },
                                {
                                    "compound": na,
                                    "role": "amine",
                                    "coefficient": 1,
                                },
                            ],
                        },
                    )
    return {
        "condensation": list(condensations.values()),
        "exchange": list(exchanges.values()),
    }


def inputs_fingerprint(cfg: Config) -> str:
    """SHA-256 over the input lists and the settings that shape steps 1-3."""
    h = hashlib.sha256()
    for key in ("aldehydes", "amines"):
        path = cfg.path(cfg["inputs"][key])
        h.update(path.read_bytes() if path.exists() else b"missing")
    for section in ("specification", "products", "coordinates"):
        h.update(repr(sorted(cfg[section].items())).encode())
    return h.hexdigest()


def up_to_date(cfg: Config) -> bool:
    """True if molecules.toml was built from the current inputs."""
    if not cfg.molecules_file.exists():
        return False
    data = read_toml(cfg.molecules_file)
    return data.get("inputs_sha256") == inputs_fingerprint(cfg)


def run(cfg: Config, force: bool = False) -> dict:
    """Run steps 1-3. Returns a summary dict."""
    # Step 1: lists within specification
    accepted, rejected = {}, []
    for role, key in (
        ("aldehyde", "aldehydes"),
        ("amine", "amines"),
    ):
        entries = compounds.read_list(cfg.path(cfg["inputs"][key]))
        ok, bad = compounds.screen(entries, role, cfg)
        accepted[role] = ok
        rejected += bad
    log.info(
        "step 1: %d aldehydes, %d amines accepted; %d rejected",
        len(accepted["aldehyde"]),
        len(accepted["amine"]),
        len(rejected),
    )

    comps: dict[str, dict] = {}
    inputs: dict[str, list[dict]] = {"aldehyde": [], "amine": []}
    for role in ("aldehyde", "amine"):
        for entry in accepted[role]:
            rec = _input_compound(entry)
            if rec["compound"] in comps:
                rejected.append(
                    {
                        **{
                            k: entry[k]
                            for k in ("name", "smiles", "source", "role")
                        },
                        "reason": "duplicate of "
                        f"{comps[rec['compound']]['name']}",
                    }
                )
                continue
            comps[rec["compound"]] = rec
            first_iso = next(iter(rec["isomers"].values()))
            inputs[role].append({**rec, "mol": first_iso["mol"]})

    # Step 2: products via SMARTS (site-tagged, E/Z enumerated)
    prod_set = products.generate(inputs["aldehyde"], inputs["amine"], cfg)
    for ckey, comp in prod_set.compounds.items():
        if (
            ckey in comps
        ):  # a product that is also an input: keep the input record
            log.warning(
                "product %s is also an input molecule; kept as input", ckey
            )
            continue
        comps[ckey] = comp
    water = _water()
    comps[water["compound"]] = water
    log.info("step 2: %d condensation products", len(prod_set.compounds))

    # Step 3: coordinates and per-molecule folders
    records = {k: _write_molecule(c, cfg, force) for k, c in comps.items()}
    n_seeds = sum(
        len(i["seeds"]) for r in records.values() for i in r["isomers"]
    )
    log.info(
        "step 3: %d seed geometries in %d molecule folders",
        n_seeds,
        len(records),
    )

    reactions = _reactions(records, water["compound"], cfg)
    header = {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "pipeline_version": __version__,
        "pipeline": pipeline_commit(),
    }

    grouped = {
        role: [r for r in records.values() if r["role"] == role]
        for role in ROLE_ORDER
    }
    write_toml(
        cfg.molecules_file,
        {
            **header,
            "inputs_sha256": inputs_fingerprint(cfg),
            "settings": {
                "coordinates": dict(cfg["coordinates"]),
                "specification": {
                    k: v
                    for k, v in cfg["specification"].items()
                    if not isinstance(v, dict)
                },
            },
            "counts": {role: len(v) for role, v in grouped.items()},
            **{role: v for role, v in grouped.items() if v},
        },
    )
    write_toml(
        cfg.reactions_file,
        {
            **header,
            "counts": {k: len(v) for k, v in reactions.items()},
            **{k: v for k, v in reactions.items() if v},
        },
    )
    write_toml(cfg.root / "rejected.toml", {**header, "rejected": rejected})
    return {
        "molecules": {r: len(v) for r, v in grouped.items()},
        "seeds": n_seeds,
        "rejected": len(rejected),
        "reactions": {k: len(v) for k, v in reactions.items()},
    }
