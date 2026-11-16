"""TS module jobs (SOP 10): autodE TS searches, minima, single points.

    ts/
      molecules.txt                       optional subset (as in the branches)
      reactions/<CND key>/
        reaction.toml                     molecules, channels, kinetics
        <step>_<shuttle>/                 one channel, e.g. addition_tfa
          channel.toml                    EVERYTHING about this TS: mechanism,
                                          state, every file path, autodE
                                          result, frequencies, energies
          reactant.xyz product.xyz        the step's complexes (same order)
          autode/                         task.toml, run.sh, ts.xyz, work/
          freq/                           ORCA Freq on the autodE TS
          sp/<tag>/{energy, solv_gas, solv_<model>}
      species/<label>/                    every reactant, product, hemiaminal
        ts.toml                           EVERYTHING about this molecule:
                                          identity, role (reactant/product/
                                          intermediate/shuttle), every file
                                          path, conformers, energies
        conformers.xyz conf<i>/ best.xyz  (optimised here: hemiaminal, TFA,
        sp/<tag>/...                       salt; trunk species reuse best.xyz)

One TOML per molecule (ts.toml, like trunk.toml, benchmark.toml and
production.toml in the other modules; index/<key>.toml links them) and one
per TS (channel.toml) hold every path and number, so a parser never has to
walk the folders. <tag> names the energy protocol: v<version>
(production lock: its method on these geometries plus its solvation model)
or "opt" (the r2SCAN-3c/CPCM energies of the Opt/Freq jobs, no extra jobs).

Stages: ts_minima (ORCA Opt+Freq of the hemiaminal, TFA and salt
conformers), ts_autode (one autodE search per channel, with xTB and ORCA),
ts_freq (ORCA Freq on the TS: G_corr at the trunk's level, one imaginary
mode), ts_sp (single points of the energy protocol).
"""

from __future__ import annotations

from pathlib import Path

from predcc_dft import SCHEMA_VERSION
from predcc_dft.backends.slurm import write_array
from predcc_dft.energies import single_point, solvation, thermo
from predcc_dft.protocol import constants, lock, project, records
from predcc_dft.protocol.config import Config
from predcc_dft.protocol.io import (
    format_xyz,
    now,
    read_toml,
    read_xyz_frames,
    write_text,
    write_toml,
)
from predcc_dft.ts import engine, geometry, mechanism, tasks

BRANCH = "ts"
STAGES = ("ts_minima", "ts_autode", "ts_freq", "ts_sp")
TFA_LABEL = "tfa"
SIDES = {  # kinetic symbol -> side of the condensation
    "A": "reactant",
    "N": "reactant",
    "I": "product",
    "W": "product",
    "H": "intermediate",
    "T": "shuttle",
    "S": "shuttle",
}


# ------------------------------------------------------------------ settings


def shuttles(cfg: Config) -> list[str]:
    """Shuttles computed for every step ([ts] shuttles)."""
    out = list(cfg["ts"]["shuttles"])
    bad = [s for s in out if s not in mechanism.SHUTTLES]
    if bad:
        raise mechanism.MechanismError(
            f"[ts] shuttles: unknown {bad}; use {mechanism.SHUTTLES}"
        )
    return out


def energy_spec(cfg: Config) -> dict:
    """Energy protocol: the production lock or the optimisation level."""
    mode = cfg["ts"].get("energy", "auto")
    if mode == "auto":
        mode = "lock" if lock.path(cfg).exists() else "opt"
    if mode == "opt":
        return {
            "tag": "opt",
            "mode": "opt",
            "method": "opt level (r2SCAN-3c/CPCM)",
            "solvation": "cpcm (in the Opt/Freq job)",
            "tasks": {},
        }
    if mode != "lock":
        raise ValueError(f"[ts] energy must be auto, lock or opt, not {mode}")
    the_lock = lock.read(cfg)
    s = the_lock["solvation"]
    model = {
        "orca_keywords": s.get("orca_keywords", ""),
        "orca_blocks": s.get("orca_blocks", ""),
        "parse": s.get("parse", ""),
    }
    e = the_lock["energy"]
    return {
        "tag": lock.version_dir(the_lock),
        "mode": "lock",
        "production_version": the_lock["production_version"],
        "method": e["method"],
        "solvation": s["label"],
        "model": model,
        "tasks": {
            "energy": (e["orca_keywords"], e.get("orca_blocks", "")),
            **solvation.job_inputs(s["level_keywords"], {s["label"]: model}),
        },
    }


# ------------------------------------------------------------------ layout


def reactions_dir(cfg: Config) -> Path:
    """ts/reactions/."""
    return cfg.ts_dir / "reactions"


def species_dir(cfg: Config, label: str) -> Path:
    """ts/species/<label>/."""
    return cfg.ts_dir / "species" / label


def molecule_file(cfg: Config, label: str) -> Path:
    """ts/species/<label>/ts.toml: the TS module's file of one molecule."""
    return species_dir(cfg, label) / "ts.toml"


def channel_file(cdir: Path) -> Path:
    """<channel>/channel.toml: one file per TS."""
    return cdir / "channel.toml"


def reaction_file(cfg: Config, key: str) -> Path:
    """ts/reactions/<key>/reaction.toml."""
    return reactions_dir(cfg) / key / "reaction.toml"


def channel_dirs(cfg: Config, rxn: dict) -> dict[tuple[str, str], Path]:
    """(step, shuttle) -> channel folder of one reaction."""
    base = reactions_dir(cfg) / rxn["key"]
    return {
        (step, sh): base / f"{step}_{sh}"
        for step in mechanism.STEPS
        for sh in shuttles(cfg)
    }


def rel(cfg: Config, path: Path) -> str:
    """Path relative to the study folder, as stored in every TOML."""
    return path.relative_to(cfg.root).as_posix()


def update_toml(path: Path, patch: dict) -> dict:
    """Merge patch into a TOML file's top-level tables and write it."""
    data = read_toml(path) if path.exists() else {}
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **val}
        else:
            data[key] = val
    write_toml(path, data)
    return data


def reactions(cfg: Config) -> list[dict]:
    """Imine condensations whose molecules are all in the TS selection."""
    if not cfg.reactions_file.exists():
        return []
    chosen = {r["compound"]: r for r in project.selected(cfg, BRANCH)}
    out = []
    for rxn in read_toml(cfg.reactions_file).get("condensation", []):
        if "imine" not in rxn["linkages"]:
            continue
        roles = {
            p["role"]: p["compound"]
            for p in rxn["reactants"] + rxn["products"]
        }
        if not all(roles.get(r) in chosen for r in ("aldehyde", "amine")):
            continue
        if roles.get("imine") not in chosen:
            continue
        recs = {role: chosen[key] for role, key in roles.items()}
        out.append(
            {
                "key": rxn["key"],
                "recs": recs,
                "hemiaminal": f"hemiaminal_{rxn['key']}",
                "salt": f"salt_{recs['amine']['compound']}",
            }
        )
    return out


def symbols(rxn: dict) -> dict[str, str]:
    """Kinetic symbol -> species label of one reaction."""
    r = rxn["recs"]
    return {
        "A": r["aldehyde"]["compound"],
        "N": r["amine"]["compound"],
        "I": r["imine"]["compound"],
        "W": r["water"]["compound"],
        "H": rxn["hemiaminal"],
        "T": TFA_LABEL,
        "S": rxn["salt"],
    }


def species(cfg: Config, rxns: list[dict]) -> dict[str, dict]:
    """Every molecule the kinetics needs: label -> definition.

    kind "trunk": aldehyde, amine, imine, water (geometry: trunk best.xyz);
    kind "aux": hemiaminal per reaction, TFA, salt per amine (ts_minima).
    Each lists its roles: reaction, symbol and side (reactant, product,
    intermediate or shuttle).
    """
    acid = bool(set(shuttles(cfg)) & {"tfa", "tfa_salt"})
    roles = {"A": "aldehyde", "N": "amine", "I": "imine", "W": "water"}
    out: dict[str, dict] = {}
    for rxn in rxns:
        r = rxn["recs"]
        defs: dict[str, dict] = {}
        for sym, role in roles.items():
            defs[sym] = {
                "kind": "trunk",
                "role": role,
                "rec": r[role],
                "name": r[role]["name"],
                "smiles": r[role]["smiles"],
            }
        defs["H"] = {
            "kind": "aux",
            "role": "hemiaminal",
            "name": f"{r['aldehyde']['name']} + {r['amine']['name']}"
            " (hemiaminal)",
            "smiles": mechanism.hemiaminal_smiles(
                r["aldehyde"]["smiles"], r["amine"]["smiles"]
            ),
        }
        if acid:
            defs["T"] = {
                "kind": "aux",
                "role": "tfa",
                "name": "trifluoroacetic acid",
                "smiles": mechanism.TFA_SMILES,
            }
            defs["S"] = {
                "kind": "aux",
                "role": "salt",
                "name": f"{r['amine']['name']} trifluoroacetate (ion pair)",
                "smiles": mechanism.shuttle_species(
                    "tfa_salt", r["amine"]["smiles"]
                )[0],
            }
        for sym, d in defs.items():
            label = symbols(rxn)[sym]
            sp = out.setdefault(
                label,
                {**d, "label": label, "symbol": sym, "charge": 0, "in": []},
            )
            sp["in"].append(
                {"reaction": rxn["key"], "symbol": sym, "side": SIDES[sym]}
            )
    return out


# ------------------------------------------------------------------ set-up


def _step(rxn: dict, step: str, shuttle: str) -> mechanism.Step:
    r = rxn["recs"]
    ald, amine = r["aldehyde"]["smiles"], r["amine"]["smiles"]
    if step == "addition":
        return mechanism.addition(ald, amine, shuttle)
    hemi = mechanism.hemiaminal_smiles(ald, amine)
    return mechanism.dehydration(hemi, amine, shuttle)


def _mechanism_record(step: mechanism.Step, handover: dict) -> dict:
    reactants, products = mechanism.kinetic_species(step.name, step.shuttle)
    return {
        "charge": step.charge,
        "multiplicity": 1,
        "n_atoms": step.reactant.GetNumAtoms(),
        "kinetic_reactants": reactants,
        "kinetic_products": products,
        "reactant_fragments": step.fragments("reactant"),
        "product_fragments": step.fragments("product"),
        "forming": [list(p) for p in step.forming],
        "breaking": [list(p) for p in step.breaking],
        "transfers": [
            {"h": t.h, "from": t.src, "to": t.dst} for t in step.transfers
        ],
        "shuttle_atoms": mechanism.shuttle_atoms(step),
        "autode_direction": handover["direction"],
        "autode_forming": handover["forming"],
        "autode_breaking": handover["breaking"],
        "atom_indices": "0-based, as in reactant.xyz, product.xyz and ts.xyz;"
        " autode_* bonds in autodE order (autode/task.toml)",
    }


def _setup_channel(cfg: Config, rxn: dict, step: str, sh: str) -> str:
    cdir = channel_dirs(cfg, rxn)[(step, sh)]
    if channel_file(cdir).exists():
        return "present"
    ident = {
        "schema_version": SCHEMA_VERSION,
        "reaction": rxn["key"],
        "step": step,
        "shuttle": sh,
        "aldehyde": rxn["recs"]["aldehyde"]["name"],
        "amine": rxn["recs"]["amine"]["name"],
    }
    try:
        st = _step(rxn, step, sh)
        reactant, product = geometry.complexes(
            st, int(cfg["coordinates"]["random_seed"])
        )
    except (mechanism.MechanismError, geometry.EmbedError) as exc:
        write_toml(
            channel_file(cdir),
            ident | {"state": "setup_failed", "reason": str(exc)},
        )
        return "failed"
    write_text(cdir / "reactant.xyz", reactant)
    write_text(cdir / "product.xyz", product)
    handover = engine.task(st, reactant, product)
    write_toml(
        cdir / "autode" / "task.toml",
        {"name": f"{rxn['key']}_{step}_{sh}", **handover},
    )
    write_toml(
        channel_file(cdir),
        ident | {"mechanism": _mechanism_record(st, handover)},
    )
    return "new"


def setup(cfg: Config) -> dict:
    """reaction.toml, the complexes of every channel, minima conformers."""
    rxns = reactions(cfg)
    counts = {"reactions": len(rxns), "channels": 0, "new": 0, "failed": 0}
    for rxn in rxns:
        r = rxn["recs"]
        update_toml(
            reaction_file(cfg, rxn["key"]),
            {
                "schema_version": SCHEMA_VERSION,
                "reaction": rxn["key"],
                "aldehyde": r["aldehyde"]["name"],
                "amine": r["amine"]["name"],
                "imine": r["imine"]["name"],
                "shuttles": shuttles(cfg),
            },
        )
        for step, sh in channel_dirs(cfg, rxn):
            counts["channels"] += 1
            res = _setup_channel(cfg, rxn, step, sh)
            if res != "present":
                counts[res] += 1
            elif (
                read_toml(
                    channel_file(channel_dirs(cfg, rxn)[(step, sh)])
                ).get("state")
                == "setup_failed"
            ):
                counts["failed"] += 1
    for label, sp in species(cfg, rxns).items():
        sdir = species_dir(cfg, label)
        if sp["kind"] != "aux" or (sdir / "conformers.xyz").exists():
            continue
        confs = geometry.conformers(
            sp["smiles"],
            int(cfg["ts"]["minima"]["conformers"]),
            int(cfg["coordinates"]["random_seed"]),
        )
        write_text(sdir / "conformers.xyz", "".join(confs))
    return counts


# ------------------------------------------------------------------ state


def mechanism_of(cdir: Path) -> dict | None:
    """The [mechanism] table of a channel (None before set-up)."""
    path = channel_file(cdir)
    return read_toml(path).get("mechanism") if path.exists() else None


def autode_result(cdir: Path) -> dict | None:
    """autode/autode_result.toml of a finished search."""
    path = cdir / "autode" / "autode_result.toml"
    return read_toml(path) if path.exists() else None


def channel_state(cfg: Config, cdir: Path) -> tuple[str, str]:
    """(state, reason) of a channel.

    no_complex -> autode -> freq -> verified; dead ends are setup_failed,
    autode_failed (no TS found) and bad_ts (the Freq job does not show
    exactly one imaginary mode below -[ts] imaginary_threshold_cm1).
    """
    path = channel_file(cdir)
    if not path.exists():
        return "no_complex", ""
    data = read_toml(path)
    if data.get("state") == "setup_failed":
        return "setup_failed", data.get("reason", "")
    res = autode_result(cdir)
    if res is None:
        return "autode", ""
    if res["status"] != "found" or not (cdir / "autode" / "ts.xyz").exists():
        return "autode_failed", res.get("reason", "")
    if not tasks.done(cdir / "freq", "freq"):
        return "freq", ""
    ok, why = tasks.ts_check(
        tasks.parsed(cdir / "freq"),
        float(cfg["ts"]["imaginary_threshold_cm1"]),
    )
    return ("verified", "") if ok else ("bad_ts", why)


# ------------------------------------------------------------------ writers


def _minima_jobs(cfg: Config, sps: dict[str, dict]) -> list[Path]:
    m = cfg["ts"]["minima"]
    out = []
    for label, sp in sps.items():
        sdir = species_dir(cfg, label)
        if sp["kind"] != "aux" or not (sdir / "conformers.xyz").exists():
            continue
        frames = read_xyz_frames(sdir / "conformers.xyz")
        for i, (comment, atoms) in enumerate(frames, start=1):
            cdir = sdir / f"conf{i}"
            if tasks.done(cdir, "opt"):
                continue
            tasks.write(
                cfg,
                "ts_minima",
                cdir,
                m["keywords"],
                m.get("blocks", ""),
                tasks.start_geometry(cdir, format_xyz(atoms, comment)),
                sp["charge"],
                f"{label} conf{i}",
            )
            out.append(cdir)
    return out


def _channel_jobs(cfg: Config, stage: str) -> list[Path]:
    out = []
    for rxn in reactions(cfg):
        for (step, sh), cdir in channel_dirs(cfg, rxn).items():
            state, _ = channel_state(cfg, cdir)
            label = f"{rxn['key']} {step}_{sh}"
            if stage == "ts_autode" and state == "autode":
                tasks.write_autode(cfg, cdir / "autode", label)
                out.append(cdir / "autode")
            elif stage == "ts_freq" and state == "freq":
                mech = mechanism_of(cdir) or {}
                tasks.write(
                    cfg,
                    stage,
                    cdir / "freq",
                    cfg["ts"]["freq_keywords"],
                    cfg["ts"].get("freq_blocks", ""),
                    (cdir / "autode" / "ts.xyz").read_text(encoding="utf-8"),
                    mech.get("charge", 0),
                    f"{label} freq",
                )
                out.append(cdir / "freq")
    return out


def points(cfg: Config) -> list[dict]:
    """Every geometry that needs energies: each TS and each molecule.

    Ready once its geometry is final: a verified TS, a chosen best
    conformer, or the trunk best.xyz.
    """
    rxns = reactions(cfg)
    out = []
    for label, sp in species(cfg, rxns).items():
        if sp["kind"] == "trunk":
            geom = project.mol_dir(cfg, sp["rec"]) / "best.xyz"
            ready = project.structure(cfg, sp["rec"]) is not None
        else:
            geom = species_dir(cfg, label) / "best.xyz"
            ready = geom.exists()
        out.append(
            {
                "label": label,
                "dir": species_dir(cfg, label),
                "geometry": geom,
                "ready": ready,
                "charge": sp["charge"],
                "kind": sp["kind"],
            }
        )
    for rxn in rxns:
        for (step, sh), cdir in channel_dirs(cfg, rxn).items():
            out.append(
                {
                    "label": f"{rxn['key']}.{step}_{sh}",
                    "dir": cdir,
                    "geometry": cdir / "autode" / "ts.xyz",
                    "ready": channel_state(cfg, cdir)[0] == "verified",
                    "charge": (mechanism_of(cdir) or {}).get("charge", 0),
                    "kind": "ts",
                }
            )
    return out


def _sp_jobs(cfg: Config, spec: dict, force: bool = False) -> list[Path]:
    out = []
    for pt in points(cfg):
        if not pt["ready"]:
            continue
        geom = pt["geometry"].read_text(encoding="utf-8")
        rec = {"compound": pt["label"], "charge": pt["charge"]}
        rec["multiplicity"] = 1
        for task, (keywords, blocks) in spec["tasks"].items():
            d = pt["dir"] / "sp" / spec["tag"] / task
            if not single_point.needs_run(
                d, single_point.NEEDS_SP, geom, force
            ):
                continue
            single_point.write(
                cfg, "ts_sp", d, keywords, blocks, geom, rec, task
            )
            out.append(d)
    return out


def write_jobs(cfg: Config, stage: str) -> dict:
    """Write the job array of one TS stage."""
    if stage == "ts_minima":
        dirs = _minima_jobs(cfg, species(cfg, reactions(cfg)))
    elif stage == "ts_sp":
        spec = energy_spec(cfg)
        dirs = _sp_jobs(cfg, spec) if spec["mode"] == "lock" else []
    else:
        dirs = _channel_jobs(cfg, stage)
    script = write_array(cfg, stage, dirs, cfg.ts_dir / "jobs")
    return {"tasks": len(dirs), "script": str(script) if script else None}


def writers() -> dict:
    """SLURM stage -> job writer of the TS module."""

    def make(stage: str):  # noqa: ANN202
        return lambda cfg: write_jobs(cfg, stage)

    return {s: make(s) for s in STAGES}


# ------------------------------------------------------------------ collect


def _conformers(cfg: Config, sdir: Path) -> tuple[str, str, list[dict]]:
    """(status, best conformer, rows) of a molecule optimised here.

    Writes best.xyz: the lowest G of the conformers that converged to a
    minimum, once every conformer has finished or been given up.
    """
    limit = int(cfg["pipeline"]["max_submissions"])
    tol = float(cfg["frequencies"]["imaginary_tolerance_cm1"])
    rows, found, waiting = [], [], False
    for cdir in sorted(p for p in sdir.glob("conf[0-9]*") if p.is_dir()):
        res = tasks.parsed(cdir)
        row: dict = {
            "name": cdir.name,
            "directory": rel(cfg, cdir),
            "input": rel(cfg, cdir / "input.inp"),
            "output": rel(cfg, cdir / "output.out"),
            "final_xyz": rel(cfg, cdir / "final.xyz"),
            "submissions": records.submissions(cdir),
        }
        if not tasks.done(cdir, "opt"):
            row["status"] = "pending" if not res.get("exists") else "failed"
            waiting |= row["submissions"] < limit
        else:
            ok, why = tasks.minimum_check(res, tol)
            row |= {
                "status": "minimum" if ok else "not a minimum",
                "reason": why,
                "electronic_hartree": res.get("final_energy"),
                "g_corr_hartree": res.get("g_minus_e"),
                "gibbs_hartree": res.get("gibbs"),
                "lowest_freq_cm1": res.get("lowest_freq_cm1"),
            }
            if ok and "gibbs" in res:
                found.append((res["gibbs"], cdir))
        rows.append(row)
    if waiting:
        return "pending", "", rows
    if not found:
        return "failed", "", rows
    _, best = min(found)
    text = (best / "final.xyz").read_text(encoding="utf-8")
    best_xyz = sdir / "best.xyz"
    if not best_xyz.exists() or best_xyz.read_text(encoding="utf-8") != text:
        write_text(best_xyz, text)
    return "done", best.name, rows


def _sp_energies(
    cfg: Config, spec: dict, pt: dict, geom: str
) -> tuple[float | None, float | None, dict, dict]:
    """(E, dG_solv kcal/mol, task states, task paths) of the single points."""
    res, state, paths = {}, {}, {}
    for task in spec["tasks"]:
        d = pt["dir"] / "sp" / spec["tag"] / task
        step = "sp" if task == "energy" else "solv"
        method = spec["method"] if task == "energy" else "solvation_level"
        solv = "gas" if task in ("energy", "solv_gas") else spec["solvation"]
        calc = single_point.calculation(
            {"compound": pt["label"]}, step, method, solv, BRANCH
        )
        calc["conformer"] = pt["kind"]
        calc["id"] = records.calc_id(
            pt["label"], pt["kind"], step, method, solv
        )
        calc["geometry_from"] = rel(cfg, pt["geometry"])
        extra = {"calculation": {"ts_energy": spec["tag"]}}
        state[task], res[task] = single_point.collect(
            d, calc, single_point.NEEDS_SP, geom, extra
        )
        paths[task] = {
            "directory": rel(cfg, d),
            "output": rel(cfg, d / "output.out"),
            "record": rel(cfg, d / "record.toml"),
            "status": state[task],
        }
    e = res.get("energy", {}).get("final_energy")
    dg = solvation.dg_solv(spec["solvation"], spec["model"], res)
    return e, dg, state, paths


def _energy(
    cfg: Config, spec: dict, pt: dict, e_opt: float | None, g_corr: float
) -> dict:
    """G of one point under the energy protocol (hartree)."""
    std = thermo.standard_state_kcal(cfg["energies"]["temperature_k"])
    if not cfg["energies"].get("standard_state_correction", True):
        std = 0.0
    out: dict = {
        "energy_protocol": spec["tag"],
        "method": spec["method"],
        "solvation_model": spec["solvation"],
        "g_corr_hartree": g_corr,
        "standard_state_kcal_mol": std,
    }
    e: float | None
    dg: float | None
    if spec["mode"] == "opt":
        e, dg = e_opt, 0.0
    else:
        geom = pt["geometry"].read_text(encoding="utf-8")
        e, dg, _, out["single_points"] = _sp_energies(cfg, spec, pt, geom)
    out["electronic_hartree"] = e
    out["dg_solv_kcal_mol"] = dg
    out["status"] = (
        "complete"
        if e is not None and dg is not None
        else ("waiting for single points")
    )
    if e is not None and dg is not None:
        out["g_hartree"] = e + g_corr + (dg + std) / constants.HARTREE_TO_KCAL
    return out


def _collect_molecule(cfg: Config, spec: dict, sp: dict, pt: dict) -> str:
    """Rewrite one ts.toml; returns its status."""
    label = sp["label"]
    sdir = species_dir(cfg, label)
    data: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated": now(),
        "label": label,
        "name": sp["name"],
        "role": sp["role"],
        "kind": sp["kind"],
        "symbol": sp["symbol"],
        "smiles": sp["smiles"],
        "charge": sp["charge"],
        "multiplicity": 1,
        "in_reactions": sp["in"],
        "module": "ts",
        "files": {"ts_toml": rel(cfg, molecule_file(cfg, label))},
    }
    old = (
        read_toml(molecule_file(cfg, label)).get("energies", {})
        if (molecule_file(cfg, label).exists())
        else {}
    )
    if sp["kind"] == "trunk":
        struct = project.structure(cfg, sp["rec"])
        mdir = project.mol_dir(cfg, sp["rec"])
        data["files"] |= {
            "trunk_directory": rel(cfg, mdir),
            "trunk_toml": rel(cfg, mdir / "trunk.toml"),
            "trunk_molecule_toml": rel(cfg, mdir / "molecule.toml"),
            "trunk_structure_toml": rel(cfg, mdir / "structure.toml"),
            "geometry": rel(cfg, mdir / "best.xyz"),
        }
        data["compound"] = sp["rec"]["compound"]
        if struct is None:
            data["status"] = "waiting for trunk"
            write_toml(molecule_file(cfg, label), data | {"energies": old})
            return data["status"]
        data |= project.identity(sp["rec"], struct)
        data["status"] = "done"
        data["trunk"] = {
            "e_hartree": struct["best"]["e_hartree"],
            "g_corr_hartree": struct["best"]["g_corr_hartree"],
            "g_corr_source": "trunk step 6 Opt+Freq",
        }
        e_opt = struct["best"]["e_hartree"]
        g_corr = struct["best"]["g_corr_hartree"]
    else:
        status, best, rows = _conformers(cfg, sdir)
        data["files"] |= {
            "conformers_xyz": rel(cfg, sdir / "conformers.xyz"),
            "geometry": rel(cfg, sdir / "best.xyz"),
        }
        data["conformer_source"] = "RDKit ETKDGv3 + MMFF94, ORCA Opt+Freq"
        data["status"] = status
        data["best_conformer"] = best
        data["conformers"] = rows
        if status != "done":
            write_toml(molecule_file(cfg, label), data | {"energies": old})
            return status
        res = tasks.parsed(sdir / best)
        e_opt, g_corr = res.get("final_energy"), res["g_minus_e"]
    old[spec["tag"]] = _energy(cfg, spec, pt, e_opt, g_corr)
    write_toml(molecule_file(cfg, label), data | {"energies": old})
    return (
        "with G" if "g_hartree" in old[spec["tag"]] else "waiting for energies"
    )


def _collect_channel(
    cfg: Config, spec: dict, rxn: dict, step: str, sh: str, pt: dict
) -> str:
    """Rewrite one channel.toml; returns its state."""
    cdir = channel_dirs(cfg, rxn)[(step, sh)]
    if not channel_file(cdir).exists():
        return "no_complex"
    data = read_toml(channel_file(cdir))
    state, why = channel_state(cfg, cdir)
    data |= {"generated": now(), "state": state, "reason": why}
    if state == "setup_failed":
        write_toml(channel_file(cdir), data)
        return state
    a = cdir / "autode"
    data["files"] = {
        "channel_toml": rel(cfg, channel_file(cdir)),
        "reactant_xyz": rel(cfg, cdir / "reactant.xyz"),
        "product_xyz": rel(cfg, cdir / "product.xyz"),
        "autode_task": rel(cfg, a / "task.toml"),
        "autode_log": rel(cfg, a / "autode.log"),
        "autode_result": rel(cfg, a / "autode_result.toml"),
        "autode_workdir": rel(cfg, a / "work"),
        "ts_xyz": rel(cfg, a / "ts.xyz"),
        "freq_input": rel(cfg, cdir / "freq" / "input.inp"),
        "freq_output": rel(cfg, cdir / "freq" / "output.out"),
    }
    data["submissions"] = {
        "autode": records.submissions(a),
        "freq": records.submissions(cdir / "freq"),
    }
    res = autode_result(cdir)
    if res is not None:
        data["autode"] = {
            k: res.get(k)
            for k in ("status", "reason", "energy_hartree", "autode_version")
            if res.get(k) is not None
        } | {
            "imaginary_cm1": res.get("imaginary_cm1", []),
            "attempts": res.get("attempts", []),
        }
    if tasks.done(cdir / "freq", "freq"):
        fr = tasks.parsed(cdir / "freq")
        freqs = fr.get("frequencies_cm1") or []
        data["freq"] = {
            "imaginary_cm1": [f for f in freqs if f < 0],
            "electronic_hartree": fr.get("final_energy"),
            "g_corr_hartree": fr.get("g_minus_e"),
            "keywords": cfg["ts"]["freq_keywords"],
        }
    energies = data.get("energies", {})
    if state == "verified":
        fr = tasks.parsed(cdir / "freq")
        energies[spec["tag"]] = _energy(
            cfg, spec, pt, fr.get("final_energy"), fr["g_minus_e"]
        )
    data["energies"] = energies
    write_toml(channel_file(cdir), data)
    return state


def collect(cfg: Config) -> dict:
    """Rewrite every ts.toml, channel.toml and reaction.toml."""
    spec = energy_spec(cfg)
    rxns = reactions(cfg)
    sps = species(cfg, rxns)
    pts = {p["label"]: p for p in points(cfg)}
    counts: dict[str, int] = {}

    def count(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    for label, sp in sps.items():
        count(f"molecules {_collect_molecule(cfg, spec, sp, pts[label])}")
    for rxn in rxns:
        channels = {}
        for (step, sh), cdir in channel_dirs(cfg, rxn).items():
            state = _collect_channel(
                cfg, spec, rxn, step, sh, pts[f"{rxn['key']}.{step}_{sh}"]
            )
            count(f"channels {state}")
            channels[f"{step}_{sh}"] = {
                "state": state,
                "channel_toml": rel(cfg, channel_file(cdir)),
            }
        update_toml(
            reaction_file(cfg, rxn["key"]),
            {
                "generated": now(),
                "molecules": {
                    sym: {
                        "label": label,
                        "side": SIDES[sym],
                        "ts_toml": rel(cfg, molecule_file(cfg, label)),
                    }
                    for sym, label in symbols(rxn).items()
                    if label in sps
                },
                "channels": channels,
            },
        )
    return {"energy protocol": spec["tag"], **dict(sorted(counts.items()))}
