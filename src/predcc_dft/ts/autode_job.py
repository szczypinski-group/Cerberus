"""One autodE TS search, run inside a SLURM task: ``python -m ... <dir>``.

Reads <dir>/task.toml (written by predcc_dft.ts.engine), builds the start and
end complexes from our atom-mapped fragments, and runs autodE's TS search with
the step's own forming/breaking bonds: a TS template, an xTB and an ORCA
adaptive path, then a NEB, each guess optimised with OptTS and accepted only
if the imaginary mode forms and breaks the right bonds (or a displacement
along it links start and end).

Writes <dir>/autode_result.toml (status, energy, imaginary frequencies) and,
on success, <dir>/ts.xyz in the atom order of the step's complexes.
autodE is imported here only, so the rest of the package runs without it.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from predcc_dft.protocol.io import format_xyz, read_toml, write_toml

RESULT = "autode_result.toml"
KEYWORD_SETS = ("low_opt", "grad", "low_sp", "opt", "opt_ts", "hess", "sp")


def _configure(ade: object, task: dict) -> None:
    """Set the autodE Config: cores, memory, programs, ORCA keywords."""
    from autode.wrappers.keywords import KeywordsSet

    cfg = ade.Config  # type: ignore[attr-defined]
    if os.environ.get("ORCA_BIN"):
        cfg.ORCA.path = os.environ["ORCA_BIN"]
    cfg.n_cores = int(task["n_cores"])
    cfg.max_core = float(task["max_core_mb"]) / 1000.0  # GB
    cfg.hcode = task["hcode"]
    cfg.lcode = task["lcode"]
    cfg.make_ts_template = False
    cfg.min_imag_freq = -float(task["min_imag_freq_cm1"])
    kw, blocks = task["orca_keywords"], task.get("orca_blocks", {})
    cfg.ORCA.keywords = KeywordsSet(
        **{
            k: kw[k].split() + ([blocks[k]] if blocks.get(k) else [])
            for k in KEYWORD_SETS
        }
    )


def _complex(ade: object, task: dict, side: str, name: str):  # noqa: ANN202
    """Reactant/ProductComplex of one side from its fragments."""
    from autode.species import ProductComplex, ReactantComplex

    mol_cls = ade.Reactant if side == "start" else ade.Product  # type: ignore[attr-defined]
    mols = []
    for k, frag in enumerate(task[side]["fragments"]):
        atoms = [ade.Atom(el, x, y, z) for el, x, y, z in frag["atoms"]]  # type: ignore[attr-defined]
        mols.append(
            mol_cls(
                f"{name}_{k}",
                atoms=atoms,
                charge=frag["charge"],
                mult=1,
                solvent_name=task["solvent"] or None,
            )
        )
    complex_cls = ReactantComplex if side == "start" else ProductComplex
    return complex_cls(*mols, name=name)


def _locate(name: str, start, end, br) -> tuple[object, list[dict]]:  # noqa: ANN001
    """Run autodE's get_ts loop, guarded: a failing method moves on.

    Same order as autodE (template, xTB adaptive path, ORCA adaptive path,
    NEB from the path); our complexes are already in a reactive arrangement,
    so there is no re-orientation or truncation.
    """
    from autode.mol_graphs import get_mapping, reac_graph_to_prod_graph
    from autode.transition_states import TransitionState
    from autode.transition_states.locate_tss import ts_guess_funcs_prms

    mapping = get_mapping(
        graph1=end.graph, graph2=reac_graph_to_prod_graph(start.graph, br)
    )
    end.reorder_atoms(mapping=mapping)
    attempts: list[dict] = []
    for func, params in ts_guess_funcs_prms(name, start, end, br):
        entry = {"method": func.__name__, "result": ""}
        attempts.append(entry)
        try:
            guess = func(*params)
            if guess is None:
                entry["result"] = "no guess"
                continue
            if not guess.could_have_correct_imag_mode:
                entry["result"] = "guess has no suitable imaginary mode"
                continue
            ts = TransitionState(guess, bond_rearr=br)
            ts.optimise()
            if not ts.is_true_ts:
                entry["result"] = "OptTS did not give the right mode"
                continue
        except Exception as exc:  # noqa: BLE001 - try the next method
            entry["result"] = f"error: {type(exc).__name__}: {exc}"
            continue
        entry["result"] = "TS found"
        return ts, attempts
    return None, attempts


def run(task_dir: Path) -> dict:
    """Run the search described in task_dir/task.toml; return the result."""
    import autode as ade
    from autode.bond_rearrangement import BondRearrangement

    task = read_toml(task_dir / "task.toml")
    _configure(ade, task)
    work = task_dir / "work"
    work.mkdir(exist_ok=True)
    os.chdir(work)
    start = _complex(ade, task, "start", "start")
    end = _complex(ade, task, "end", "end")
    br = BondRearrangement(
        forming_bonds=[tuple(b) for b in task["forming"]],
        breaking_bonds=[tuple(b) for b in task["breaking"]],
    )
    ts, attempts = _locate(task["name"], start, end, br)
    if ts is None:
        return {
            "status": "failed",
            "reason": "autodE found no TS",
            "attempts": attempts,
        }
    order = task["start"]["order"]  # autodE index -> complex index
    atoms: list = [None] * len(order)
    for k, atom in enumerate(ts.atoms):  # type: ignore[attr-defined]
        atoms[order[k]] = (atom.label, *(float(c) for c in atom.coord))
    (task_dir / "ts.xyz").write_text(
        format_xyz(atoms, f"autodE TS {task['name']}"),
        encoding="utf-8",
        newline="\n",
    )
    freqs = ts.imaginary_frequencies or []  # type: ignore[attr-defined]
    energy = ts.energy  # type: ignore[attr-defined]
    return {
        "status": "found",
        "reason": "",
        "energy_hartree": float(energy) if energy is not None else None,
        "imaginary_cm1": [float(f) for f in freqs],
        "autode_version": ade.__version__,
        "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point of the SLURM task; always writes autode_result.toml."""
    task_dir = Path((argv or sys.argv[1:])[0]).resolve()
    try:
        result = run(task_dir)
    except Exception as exc:  # noqa: BLE001 - recorded for the pipeline
        result = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    write_toml(task_dir / RESULT, result)
    return 0 if result["status"] == "found" else 1


if __name__ == "__main__":
    sys.exit(main())
