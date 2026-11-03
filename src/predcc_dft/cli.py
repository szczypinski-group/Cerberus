"""Command-line interface: ``predcc <command>`` (or ``python -m predcc_dft``).

Everyday use is two commands, run from the project directory:

    predcc init                 once: config/, inputs/, experimental/
    predcc run --submit         everything else; repeats itself on SLURM
    predcc level double_hybrid --submit
                                one level of theory (functionals x basis sets)
    predcc functional pwpb95d4 --submit
                                one functional with all of its basis sets

Individual stages stay available for debugging and pilots (prepare,
conformers, dft-opt, frequencies, check, reference, benchmark, analyse,
orca-input); every command is idempotent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from predcc_dft import __version__
from predcc_dft.io_utils import DATA_DIR, Config, load_config, write_text

STARTER_SETTINGS = """\
# Project settings: merged over the packaged defaults
# ({defaults}).
# Any key from that file can be overridden here; only list what you change.

[inputs]
# One SMILES per line; an optional name may follow after a space.
aldehydes = "inputs/aldehydes.txt"
amines = "inputs/amines.txt"

[slurm]
account = ""
partition = ""
# e.g. ["orca/6.0.1", "openmpi/4.1.6", "crest/3.0", "xtb/6.7"]
modules = []
# Activate the environment that has predcc (and CENSO) installed:
setup_lines = ["source {venv}/bin/activate"]
scratch_root = ""   # e.g. "$TMPDIR"; empty = run ORCA in the task folder

# [conformers]
# dft_engine = "censo"          # or "orca" (plain ORCA optimisations)

# [benchmark]
# methods = []    # subset of config/methods.toml labels; [] = all

# [analysis.parents]
# imine = "benzaldehyde + aniline"
"""

EXPERIMENTAL_EXAMPLE = """\
# One file per measured equilibrium, named <reaction key>.toml (SOP 2.5).
# Reaction keys are in reactions.toml / derived/reaction_energies.csv.
# Rename to CND-<16 hex>.toml and fill in; files starting with '_' are ignored.
schema_version = 1
reaction = "CND-0000000000000000"
log_k = 0.0
log_k_uncertainty = 0.05
k_definition = "[imine][H2O]/([aldehyde][amine])"
k_units = "dimensionless"   # dimensionless | L/mol (only dimensionless used)
water_included = true
source_method = "nmr_equilibrium"
temperature_k = 298.15
solvent = "CD3CN"
water_ppm = 0
notebook = ""
doi = ""
"""


def _print(summary: dict, indent: str = "  ") -> None:
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"{indent}{key}:")
            _print(val, indent + "  ")
        else:
            print(f"{indent}{key}: {val}")


def cmd_init(root: Path, example: bool) -> int:
    """Create config/, inputs/ and experimental/ in a project directory."""
    root.mkdir(parents=True, exist_ok=True)
    cdir = root / "config"
    made = []
    venv = Path(sys.prefix)
    files = {
        cdir / "settings.toml": STARTER_SETTINGS.format(
            defaults=DATA_DIR / "settings.toml", venv=venv
        ),
        cdir / "methods.toml": (DATA_DIR / "methods.toml").read_text(),
        cdir / "censo2rc": (DATA_DIR / "censo2rc").read_text(),
        root / "experimental" / "_example.toml": EXPERIMENTAL_EXAMPLE,
    }
    for name in ("aldehydes.txt", "amines.txt"):
        src = DATA_DIR / "examples" / name
        files[root / "inputs" / name] = (
            src.read_text()
            if example
            else src.read_text().split("\n", 2)[0] + "\n"  # header only
        )
    for path, text in files.items():
        if path.exists():
            continue
        write_text(path, text)
        made.append(str(path.relative_to(root)))
    print(f"project: {root.resolve()}")
    print(
        "  created: " + (", ".join(made) if made else "nothing (all present)")
    )
    print("next: edit config/settings.toml ([slurm]) and inputs/*.txt, then")
    print("      predcc run            # check locally what will be submitted")
    print("      predcc run --submit   # on the cluster login node")
    return 0


def _print_levels(cfg: Config) -> None:
    """Print every level with its methods and ORCA keyword lines."""
    from predcc_dft import method_matrix

    table = cfg.methods.get("methods", {})
    levels = method_matrix.all_levels(cfg.methods)
    for name, labels in levels.items():
        print(f"{name}  ({len(labels)} methods)")
        for label in labels:
            print(f"    {label:<32s} ! {table[label]['orca_keywords']}")
    print("\nfunctionals (each with all of its basis sets):")
    for f in cfg.methods.get("functionals", {}):
        bases = [
            table[m]["basis_label"]
            for m in method_matrix.functional_methods(cfg.methods, f)
        ]
        print(f"    {f:<20s} {', '.join(bases)}")
    print("\nrun:  predcc level <level> --submit")
    print("      predcc functional <functional> --submit")


def _need_molecules(cfg: Config) -> bool:
    if not cfg.molecules_file.exists():
        print("molecules.toml not found: run 'predcc run' or 'predcc prepare'")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for every command."""
    ap = argparse.ArgumentParser(
        prog="predcc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument(
        "--root", type=Path, help="project directory (default: current)"
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="extra TOML merged over config/settings.toml",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create config/, inputs/, experimental/")
    p.add_argument("directory", nargs="?", type=Path)
    p.add_argument(
        "--example",
        action="store_true",
        help="fill inputs/ with example lists",
    )

    p = sub.add_parser(
        "run", help="do everything possible now; --submit to run on SLURM"
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="submit job arrays + a driver job that runs the next round",
    )
    p.add_argument("--round", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument(
        "--force-prepare", action="store_true", help="rerun steps 1-3"
    )
    p.add_argument(
        "--level",
        action="append",
        default=[],
        help="benchmark only this level of theory (repeatable)",
    )
    p.add_argument(
        "--functional",
        action="append",
        default=[],
        help="benchmark only this functional, all its basis sets (repeatable)",
    )

    p = sub.add_parser(
        "level",
        help="run whole levels of theory (functionals x basis sets)",
        description="Same as `predcc run --level NAME ...`: every functional "
        "x basis set of the level(s) on every molecule, then the ranking. "
        "Earlier steps still run for molecules that need them.",
    )
    p.add_argument("level", nargs="+", help="level name(s) from methods.toml")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--round", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument(
        "--force-prepare", action="store_true", help=argparse.SUPPRESS
    )
    p = sub.add_parser(
        "functional",
        help="run functionals with every one of their basis sets",
        description="Same as `predcc run --functional NAME ...`.",
    )
    p.add_argument("functional", nargs="+", help="functional label(s)")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--round", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument(
        "--force-prepare", action="store_true", help=argparse.SUPPRESS
    )
    sub.add_parser("levels", help="list levels, functionals and their methods")

    p = sub.add_parser("status", help="progress of every molecule")
    p.add_argument("--summary", action="store_true", help="counts only")
    sub.add_parser("cancel", help="scancel this project's queued jobs")

    p = sub.add_parser("prepare", help="steps 1-3 only")
    p.add_argument("--force", action="store_true", help="re-embed seeds")
    for name, hlp in (
        ("conformers", "steps 4-5: write CREST (+ CENSO) job array"),
        ("dft-opt", "step 5, ORCA route: write optimisation job array"),
        ("frequencies", "step 6: write Opt+Freq job array (and retries)"),
        ("reference", "step 8/10: write DLPNO + solvation job array"),
        ("benchmark", "step 9: write method-matrix single-point job array"),
    ):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--force", action="store_true")
    sub.add_parser("check", help="step 7 only: sanity check and collapse")
    sub.add_parser(
        "analyse", help="steps 7, 10-11: collect results, rebuild derived/"
    )

    p = sub.add_parser(
        "orca-input",
        help="one-off ORCA input + run.sh + job.slurm for any step",
        description="Steps: dft_opt, freq, freq_retry, ref_x, ref_y, "
        "solv_gas, solv_<solvation label>, sp_<method label>, custom",
    )
    p.add_argument("--step", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--molecule", help="compound key; uses best.xyz, else the first seed"
    )
    src.add_argument("--xyz", type=Path, help="any (multi-frame) xyz file")
    p.add_argument("--frame", type=int, default=-1, help="default: last")
    p.add_argument("--out", type=Path, help="output folder")
    p.add_argument("--charge", type=int)
    p.add_argument("--mult", type=int, help="spin multiplicity")
    p.add_argument("--keywords", help="override the step's keyword line")
    p.add_argument("--blocks", help="override the step's %% blocks")
    return ap


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0911, PLR0912
    """Parse arguments and run one command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.command == "init":
        return cmd_init(
            args.directory or args.root or Path.cwd(), args.example
        )

    cfg = load_config(args.config, args.root)
    # Imported here so `predcc init` and --help work without RDKit.
    from predcc_dft import (
        benchmark,
        conformers,
        dft_opt,
        frequencies,
        method_matrix,
        orca_inputs,
        pipeline,
        prepare,
        reference,
        sanity,
        status,
    )

    cmd = args.command
    if cmd == "levels":
        _print_levels(cfg)
        return 0
    if cmd in ("run", "level", "functional"):
        levels = list(getattr(args, "level", []) or [])
        functionals = list(getattr(args, "functional", []) or [])
        try:  # fail early on a typo
            for name in levels:
                method_matrix.level_methods(cfg.methods, name)
            for name in functionals:
                method_matrix.functional_methods(cfg.methods, name)
        except KeyError as err:
            print(err.args[0])
            return 2
        if levels or functionals:
            cfg["benchmark"] = {
                **cfg["benchmark"],
                "levels": levels,
                "functionals": functionals,
            }
        report = pipeline.advance(
            cfg, args.submit, args.round, args.config, args.force_prepare
        )
        print(f"PreDCC round {report.pop('round')} in {cfg.root}")
        _print(report)
        return 0
    if cmd == "cancel":
        ids = pipeline.cancel(cfg)
        print("cancelled: " + (", ".join(ids) if ids else "nothing active"))
        return 0
    if cmd == "prepare":
        _print(prepare.run(cfg, force=args.force))
        return 0
    if cmd == "orca-input":
        if args.molecule and not _need_molecules(cfg):
            return 1
        res = orca_inputs.make(
            cfg,
            args.step,
            args.out,
            args.xyz,
            args.molecule,
            args.frame,
            args.charge,
            args.mult,
            args.keywords,
            args.blocks,
        )
        print(f"ORCA input for {args.step}:")
        _print(res)
        print(f"  submit: sbatch {res['slurm']}")
        return 0
    if not _need_molecules(cfg):
        return 1
    if cmd == "status":
        if args.summary:
            _print(status.summary(cfg))
        else:
            print(status.table(status.run(cfg)))
            print(f"\nwritten: {cfg.derived_dir / 'setup_status.csv'}")
        return 0
    if cmd == "dft-opt" and conformers.engine(cfg) != "orca":
        print('dft-opt needs [conformers] dft_engine = "orca"')
        return 1
    writers = {
        "conformers": conformers.write_jobs,
        "dft-opt": dft_opt.write_jobs,
        "frequencies": frequencies.write_jobs,
        "reference": reference.write_jobs,
        "benchmark": benchmark.write_jobs,
    }
    if cmd in writers:
        _print(writers[cmd](cfg, args.force))
        return 0
    if cmd == "check":
        _print(sanity.run(cfg))
        return 0
    if cmd == "analyse":
        _print(pipeline.local_steps(cfg))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
