"""methods.toml (SOP 2.5, 6): functionals x basis sets, levels, solvation.

methods.toml may define, besides explicit ``[methods.<label>]`` tables,

    [functionals.<f>]   orca_keywords, functional, family, dispersion,
                        needs_aux_c (double hybrids: add the /C basis)
    [basis_sets.<b>]    orca_keywords, basis, zeta, aux_c
    [levels.<name>]     functionals = [...], basis_sets = [...]

Every level expands into one method per functional x basis set, labelled
``<f>_<b>`` (e.g. ``revdsdpbep86d4_def2tzvp``), so the rest of the pipeline
only ever sees ``[methods.<label>]``. A functional may also list its own
``basis_sets``; one in no level is crossed with every basis set defined.
An explicit method joins a level with ``level = "<name>"`` (composites such
as r2SCAN-3c have no basis to vary).
"""

from __future__ import annotations

import copy

FUNCTIONAL_META = ("functional", "family", "dispersion", "baseline")


def _expand_one(methods: dict, f_label: str, b_label: str) -> dict:
    f = methods["functionals"][f_label]
    b = methods["basis_sets"][b_label]
    keywords = f"{f['orca_keywords']} {b['orca_keywords']}"
    if f.get("needs_aux_c"):
        aux = b.get("aux_c")
        if not aux:
            raise ValueError(
                f"basis set {b_label} has no aux_c, needed by {f_label}"
            )
        keywords += f" {aux}"
    return {
        **{k: f[k] for k in FUNCTIONAL_META if k in f},
        "baseline": f.get("baseline", False),
        "basis": b.get("basis", b["orca_keywords"]),
        "zeta": b.get("zeta", ""),
        "orca_keywords": keywords,
        "orca_blocks": "\n".join(
            x
            for x in (f.get("orca_blocks", ""), b.get("orca_blocks", ""))
            if x
        ),
        "functional_label": f_label,
        "basis_label": b_label,
    }


def expand(methods: dict) -> dict:
    """Return a copy of methods.toml with every level expanded into methods."""
    out = copy.deepcopy(methods)
    table = out.setdefault("methods", {})
    explicit = set(table)
    for level, spec in out.get("levels", {}).items():
        unknown = [
            f
            for f in spec.get("functionals", [])
            if f not in out.get("functionals", {})
        ] + [
            b
            for b in spec.get("basis_sets", [])
            if b not in out.get("basis_sets", {})
        ]
        if unknown:
            raise KeyError(
                f"level {level}: not defined in methods.toml: "
                + ", ".join(unknown)
            )
        for f in spec.get("functionals", []):
            for b in spec.get("basis_sets", []):
                label = f"{f}_{b}"
                if label in explicit:
                    raise ValueError(
                        f"level {level} makes {label}, which is also an "
                        "explicit [methods] entry; rename one of them"
                    )
                entry = table.setdefault(label, _expand_one(out, f, b))
                levels = entry.setdefault("levels", [])
                if level not in levels:
                    levels.append(level)
    # Every functional also gets its own grid, so `predcc functional <f>`
    # covers it: its own basis_sets list if given, else the basis sets of
    # its levels, else every basis set defined.
    level_bases: dict[str, list[str]] = {}
    for spec in out.get("levels", {}).values():
        for f in spec.get("functionals", []):
            bases = level_bases.setdefault(f, [])
            bases += [b for b in spec.get("basis_sets", []) if b not in bases]
    for f, fspec in out.get("functionals", {}).items():
        bases = (
            fspec.get("basis_sets")
            or level_bases.get(f)
            or list(out.get("basis_sets", {}))
        )
        for b in bases:
            if b not in out.get("basis_sets", {}):
                raise KeyError(f"functional {f}: unknown basis set {b}")
            label = f"{f}_{b}"
            if label in explicit:
                raise ValueError(
                    f"functional {f} makes {label}, which is also an "
                    "explicit [methods] entry; rename one of them"
                )
            table.setdefault(label, _expand_one(out, f, b)).setdefault(
                "levels", []
            )
    for label in explicit:
        lvl = table[label].get("level")
        if lvl:
            table[label]["levels"] = [lvl] if isinstance(lvl, str) else lvl
    return out


def level_methods(methods: dict, level: str) -> list[str]:
    """Method labels belonging to one level (expanded methods.toml)."""
    names = set(methods.get("levels", {}))
    names |= {
        lv
        for m in methods.get("methods", {}).values()
        for lv in m.get("levels", [])
    }
    if level not in names:
        raise KeyError(
            f"unknown level {level!r}; defined: {', '.join(sorted(names))}"
        )
    return [
        label
        for label, m in methods.get("methods", {}).items()
        if level in m.get("levels", [])
    ]


def all_levels(methods: dict) -> dict[str, list[str]]:
    """Level name -> method labels, in definition order."""
    names: list[str] = list(methods.get("levels", {}))
    for m in methods.get("methods", {}).values():
        names += [lv for lv in m.get("levels", []) if lv not in names]
    return {n: level_methods(methods, n) for n in names}


def functional_methods(methods: dict, functional: str) -> list[str]:
    """Method labels of one functional with each of its basis sets."""
    labels = [
        label
        for label, m in methods.get("methods", {}).items()
        if m.get("functional_label") == functional
    ]
    if not labels:
        raise KeyError(
            f"unknown functional {functional!r}; defined: "
            + ", ".join(methods.get("functionals", {}))
        )
    return labels


def method_inputs(methods: dict, label: str) -> tuple[str, str]:
    """ORCA keywords and % blocks of one method (common_* appended)."""
    m = methods["methods"][label]
    keywords = f"{m['orca_keywords']} {methods.get('common_keywords', '')}"
    blocks = "\n".join(
        b.strip()
        for b in (m.get("orca_blocks", ""), methods.get("common_blocks", ""))
        if b and b.strip()
    )
    return keywords.strip(), blocks


def solvation_models(methods: dict) -> dict[str, dict]:
    """[solvation.<label>] tables of methods.toml."""
    return dict(methods.get("solvation", {}))
