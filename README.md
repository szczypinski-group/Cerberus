# PreDCC DFT

DFT for imine-type (C=N) condensation and exchange equilibria in acetonitrile.
From aldehyde and amine lists, the pipeline builds every product, finds
conformers, optimises them and checks their structures. From that one set of
geometries it runs either a **benchmark** (many functionals and basis sets
against DLPNO-CCSD(T)/CBS, ranked) or **production** (one locked method for
many molecules), or both.

## Install

```bash
git clone git@github.com:szczypinski-group/PreDCC_DFT.git
cd PreDCC_DFT
uv sync                    # add --extra censo to install CENSO into the same env
```

CREST 3, xTB and ORCA (5 or 6) must be available on the cluster. Load them with
`[slurm] modules` in the study settings.

## One study, one trunk, two branches

```
inputs → seeds → CREST (xTB) → CENSO / r²SCAN-3c opt → Opt+Freq → structure check
       → best.xyz  (+ r²SCAN-3c thermal correction, isomer populations)      TRUNK
                 │
        ┌────────┴──────────────────────────┐
   BENCHMARK                            PRODUCTION
   DLPNO-CCSD(T)/CBS reference          one locked functional/basis set
   every functional × basis set         one locked solvation model
   every solvation model                G, ΔG, log K
   errors, ranking, candidate ──promote──► production/production.toml
```

The geometry, and the thermal correction from its frequency calculation, are
computed once in the trunk. Both branches run single points on exactly the same
`best.xyz` (every record stores its SHA-256) and never write into the trunk or
into each other. The only link between them is the production lock.

## Usage

```bash
uv run predcc init ~/imine_study      # once: config/, inputs/, benchmark/, production/
cd ~/imine_study
# edit config/settings.toml ([slurm] account/partition/modules) and inputs/*.txt
```

| Command | Runs |
| --- | --- |
| `predcc run [--submit]` | trunk only: `best.xyz` for every molecule |
| `predcc benchmark [--submit]` | trunk (where needed) + the benchmark branch |
| `predcc benchmark --level double_hybrid --submit` | … only one level of theory |
| `predcc benchmark --functional pwpb95d4 --submit` | … only one functional, all its basis sets |
| `predcc promote` | benchmark candidate → `production/production.toml` |
| `predcc promote --method wb97mv_def2tzvp --solvation smd_mecn` | … or choose by hand |
| `predcc production [--submit]` | trunk (where needed) + the production branch |
| `predcc status` / `predcc cancel` | progress of trunk and branches / stop all jobs |

Without `--submit` a command is a dry run: it does the local steps, writes the
job scripts and prints what it would submit. With `--submit` it submits the
ready job arrays and queues a small driver job behind them that runs the same
command again, so each chain runs to completion by itself. A benchmark chain
and a production chain can run at the same time; each waits only for the
trunk and its own jobs. `predcc level …` and `predcc functional …` still work
as short forms of `predcc benchmark --level/--functional`.

### Input lists

`inputs/aldehydes.txt` and `inputs/amines.txt` are plain text files with one
SMILES per line. A name may follow after a space; without one, the SMILES is
used as the name. Blank lines and lines starting with `#` are skipped. A `#`
inside a SMILES is a triple bond and is kept.

```
# aldehydes
O=Cc1ccccc1 benzaldehyde
O=Cc1ccc(cc1)[N+](=O)[O-] 4-nitrobenzaldehyde
O=Cc1ccc(C=O)cc1
```

By default the chemistry is **aldehyde + primary amine → imine + water**: every
aldehyde with every amine, one imine per pair (E and Z isomers kept inside each
imine entry). `molecules.toml` groups the molecules as `[[aldehyde]]`,
`[[amine]]`, `[[imine]]` and `[[water]]`. Inputs that are not primary amines
are listed in `rejected.toml` with the reason. The other SOP linkage classes
are implemented but switched off; to add them:

```toml
[products]
linkages = ["imine", "oxime", "hydrazone", "acylhydrazone", "azine"]
```

### Branch subsets

The trunk computes every input molecule. Each branch can be limited with
`<branch>/molecules.txt`, one entry per line (name, SMILES, compound key or
InChIKey). Selected aldehyde/amine pairs bring their imines, an imine brings its
aldehyde and amine, and water is always included. An empty file means every
molecule; an entry that matches nothing is an error. Typically the benchmark
uses a small hand-picked set and production uses everything:

```
# benchmark/molecules.txt
benzaldehyde
4-nitrobenzaldehyde
aniline
NCc1ccccc1
```

### Levels of theory (benchmark)

`benchmark/methods.toml` defines functionals, basis sets and **levels**. A level
is a set of functionals crossed with a set of basis sets; each combination is
one method, labelled `<functional>_<basis>` (e.g. `pwpb95d4_def2qzvp`). Double
hybrids get the matching `/C` auxiliary basis automatically.

```toml
[levels.double_hybrid]
functionals = ["revdsdpbep86d4", "pwpb95d4"]
basis_sets  = ["def2svp", "def2tzvp", "def2qzvp", "madef2tzvp"]
```

`predcc levels` lists every level and functional with its ORCA keyword lines.
A functional runs with the basis sets of every level it is in, or with its own
`basis_sets = [...]`; one in no level gets every basis set defined. Running the
same level again only submits what is missing.

### Production lock

`production/production.toml` is self-contained: it holds the exact ORCA
keyword lines of the locked method and solvation model, the version, and the
benchmark evidence. Production never reads the benchmark's `methods.toml`.
A version is frozen: promoting a different method to an existing version is
refused, so choose a new one (`predcc promote --version 1.1`). Results of each
version live in their own `v<version>/` folders and `derived/v<version>/`
tables, so versions are never mixed. Every lock is also archived in
`production/locks/`.

### Safety rails

- **Idempotent:** a calculation that finished normally is never rerun.
- **No double submission:** while a stage a chain depends on is still queued,
  that chain writes nothing; partial outputs would look like failures.
- **No endless loops:** each task folder is submitted at most
  `[pipeline] max_submissions` times (default 2), then listed in
  `<stage>/given_up.txt`. A chain stops after `max_rounds`.
- **No queue races:** the task list is written into each array script, so
  rewriting the script cannot move a queued task onto another folder.

## Study folder

```
imine_study/
├── config/            settings.toml, censo2rc
├── inputs/            aldehydes.txt, amines.txt
├── molecules.toml  reactions.toml  rejected.toml
├── molecules/<key>/                         TRUNK (shared, read-only to branches)
│   ├── molecule.toml  seeds/  conformers/  frequencies/
│   └── structure.toml  best.xyz
├── jobs/<stage>/  jobs/driver/  jobs/submitted.toml
├── derived/       calculations.csv, setup_status.csv
│
├── benchmark/                               BRANCH 1
│   ├── methods.toml  molecules.txt  molecules.toml  experimental/
│   ├── molecules/<key>/
│   │   ├── reference/  ref_x/ ref_y/ solv_gas/ solv_<model>/ reference.toml
│   │   └── methods/    <method>/record.toml …  summary.toml
│   ├── jobs/reference/  jobs/benchmark/
│   └── derived/   species_energies.csv  reaction_energies.csv  errors.csv
│                  errors_wide.csv  method_ranking.csv  level_summary.csv
│                  calculations.csv  production_candidate.toml
│
└── production/                              BRANCH 2
    ├── production.toml  locks/  molecules.txt  molecules.toml  experimental/
    ├── molecules/<key>/v<version>/
    │   ├── energy/  solv_gas/  solv_<model>/      record.toml each
    │   └── result.toml
    ├── jobs/production/
    └── derived/v<version>/  species_energies.csv  reaction_energies.csv
                             calculations.csv  (errors.csv with experiment)
```

**The same labels everywhere.** Every molecule is the same compound key
(first InChIKey block) in the trunk and in both branches, and every
per-molecule file (`reference.toml`, `summary.toml`, `result.toml`) carries
`compound`, `name`, `role` (aldehyde/amine/imine/water), `isomer`, `inchikey`
and the geometry hash. Each branch's `molecules.toml` lists its molecules,
grouped by role. The CSV tables of both branches have the same columns, so one
analysis script reads either.

Records and tables use paths relative to the study folder, so it can be copied
or archived as a whole. The SLURM scripts contain absolute paths, but each
round rewrites them. Keep the study on a filesystem the compute nodes can see.

## Pipeline steps (SOP 5.1)

| Step | Where | Module | Output |
| --- | --- | --- | --- |
| 1 Lists, classes by SMARTS | trunk | `species/inputs.py` | `molecules.toml`, `rejected.toml` |
| 2 Products by reaction SMARTS (5.2), E/Z | trunk | `species/products.py` | `reactions.toml` (CND-/TRN- keys, 2.3) |
| 3 ETKDGv4 + MMFF seeds | trunk | `species/coordinates.py` | `molecules/<key>/seeds/` |
| 4–5 CREST (ALPB MeCN) → CENSO r²SCAN-3c/CPCM | trunk | `structures/conformers.py`, `dft_opt.py` | `conformers/` |
| 6 Opt + Freq r²SCAN-3c/CPCM, qRRHO, retry | trunk | `structures/frequencies.py` | `frequencies/…/record.toml` |
| 7 Structure check, E/Z/rotamer collapse | trunk | `structures/sanity.py` | `structure.toml`, `best.xyz` |
| 8 DLPNO-CCSD(T)/CBS(3/4) | benchmark | `benchmark/reference.py` | `reference/ref_x`, `ref_y` |
| 9 Method grid single points (6.1) | benchmark | `benchmark/grid.py` | `methods/<method>/`, `summary.toml` |
| 10 ΔG_solv, every model (6.2) | benchmark | `benchmark/reference.py` | `reference/solv_*`, `reference.toml` |
| 11 + 8 ΔG, errors, ranking, candidate | benchmark | `benchmark/report.py` | `benchmark/derived/` |
| locked SP + ΔG_solv (12) | production | `production/stage.py` | `v<version>/`, `result.toml` |
| G, ΔG, log K (12) | production | `production/report.py` | `production/derived/v<version>/` |

Every ORCA task writes a `record.toml` (SOP 2.5): calculation ID, `branch`,
`production_version` (production), status (`completed | failed | rejected |
stale`), `geometry_from`, geometry SHA-256, ORCA version, pipeline commit,
SLURM job ID, cores and wall time. Every table is rebuilt from these records.

| Benchmark table | Content |
| --- | --- |
| `species_energies.csv` | compound × method × solvation: E, G_corr, ΔG_solv, standard state, ensemble term, G |
| `reaction_energies.csv` | reaction × method × solvation: ΔE, ΔG, ΔΔG vs. the class parent, log K |
| `errors.csv` | the same rows × reference: gas phase vs. DLPNO-CCSD(T), solution vs. experiment |
| `method_ranking.csv` | MAE, RMSE, MSE, SD, max, bootstrap 95 % CI, MAE in log K and ΔΔG, gain over the B3LYP/6-31+G(d,p) baseline, per linkage class and overall |
| `level_summary.csv` | per level: number of methods, best and worst method, mean MAE |
| `errors_wide.csv` | one row per reaction, one column per method (gas-phase error vs. DLPNO) |
| `production_candidate.toml` | lock draft with the evidence (SOP 2.8, 8.1); input of `predcc promote` |

Production writes `species_energies.csv` and `reaction_energies.csv` with the
same columns (one method, gas and the locked solvation model), plus
`errors.csv` if `production/experimental/` holds measured K values.

G(species) = E(method, gas, best geometry) + G_corr(r²SCAN-3c) + ΔG_solv(model)
+ RT ln 24.46 + (G_ens − G_best). The thermal, solvation and ensemble terms are
the same for every method, as the SOP prescribes.

### Data integrity rules

- **Structures are final before single points.** A molecule's best geometry
  is chosen only after every conformer has a final frequency result.
- **Single points are tied to one geometry.** If `best.xyz` changes, older
  single-point results in either branch are marked `stale`, excluded from every
  table, and rerun.
- **T1 check (SOP 7).** Benchmark reactions with a species whose DLPNO T1
  diagnostic is above `[analysis] t1_threshold` (default 0.02) are kept in
  `errors.csv` with `t1_flag = True`, but left out of `method_ranking.csv`.
- **Ensemble correction (SOP 2.7, 5.1 step 11).** G includes G_ens − G_best
  over the r²SCAN-3c conformers, E/Z isomers and rotamers
  (`[analysis] ensemble_correction`).
- **Error split (SOP 8).** Rows compared with experiment split the error into
  `error_electronic_kcal_mol` and `error_thermal_solvation_kcal_mol`.
- **Fair comparison.** In `method_ranking.csv`, `n` is the number of reactions
  and `complete = True` means the method covers every reaction of its group.
- **Full precision.** Hartree energies in the CSVs have 10 decimals.

## Configuration

`predcc init` writes `config/settings.toml`, `config/censo2rc`,
`benchmark/methods.toml` and an empty `molecules.txt` per branch. Settings are
merged over the packaged defaults in `src/predcc_dft/data/settings.toml`, so the
study file only needs the keys you change. `[energies]` (solvation level,
temperature, standard state) applies to both branches; `[reference]`,
`[benchmark]` and `[analysis]` to the benchmark.

One-off inputs for pilots use exactly the pipeline's settings:

```bash
predcc orca-input --step freq --molecule RISHSDLMXPLIHT
predcc orca-input --step sp_wb97mv_def2tzvp --molecule RISHSDLMXPLIHT
predcc orca-input --step prod_energy --molecule RISHSDLMXPLIHT      # needs a lock
predcc orca-input --step custom --keywords "wB97M-V def2-TZVP SP" --xyz my.xyz
```

The individual trunk stages (`prepare`, `conformers`, `dft-opt`,
`frequencies`, `check`) and `analyse [trunk|benchmark|production]` remain
available for debugging.

## Check before the first real run

- **ORCA keywords:** run one small molecule per keyword line (`orca-input`),
  especially DLPNO, `revDSD-PBEP86-D4/2021`, `STABPerform` and the SMD block.
- **Output parsing:** the strings matched in `backends/orca.py` (`E(0)`,
  `Final correlation energy`, `G-E(el)`, orbital energies, the COSMO-RS
  `dGsolv`) were tested only against mocked output. Compare them with one real
  output from your ORCA version.
- **CENSO:** `censo2rc` was validated with CENSO 3.0.8.
- **Pilot (SOP 13.2):** check that E and Z get different InChIKeys (collisions
  get a `-E`/`-Z` suffix and are flagged in `molecules.toml`) and that every
  SMARTS template gives the expected products.

## Deviations from the SOP

- Folders are per **compound** (`molecules/<first InChIKey block>/`), not per
  species. E/Z isomers and rotamers are pooled in one conformer ensemble and
  collapsed to the lowest-G geometry. Isomer populations are kept in
  `structure.toml`.
- Single points run on one geometry per compound, the lowest-G conformer.
  The ensemble enters only through the r²SCAN-3c correction above.
- ΔG_solv is computed once, at r²SCAN-3c, for every method, including the
  B3LYP/6-31+G(d,p) baseline. Morgan et al. used SMD at the B3LYP level.
- Production (SOP 12) shares the study folder with the benchmark (as a branch)
  rather than living in a separate project root; the branches are still
  insular. Campaigns, budgets and the cheap xTB pre-screen of SOP 12.3 are not
  implemented yet.
- Reactions live in one `reactions.toml`, not one file per reaction.
- ORCA outputs are not gzipped, and figures (SOP 2.7) are not generated yet.

## Code layout

```
src/predcc_dft/
├── protocol/     SOP Part I §2: config, file I/O, record.toml, methods.toml,
│                 the production lock, constants, project (registry, subsets)
├── chemistry/    SOP 2.2/2.3 identifiers and 5.2 SMARTS, shared chemistry
├── backends/     ORCA and SLURM adapters (no science)
├── species/      trunk, SOP 5.1 steps 1-3: inputs, imines, 3D seeds
├── structures/   trunk, SOP 5.1 steps 4-7: CREST/CENSO, Opt+Freq, best.xyz
├── energies/     shared single-point building blocks: one SP, DLPNO, solvation,
│                 trunk thermo terms
├── analysis/     shared, branch-neutral: species/reaction ΔG, errors, CSV tables
├── benchmark/    branch: DLPNO + solvation, method grid, ranking, candidate
├── production/   branch: locked method + solvation, G/ΔG/log K per version
└── workflow/     the predcc command, chains and SLURM drivers, status
```

**The dependency rule.** Imports only go downward: `workflow` → the branches
→ the shared levels → the foundations. The trunk levels never import each
other or a branch, and **`benchmark/` and `production/` never import each
other**: a later level reads only files an earlier one wrote, through
`protocol/project.py`. `tests/test_layers.py` fails if any module breaks this.
Later SOP parts (descriptors, TS, ML) belong in new sibling packages.

## Development

```bash
uv sync
uvx pre-commit install
uv run pytest            # end-to-end with mocked CREST/CENSO/ORCA/SLURM
```
