# PreDCC DFT

DFT benchmarking of imine-type (C=N) condensation and exchange equilibria in
acetonitrile. From aldehyde and amine lists, the pipeline
builds every product, finds conformers, optimises them and checks their
structures. It then computes the DLPNO-CCSD(T)/CBS reference and runs every
method in the benchmark matrix. It finishes by ranking the methods and writing
a draft of the production lock.

## Install

```bash
git clone git@github.com:szczypinski-group/PreDCC_DFT.git
cd PreDCC_DFT
uv sync                    # add --extra censo to install CENSO into the same env
```

CREST 3, xTB and ORCA (5 or 6) must be available on the cluster. Load them with
`[slurm] modules` in the project settings.

## Usage: two commands

```bash
uv run predcc init ~/imine_benchmark          # once: creates config/, inputs/, experimental/
cd ~/imine_benchmark
# edit config/settings.toml ([slurm] account/partition/modules) and inputs/*.txt
uv run predcc run                             # dry run: does local steps, writes jobs, submits nothing
uv run predcc run --submit                    # the whole benchmark, hands-off
```

### Input lists

`inputs/aldehydes.txt` and `inputs/amines.txt` are plain text files
with one SMILES per line. A name may follow after a space; without one, the
SMILES is used as the name. Blank lines and lines starting with `#` are
skipped. A `#` inside a SMILES is a triple bond and is kept.

```
# aldehydes
O=Cc1ccccc1 benzaldehyde
O=Cc1ccc(cc1)[N+](=O)[O-] 4-nitrobenzaldehyde
O=Cc1ccc(C=O)cc1
```

By default the benchmark is **aldehyde + primary amine → imine + water**:
every aldehyde with every amine, one imine per pair (E and Z isomers kept
inside each imine entry). `molecules.toml` groups the molecules as
`[[aldehyde]]`, `[[amine]]`, `[[imine]]` and `[[water]]`. Inputs that are not
primary amines (hydrazines, hydroxylamines, hydrazides) are listed in
`rejected.toml` with the reason.

The other SOP linkage classes are implemented but switched off. To add them
later, list them in `config/settings.toml`:

```toml
[products]
linkages = ["imine", "oxime", "hydrazone", "acylhydrazone", "azine"]
```

They are then stored under `[[imine]]` too, with the class in each entry's
`linkage` field. CSV
(`name,smiles`) and TOML lists still work: point `[inputs]` in
`config/settings.toml` at them.

`predcc run --submit` does everything it can with the results on disk, then
submits the job arrays that are ready. It also queues a small driver job behind
them (`--dependency=afterany`), and that job runs `predcc run --submit` again.
Each round therefore starts the next one, and the benchmark goes from SMILES to
method ranking with no further commands:

| Round | What happens |
| --- | --- |
| 1 | steps 1–3 (lists, SMARTS products, 3D seeds) → submits CREST + CENSO |
| 2 | → submits ORCA Opt+Freq for every conformer in the window |
| 3 | failed or imaginary-frequency conformers → one retry (VeryTightOpt, exact Hessian) |
| 4 | step 7 structure check + E/Z/rotamer collapse → submits DLPNO, solvation and benchmark single points |
| 5 | everything parsed → `derived/` tables, method ranking, `production_candidate.toml`; the chain stops |

### One command per level of theory

`config/methods.toml` defines functionals, basis sets and **levels**. A level
is a set of functionals crossed with a set of basis sets:

```toml
[levels.double_hybrid]
functionals = ["revdsdpbep86d4", "pwpb95d4"]
basis_sets  = ["def2svp", "def2tzvp", "def2qzvp", "madef2tzvp"]
```

Each combination becomes one method, labelled `<functional>_<basis>` (for
example `pwpb95d4_def2qzvp`). Double hybrids get the matching `/C` auxiliary
basis automatically.

```bash
predcc levels                          # list every level, its methods and ORCA lines
predcc level double_hybrid --submit    # run the whole level on every molecule
predcc level hybrid double_hybrid --submit   # several levels at once
```

One functional with all of its basis sets (the basis sets of every level it
is in, or its own `basis_sets = [...]` list; a functional in no level gets
every basis set defined):

```bash
predcc functional pwpb95d4 --submit
predcc functional pwpb95d4 revdsdpbep86d4 --submit
```

`predcc level` and `predcc functional` are `predcc run` restricted to those
methods. Molecules without a
best geometry yet go through the earlier steps first. When the single points
finish, the ranking in `derived/` includes the new methods, and the `level`
column lets you filter by level. Running the same level again only submits
what is missing. To always run a fixed selection, set `[benchmark] levels` in
`config/settings.toml`. With no selection, `predcc run` benchmarks every
method.

To add a functional or basis set, add a `[functionals.*]` or `[basis_sets.*]`
entry and list it in a level.

Other commands:

```bash
predcc status            # per-molecule progress table (also derived/setup_status.csv)
predcc cancel            # scancel this project's queued jobs, driver included
predcc run               # safe at any time; rebuilds derived/ and shows what is pending
```

If you add molecules to `inputs/`, run `predcc run --submit` again. Steps 1–3
rerun because the input fingerprint changed, finished work is kept, and only
the new molecules are computed. If you add an `experimental/<reaction key>.toml`
file, `predcc run` adds the comparison with experiment.

### Safety rails

- **Idempotent:** a calculation that finished normally is never rerun.
- **No double submission:** while a stage of the project is still queued,
  `run` writes nothing. Partial outputs would look like failures.
- **No endless loops:** each task folder is submitted at most
  `[pipeline] max_submissions` times (default 2), then listed in
  `jobs/<stage>/given_up.txt`. The chain stops after `max_rounds`.
- **No queue races:** the task list is written into each array script, so
  rewriting the script cannot move a queued task onto another folder.

## Pipeline steps (SOP 5.1)

| Step | Module | Output |
| --- | --- | --- |
| 1 Lists within specification, classes by SMARTS | `compounds.py` | `molecules.toml`, `rejected.toml` |
| 2 Products by reaction SMARTS (5.2), E/Z, azines, rotamers | `products.py` | `reactions.toml` (CND-/TRN- keys, 2.3) |
| 3 ETKDGv4 + MMFF seeds | `coordinates.py` | `molecules/<key>/seeds/` |
| 4–5 CREST (ALPB MeCN) → CENSO r²SCAN-3c/CPCM (or plain ORCA) | `conformers.py`, `dft_opt.py` | `conformers/` |
| 6 Opt + Freq r²SCAN-3c/CPCM, qRRHO, retry | `frequencies.py` | `frequencies/CONF*/attempt*/record.toml` |
| 7 Connectivity, E/Z and rotamer check; collapse | `sanity.py` | `structure.toml`, `best.xyz` |
| 8 DLPNO-CCSD(T)/CBS(3/4) | `reference.py` | `reference/ref_x`, `ref_y` |
| 9 Method matrix single points (6.1), by level | `benchmark.py`, `method_matrix.py` | `benchmark/<method>/record.toml` |
| 10 ΔG_solv: SMD, CPCM (+ COSMO-RS) at r²SCAN-3c (6.2) | `reference.py` | `reference/solv_*`, `reference.toml` |
| 11 + 8 ΔG assembly, errors, ranking, lock draft | `analysis.py` | `derived/` |
| driver | `pipeline.py` | `jobs/` |

Every ORCA task writes a `record.toml` (SOP 2.5): a deterministic calculation
ID, status (`completed | failed | rejected`), `geometry_from`, ORCA version, the
pipeline's git commit, and the SLURM job ID, core count and wall time. Every
table in `derived/` is rebuilt from these records:

| File | Content |
| --- | --- |
| `calculations.csv` | one row per record |
| `species_energies.csv` | compound × method × solvation: E, G_corr, ΔG_solv, G |
| `reaction_energies.csv` | reaction × method × solvation: ΔE, ΔG, ΔΔG vs. the class parent, log K |
| `errors.csv` | the same rows × reference: gas-phase vs. DLPNO-CCSD(T), solution vs. experiment |
| `method_ranking.csv` | MAE, RMSE, MSE, SD, max, bootstrap 95 % CI, MAE in log K and ΔΔG, gain over the B3LYP/6-31+G(d,p) baseline, per linkage class and overall |
| `level_summary.csv` | per level × reference × solvation × linkage: number of methods, best and worst method, mean MAE |
| `errors_wide.csv` | spreadsheet view: one row per reaction, one column per method, gas-phase error vs. DLPNO (kcal/mol) |
| `production_candidate.toml` | SOP 2.8 lock draft, with the evidence and the solvation-spread check of 6.2. Review it, then copy to `config/production.toml` (8.1) |

### Data integrity rules

- **Structures are final before single points.** A molecule's best geometry
  is chosen only after every conformer has a final frequency result, so a
  pending retry can't change it after the single points have started.
- **Single points are tied to one geometry.** Every single-point record stores
  the SHA-256 of the geometry it used. If `best.xyz` changes, older results
  are marked `stale` and excluded from every table, and the jobs are
  rewritten and rerun.
- **T1 check (SOP 7).** Reactions with a species whose DLPNO T1 diagnostic
  is above `[analysis] t1_threshold` (default 0.02) are kept in `errors.csv`
  with `t1_flag = True`, but left out of `method_ranking.csv`.
- **Ensemble correction (SOP 2.7, 5.1 step 11).** Each species' G includes
  G_ens − G_best, the Boltzmann sum over its r²SCAN-3c conformers, E/Z
  isomers and rotamers (`[analysis] ensemble_correction`). The correction is
  the same for every method, so it cancels in the comparison with DLPNO.
- **Error split (SOP 8).** For rows compared with experiment, `errors.csv`
  splits the error into `error_electronic_kcal_mol` (ΔE of the method − ΔE of
  DLPNO) and `error_thermal_solvation_kcal_mol` (DLPNO-based ΔG −
  experiment).
- **Full precision.** Hartree energies in the CSVs have 10 decimals.

Every method row in these tables carries `method`, `functional_label`,
`basis_label`, `level`, `family` and `zeta`, so tallying a level, a functional
or a basis-set series is a filter plus a group-by (pandas, R or a
spreadsheet). In `method_ranking.csv`, `n` is the number of reactions and
`complete = True` means the method covers every reaction of its group. A
method with failed single points is ranked on fewer reactions, so compare
MAEs only between complete rows.

G(species) = E(method, gas, best geometry) + G_corr(r²SCAN-3c) + ΔG_solv(model)
+ RT ln 24.46. The thermal and solvation terms are computed once and reused for
every method, as the SOP prescribes.

## One self-contained project directory

Each benchmark is one directory. Every input, setting, calculation, record,
job script, log and table is written inside it, and the tests check that
nothing is written elsewhere (home directory or current directory):

```
my_benchmark/
├── config/            settings.toml, methods.toml, censo2rc
├── inputs/            aldehydes.txt, amines.txt
├── experimental/      measured K, one <reaction key>.toml each
├── molecules.toml, reactions.toml, rejected.toml
├── molecules/<key>/   seeds/, conformers/, frequencies/, best.xyz,
│                      structure.toml, reference/, benchmark/<method>/
├── jobs/<stage>/      array scripts, manifests, SLURM logs; jobs/driver/
└── derived/           every CSV table and production_candidate.toml
```

Records and tables use paths relative to the project directory, so it can be
copied or archived as a whole. The SLURM scripts in `jobs/` contain absolute
paths, but each round rewrites them. The code itself lives in this repository
or its virtual environment, not in the project. Keep the project on a
filesystem the compute nodes can see. If you set `[slurm] scratch_root`,
ORCA runs there temporarily and its results are copied back.

## Configuration

`predcc init` writes `config/settings.toml`, `config/methods.toml` and
`config/censo2rc` into the project. Settings are merged over the packaged
defaults in `src/predcc_dft/data/settings.toml`, so the project file only needs
the keys you change. The method matrix is the project's
`config/methods.toml`. It holds `[functionals.*]`, `[basis_sets.*]` and
`[levels.*]` (the grid described above), explicit `[methods.*]` (composites
and the literature baseline), and `[solvation.*]` (the solvation models).

One-off inputs for pilots use exactly the pipeline's settings:

```bash
predcc orca-input --step freq --molecule RISHSDLMXPLIHT
predcc orca-input --step sp_wb97mv_def2tzvp --molecule RISHSDLMXPLIHT
predcc orca-input --step custom --keywords "wB97M-V def2-TZVP SP" --xyz my.xyz
```

The individual stages (`prepare`, `conformers`, `dft-opt`, `frequencies`,
`check`, `reference`, `benchmark`, `analyse`) remain available for debugging.

## Check before the first real run

- **ORCA keywords:** run one small molecule per keyword line (`orca-input`),
  especially DLPNO, `revDSD-PBEP86-D4/2021`, `STABPerform` and the SMD block.
- **Output parsing:** the strings matched in `orca.py` (`E(0)`,
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
  The ensemble enters only through the r²SCAN-3c correction above; the SOP's
  full Boltzmann weighting would run every method on every conformer.
- ΔG_solv is computed once, at r²SCAN-3c, for every method, including the
  B3LYP/6-31+G(d,p) baseline. Morgan et al. used SMD at the B3LYP level
  itself.
- Reactions live in one `reactions.toml`, not one file per reaction.
- ORCA outputs are not gzipped, and figures (SOP 2.7) are not generated yet.
  The tidy CSVs in `derived/` are ready for them.

## Development

```bash
uv sync
uvx pre-commit install
uv run pytest            # end-to-end with mocked CREST/CENSO/ORCA/SLURM
```
