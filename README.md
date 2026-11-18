# PreDCC DFT

DFT for imine-type (C=N) condensation and exchange equilibria in acetonitrile.
From aldehyde and amine lists, the pipeline builds every product, finds
conformers, optimises them and checks their structures. From that one set of
geometries it runs either a **benchmark** (many functionals and basis sets
against DLPNO-CCSD(T)/CBS, ranked) or **production** (one locked method for
many molecules), or both. A separate **TS module** locates the transition
states of each condensation (with proton shuttles, including TFA) and turns
them into rate constants and concentrations over time.

## Install

```bash
git clone git@github.com:szczypinski-group/PreDCC_DFT.git
cd PreDCC_DFT
uv sync                    # add --extra censo to install CENSO into the same env
uv sync --extra ts         # on the cluster, for `predcc ts`: builds autodE 1.4.5
uv sync --extra nmr        # for `predcc nmr`: nmrglue
```

autodE is not on PyPI, so `--extra ts` builds it from its GitHub tag (a C++
compiler is needed; the cluster's usual `gcc` is enough). Combine extras with
`uv sync --extra censo --extra ts`.

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
uv run predcc init ~/imine_study      # once: config/, inputs/, benchmark/, production/, ts/
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
| `predcc ts [--submit]` | trunk (where needed) + the TS module: barriers, rates, kinetics |
| `predcc nmr [benchmark\|production]` | measured log K from the NMR spectra listed in `<branch>/nmr/samples.toml` |
| `predcc show benzaldehyde` | a molecule's `index/<key>.toml` (name, SMILES, key or InChIKey) |
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

## TS module and kinetics (SOP 10)

`predcc ts --submit` takes every imine condensation of `ts/molecules.txt`
(same format as the branch subsets; no entries = all, which is a lot of TSs)
and runs both steps of the mechanism

```
aldehyde + amine  ──addition──►  hemiaminal  ──dehydration──►  imine + water
```

once per proton shuttle in `[ts] shuttles`. In each step one proton moves from
the amine N–H to an oxygen, either directly (`none`, a strained 4-ring TS) or
relayed by a shuttle that comes out unchanged:

| Shuttle | Species | Relay |
| --- | --- | --- |
| `water` | H₂O | accepts on O, gives an O–H |
| `amine` | a second amine molecule | accepts on N, gives an N–H |
| `tfa` | CF₃COOH | accepts on C=O, gives the O–H (8-membered ring) |
| `tfa_salt` | R-NH₃⁺ ·CF₃COO⁻ ion pair | accepts on O⁻, gives an N⁺–H; leaves as amine + TFA |

**Per TS (one channel), with autodE:**

1. RDKit builds the reactant and product complexes with the same atom order,
   and the step's exact forming and breaking bonds are written down
   (`channel.toml` → `[mechanism]`).
2. **autodE** (`ts_autode` stage, one SLURM task per TS) gets these complexes
   and bonds. It tries a TS template, an xTB adaptive path, an ORCA adaptive
   path and a NEB, optimises each guess with OptTS (r²SCAN-3c, CPCM
   acetonitrile, exact Hessian), and accepts it only if the imaginary mode
   forms and breaks the right bonds, or a displacement along it links the two
   complexes. A method that fails moves on to the next; every attempt is
   recorded.
3. autodE refuses more forming than breaking bonds, so **additions are
   searched in reverse** (hemiaminal → aldehyde + amine), which is the same
   TS. Dehydrations run forward. autodE's automatic bond search stops at 2
   forming + 2 breaking bonds; giving it our own bonds is what lets the
   shuttled steps (2 forming + 3 breaking in the searched direction) work.
4. `ts_freq`: one ORCA Freq at the trunk's level on the autodE TS gives G_corr
   with the same thermochemistry as the trunk species, and must show exactly
   one imaginary mode below −100 cm⁻¹ (state `verified`; otherwise
   `bad_ts`, or `autode_failed` if no TS was found).

The minima the kinetics needs come from the trunk (aldehyde, amine, imine,
water: `best.xyz`) or are optimised here (hemiaminal, TFA and the
ammonium–TFA salt: a few RDKit conformers, ORCA Opt+Freq, lowest G).

**One TOML per molecule and per TS.** Every path and number about a molecule,
reactant or product alike, is in its `ts/species/<label>/ts.toml`:
name, role (aldehyde, amine, imine, water, hemiaminal, tfa, salt), its side in
each reaction (`reactant`, `product`, `intermediate`, `shuttle`), the trunk
files it comes from, every conformer job (input, output, final geometry,
energies), every single point (output, record) and its energies per protocol.
Every TS has the same in its channel's `channel.toml` (mechanism, autodE task,
log, result and attempts, `ts.xyz`, the Freq job, energies), and
`reaction.toml` points to all of them and holds the kinetics. All paths are
relative to the study folder.

**Energies** (`[ts] energy`): with a production lock, its method and solvation
model run as single points on every TS and minimum (folders `v<version>`);
without one, the r²SCAN-3c/CPCM energies of the optimisations are used
(folders `opt`). G = E + G_corr (OptTS/Opt Freq) + ΔG_solv + RT ln 24.46,
so every barrier and reaction free energy refers to 1 mol/L.

**Kinetics:** a mass-action model of A (aldehyde), N (free amine), H
(hemiaminal), I (imine), W (water), T (free TFA) and S (salt), one elementary
reaction per verified channel:

- k_f = (k_B T/h) exp(−ΔG‡/RT), with ΔG‡ from the separated reactants and
  the shuttle. A shuttled addition is termolecular (M⁻² s⁻¹), so its rate
  scales with the shuttle's concentration. That is how 2 mol% TFA enters:
  through [T] and [S], not through a changed barrier.
- k_r = k_f / K, with K from the computed ΔG. Every channel therefore leads to
  the same equilibrium, and the long-time ratio [I][W]/([A][N]) is checked
  against K (`log_q_eq` vs `log_k_condensation`).
- TFA + amine ⇌ salt runs at the diffusion limit, with K from the computed
  salt free energy (or `[ts.kinetics] salt_dg_kcal_mol`). Aliphatic amines
  (pKa(MeCN) ≈ 17–18) bind most of the TFA (pKa ≈ 12.7) as the salt; anilines
  (anilinium ≈ 10.6) mostly leave it free.
- Water starts at `[ts.conditions] water_mol_l` (default 0) and is produced
  with the imine, so [W] = [I] throughout: no Karl Fischer value is needed.
- Every barrier is shifted by ±1 kcal/mol (`barrier_shift_kcal_mol`) for an
  uncertainty band on t50, t95 and the 24 h yield.

Default conditions (`[ts.conditions]`): 298.15 K, aldehyde and amine 0.050
mol/L each (50 mmol in 1 L; change both if your 50 mmol is in another
volume), TFA 2 mol% of the aldehyde (1.0 mM), 24 h.

**Outputs** in `ts/derived/<tag>/`:

| File | One row per | Columns |
| --- | --- | --- |
| `summary.csv` | reaction | ΔG and log K, equilibrium and 24 h concentrations, yield, t50/t95 (central, fast, slow), main channel per step, fraction of TFA as salt |
| `barriers.csv` | reaction × step × shuttle | state, imaginary mode, ΔG‡, ΔG, molecularity, k_f, k_r, units, share of the step's flux |
| `kinetics/<reaction>.csv` | time point | t (s, h), every concentration, imine yield |
| `species.csv` | minimum | E, G_corr, ΔG_solv, standard state, G |

`reaction.toml` of each reaction holds the same summary and channel rows under
`[kinetics.<tag>]`, with the path of its concentration table.

## Study folder

```
imine_study/
├── config/            settings.toml, censo2rc
├── inputs/            aldehydes.txt, amines.txt
├── molecules.toml  reactions.toml  rejected.toml
├── index/<key>.toml                         ROOT FILE of each molecule → every
│                                            module's file below
├── molecules/<key>/                         TRUNK (shared, read-only to branches)
│   ├── trunk.toml                           everything the trunk has
│   ├── molecule.toml  seeds/  conformers/  frequencies/
│   └── structure.toml  best.xyz
├── jobs/<stage>/  jobs/driver/  jobs/submitted.toml
├── derived/       calculations.csv, setup_status.csv
│
├── benchmark/                               BRANCH 1
│   ├── methods.toml  molecules.txt  molecules.toml  experimental/
│   ├── molecules/<key>/
│   │   ├── benchmark.toml                   everything the benchmark has
│   │   ├── reference/  ref_x/ ref_y/ solv_gas/ solv_<model>/ reference.toml
│   │   └── methods/    <method>/record.toml …  summary.toml
│   ├── jobs/reference/  jobs/benchmark/
│   └── derived/   species_energies.csv  reaction_energies.csv  errors.csv
│                  errors_wide.csv  method_ranking.csv  level_summary.csv
│                  calculations.csv  production_candidate.toml
│
├── production/                              BRANCH 2
│   ├── production.toml  locks/  molecules.txt  molecules.toml  experimental/
│   ├── molecules/<key>/
│   │   ├── production.toml                  everything, every version
│   │   └── v<version>/  energy/ solv_gas/ solv_<model>/  result.toml
│   ├── jobs/production/
│   └── derived/v<version>/  species_energies.csv  reaction_energies.csv
│                            calculations.csv  (errors.csv with experiment)
│
└── ts/                                      TS MODULE
    ├── molecules.txt  molecules.toml
    ├── reactions/<CND key>/
    │   ├── reaction.toml                 molecules, channels, [kinetics.<tag>]
    │   └── <step>_<shuttle>/             e.g. addition_tfa
    │       ├── channel.toml              everything about this TS
    │       ├── reactant.xyz  product.xyz
    │       ├── autode/  task.toml run.sh autode.log autode_result.toml
    │       │            ts.xyz  work/ (every autodE ORCA/xTB file)
    │       ├── freq/
    │       └── sp/<tag>/
    ├── species/<label>/                  compound keys (trunk), hemiaminal_<key>,
    │   ├── ts.toml                       tfa, salt_<amine>: everything about it
    │   ├── conformers.xyz  conf<i>/  best.xyz   (optimised here only)
    │   └── sp/<tag>/
    ├── jobs/ts_minima/ ts_autode/ ts_freq/ ts_sp/
    └── derived/<tag>/  summary.csv  barriers.csv  species.csv  kinetics/
```

**Keeping the data in its own folder layout.** The package holds no study
data; a study is any folder outside the repository (`--root`, or the current
folder). The branch folders can sit anywhere inside it:

```bash
uv run predcc init C:/Szcz_Projects/project --data-dir data --experimental-dir nmr
```

```
project/
├── config/  inputs/  molecules/  jobs/  derived/  index/     trunk
└── data/
    ├── benchmark/   molecules/  nmr/  jobs/  derived/  methods.toml
    ├── production/  molecules/  nmr/  jobs/  derived/  production.toml
    └── ts/
```

This writes `benchmark_dir`, `production_dir`, `ts_dir` and
`experimental_dir` into `config/settings.toml` (`[project]`); writing those
keys yourself before `predcc init` works the same. Every command then reads
and writes there. The measured-data folder (`nmr/`) is read for
`<reaction key>.toml` files directly in it (measured log K); raw data
(spectra, integrals) can sit beside them in sub-folders and is not touched.

**Measured K from NMR.** Put each Bruker experiment folder (`fid`,
`acqus`, `pdata/1/1r`, …) anywhere below a branch's measured-data folder and
list it in that folder's `samples.toml` (created by `predcc init`):

```toml
[defaults]                      # any key can also be set per sample
aldehyde_ppm = [9.90, 10.10]    # aldehyde CHO signal
imine_ppm = [8.30, 8.60]        # imine CH=N signal
aldehyde_protons = 1            # protons in each integrated signal
imine_protons = 1
spectrum = "auto"               # processed (pdata) if present, else raw FID

[[sample]]
name = "benzaldehyde + aniline, 24 h"
data = "raw/benzaldehyde_aniline/10"   # relative to the nmr/ folder
imine = "benzaldehyde + aniline (imine)"  # name, SMILES, key or InChIKey
aldehyde_mol_l = 0.050
amine_mol_l = 0.050
```

The imine is looked up in `molecules.toml`; its stored origins give the
aldehyde, the amine and the reaction (`aldehyde` and `amine` can be named
instead, or as well, to pick one of several origins). Imines formed at one
of several sites (`multi_site`, e.g. from a dialdehyde) are refused, since
the conversion assumes one aldehyde group.

`uv run predcc nmr` reads each spectrum with nmrglue (TopSpin's processed
spectrum when present; otherwise the raw FID with automatic phasing),
integrates both ranges above a straight local baseline, and computes the
conversion x = (I_imine/n_imine) / (I_imine/n_imine + I_aldehyde/n_aldehyde).
With [imine] = x A₀, [aldehyde] = (1 − x) A₀, [amine] = N₀ − x A₀ and
[water] = W₀ + x A₀ (water is produced with the imine), K = [imine][water] /
([aldehyde][amine]). Samples of the same aldehyde + amine are averaged into
`<reaction key>.toml` (log K, uncertainty from replicates and integration,
every sample's integrals and conversion), which the benchmark and production
analyses then compare against. `derived/nmr_samples.csv` lists every sample.
Each result file carries the imine's, aldehyde's and amine's identity from
`molecules.toml` (compound key, name, SMILES, InChIKeys, links to their
`index/<key>.toml`), and the imine's `index/<key>.toml` gets a `[measured]`
table per branch (log K, uncertainty, number of samples, file).
A `<reaction key>.toml` that you wrote by hand is never overwritten. Check
the integration ranges on a real spectrum first: overlapping signals or a
poor phase/baseline in the range go straight into K.

**One file per molecule per module, one root file per molecule.** Each module
writes one TOML per molecule in its own folder, named after the module, with
the molecule's identity, status, every file path (inputs, outputs, records,
geometries) and its results:

| Module | File |
| --- | --- |
| trunk | `molecules/<key>/trunk.toml`: seeds, CREST/CENSO files, every Opt+Freq attempt, structure, best geometry and energies |
| benchmark | `benchmark/molecules/<key>/benchmark.toml`: every reference, solvation and method job, the DLPNO/CBS reference, ΔG_solv, method energies |
| production | `production/molecules/<key>/production.toml`: per version, every job, method, solvation model, E, ΔG_solv, G |
| ts | `ts/species/<label>/ts.toml`: role and side in each reaction, conformers, single points, energies |

`index/<key>.toml` is the root file of the molecule: name, role, SMILES and
InChIKeys once, then one table per module with the path of that module's
file, whether the molecule is in the module's selection, and a one-line
status. Hemiaminals, TFA and salts, which exist only in the TS module, get
their own `index/<label>.toml`. The index is rebuilt after every round of
every chain and by `predcc analyse`; the modules never write into it or into
each other. `predcc show <name | SMILES | key | InChIKey>` prints the
matching index file. Everything is plain TOML, so `grep -r benzaldehyde
index/` or a few lines of Python finds any molecule.

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
- **TS module:** run `predcc ts` on one reaction with one shuttle first and
  read `autode/autode.log` and `channel.toml` → `[autode] attempts`. Check
  that autodE finds `orca` and `xtb` on the node (or set `ORCA_BIN`), that
  r²SCAN-3c works in autodE's ORCA inputs (`autode/work/*.inp`), and look at a
  few `addition_tfa_salt` TSs by eye: the ion pair can rearrange during OptTS.
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
- **TS module:** SOP 1.3 has no catalyst; here 2 mol% TFA is modelled
  explicitly (free acid and ammonium salt as shuttles). TSs are validated by
  autodE's imaginary-mode check plus an ORCA Freq, not by a full IRC. One TS
  conformer per channel, at most one shuttle per TS, no DLPNO check of the
  barriers, no iminium/protonated-imine pathway, and kinetics per single
  condensation (exchange reactions and whole-library DCC kinetics come later).
  Auxiliary minima use RDKit conformers, not CREST.

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
├── ts/           SOP 10: mechanisms, autodE TS searches, rates, kinetics
└── workflow/     the predcc command, chains and SLURM drivers, status
```

**The dependency rule.** Imports only go downward: `workflow` → the branches
→ the shared levels → the foundations. The trunk levels never import each
other or a branch, and **`benchmark/` and `production/` never import each
other**, nor does `ts/`: a later level reads only files an earlier one wrote,
through `protocol/project.py`. `tests/test_layers.py` fails if any module
breaks this. Later SOP parts (descriptors, ML) belong in new sibling packages.

## Development

```bash
uv sync
uvx pre-commit install
uv run pytest            # end-to-end with mocked CREST/CENSO/ORCA/SLURM
```
