# Reproduction sequence

## Prepare

Combine code with its matching Release ZIP. Use Python 3.12 with
`ml/requirements.lock.txt` for ML; use Python 3.11 and `requirements-analysis.txt`
for transport/microscopic tables. Counters use C++17 and Windows memory mapping
(tested compiler: MinGW-W64 g++ 8.1.0).

```powershell
python -X utf8 revision/run.py prepare --dest work/main
python -X utf8 revision/run.py compile --work work/main
```

Use the desired interpreter through `--python` on each `stage` command. Examples
below use the active Python; choose the ML or analysis environment as appropriate.

## 1. MD simulation

Copy a supplied `simulation/screening/...` or `simulation/optimized/...` case
into a job directory. Ordinary input folders contain `TbHz.in` and `TbHz.data`;
PMF folders also contain `TbHz.colvars`. No nested archives need extraction.
Run `lmp -in TbHz.in` to generate a new trajectory. The archived naming does not imply a separate undriven
equilibration stage. No scheduler-specific submission script is required.

## 2. Screening event series and paper labels

The existing CSV series cover all 196 cases. Validate them without raw trajectories:

```powershell
python revision/run.py stage screening-check --work work/main
```

To recalculate from raw trajectories, use the root containing `0.5q`, `0.8q`,
`1.0q`, `1.2q`. This reads approximately 467 GB:

```powershell
python work/main/screening/export_events.py --raw-root RAW_SCREENING_ROOT --training-csv work/main/ml/data.csv --counter work/main/screening/recount_screening.exe --workers 4
```

Consult `screening/DATA_DICTIONARY.md`. Saved frames span 0.01–120 ns; observed
event exposure is 119.99 ns. Legacy counts, signed plane crossings and completed
transits are different quantities. Keep the author-selected paper training table.

## 3. Train, validate, optimize and interpret

```powershell
python revision/run.py stage ml-check --work work/main
python revision/run.py stage ml-fit --work work/main
```

`ml-check` runs six behavioral tests and verifies saved models, folds and metrics.
`ml-fit` runs nested/blocked/learning validation, full-data and separately calibrated
models, diagnostics, mixed-integer optimization, attribution/PDP, optimizer-seed
checks and seed-specific numeric tables. A complete fit
can take substantial time. Raw saved results and searches are included in the
Release. Use fresh work directories for changed inputs or fitting protocols.

`ml/export_publication_tables.py` exports the main seed-42 predictions, fold
metrics, performance, coverage and QQ quantiles separately from all five seeds.
`ml/published_tables/` contains the frozen numeric presentation tables for reference.
ML output consists of models, numerical data and metadata; this workflow does not export images.

## 4. Optimized-condition transport

Seven matching MD inputs are under `simulation/optimized/`. To count new complete
trajectories, run `optimized_transport/recount.py --raw-root RAW_OPTIMIZED_CASES`,
where the directory contains the seven case subdirectories. Then, or directly
using supplied event tables:

```powershell
python revision/run.py stage transport-tables --work work/main
```

Outputs: `transport_counts.csv`, `physical_flux.csv`, `flux_blocks_10ns.csv` and
`results.json`. The frozen candidate is H=0.5691, Charge=-2, Sigma=4.2143 Å.

## 5. Microscopic analysis

```powershell
python revision/run.py stage topology --work work/main
python revision/run.py stage microscopic-tables --work work/main
```

For new raw trajectories, run `stage micro-recount --optimized-raw RAW_OPTIMIZED_CASES`
before `microscopic-tables`. It extracts water bonds and invokes the microscopic
and angle counters. Supplied block/episode data permit the table stage directly.
The final stage also converts water number density to g cm^-3 and computes mean
water-dipole angle profiles using molecule-frame weighting. Bootstrap parameters
are fixed in `microscopic/statistics_protocol.json`.

## Additional supplied simulation inputs

`simulation/pmf/` preserves the supplied `baseline`, `high_flux` and `optimized`
groups, each with Li/Mg and the intervals `28-35`, `36-41`, `42-47`. Each folder
contains its original `TbHz.in`, `TbHz.data` and `TbHz.colvars`. Run from that
folder with a LAMMPS build supporting the COLVARS fix. The supplied configuration
sets the restrained atom, center progression and force constant. Inputs were
copied without changing the simulation protocol. PMF trajectories and processing
outputs are not included in this input-only addition.

`simulation/additional/` retains source folder names for 18 additional input
sets: four fixed/flexible framework sets, three prediction-candidate sets,
three optimized pressure batches and eight baseline pressure sets. A folder
may run several pressure values or seeds according to its own `TbHz.in`.

Four original `rerun.in` templates remain under `simulation/screening/<H>q/`.
They are auxiliary trajectory-replay inputs, not independent simulation cases.
They require an individual case's `TbHz.data` and a trajectory passed as
`TRAJ_FILE`. They are archived as supplied and were not executed or scientifically
validated as part of this packaging update. The existing main analysis stages
do not depend on these templates. No new MD or PMF calculations were run here.
