# HTMD Data and Code

This repository contains the manuscript revision's reproducible calculation code and simulation inputs.

The main workflow is MD simulation and event counting, ML training and validation, optimization and interpretation, and optimized-condition transport and microscopic analysis. The ML workflow exports models and numerical tables.

## Get the data

Download the ZIP files from [Releases](https://github.com/KirikozZ/HTMD-DATA-and-CODE/releases):

- `Revision_Reproducibility_Data.zip`: matching simulation inputs, screening event series, saved ML results, optimization results and microscopic data. Extract it at this repository root so its `revision/` directory merges with the code.
- `Supplementary_Source_Data.zip`: manuscript and SI figure source files and screening summary tables. Extract separately to browse the supplied source data.

## Run

Read [the workflow instructions](revision/WORKFLOW.md) for the Python environments and compiler requirements. After extracting the matching Release data:

```powershell
python -X utf8 revision/run.py prepare --dest work/main
python -X utf8 revision/run.py compile --work work/main
```

The C++ counters use Windows memory mapping. PMF simulation inputs require a LAMMPS build supporting the COLVARS fix. Complete atomic trajectories are not bundled; saved event and microscopic block data support statistical reproduction.

## Included inputs

The Release contains 239 complete simulation input folders, plus four auxiliary trajectory-replay inputs: 243 `.in`, 239 `.data` and 18 `.colvars` files. Ordinary simulation folders contain the input script and coordinates; each of the 18 PMF folders also contains its colvars configuration. Simulation logs and restarts are omitted.

The paper's original 196-row training table remains in `revision/ml/data.csv`. Recounts agree with 194 rows; the S112/S119 differences are documented separately without replacing the paper's labels.

See [revision/README.md](revision/README.md) for the directory overview. The data packages retain the scientific results; this release does not imply that complete MD or ML training was rerun during packaging.
