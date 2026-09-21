# Main scientific workflow

The calculation sequence is:

`MD initialization and simulation → event counting → ML training/validation → optimization and interpretation → optimized-condition transport and microscopic analysis`.

| Directory | Purpose |
| --- | --- |
| `simulation/screening/` | LAMMPS inputs for the 196 screening systems |
| `simulation/optimized/` | Optimized condition and six parameter perturbations |
| `simulation/pmf/` | 18 PMF input folders: baseline/high_flux/optimized × Li/Mg × three position intervals |
| `simulation/additional/` | 18 additional input folders: pressure, fixed/flexible frameworks and prediction candidates |
| `screening/` | C++ counting, per-frame CSV export and mass-balance checks |
| `ml/` | Model fitting, validation, calibration, optimization, SHAP/PDP and numerical publication exports |
| `optimized_transport/` | Seven-case counting and aggregate transport statistics |
| `microscopic/` | Hydration, residence, density, orientation and water-angle calculations |

Extract `Revision_Reproducibility_Data.zip` at the repository root to merge its
`revision/` directory with this code. Its 239 input folders contain `TbHz.in` and
`TbHz.data`; the 18 PMF folders also contain `TbHz.colvars`. The `.in` and `.colvars`
copies match the code version. Four original `rerun.in` templates are retained
at the screening hydrophilicity-directory level; use them from an individual
case directory with its topology and an explicitly supplied trajectory.
Simulation logs, restarts and per-case JSON files are omitted. Paths are relative.

Read `WORKFLOW.md` for dependencies and commands. Start with:

```powershell
python -X utf8 revision/run.py prepare --dest work/main
python -X utf8 revision/run.py compile --work work/main
```

`ml/data.csv` is the unchanged 196-row paper training input. Recounts agree with
194 rows; S112/S119 differences are retained separately and never replace paper
labels. Main ML results use seed 42; repeated-seed results retain seeds 42–46.

Final main-text and SI figure source files are in `Supplementary_Source_Data.zip`.
Its figure index is a data lookup, not a code dependency. Internal script
inventories and machine-specific source paths are not part of this code release.

The analysis scripts follow the author's main calculation chain. The simulation
input archive also includes PMF, pressure and fixed/flexible-framework inputs
explicitly selected by the author. Response-writing utilities, reviewer-control
analysis scripts and native-layout automation are outside this code package. Full atomic trajectories are external. Saved events and microscopic
block data support statistical reproduction without rerunning MD. Existing
initialization coordinates are supplied; an unrecovered earlier ion-placement
seed is not reconstructed. Time-block analysis alone does not establish
pre-counting stationarity.
