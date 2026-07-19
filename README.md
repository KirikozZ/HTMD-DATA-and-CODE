# Microenvironment-Controlled Li+/Mg2+ Separation in COF Nanochannels

This repository contains the LAMMPS input files and Python analysis scripts supporting the study **"Microenvironment-controlled Li+/Mg2+ separation in COFs: revealed by high-throughput simulations and machine learning."**

The study combines high-throughput non-equilibrium molecular dynamics (NEMD), Gaussian process regression (GPR), mixed-integer optimization, and SHAP analysis to examine how pore-wall hydrophilicity, framework charge, and steric hindrance affect Li+/Mg2+ separation in TbHz COF nanochannels.

## Repository contents

```text
.
|-- Additional MD validation/
|   |-- TbHz.in
|   `-- TbHz.data
|-- Hydrophilicity_0.5/
|-- Hydrophilicity_0.8/
|-- Hydrophilicity_1.0/
|-- Hydrophilicity_1.2/
|-- python_reproducibility_scripts/
|   |-- data.csv
|   |-- train_gpr_selectivity.py
|   |-- predict_gpr_selectivity.py
|   |-- optimize_gpr_selectivity.py
|   |-- shap_gpr_selectivity.py
|   |-- plot_shap_reference.py
|   |-- plot_log_selectivity_shap_custom.py
|   |-- generate_train_test_scatter_charts.py
|   |-- generate_distribution_intervals.py
|   `-- generate_distribution_publication_charts.py
`-- README.md
```

The four `Hydrophilicity_*` folders contain the 196 high-throughput parameter combinations. Together with `Additional MD validation/`, the repository provides 197 simulation cases. Each case includes a `TbHz.in` LAMMPS input script and a matching `TbHz.data` structure/topology file.

The `python_reproducibility_scripts/` directory contains **9 Python scripts** for data analysis and figure reproduction. Its `data.csv` file is required by the Python workflow and should remain in that directory.

## High-throughput simulation design

The 196 NEMD systems were generated using a full-factorial design:

| Descriptor | Simulated values |
|---|---|
| Hydrophilicity | 0.5, 0.8, 1.0, 1.2 |
| Framework charge | -9, -6, -3, 0, +3, +6, +9 e |
| Sigma | 1.346421, 1.846421, 2.346421, 2.846421, 3.346421, 3.846421, 4.346421 A |

Each NEMD simulation uses a TbHz COF membrane separating a feed solution containing 0.5 M LiCl and 0.5 M MgCl2 from pure water. Simulations were performed with periodic boundary conditions at 300 K, a 1 fs timestep, and a 120 ns production run. The LAMMPS input files specify the complete simulation protocol, including force-field settings, PPPM electrostatics, SHAKE constraints, and the pressure-driving force.

## Python workflow

Run the following commands from `python_reproducibility_scripts/`. The scripts use Python 3 with `numpy`, `pandas`, and `Pillow`.

1. Train and serialize the GPR models:

   ```bash
   python train_gpr_selectivity.py
   ```

2. Optimize the predicted Li+/Mg2+ selectivity:

   ```bash
   python optimize_gpr_selectivity.py
   ```

3. Calculate SHAP values:

   ```bash
   python shap_gpr_selectivity.py
   ```

4. Generate model-performance and distribution data/figures:

   ```bash
   python generate_train_test_scatter_charts.py
   python generate_distribution_intervals.py
   python generate_distribution_publication_charts.py
   ```

The scripts write derived files to `python_reproducibility_scripts/outputs/gpr_selectivity/`.

## Analysis details

The GPR inputs are `Hydrophilicity`, `Charge`, and `Sigma`; the predicted outputs are the permeation counts of water (`FWater`), Li+ (`FLi`), Cl- (`FCl`), and Mg2+ (`FMg`). The selectivity objective is

```text
ln(Selectivity) = ln((FLi + 1) / (FMg + 1)).
```

Hyperparameters are selected by fivefold cross-validation over the 196-system dataset. Final GPR models are refitted using all 196 systems and used for optimization and SHAP analysis. For the Figure 3 prediction-versus-simulation visualization, the scripts additionally use a fixed 80/20 random split with random seed 42.

Mixed-integer optimization uses the ranges Hydrophilicity = 0.5-1.2, Sigma = 1.346421-4.346421 A, and integer Charge = -9 to +9 e. The default settings use beta = 0, random seed = 42, 12,000 initial Latin-hypercube candidates, 30 starts, and 80 local-search iterations.

Using the provided data and scripts, the mixed-integer optimum is reproduced as:

```text
Hydrophilicity = 0.6554
Charge          = -2 e
Sigma           = 4.2608 A
ln(Selectivity) = 3.2269
```

## Notes

- LAMMPS trajectories and restart files are not included; they can be regenerated from the provided input and data files.
- Please cite the associated article when using these data, input files, or scripts.
- For questions regarding the repository, please contact the corresponding authors listed in the manuscript.
