# Multi-hazard Lifecycle RL

Lifecycle management of infrastructure under multi-recurrent hazards using simulation-informed modelling and CEM-based threshold-policy search.

This repository provides the source code, case configurations, and paper results for the lifecycle cases, sensitivity/ablation analysis, uncertainty evaluation, and CEM–GA–PSO comparison.

## Key parameters

| Item | Value |
|---|---|
| Fire rates (baseline / fire-dominant / earthquake-dominant) | `0.20 / 0.40 / 0.10` per year |
| Earthquake rates (baseline / fire-dominant / earthquake-dominant) | `0.025 / 0.0125 / 0.05` per year |
| Maintenance costs | `[0.0, 0.25, 0.6]` |
| Deterioration rates `det_alpha_T_levels` | `[0.8, 0.4, 0.2]` |
| Repair recovery rate | `0.25 / year` |
| Repair durations | `[0.0, 1.0, 1.5]` years |
| CEM settings | 200 iterations, population 80, 30 evaluation episodes |
| Holdout / uncertainty episodes | 400 |

See `results/RL_Code/lcc_rerun/PARAMETERS.json`.

## Repository layout

```
code/RL_Code/                 # source code and case configs
results/RL_Code/lcc_rerun/    # figures and numerical summaries
```

### Main scripts

- `train_lifecycle_rl.py` — lifecycle simulation, CEM training, and case plotting
- `ablation_experiment.py` — sensitivity and ablation analysis
- `optimizer_baseline_experiment.py` — GA / PSO / CEM comparison
- `evaluate_uncertainty.py` — mean ± 95% CI half-width evaluation
- `plot_lcc_combined.py` — training-dynamics figures
- `run_all_experiments.sh` — sequential experiment suite

### Results

- Case 1–3 lifecycle figures
- Training-dynamics panels
- Ablation metric bars and multi-objective trade-off
- Uncertainty table
- Optimizer comparison summary
- Result JSON files with configs, best parameters, holdout metrics, and training history

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
export PYTHONPATH="$PWD/code/RL_Code"
export MPLCONFIGDIR="$PWD/.mplconfig"

# Full experiment suite
bash code/RL_Code/run_all_experiments.sh

# Replot ablation figures from existing results
python code/RL_Code/ablation_experiment.py --plot-only \
  --out-root results/RL_Code/lcc_rerun/ablation_results
```

## Notes

- Return values in objective-sensitivity cases use each case’s own weight setting and should not be ranked across different weight formulations.
- Optimizer comparison uses a common holdout seed set under the baseline lifecycle configuration.
