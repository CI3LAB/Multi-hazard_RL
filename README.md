# Multi-hazard Lifecycle RL (Final S7)

Lifecycle management of infrastructure under multi-recurrent hazards using simulation-informed modelling and CEM-based threshold-policy search.

This package contains the **final frozen experimental setup (S7)** used for the paper figures, ablation/sensitivity tables, uncertainty estimates, and CEM–GA–PSO comparison.

## Frozen parameters (S7)

| Item | Value |
|---|---|
| Fire rates (baseline / fire-dom. / eq-dom.) | `0.20 / 0.40 / 0.10` per year |
| Earthquake rates (baseline / fire-dom. / eq-dom.) | `0.025 / 0.0125 / 0.05` per year |
| Maintenance costs | `[0.0, 0.25, 0.6]` |
| Deterioration rates `det_alpha_T_levels` | `[0.8, 0.4, 0.2]` |
| Repair rate | `0.25 / year` (continuous recovery) |
| Repair durations | `[0.0, 1.0, 1.5]` years |
| CEM budget | 200 iterations, population 80, 30 eval episodes |
| Holdout / uncertainty | 400 episodes |

See `results/RL_Code/lcc_rerun/FINAL_S7_PARAMETERS.json`.

## Repository layout

```
code/RL_Code/                 # runnable source + case configs
results/RL_Code/lcc_rerun/    # final paper figures + slim result summaries
```

### Code (main entry points)

- `train_lifecycle_rl.py` — lifecycle simulation + CEM training / plotting
- `ablation_experiment.py` — sensitivity / ablation study
- `optimizer_baseline_experiment.py` — GA / PSO / CEM comparison
- `evaluate_uncertainty.py` — mean ± 95% CI half-width (N=400)
- `plot_lcc_combined.py` — training-dynamics combined figures
- `run_final_s7_experiments.sh` — full sequential experiment suite

### Results included

- Case 1–3 paper figures (`process_iterations`, `cost/lr/risk_over_time`)
- Training-dynamics panels (`fig/rl_eval_*.png`)
- Ablation metric bars + multi-objective trade-off
- Uncertainty table (`uncertainty_results/`)
- Optimizer comparison summary (`optimizer_baselines/`)
- Slim JSON summaries (`*.slim.json`) with configs, best params, holdout metrics, and training history  
  (full episode trajectories are omitted to keep the repo size practical)

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Re-run / replot (optional)

From the repository root:

```bash
export PYTHONPATH="$PWD/code/RL_Code"
export MPLCONFIGDIR="$PWD/.mplconfig"

# Full formal suite (long-running)
bash code/RL_Code/run_final_s7_experiments.sh

# Or replot ablation figures from existing summaries
python code/RL_Code/ablation_experiment.py --plot-only \
  --out-root results/RL_Code/lcc_rerun/ablation_results
```

## Notes

- Large intermediate probes, hazard-sweep trials, `old coding/`, logs, and `.venv` are **not** included.
- Ablation/uncertainty returns use the paper-comparable raw scalarisation under each setting’s weights; do not rank return values across different weight settings.
- Optimizer comparison uses a common holdout seed set under the baseline lifecycle configuration.
