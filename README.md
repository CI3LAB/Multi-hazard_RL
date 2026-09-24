# Multi-hazard Lifecycle RL

Lifecycle management of infrastructure under multi-recurrent hazards using simulation-informed modelling and CEM-based threshold-policy search.

This repository contains the **adopted** training framework and the paper results. 
## Adopted model

| Item | Adopted value |
|---|---|
| Pre-reinforcement | Costs `5 / 10 / 20`; hazard-damage multipliers `1.00 / 0.95 / 0.88`|
| Earthquake / fire rates | Baseline `(0.025, 0.20)`; fire-dominant `(0.0125, 0.40)`; earthquake-dominant `(0.05, 0.10)` per year |
| Intensity mix | Low / medium / high = `0.60 / 0.30 / 0.10` |
| Earthquake ΔF | `0.10 / 0.30 / 0.60` |
| Fire ΔF | `0.10 / 0.20 / 0.30` |
| Weibull deterioration `α_T` | `[0.80, 0.45, 0.20]` for maintenance levels 0 / 1 / 2; `τ = 55` years, `k = 2.2` |
| Maintenance costs | `[0.0, 0.25, 0.6]` every 10 years |
| Repair | Recovery `0 / 0.15 / 0.30` over `0 / 1.0 / 1.5` years; costs `[0, 3, 10]` |
| Risk | `F_crit / F1 / F2 = 0.70 / 0.50 / 0.30`; penalties `1 / 4 / 10`; plus an instantaneous jump term at hazard drops |
| Discount | `γ = 0.03` on cost (NPV) only; resilience loss and risk are undiscounted |
| Objective | Weighted sum of reference-normalized LR, risk, and NPV; feasibility penalty weight = 0 |
| CEM | 200 iterations, population 80, 30 eval episodes; per-`pre_index` Gaussians; delayed elitist after 25% of iterations; `σ` floor `0.08 → 0.02` |
| Holdout | `N = 400` common seeds |

Figures 5–10 plot **undiscounted lifecycle cash** (`LCC`). Tables that report Cost use **NPV**. See `results/RL_Code/lcc_rerun/PARAMETERS.json`.

## Layout

```
code/RL_Code/                 training code and case configs
results/RL_Code/lcc_rerun/    paper figures and numerical summaries
```

`code/RL_Code/lcc_rerun` is a symlink to `results/RL_Code/lcc_rerun`, so training writes into the results tree.

### Scripts

- `train_lifecycle_rl.py` — lifecycle environment, CEM, and case figures
- `plot_lcc_combined.py` — six-case training/evaluation curves
- `ablation_experiment.py` — objective-sensitivity and intervention ablation
- `optimizer_baseline_experiment.py` — fixed-rule / GA / PSO / CEM on the same threshold policy
- `evaluate_uncertainty.py` — holdout mean ± 95% CI
- `train_deeprl_ppo.py` — sequential PPO reference (appendix); not the reported inspection policy
- `run_all_experiments.sh` — full suite

### Results in this snapshot

- Cases 1, 2a, 2b, 3a, 3b, 3c: functionality / cost / resilience-loss / risk trajectories
- Combined training curves
- Ablation bars and trade-off (no risk-compensated variant)
- Optimizer comparison with uncertainty, plus PPO holdout numbers

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is optional for CEM evaluation speed and required only for the PPO reference.

## Usage

```bash
export PYTHONPATH="$PWD/code/RL_Code"
export MPLCONFIGDIR="$PWD/.mplconfig"

# Full suite (long)
bash code/RL_Code/run_all_experiments.sh

# Train the six paper cases
python code/RL_Code/train_lifecycle_rl.py all

# Replot case figures from saved JSON
python code/RL_Code/train_lifecycle_rl.py all --replot

# Replot ablation figures
python code/RL_Code/ablation_experiment.py --plot-only \
  --out-root results/RL_Code/lcc_rerun/ablation_results
```

## Notes

- Objective-sensitivity Return values use each variant’s own weights and should not be ranked across formulations.
- Optimizer comparison uses a common holdout seed set under the Case 1 environment.
- PPO attains a higher holdout return but is a sequential black-box policy, so CEM remains the reported solver for the inspectable threshold rule.
