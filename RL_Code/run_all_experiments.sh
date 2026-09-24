#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="${PYTHON:-python3}"
fi
code_dir="${repo_root}/code/RL_Code"

cd "${repo_root}"
export PYTHONPATH="${code_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${repo_root}/.mplconfig"

echo "=== Experiment suite started: $(date -Is) ==="

echo "=== Main lifecycle cases (1, 2a, 2b, 3a, 3b, 3c) ==="
"${python_bin}" -u "${code_dir}/train_lifecycle_rl.py" all

echo "=== Ablation variants ==="
"${python_bin}" -u "${code_dir}/ablation_experiment.py" --holdout-episodes 400

echo "=== GA/PSO/CEM threshold-policy comparison ==="
"${python_bin}" -u "${code_dir}/optimizer_baseline_experiment.py" \
  --config "${code_dir}/lcc_rerun/configs/case1_baseline.json" \
  --out-dir "${code_dir}/lcc_rerun/optimizer_baselines" \
  --iterations 200 --population 80 --eval-episodes 30 --holdout-episodes 400

echo "=== Uncertainty evaluation (N=400) ==="
"${python_bin}" -u "${code_dir}/evaluate_uncertainty.py" \
  --episodes 400 \
  --out-dir "${code_dir}/lcc_rerun/uncertainty_results"

echo "=== Optional sequential PPO reference (same budget as CEM) ==="
"${python_bin}" -u "${code_dir}/train_deeprl_ppo.py" \
  --config "${code_dir}/lcc_rerun/configs/case1_baseline.json" \
  --out-dir "${code_dir}/lcc_rerun/deeprl_ppo" \
  --episodes 480000

echo "=== Combined training-curve figures ==="
"${python_bin}" -u "${code_dir}/plot_lcc_combined.py"

echo "=== Experiment suite completed: $(date -Is) ==="
