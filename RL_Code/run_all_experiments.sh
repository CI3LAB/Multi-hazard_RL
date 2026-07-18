#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
code_dir="${repo_root}/code/RL_Code"

cd "${repo_root}"
export PYTHONPATH="${code_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${repo_root}/.mplconfig"

echo "=== Experiment suite started: $(date -Is) ==="

echo "=== Main lifecycle cases ==="
"${python_bin}" -u "${code_dir}/train_lifecycle_rl.py" all

echo "=== Case 4 weight-reallocation variants ==="
for config in \
  "${code_dir}/lcc_rerun/case4_weight_reallocation/configs/case4a_no_resilience_equal.json" \
  "${code_dir}/lcc_rerun/case4_weight_reallocation/configs/case4b_no_resilience_risk_replacement.json" \
  "${code_dir}/lcc_rerun/case4_weight_reallocation/configs/case4c_no_risk_cost_emphasis.json"
do
  "${python_bin}" -u "${code_dir}/train_lifecycle_rl.py" "${config}"
done

echo "=== Ablation variants ==="
"${python_bin}" -u "${code_dir}/ablation_experiment.py" --holdout-episodes 400

echo "=== GA/PSO baseline comparison ==="
"${python_bin}" -u "${code_dir}/optimizer_baseline_experiment.py" \
  --config "${code_dir}/lcc_rerun/configs/case1_baseline.json" \
  --out-dir "${code_dir}/lcc_rerun/optimizer_baselines" \
  --iterations 200 --population 80 --eval-episodes 30 --holdout-episodes 400

echo "=== Uncertainty evaluation (N=400) ==="
"${python_bin}" -u "${code_dir}/evaluate_uncertainty.py" \
  --episodes 400 \
  --out-dir "${code_dir}/lcc_rerun/uncertainty_results"

echo "=== Experiment suite completed: $(date -Is) ==="
