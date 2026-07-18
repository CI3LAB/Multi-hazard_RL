from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import train_lifecycle_rl as tl


ROOT = Path(__file__).resolve().parent
LCC_ROOT = ROOT / "lcc_rerun"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    result_json: Path
    weights: tuple[float, float, float]
    eval_seed_offset: int = 700_000


def _t_crit_95(df: int) -> float:
    # Two-sided 95% critical values for small df; large df -> 1.96.
    table = {
        1: 12.706, 2: 4.303, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df >= 200:
        return 1.972
    if df in table:
        return table[df]
    keys = sorted(table)
    for k in keys:
        if df <= k:
            return table[k]
    return 1.972


def _ci_halfwidth(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n <= 1:
        return 0.0
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    return float(_t_crit_95(n - 1) * se)


def _reported_return(lr: float, risk: float, cost: float, weights: tuple[float, float, float]) -> float:
    w_lr, w_risk, w_cost = weights
    return float(-(w_lr * lr + w_risk * risk + w_cost * cost))


def _load_case(spec: CaseSpec) -> tuple[tl.LifecycleConfig, tl.PolicyParams, int]:
    raw = json.loads(spec.result_json.read_text(encoding="utf-8"))
    lifecycle_raw = raw.get("lifecycle_config", raw.get("lifecycle", {}))
    if not isinstance(lifecycle_raw, dict):
        raise ValueError(f"Missing lifecycle config in {spec.result_json}")
    cfg = tl.LifecycleConfig(**tl._coerce_lifecycle_config_dict(lifecycle_raw))  # type: ignore[attr-defined]
    if "best" in raw:
        p = raw["best"]["params"]
    else:
        p = raw["best_params"]
    params = tl.PolicyParams(
        pre_index=int(p["pre_index"]),
        maint_t1=float(p["maint_t1"]),
        maint_t2=float(p["maint_t2"]),
        repair_t1=float(p["repair_t1"]),
        repair_t2=float(p["repair_t2"]),
    )
    cem_raw = raw.get("cem_config", raw.get("cem", {}))
    seed = int(cem_raw.get("seed", raw.get("seed", 20260301)))
    return cfg, params, seed


def _evaluate_episodes(
    cfg: tl.LifecycleConfig,
    params: tl.PolicyParams,
    *,
    base_seed: int,
    episodes: int,
    weights: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    returns: list[float] = []
    reported: list[float] = []
    lrs: list[float] = []
    risks: list[float] = []
    costs: list[float] = []
    min_fs: list[float] = []
    for i in range(int(episodes)):
        ep = tl.simulate_episode(cfg, params, seed=int(base_seed + i * 10007))
        returns.append(float(ep.return_value))
        lrs.append(float(ep.lr))
        risks.append(float(ep.risk))
        costs.append(float(ep.lcc))
        min_fs.append(float(ep.min_f))
        reported.append(_reported_return(ep.lr, ep.risk, ep.lcc, weights))
    return {
        "return_norm": np.asarray(reported, dtype=float),
        "return": np.asarray(returns, dtype=float),
        "lr": np.asarray(lrs, dtype=float),
        "risk": np.asarray(risks, dtype=float),
        "cost": np.asarray(costs, dtype=float),
        "min_f": np.asarray(min_fs, dtype=float),
    }


def _summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "ci_halfwidth": _ci_halfwidth(arr),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": float(arr.size),
    }


def _case_specs() -> list[CaseSpec]:
    ab = LCC_ROOT / "ablation_results"
    case4 = LCC_ROOT / "case4_weight_reallocation" / "results"
    w3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return [
        CaseSpec("baseline_full", "Full model (Baseline)", ab / "baseline_full" / "cem_result.json", w3),
        CaseSpec(
            "no_resilience_equal",
            "No resilience (w2=1/2, w3=1/2)",
            case4 / "cem_results_case4a_no_resilience_equal.json",
            (0.0, 0.5, 0.5),
        ),
        CaseSpec(
            "no_resilience_risk_comp",
            "No resilience risk-comp. (w2=2/3, w3=1/3)",
            ab / "no_resilience_risk_compensated" / "cem_result.json",
            (0.0, 2.0 / 3.0, 1.0 / 3.0),
        ),
        CaseSpec("no_cost", "No cost", ab / "no_cost_term" / "cem_result.json", (0.5, 0.5, 0.0)),
        CaseSpec("no_risk", "No risk", ab / "no_risk_term" / "cem_result.json", (0.5, 0.0, 0.5)),
        CaseSpec("no_pre_reinforcement", "No pre-reinforcement", ab / "no_pre_reinforcement" / "cem_result.json", w3),
        CaseSpec("no_maintenance", "No maintenance", ab / "no_maintenance" / "cem_result.json", w3),
        CaseSpec("no_repair", "No repair", ab / "no_repair" / "cem_result.json", w3),
    ]


def _fmt(mean: float, hw: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {hw:.{digits}f}"


def run_uncertainty(*, episodes: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["return_norm", "lr", "risk", "cost", "min_f"]
    metric_labels = {
        "return_norm": "Return",
        "lr": "Resilience loss",
        "risk": "Risk",
        "cost": "Cost",
        "min_f": "Min F",
    }
    rows_out: list[dict[str, Any]] = []
    table_lines: list[str] = []
    table_lines.append(
        "Uncertainty estimates: mean ± 95% CI half-width "
        f"(N={episodes} independent lifecycle episodes per case)"
    )
    table_lines.append("")

    for spec in _case_specs():
        if not spec.result_json.exists():
            raise FileNotFoundError(f"Missing result JSON: {spec.result_json}")
        cfg, params, seed = _load_case(spec)
        samples = _evaluate_episodes(
            cfg,
            params,
            base_seed=int(seed + spec.eval_seed_offset),
            episodes=int(episodes),
            weights=spec.weights,
        )
        case_row: dict[str, Any] = {
            "case_key": spec.key,
            "case_label": spec.label,
            "result_json": str(spec.result_json),
            "eval_base_seed": int(seed + spec.eval_seed_offset),
            "episodes": int(episodes),
            "weights_lr_risk_cost": list(spec.weights),
        }
        display_parts: list[str] = [spec.label]
        for key in metrics:
            stats = _summarize(samples[key])
            case_row[f"{key}_mean"] = stats["mean"]
            case_row[f"{key}_ci_halfwidth"] = stats["ci_halfwidth"]
            case_row[f"{key}_std"] = stats["std"]
            digits = 4 if key == "return_norm" else 3
            if key == "cost":
                digits = 2
            display_parts.append(f"{metric_labels[key]}: {_fmt(stats['mean'], stats['ci_halfwidth'], digits)}")
        rows_out.append(case_row)
        table_lines.append(" | ".join(display_parts))

    csv_path = out_dir / "uncertainty_mean_ci_halfwidth.csv"
    json_path = out_dir / "uncertainty_mean_ci_halfwidth.json"
    txt_path = out_dir / "uncertainty_mean_ci_halfwidth.txt"

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    json_path.write_text(
        json.dumps(
            {
                "episodes": int(episodes),
                "ci_level": 0.95,
                "ci_type": "two_sided_t_halfwidth",
                "records": rows_out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    txt_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mean ± 95% CI half-width for saved policies.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(LCC_ROOT / "uncertainty_results"),
    )
    args = parser.parse_args()
    out = run_uncertainty(episodes=int(args.episodes), out_dir=Path(args.out_dir).resolve())
    print(out)


if __name__ == "__main__":
    main()
