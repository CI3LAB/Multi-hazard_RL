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
DEFAULT_CONFIG = LCC_ROOT / "configs" / "case1_baseline.json"
OPT_DIR = LCC_ROOT / "optimizer_baselines"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    result_json: Path
    weights: tuple[float, float, float]
    eval_seed_offset: int = 700_000
    group: str = "ablation"
    fallback_config: Path | None = None
    eval_base_seed: int | None = None
    force_fallback_config: bool = False


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
    lifecycle_raw = raw.get("lifecycle_config", raw.get("lifecycle"))
    use_embedded = (
        (not spec.force_fallback_config)
        and isinstance(lifecycle_raw, dict)
        and bool(lifecycle_raw)
    )
    if use_embedded:
        cfg = tl.LifecycleConfig(**tl._coerce_lifecycle_config_dict(lifecycle_raw))  # type: ignore[attr-defined]
    else:
        fallback = spec.fallback_config if spec.fallback_config is not None else DEFAULT_CONFIG
        if not fallback.exists():
            raise ValueError(f"Missing lifecycle config in {spec.result_json} and fallback {fallback}")
        cfg, _cem_cfg, _paths = tl._load_run_config(fallback, ROOT)  # type: ignore[attr-defined]
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
    if spec.eval_base_seed is not None:
        seed = int(spec.eval_base_seed)
    else:
        cem_raw = raw.get("cem_config", raw.get("cem", {}))
        if not isinstance(cem_raw, dict):
            cem_raw = {}
        seed = int(cem_raw.get("seed", raw.get("seed", raw.get("config", {}).get("seed", 20260301))))
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
    lccs: list[float] = []
    min_fs: list[float] = []
    feasibles: list[float] = []
    for i in range(int(episodes)):
        ep = tl.simulate_episode(cfg, params, seed=int(base_seed + i * 10007))
        returns.append(float(ep.return_value))
        lrs.append(float(ep.lr))
        risks.append(float(ep.risk))
        costs.append(float(ep.npv))
        lccs.append(float(ep.lcc))
        min_fs.append(float(ep.min_f))
        feasibles.append(1.0 if ep.feasible else 0.0)
        # Return plotted in ablation/uncertainty tables is the same normalized
        # objective CEM/GA/PSO actually optimize, not the unscaled raw mix.
        reported.append(float(ep.return_value))
    return {
        "return_norm": np.asarray(reported, dtype=float),
        "return": np.asarray(returns, dtype=float),
        "lr": np.asarray(lrs, dtype=float),
        "risk": np.asarray(risks, dtype=float),
        "cost": np.asarray(costs, dtype=float),
        "lcc": np.asarray(lccs, dtype=float),
        "min_f": np.asarray(min_fs, dtype=float),
        "feasible": np.asarray(feasibles, dtype=float),
    }


def _summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "ci_halfwidth": _ci_halfwidth(arr),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": float(arr.size),
    }


def evaluate_policy_uncertainty(
    cfg: tl.LifecycleConfig,
    params: tl.PolicyParams,
    *,
    base_seed: int,
    episodes: int,
    weights: tuple[float, float, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    if weights is None:
        weights = (float(cfg.w_lr), float(cfg.w_risk), float(cfg.w_cost))
    samples = _evaluate_episodes(
        cfg,
        params,
        base_seed=int(base_seed),
        episodes=int(episodes),
        weights=weights,
    )
    stats = {key: _summarize(values) for key, values in samples.items()}
    return samples, stats


def holdout_with_uncertainty(
    cfg: tl.LifecycleConfig,
    params: tl.PolicyParams,
    *,
    base_seed: int,
    episodes: int,
) -> dict[str, Any]:
    samples, stats = evaluate_policy_uncertainty(
        cfg,
        params,
        base_seed=int(base_seed),
        episodes=int(episodes),
    )
    return {
        "return": stats["return"]["mean"],
        "lr": stats["lr"]["mean"],
        "risk": stats["risk"]["mean"],
        "lcc": stats["lcc"]["mean"],
        "npv": stats["cost"]["mean"],
        "min_f": stats["min_f"]["mean"],
        "feasible_frac": stats["feasible"]["mean"],
        "reported_return": stats["return"]["mean"],
        "uncertainty": {
            "episodes": int(episodes),
            "ci_level": 0.95,
            "ci_type": "two_sided_t_halfwidth",
            "eval_base_seed": int(base_seed),
            "return": stats["return"],
            "reported_return": stats["return"],
            "lr": stats["lr"],
            "risk": stats["risk"],
            "lcc": stats["lcc"],
            "npv": stats["cost"],
            "min_f": stats["min_f"],
            "feasible_frac": stats["feasible"],
        },
    }


def _case_specs() -> list[CaseSpec]:
    ab = LCC_ROOT / "ablation_results"
    w3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return [
        CaseSpec("baseline_full", "Full model (Baseline)", ab / "baseline_full" / "cem_result.json", w3),
        CaseSpec("no_resilience_equal", "No resilience (w2=1/2, w3=1/2)", ab / "no_resilience_term" / "cem_result.json", (0.0, 0.5, 0.5)),
        CaseSpec("no_cost", "No cost", ab / "no_cost_term" / "cem_result.json", (0.5, 0.5, 0.0)),
        CaseSpec("no_risk", "No risk", ab / "no_risk_term" / "cem_result.json", (0.5, 0.0, 0.5)),
        CaseSpec("no_pre_reinforcement", "No pre-reinforcement", ab / "no_pre_reinforcement" / "cem_result.json", w3),
        CaseSpec("no_maintenance", "No maintenance", ab / "no_maintenance" / "cem_result.json", w3),
        CaseSpec("no_repair", "No repair", ab / "no_repair" / "cem_result.json", w3),
    ]


def _optimizer_eval_seed(opt_dir: Path) -> int:
    summary = opt_dir / "optimizer_baseline_results.json"
    if summary.exists():
        raw = json.loads(summary.read_text(encoding="utf-8"))
        cfg_block = raw.get("optimizer_config", {})
        if isinstance(cfg_block, dict) and "common_holdout_seed" in cfg_block:
            return int(cfg_block["common_holdout_seed"])
    ga = opt_dir / "ga_result.json"
    if ga.exists():
        raw = json.loads(ga.read_text(encoding="utf-8"))
        cfg_block = raw.get("config", {})
        if isinstance(cfg_block, dict) and "holdout_seed" in cfg_block:
            return int(cfg_block["holdout_seed"])
    return 21160301


def _optimizer_case_specs() -> list[CaseSpec]:
    w3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    holdout_seed = _optimizer_eval_seed(OPT_DIR)
    cem_path = OPT_DIR / "cem_result.json"
    if not cem_path.exists():
        cem_path = LCC_ROOT / "ablation_results" / "baseline_full" / "cem_result.json"
    return [
        CaseSpec(
            "opt_fixed",
            "Fixed heuristic",
            OPT_DIR / "fixed_heuristic_result.json",
            w3,
            group="optimizer",
            fallback_config=DEFAULT_CONFIG,
            eval_base_seed=holdout_seed,
            force_fallback_config=True,
        ),
        CaseSpec(
            "opt_ga",
            "GA",
            OPT_DIR / "ga_result.json",
            w3,
            group="optimizer",
            fallback_config=DEFAULT_CONFIG,
            eval_base_seed=holdout_seed,
            force_fallback_config=True,
        ),
        CaseSpec(
            "opt_pso",
            "PSO",
            OPT_DIR / "pso_result.json",
            w3,
            group="optimizer",
            fallback_config=DEFAULT_CONFIG,
            eval_base_seed=holdout_seed,
            force_fallback_config=True,
        ),
        CaseSpec(
            "opt_cem",
            "CEM",
            cem_path,
            w3,
            group="optimizer",
            fallback_config=DEFAULT_CONFIG,
            eval_base_seed=holdout_seed,
            force_fallback_config=True,
        ),
    ]


def _fmt(mean: float, hw: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {hw:.{digits}f}"


def _eval_seed(spec: CaseSpec, loaded_seed: int) -> int:
    if spec.eval_base_seed is not None:
        return int(spec.eval_base_seed)
    return int(loaded_seed + spec.eval_seed_offset)


def _evaluate_specs(
    specs: list[CaseSpec],
    *,
    episodes: int,
    require_existing: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
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
    for spec in specs:
        if not spec.result_json.exists():
            if require_existing:
                raise FileNotFoundError(f"Missing result JSON: {spec.result_json}")
            continue
        cfg, params, seed = _load_case(spec)
        eval_seed = _eval_seed(spec, seed)
        samples = _evaluate_episodes(
            cfg,
            params,
            base_seed=int(eval_seed),
            episodes=int(episodes),
            weights=spec.weights,
        )
        case_row: dict[str, Any] = {
            "case_key": spec.key,
            "case_label": spec.label,
            "group": spec.group,
            "result_json": str(spec.result_json),
            "eval_base_seed": int(eval_seed),
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
    return rows_out, table_lines


def _write_outputs(
    *,
    out_dir: Path,
    stem: str,
    episodes: int,
    header: str,
    rows_out: list[dict[str, Any]],
    table_lines: list[str],
) -> Path:
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    txt_path = out_dir / f"{stem}.txt"
    lines = [header, ""] + table_lines
    if rows_out:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
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
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def run_uncertainty(*, episodes: int, out_dir: Path, group: str = "all") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "Uncertainty estimates: mean ± 95% CI half-width "
        f"(N={episodes} independent lifecycle episodes per case). "
        "Return is the normalized training objective."
    )
    last_path = out_dir / "uncertainty_mean_ci_halfwidth.txt"
    want_ablation = group in ("all", "ablation")
    want_optimizer = group in ("all", "optimizer")

    if want_ablation:
        rows, lines = _evaluate_specs(_case_specs(), episodes=episodes, require_existing=True)
        last_path = _write_outputs(
            out_dir=out_dir,
            stem="uncertainty_mean_ci_halfwidth",
            episodes=episodes,
            header=header,
            rows_out=rows,
            table_lines=lines,
        )

    if want_optimizer:
        opt_header = (
            "Optimizer comparison uncertainty estimates: mean ± 95% CI half-width "
            f"(N={episodes} independent lifecycle episodes, common holdout seeds). "
            "Return is the normalized training objective."
        )
        rows, lines = _evaluate_specs(
            _optimizer_case_specs(),
            episodes=episodes,
            require_existing=False,
        )
        if not rows:
            if group == "optimizer":
                raise FileNotFoundError(
                    f"No optimizer result JSONs found under {OPT_DIR}. "
                    "Run optimizer_baseline_experiment.py first."
                )
        else:
            last_path = _write_outputs(
                out_dir=out_dir,
                stem="optimizer_uncertainty_mean_ci_halfwidth",
                episodes=episodes,
                header=opt_header,
                rows_out=rows,
                table_lines=lines,
            )
    return last_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mean ± 95% CI half-width for saved policies.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(LCC_ROOT / "uncertainty_results"),
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["all", "ablation", "optimizer"],
        default="all",
        help="Which saved-policy group to evaluate.",
    )
    args = parser.parse_args()
    out = run_uncertainty(
        episodes=int(args.episodes),
        out_dir=Path(args.out_dir).resolve(),
        group=str(args.group),
    )
    print(out)


if __name__ == "__main__":
    main()
