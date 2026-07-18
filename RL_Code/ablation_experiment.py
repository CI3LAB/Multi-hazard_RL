from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager

import train_lifecycle_rl as tl

# ---------------------------
# Centralized plot font config
# Tune all figure text sizes here.
# ---------------------------
_PLOT_FONT: dict[str, float] = {
    # Metric panel fonts
    "metric_title": 34.0,
    "metric_axis": 33.0,
    "metric_ticks": 29.0,
    "metric_value": 27.0,
    # Convergence fonts
    "conv_title": 21.0,
    "conv_axis": 18.0,
    "conv_ticks": 16.0,
    "conv_legend": 16.0,
    # Tradeoff fonts
    "trade_title": 32.0,
    "trade_axis": 31.0,
    "trade_ticks": 28.0,
    "trade_label": 26.0,
    "trade_legend": 26.0,
    "trade_legend_title": 26.0,
    "trade_bubble_legend": 25.0,
    "trade_bubble_legend_title": 25.0,
    "trade_cbar_label": 28.0,
    "trade_cbar_ticks": 26.0,
    "trade_footer": 26.0,
}

# Muted fills with restrained academic hatch patterns.
_BAR_COLORS: list[str] = [
    "#6F8FBF",
    "#79AFC4",
    "#98C7A3",
    "#E5D48F",
    "#D78895",
    "#A996D2",
    "#C79E74",
]
_BAR_HATCHES: list[str] = [
    "////",
    "\\\\\\\\",
    "----",
    "||||",
    "xxxx",
    "....",
    "++",
]
_BAR_EDGE = "#222222"
_BAR_STEP = 1.00
_BAR_WIDTH = 0.68

_TIMES_NEW_ROMAN = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if _TIMES_NEW_ROMAN.exists():
    font_manager.fontManager.addfont(str(_TIMES_NEW_ROMAN))
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10.0,
        "hatch.linewidth": 0.7,
        "savefig.dpi": 300,
    }
)


def _font(name: str) -> float:
    return float(_PLOT_FONT[name])


def _apply_full_frame(ax: Any, *, linewidth: float = 0.9, color: str = "#333333") -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(float(linewidth))
        ax.spines[side].set_color(color)


def _plain_log_tick(value: float, _pos: int | None = None) -> str:
    value = float(value)
    if value <= 0.0:
        return ""
    if value >= 10.0:
        if abs(value - round(value)) < 1.0e-6:
            return str(int(round(value)))
        return f"{value:.0f}"
    if value >= 1.0:
        return f"{value:.0f}" if abs(value - round(value)) < 1.0e-6 else f"{value:.1f}"
    return f"{value:.1g}"


def _format_metric_yaxis(ax: Any, axis_spec: dict[str, Any]) -> None:
    ticks = axis_spec.get("ticks")
    if ticks is not None:
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    scale = str(axis_spec["scale"])
    if scale == "log":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_plain_log_tick))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    elif scale == "symlog":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _pos: f"{value:.0f}" if abs(value) >= 1.0 else f"{value:g}"
            )
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _lcc_rerun_root() -> Path:
    return _script_dir() / "lcc_rerun"


def _display_variant_name(name: str) -> str:
    name_map = {
        "baseline_full": "Default",
        "no_risk_term": "No risk",
        "no_resilience_term": "No resilience",
        "no_resilience_risk_compensated": "No resilience\n(risk-comp.)",
        "no_cost_term": "No cost",
        "no_pre_reinforcement": "No pre-reinforcement",
        "no_maintenance": "No maintenance",
        "no_repair": "No repair",
        "baseline_equal": "Equal\nweights",
        "cost_oriented": "Cost\noriented",
        "risk_oriented": "Risk\noriented",
        "resilience_oriented": "Resilience\noriented",
        "case1_baseline": "Default",
    }
    return name_map.get(str(name), str(name))


def _make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"ablation_{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _default_base_config_path() -> Path:
    return _lcc_rerun_root() / "configs" / "case1_baseline.json"


def _base_cfg(config_path: Path | None = None) -> tuple[tl.LifecycleConfig, tl.CEMConfig, Path]:
    path = _default_base_config_path() if config_path is None else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized baseline config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    lifecycle_raw = raw.get("lifecycle", {})
    cem_raw = raw.get("cem", {})
    if not isinstance(lifecycle_raw, dict):
        raise ValueError(f"'lifecycle' must be an object in {path}")
    if not isinstance(cem_raw, dict):
        raise ValueError(f"'cem' must be an object in {path}")

    lifecycle_cfg = tl.LifecycleConfig(
        **tl._coerce_lifecycle_config_dict(lifecycle_raw)  # type: ignore[attr-defined]
    )
    cem_cfg = tl.CEMConfig(**cem_raw)
    return lifecycle_cfg, cem_cfg, path.resolve()


def _variants(base: tl.LifecycleConfig) -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_full",
            "desc": "Full objective and full intervention modules.",
            "cfg": base,
        },
        {
            "name": "no_risk_term",
            "desc": "Remove risk term from reward and re-balance to w=(resilience, cost, risk)=(0.50, 0.50, 0.00).",
            "cfg": replace(base, w_lr=0.5, w_risk=0.0, w_cost=0.5),
        },
        {
            "name": "no_resilience_term",
            "desc": "Remove resilience-loss term from reward and re-balance to w=(resilience, cost, risk)=(0.00, 0.50, 0.50).",
            "cfg": replace(base, w_lr=0.0, w_risk=0.5, w_cost=0.5),
        },
        {
            "name": "no_resilience_risk_compensated",
            "desc": "Remove resilience-loss term with risk-compensated weights, w=(resilience, cost, risk)=(0.00, 0.33, 0.67).",
            "cfg": replace(base, w_lr=0.0, w_risk=2.0 / 3.0, w_cost=1.0 / 3.0),
        },
        {
            "name": "no_cost_term",
            "desc": "Remove cost term from reward.",
            "cfg": replace(base, w_lr=0.5, w_risk=0.5, w_cost=0.0),
        },
        {
            "name": "no_pre_reinforcement",
            "desc": "Disable pre-reinforcement effect and pre-cost differentiation.",
            "cfg": replace(
                base,
                pre_cost_options=(10.0, 10.0, 10.0),
                pre_fcrit_multipliers=(1.0, 1.0, 1.0),
                pre_hazard_damage_multipliers=(1.0, 1.0, 1.0),
                pre_deterioration_multipliers=(1.0, 1.0, 1.0),
            ),
        },
        {
            "name": "no_maintenance",
            "desc": "Disable periodic maintenance actions throughout the lifecycle.",
            "cfg": replace(
                base,
                maintenance_interval_years=0,
            ),
        },
        {
            "name": "no_repair",
            "desc": "Disable post-disaster repair recovery and repair cost actions.",
            "cfg": replace(
                base,
                repair_costs=(0.0, 0.0, 0.0),
                repair_recovery_deltas=(0.0, 0.0, 0.0),
                repair_recovery_rate_per_year=0.0,
                repair_durations_years=(0.0, 0.0, 0.0),
            ),
        },
    ]


def _best_params_from_result(result: dict[str, Any]) -> tl.PolicyParams:
    p = result["best"]["params"]
    return tl.PolicyParams(
        pre_index=int(p["pre_index"]),
        maint_t1=float(p["maint_t1"]),
        maint_t2=float(p["maint_t2"]),
        repair_t1=float(p["repair_t1"]),
        repair_t2=float(p["repair_t2"]),
    )


def _reported_return_from_terms(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Raw scalarisation used for paper-comparable ablation return plots."""
    w_lr = float(cfg.get("w_lr", 1.0 / 3.0))
    w_risk = float(cfg.get("w_risk", 1.0 / 3.0))
    w_cost = float(cfg.get("w_cost", 1.0 / 3.0))
    lr = float(metrics["lr"])
    risk = float(metrics["risk"])
    cost = float(metrics.get("lcc", metrics.get("npv", 0.0)))
    return float(-(w_lr * lr + w_risk * risk + w_cost * cost))


def _ensure_reported_return(record: dict[str, Any]) -> None:
    cfg = record.get("cfg", {})
    if not isinstance(cfg, dict):
        cfg = {}
    for key in ("best_eval_train", "best_eval_holdout"):
        metrics = record.get(key)
        if isinstance(metrics, dict):
            metrics["reported_return"] = _reported_return_from_terms(metrics, cfg)


def _weight_string(cfg: dict[str, Any]) -> str:
    return (
        f"{float(cfg.get('w_lr', 0.0)):.2f}/"
        f"{float(cfg.get('w_cost', 0.0)):.2f}/"
        f"{float(cfg.get('w_risk', 0.0)):.2f}"
    )


def _main_ablation_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        rec for rec in records
        if str(rec.get("name", "")) not in {
            "no_resilience_risk_compensated",
        }
    ]


def _weight_sensitivity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return []


def _resilience_sensitivity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "baseline_full",
        "no_resilience_term",
        "no_resilience_risk_compensated",
    }
    picked = [rec for rec in records if str(rec.get("name", "")) in wanted]
    order = {
        "baseline_full": 0,
        "no_resilience_term": 1,
        "no_resilience_risk_compensated": 2,
    }
    return sorted(picked, key=lambda rec: order.get(str(rec.get("name", "")), 999))


def _weight_preference_case_specs(script_dir: Path) -> list[dict[str, Any]]:
    root = Path(script_dir).resolve() / "lcc_rerun"
    return [
        {
            "name": "baseline_equal",
            "desc": "Baseline equal-weight policy under the normalized objective.",
            "result_json": root / "case1_baseline" / "cem_results_case1_baseline.json",
        },
        {
            "name": "cost_oriented",
            "desc": "Preference case with higher cost weight.",
            "result_json": root / "case3a_cost_oriented" / "cem_results_case3a_cost_oriented.json",
        },
        {
            "name": "risk_oriented",
            "desc": "Preference case with higher risk weight.",
            "result_json": root / "case3b_risk_oriented" / "cem_results_case3b_risk_oriented.json",
        },
        {
            "name": "resilience_oriented",
            "desc": "Preference case with higher resilience-loss weight.",
            "result_json": root / "case3c_resilience_oriented" / "cem_results_case3c_resilience_oriented.json",
        },
    ]


def _record_from_case_result(
    *,
    name: str,
    desc: str,
    result_json: Path,
    holdout_episodes: int,
    holdout_seed: int,
) -> dict[str, Any]:
    if not result_json.exists():
        raise FileNotFoundError(f"Case-result JSON not found: {result_json}")
    raw = json.loads(result_json.read_text(encoding="utf-8"))
    lifecycle_raw = raw.get("lifecycle_config", {})
    cem_raw = raw.get("cem_config", {})
    if not isinstance(lifecycle_raw, dict):
        raise ValueError(f"'lifecycle_config' must be an object in {result_json}")
    if not isinstance(cem_raw, dict):
        raise ValueError(f"'cem_config' must be an object in {result_json}")

    cfg = tl.LifecycleConfig(**tl._coerce_lifecycle_config_dict(lifecycle_raw))  # type: ignore[attr-defined]
    best_params_raw = raw["best"]["params"]
    best_params = tl.PolicyParams(
        pre_index=int(best_params_raw["pre_index"]),
        maint_t1=float(best_params_raw["maint_t1"]),
        maint_t2=float(best_params_raw["maint_t2"]),
        repair_t1=float(best_params_raw["repair_t1"]),
        repair_t2=float(best_params_raw["repair_t2"]),
    )
    holdout = dict(
        tl._evaluate_params(  # type: ignore[attr-defined]
            cfg,
            best_params,
            base_seed=int(holdout_seed),
            episodes=int(holdout_episodes),
        )
    )
    payload = {
        "name": str(name),
        "desc": str(desc),
        "seed": int(cem_raw.get("seed", 0)),
        "cfg": {
            "w_lr": float(cfg.w_lr),
            "w_risk": float(cfg.w_risk),
            "w_cost": float(cfg.w_cost),
            "objective_cost_metric": str(cfg.objective_cost_metric),
            "objective_normalization": str(cfg.objective_normalization),
            "lr_ref": float(cfg.lr_ref),
            "risk_ref": float(cfg.risk_ref),
            "cost_ref": float(cfg.cost_ref),
            "floor_ref": float(cfg.floor_ref),
            "w_floor": float(cfg.w_floor),
        },
        "cem_cfg": {
            "iterations": int(cem_raw.get("iterations", 0)),
            "population": int(cem_raw.get("population", 0)),
            "elite_frac": float(cem_raw.get("elite_frac", 0.0)),
            "eval_episodes": int(cem_raw.get("eval_episodes", 0)),
        },
        "best_eval_train": dict(raw["best"]["eval"]),
        "best_eval_holdout": holdout,
        "best_params": dict(best_params_raw),
        "best_selection": dict(raw.get("best", {}).get("selection", {})),
        "compare_seed_episode": dict(raw["best"]["episode_compare_seed"]),
        "history": list(raw.get("history", [])),
        "source_result_json": str(result_json),
    }
    _ensure_reported_return(payload)
    return payload


def _weight_preference_records(
    *,
    script_dir: Path,
    holdout_episodes: int,
    holdout_seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _weight_preference_case_specs(script_dir):
        rec = _record_from_case_result(
                name=str(spec["name"]),
                desc=str(spec["desc"]),
                result_json=Path(spec["result_json"]),
                holdout_episodes=int(holdout_episodes),
                holdout_seed=int(holdout_seed),
            )
        # For the weight-preference analysis, the primary return should follow
        # the normalized objective actually used in training/evaluation rather
        # than the paper-comparable raw scalarisation used in the ablation view.
        for key in ("best_eval_train", "best_eval_holdout"):
            metrics = rec.get(key)
            if isinstance(metrics, dict):
                metrics["reported_return"] = float(metrics["return"])
        records.append(rec)
    return records


def _global_lr_scale(records: list[dict[str, Any]]) -> dict[str, Any]:
    lr_all = np.array(
        [float(r["best_eval_holdout"]["lr"]) for r in records],
        dtype=float,
    )
    lr_min = float(np.min(lr_all))
    sorted_lr = np.sort(lr_all)
    raw_lr_max = float(sorted_lr[-1])
    second_largest = float(sorted_lr[-2]) if sorted_lr.size >= 2 else raw_lr_max
    # A catastrophic no-repair outcome can be several times larger than all
    # other cases. Cap that outlier at the smallest bubble, preserving the
    # visually informative proportional differences among the normal cases.
    lr_max = second_largest if raw_lr_max > 3.0 * max(second_largest, 1.0e-12) else raw_lr_max
    lr_refs = np.array([lr_min, float(np.median(np.clip(lr_all, lr_min, lr_max))), lr_max], dtype=float)
    return {
        "lr_min": lr_min,
        "lr_max": lr_max,
        "lr_refs": lr_refs,
        "size_min": 120.0,
        "size_span": 420.0,
    }


def _bubble_sizes_from_lr(lr_values: np.ndarray, scale: dict[str, Any]) -> np.ndarray:
    lr_values = np.asarray(lr_values, dtype=float)
    lr_min = float(scale["lr_min"])
    lr_max = float(scale["lr_max"])
    size_min = float(scale["size_min"])
    size_span = float(scale["size_span"])
    if lr_max - lr_min <= 1.0e-12:
        return np.full_like(lr_values, size_min + 0.5 * size_span)
    clipped_lr = np.clip(lr_values, lr_min, lr_max)
    return size_min + size_span * (lr_max - clipped_lr) / (lr_max - lr_min)


def _metric_value(record: dict[str, Any], metric_key: str) -> float:
    metrics = record["best_eval_holdout"]
    if metric_key == "reported_return":
        if "reported_return" not in metrics:
            _ensure_reported_return(record)
        return float(record["best_eval_holdout"]["reported_return"])
    return float(metrics[metric_key])


def _metric_axis_spec(metric_key: str, vals: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(vals, dtype=float)
    if metric_key == "reported_return":
        return {
            "scale": "symlog",
            "scale_kwargs": {"linthresh": 1.0, "linscale": 1.0, "base": 10},
            "ymin": -320.0,
            "ymax": 0.0,
            "label_suffix": " symlog",
            "ticks": [-300.0, -100.0, -30.0, -10.0, -3.0, -1.0, 0.0],
        }

    if metric_key == "lcc":
        return {
            "scale": "linear",
            "scale_kwargs": {},
            "ymin": 0.0,
            "ymax": 120.0,
            "label_suffix": "",
            "ticks": list(np.arange(0.0, 121.0, 20.0)),
        }

    positive = vals[vals > 0.0]
    if positive.size == 0:
        return {
            "scale": "linear",
            "scale_kwargs": {},
            "ymin": 0.0,
            "ymax": 1.0,
            "label_suffix": "",
        }

    if metric_key == "lr":
        return {
            "scale": "log",
            "scale_kwargs": {"base": 10},
            "ymin": 2.0,
            "ymax": 200.0,
            "label_suffix": " log",
            "ticks": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
        }
    if metric_key == "risk":
        return {
            "scale": "log",
            "scale_kwargs": {"base": 10},
            "ymin": 1.0,
            "ymax": 2000.0,
            "label_suffix": " log",
            "ticks": [1.0, 10.0, 100.0, 1000.0],
        }

    vmin = float(np.min(positive))
    vmax = float(np.max(vals))
    return {
        "scale": "log",
        "scale_kwargs": {"base": 10},
        "ymin": max(vmin / 1.8, 1.0e-3),
        "ymax": max(vmax * 1.8, vmin * 2.0),
        "label_suffix": " log",
    }


def _annotation_style_for_metric(metric_key: str, value: float) -> tuple[int, str]:
    if metric_key == "reported_return":
        return (-8, "top") if value <= 0.0 else (8, "bottom")
    return (8, "bottom")


def _metric_annotation_override(
    metric_key: str,
    variant_name: str,
    *,
    panel_kind: str,
) -> dict[str, Any] | None:
    if metric_key == "reported_return" and variant_name == "no_repair":
        return {"xytext": (-48, 8), "ha": "right", "va": "bottom"}
    if metric_key == "lcc":
        if panel_kind == "single":
            overrides = {
                "baseline_full": {"xytext": (-28, 12), "ha": "center", "va": "bottom"},
                "no_risk_term": {"xytext": (-12, 28), "ha": "center", "va": "bottom"},
                "no_resilience_term": {"xytext": (20, 14), "ha": "center", "va": "bottom"},
                "no_cost_term": {"xytext": (0, 12), "ha": "center", "va": "bottom"},
                "no_pre_reinforcement": {"xytext": (0, 24), "ha": "center", "va": "bottom"},
                "no_maintenance": {"xytext": (16, 12), "ha": "center", "va": "bottom"},
                "no_repair": {"xytext": (0, 8), "ha": "center", "va": "bottom"},
            }
            return overrides.get(variant_name)
        if panel_kind == "grid":
            overrides = {
                "baseline_full": {"xytext": (-10, 5), "ha": "center", "va": "bottom"},
                "no_risk_term": {"xytext": (-4, 13), "ha": "center", "va": "bottom"},
                "no_resilience_term": {"xytext": (10, 5), "ha": "center", "va": "bottom"},
                "no_pre_reinforcement": {"xytext": (0, 12), "ha": "center", "va": "bottom"},
                "no_repair": {"xytext": (0, 5), "ha": "center", "va": "bottom"},
            }
            return overrides.get(variant_name)
    return None


def _run_one_variant(
    *,
    variant: dict[str, Any],
    cem_cfg: tl.CEMConfig,
    seed: int,
    holdout_episodes: int,
    out_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    name = str(variant["name"])
    cfg = variant["cfg"]
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cem_cfg_v = replace(cem_cfg, seed=int(seed))
    logger.info("Ablation run start | variant=%s seed=%d", name, seed)
    result = tl.train_cem(cfg, cem_cfg_v, logger=logger)

    (run_dir / "cem_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_params = _best_params_from_result(result)
    holdout = tl._evaluate_params(  # type: ignore[attr-defined]
        cfg,
        best_params,
        base_seed=int(seed + 700_000),
        episodes=int(holdout_episodes),
    )
    holdout = dict(holdout)
    compare_ep = result["best"]["episode_compare_seed"]
    payload = {
        "name": name,
        "desc": str(variant["desc"]),
        "seed": int(seed),
        "cfg": {
            "w_lr": float(cfg.w_lr),
            "w_risk": float(cfg.w_risk),
            "w_cost": float(cfg.w_cost),
            "objective_cost_metric": str(cfg.objective_cost_metric),
            "objective_normalization": str(cfg.objective_normalization),
            "lr_ref": float(cfg.lr_ref),
            "risk_ref": float(cfg.risk_ref),
            "cost_ref": float(cfg.cost_ref),
            "floor_ref": float(cfg.floor_ref),
            "w_floor": float(cfg.w_floor),
            "pre_cost_options": list(map(float, cfg.pre_cost_options)),
            "pre_fcrit_multipliers": list(map(float, cfg.pre_fcrit_multipliers)),
            "pre_hazard_damage_multipliers": list(
                map(float, cfg.pre_hazard_damage_multipliers)
            ),
            "pre_deterioration_multipliers": list(
                map(float, cfg.pre_deterioration_multipliers)
            ),
        },
        "cem_cfg": {
            "iterations": int(cem_cfg_v.iterations),
            "population": int(cem_cfg_v.population),
            "elite_frac": float(cem_cfg_v.elite_frac),
            "eval_episodes": int(cem_cfg_v.eval_episodes),
        },
        "best_eval_train": result["best"]["eval"],
        "best_eval_holdout": holdout,
        "best_params": result["best"]["params"],
        "best_selection": dict(result.get("best", {}).get("selection", {})),
        "compare_seed_episode": {
            "lr": float(compare_ep["lr"]),
            "risk": float(compare_ep["risk"]),
            "npv": float(compare_ep["npv"]),
            "return_value": float(compare_ep["return_value"]),
        },
        "history": result["history"],
    }
    _ensure_reported_return(payload)
    (run_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Ablation run done | variant=%s holdout_return=%.4f holdout_risk=%.3f holdout_cost=%.2f",
        name,
        float(holdout["return"]),
        float(holdout["risk"]),
        float(holdout["lcc"]),
    )
    return payload


def _plot_one_metric_bar(
    records: list[dict[str, Any]],
    *,
    metric_key: str,
    metric_ylabel: str,
    higher_better: bool,
    out_path: Path,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    names = [r["name"] for r in records]
    labels = []
    for n in names:
        txt = _display_variant_name(n)
        txt = txt.replace(" No ", "\nNo ")
        labels.append(txt)
    x = np.arange(len(names), dtype=float) * _BAR_STEP
    fig, ax = plt.subplots(figsize=(16.0, 9.8))
    vals = np.array([_metric_value(r, metric_key) for r in records], dtype=float)
    bars = ax.bar(
        x,
        vals,
        width=_BAR_WIDTH,
        color=[_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(vals))],
        edgecolor=_BAR_EDGE,
        linewidth=0.95,
        alpha=0.96,
        zorder=3,
    )
    for i, b in enumerate(bars):
        b.set_hatch(_BAR_HATCHES[i % len(_BAR_HATCHES)])
    ax.set_facecolor("#F7F7F7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", labelsize=_font("metric_ticks"))
    ax.tick_params(axis="x", pad=8)
    axis_spec = _metric_axis_spec(metric_key, vals)
    ax.set_yscale(axis_spec["scale"], **axis_spec["scale_kwargs"])
    ax.set_ylim(float(axis_spec["ymin"]), float(axis_spec["ymax"]))
    _format_metric_yaxis(ax, axis_spec)
    ax.set_ylabel(metric_ylabel + str(axis_spec["label_suffix"]), fontsize=_font("metric_axis"))
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    _apply_full_frame(ax, linewidth=0.95)
    ax.margins(x=0.04 if metric_key == "reported_return" else 0.06)
    baseline = float(vals[0])
    ax.axhline(
        baseline,
        color="#2F4F4F",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
    )
    for i, b in enumerate(bars):
        v = float(vals[i])
        delta = v - baseline
        sign = "+" if delta >= 0 else ""
        marker = ""
        if i != 0:
            marker = " \u2191" if higher_better and delta >= 0 else ""
            marker = marker or (" \u2193" if (not higher_better and delta <= 0) else "")
        label = f"{v:.2f}\n({sign}{delta:.2f}){marker}"
        y_shift, va = _annotation_style_for_metric(metric_key, v)
        label_font = _font("metric_value")
        override = _metric_annotation_override(metric_key, str(names[i]), panel_kind="single")
        xytext = (0, y_shift)
        ha = "center"
        if override is not None:
            xytext = tuple(override["xytext"])
            ha = str(override["ha"])
            va = str(override["va"])
        ax.annotate(
            label,
            xy=(b.get_x() + b.get_width() / 2.0, v),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=label_font,
            clip_on=False,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _metric_specs() -> list[tuple[str, str, bool, str, str]]:
    return [
        ("reported_return", "Return", True, "ablation_metric_return.png", "(a) Return"),
        ("lr", "Resilience loss", False, "ablation_metric_lr.png", "(b) Resilience loss"),
        ("risk", "Risk", False, "ablation_metric_risk.png", "(c) Risk"),
        ("lcc", "Cost", False, "ablation_metric_cost.png", "(d) Cost"),
    ]


def _plot_metric_grid(records: list[dict[str, Any]], out_path: Path) -> None:
    labels = []
    for rec in records:
        txt = _display_variant_name(str(rec["name"]))
        labels.append(txt.replace("\n", " "))
    x = np.arange(len(records), dtype=float) * _BAR_STEP

    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.4))
    for ax, (metric_key, metric_ylabel, higher_better, _filename, caption) in zip(
        axes.flat, _metric_specs()
    ):
        vals = np.array(
            [_metric_value(r, metric_key) for r in records],
            dtype=float,
        )
        bars = ax.bar(
            x,
            vals,
            width=_BAR_WIDTH,
            color=[_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(vals))],
            edgecolor=_BAR_EDGE,
            linewidth=0.6,
            alpha=0.96,
            zorder=3,
        )
        for i, b in enumerate(bars):
            b.set_hatch(_BAR_HATCHES[i % len(_BAR_HATCHES)])

        axis_spec = _metric_axis_spec(metric_key, vals)
        ax.set_yscale(axis_spec["scale"], **axis_spec["scale_kwargs"])
        ax.set_ylim(float(axis_spec["ymin"]), float(axis_spec["ymax"]))
        _format_metric_yaxis(ax, axis_spec)

        baseline = float(vals[0])
        ax.axhline(
            baseline,
            color="#2F4F4F",
            linestyle=":",
            linewidth=0.6,
            alpha=0.8,
            zorder=2,
        )
        for i, b in enumerate(bars):
            v = float(vals[i])
            delta = v - baseline
            sign = "+" if delta >= 0 else ""
            marker = ""
            if i != 0:
                marker = " \u2191" if higher_better and delta >= 0 else ""
                marker = marker or (" \u2193" if (not higher_better and delta <= 0) else "")
            label = f"{v:.2f}\n({sign}{delta:.2f}){marker}"
            y_shift, va = _annotation_style_for_metric(metric_key, v)
            override = _metric_annotation_override(metric_key, str(records[i]["name"]), panel_kind="grid")
            xytext = (0, 4 if y_shift > 0 else -4)
            ha = "center"
            if override is not None:
                raw_xytext = tuple(override["xytext"])
                xytext = (int(raw_xytext[0]), int(raw_xytext[1]))
                ha = str(override["ha"])
                va = str(override["va"])
            ax.annotate(
                label,
                xy=(b.get_x() + b.get_width() / 2.0, v),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=18.8,
                clip_on=False,
            )

        ax.set_facecolor("#F7F7F7")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", rotation_mode="anchor", fontsize=18.0)
        ax.tick_params(axis="y", labelsize=19.4)
        ax.tick_params(axis="x", pad=7)
        ax.set_ylabel(metric_ylabel + str(axis_spec["label_suffix"]), fontsize=20.5)
        ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.40, zorder=0)
        _apply_full_frame(ax, linewidth=0.8)
        ax.margins(x=0.12 if metric_key == "reported_return" else 0.06)
        ax.text(0.5, -0.34, caption, transform=ax.transAxes, ha="center", va="top", fontsize=21.5)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.16, wspace=0.30, hspace=0.58)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panels(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, ylabel, higher_better, filename, _caption in _metric_specs():
        _plot_one_metric_bar(
            records,
            metric_key=key,
            metric_ylabel=ylabel,
            higher_better=higher_better,
            out_path=out_dir / filename,
        )
    shutil.copy2(out_dir / "ablation_metric_cost.png", out_dir / "ablation_metric_npv.png")
    _plot_metric_grid(records, out_dir / "ablation_metrics.png")


def _copy_key_ablation_figures(out_dir: Path, *, dst_subdir: str = "") -> None:
    src_dir = out_dir
    dst_dir = _lcc_rerun_root() / "fig" / "ablation"
    if str(dst_subdir).strip():
        dst_dir = dst_dir / str(dst_subdir).strip()
    dst_dir.mkdir(parents=True, exist_ok=True)
    fig_names = [
        "ablation_metric_return.png",
        "ablation_metric_lr.png",
        "ablation_metric_risk.png",
        "ablation_metric_cost.png",
        "ablation_metric_npv.png",
        "ablation_metrics.png",
        "ablation_tradeoff.png",
    ]
    for name in fig_names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _plot_convergence(records: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for rec in records:
        hist = rec["history"]
        xs = [int(h["iteration"]) for h in hist]
        ys = [float(h["best_of_iter"]["return"]) for h in hist]
        ax.plot(xs, ys, linewidth=1.8, label=rec["name"])
    ax.set_xlabel("Iteration", fontsize=_font("conv_axis"))
    ax.set_ylabel("Best return of iteration", fontsize=_font("conv_axis"))
    ax.set_title("Ablation Convergence", fontsize=_font("conv_title"))
    ax.tick_params(axis="both", labelsize=_font("conv_ticks"))
    ax.grid(True, alpha=0.25)
    _apply_full_frame(ax, linewidth=0.9)
    ax.legend(loc="best", fontsize=_font("conv_legend"), frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_tradeoff(
    records: list[dict[str, Any]],
    out_path: Path,
    *,
    bubble_scale: dict[str, Any] | None = None,
) -> None:
    names = [str(r["name"]) for r in records]
    cost = np.array([float(r["best_eval_holdout"]["lcc"]) for r in records], dtype=float)
    risk = np.array([float(r["best_eval_holdout"]["risk"]) for r in records], dtype=float)
    lr = np.array([float(r["best_eval_holdout"]["lr"]) for r in records], dtype=float)
    ret = np.array([_metric_value(r, "reported_return") for r in records], dtype=float)

    scale = _global_lr_scale(records) if bubble_scale is None else dict(bubble_scale)
    lr_min = float(scale["lr_min"])
    lr_max = float(scale["lr_max"])
    bubble_sizes = _bubble_sizes_from_lr(lr, scale)

    display_cost = np.array(cost, dtype=float, copy=True)
    display_risk = np.array(risk, dtype=float, copy=True)
    name_set = set(names)

    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    scatter_ref = ax.scatter(
        display_cost,
        display_risk,
        s=bubble_sizes,
        c=ret,
        cmap="viridis",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    _apply_full_frame(ax, linewidth=0.9)
    ax.tick_params(axis="both", labelsize=_font("trade_ticks"))

    risk_positive = display_risk[display_risk > 0.0]
    if risk_positive.size > 0:
        ax.set_yscale("log", base=10)
        ax.set_ylim(0.8, 2000.0)
        ax.yaxis.set_major_locator(mticker.FixedLocator([1.0, 10.0, 100.0, 1000.0]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_plain_log_tick))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    if name_set == {"baseline_full", "no_resilience_term", "no_resilience_risk_compensated"}:
        label_offsets = {
            "baseline_full": (26, -6),
            "no_resilience_term": (22, 18),
            "no_resilience_risk_compensated": (-30, 12),
        }
        label_align = {
            "baseline_full": ("left", "center"),
            "no_resilience_term": ("left", "bottom"),
            "no_resilience_risk_compensated": ("right", "bottom"),
        }
    elif name_set == {"baseline_equal", "cost_oriented", "risk_oriented", "resilience_oriented"}:
        label_offsets = {
            "baseline_equal": (20, -10),
            "cost_oriented": (-24, 12),
            "risk_oriented": (18, 16),
            "resilience_oriented": (18, -16),
        }
        label_align = {
            "baseline_equal": ("left", "top"),
            "cost_oriented": ("right", "bottom"),
            "risk_oriented": ("left", "bottom"),
            "resilience_oriented": ("left", "top"),
        }
    else:
        label_offsets = {
            "baseline_full": (12, 10),
            "no_risk_term": (10, 10),
            "no_resilience_term": (-25, -14),
            "no_cost_term": (14, 0),
            "no_pre_reinforcement": (-12, 10),
            "no_maintenance": (10, 8),
            "no_repair": (10, -8),
        }
        label_align = {
            "baseline_full": ("left", "top"),
            "no_risk_term": ("left", "bottom"),
            "no_resilience_term": ("right", "top"),
            "no_cost_term": ("left", "center"),
            "no_pre_reinforcement": ("right", "top"),
            "no_maintenance": ("left", "bottom"),
            "no_repair": ("left", "top"),
        }
    for i, name in enumerate(names):
        dx, dy = label_offsets.get(name, (8, 8))
        ha, va = label_align.get(name, ("left", "bottom"))
        ax.annotate(
            _display_variant_name(name),
            xy=(display_cost[i], display_risk[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=_font("trade_label"),
            ha=ha,
            va=va,
            annotation_clip=False,
        )

    cbar = fig.colorbar(scatter_ref, ax=ax, pad=0.03, fraction=0.05)
    cbar.set_label("Holdout return", rotation=90, fontsize=_font("trade_cbar_label"))
    cbar.ax.tick_params(labelsize=_font("trade_cbar_ticks"))

    # Bubble-size legend for LR values.
    lr_refs = np.array(scale["lr_refs"], dtype=float)
    size_refs = _bubble_sizes_from_lr(lr_refs, scale)
    size_handles = [
        ax.scatter([], [], s=float(size_refs[i]), color="#6A6A6A", alpha=0.78, edgecolors="none")
        for i in range(3)
    ]
    ax.legend(
        size_handles,
        [f"LR={float(v):.2f}" for v in lr_refs],
        title="Bubble size (LR)",
        loc="upper right",
        bbox_to_anchor=(0.96, 0.98),
        ncols=1,
        frameon=False,
        markerscale=0.58,
        labelspacing=1.15,
        handletextpad=1.0,
        borderpad=0.3,
        fontsize=_font("trade_bubble_legend"),
        title_fontsize=_font("trade_bubble_legend_title"),
    )

    ax.set_xlim(25.0, 125.0)
    ax.xaxis.set_major_locator(mticker.FixedLocator([30.0, 50.0, 70.0, 90.0, 110.0]))
    ax.set_xlabel("Cost", fontsize=_font("trade_axis"))
    ax.set_ylabel("Risk log", fontsize=_font("trade_axis"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _build_report(
    records: list[dict[str, Any]],
    out_dir: Path,
    settings: dict[str, Any],
    *,
    title: str = "Ablation Study Report",
    note_lines: list[str] | None = None,
    return_note: str = "Return is reported with the paper-comparable raw scalarisation of LR, risk, and cost under each ablation weight setting.",
) -> None:
    baseline = records[0]["best_eval_holdout"]
    baseline_return = _metric_value(records[0], "reported_return")
    rank = sorted(
        records,
        key=lambda r: _metric_value(r, "reported_return"),
        reverse=True,
    )
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- run_time: `{settings['run_time']}`")
    lines.append(f"- iterations: `{settings['iterations']}`")
    lines.append(f"- population: `{settings['population']}`")
    lines.append(f"- elite_frac: `{settings['elite_frac']}`")
    lines.append(f"- eval_episodes: `{settings['eval_episodes']}`")
    lines.append(f"- holdout_episodes: `{settings['holdout_episodes']}`")
    lines.append(f"- {return_note}")
    if note_lines:
        for note in note_lines:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## Holdout Metrics")
    lines.append("")
    lines.append("| Variant | Weights (Resilience/Cost/Risk) | Return | LR | Risk | Cost | Min F | Feasible |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        m = rec["best_eval_holdout"]
        cfg = rec.get("cfg", {})
        lines.append(
            f"| {rec['name']} | {_weight_string(cfg)} | {_metric_value(rec, 'reported_return'):.4f} | {float(m['lr']):.3f} | {float(m['risk']):.3f} | {float(m['lcc']):.2f} | {float(m['min_f']):.3f} | {float(m['feasible_frac']):.3f} |"
        )
    lines.append("")
    lines.append("## Relative To Baseline")
    lines.append("")
    lines.append("| Variant | dReturn | dLR | dRisk | dCost |")
    lines.append("|---|---:|---:|---:|---:|")
    for rec in records[1:]:
        m = rec["best_eval_holdout"]
        lines.append(
            f"| {rec['name']} | {_metric_value(rec, 'reported_return') - baseline_return:+.4f} | {float(m['lr']) - float(baseline['lr']):+.3f} | {float(m['risk']) - float(baseline['risk']):+.3f} | {float(m['lcc']) - float(baseline['lcc']):+.2f} |"
        )
    lines.append("")
    lines.append("## Quick Findings")
    lines.append("")
    lines.append(
        f"- Best holdout return: `{rank[0]['name']}` ({_metric_value(rank[0], 'reported_return'):.4f})."
    )
    worst_risk = max(records, key=lambda r: float(r["best_eval_holdout"]["risk"]))
    best_risk = min(records, key=lambda r: float(r["best_eval_holdout"]["risk"]))
    lines.append(
        f"- Lowest risk: `{best_risk['name']}` ({float(best_risk['best_eval_holdout']['risk']):.3f}); highest risk: `{worst_risk['name']}` ({float(worst_risk['best_eval_holdout']['risk']):.3f})."
    )
    best_cost = min(records, key=lambda r: float(r["best_eval_holdout"]["lcc"]))
    worst_cost = max(records, key=lambda r: float(r["best_eval_holdout"]["lcc"]))
    lines.append(
        f"- Lowest cost: `{best_cost['name']}` ({float(best_cost['best_eval_holdout']['lcc']):.2f}); highest cost: `{worst_cost['name']}` ({float(worst_cost['best_eval_holdout']['lcc']):.2f})."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `ablation_metric_return.png`, `ablation_metric_lr.png`, `ablation_metric_risk.png`, `ablation_metric_cost.png`: metric-wise bar comparisons.")
    lines.append("- `ablation_metrics.png`: 2x2 metric panel.")
    lines.append("- `ablation_convergence.png`: best-of-iteration return curves.")
    lines.append("- `ablation_tradeoff.png`: risk-cost trade-off scatter.")
    lines.append("")
    (out_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _render_weight_preference_analysis(
    *,
    script_dir: Path,
    out_dir: Path,
    holdout_episodes: int,
    holdout_seed: int,
    run_time: str,
) -> list[dict[str, Any]]:
    records = _weight_preference_records(
        script_dir=script_dir,
        holdout_episodes=int(holdout_episodes),
        holdout_seed=int(holdout_seed),
    )
    bubble_scale = _global_lr_scale(records)
    _plot_metric_panels(records, out_dir)
    _plot_convergence(records, out_dir / "ablation_convergence.png")
    _plot_tradeoff(records, out_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
    _copy_key_ablation_figures(out_dir, dst_subdir="weight_preference_cases")
    _build_report(
        records,
        out_dir,
        {
            "run_time": run_time,
            "iterations": int(records[0]["cem_cfg"]["iterations"]),
            "population": int(records[0]["cem_cfg"]["population"]),
            "elite_frac": float(records[0]["cem_cfg"]["elite_frac"]),
            "eval_episodes": int(records[0]["cem_cfg"]["eval_episodes"]),
            "holdout_episodes": int(holdout_episodes),
        },
        title="Three-weight Sensitivity Report",
        return_note="Return is reported with the normalized holdout objective used in each case, consistent with the revised normalized Eq. (16).",
        note_lines=[
            "This report compares the baseline equal-weight case and the three preference-driven cases under the same hazard setting.",
            "All four cases use the normalized objective, so the weights act on normalized LR, risk, and cost terms.",
            "The cases correspond to equal-weight, cost-oriented, risk-oriented, and resilience-oriented preferences.",
        ],
    )
    payload = {
        "run_time": run_time,
        "settings": {
            "holdout_episodes": int(holdout_episodes),
            "holdout_seed": int(holdout_seed),
        },
        "records": records,
    }
    (out_dir / "weight_preference_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def run_ablation(
    *,
    out_root: Path,
    iterations: int | None,
    population: int | None,
    eval_episodes: int | None,
    holdout_episodes: int,
    elite_frac: float | None,
    base_seed: int | None,
    base_config_path: Path | None = None,
) -> Path:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = out_root
    script_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(out_dir / "ablation.log")

    base_cfg, source_cem_cfg, source_config_path = _base_cfg(base_config_path)
    iterations_v = int(source_cem_cfg.iterations if iterations is None else iterations)
    population_v = int(source_cem_cfg.population if population is None else population)
    eval_episodes_v = int(source_cem_cfg.eval_episodes if eval_episodes is None else eval_episodes)
    elite_frac_v = float(source_cem_cfg.elite_frac if elite_frac is None else elite_frac)
    base_seed_v = int(source_cem_cfg.seed if base_seed is None else base_seed)

    variants = _variants(base_cfg)
    cem_cfg = tl.CEMConfig(
        iterations=iterations_v,
        population=population_v,
        elite_frac=elite_frac_v,
        eval_episodes=eval_episodes_v,
        seed=base_seed_v,
        plot_first_n=0,
        plot_interval=9999,
    )

    logger.info("Ablation setup | out_dir=%s base_config=%s", str(out_dir), str(source_config_path))
    records: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        seed = int(base_seed_v + idx * 1000)
        rec = _run_one_variant(
            variant=variant,
            cem_cfg=cem_cfg,
            seed=seed,
            holdout_episodes=holdout_episodes,
            out_dir=out_dir,
            logger=logger,
        )
        records.append(rec)
    for rec in records:
        _ensure_reported_return(rec)

    main_records = _main_ablation_records(records)
    weight_records = _weight_sensitivity_records(records)
    resilience_records = _resilience_sensitivity_records(records)
    bubble_scale = _global_lr_scale(records)

    _plot_metric_panels(main_records, out_dir)
    _plot_convergence(main_records, out_dir / "ablation_convergence.png")
    _plot_tradeoff(main_records, out_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
    _copy_key_ablation_figures(out_dir)

    if weight_records:
        weight_dir = out_dir / "weight_sensitivity"
        _plot_metric_panels(weight_records, weight_dir)
        _plot_convergence(weight_records, weight_dir / "ablation_convergence.png")
        _plot_tradeoff(weight_records, weight_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
        _copy_key_ablation_figures(weight_dir, dst_subdir="weight_sensitivity")

    if resilience_records:
        resilience_dir = out_dir / "resilience_sensitivity"
        _plot_metric_panels(resilience_records, resilience_dir)
        _plot_convergence(resilience_records, resilience_dir / "ablation_convergence.png")
        _plot_tradeoff(resilience_records, resilience_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
        _copy_key_ablation_figures(resilience_dir, dst_subdir="resilience_sensitivity")

    summary_payload = {
        "run_time": run_time,
        "settings": {
            "base_config": str(source_config_path),
            "iterations": iterations_v,
            "population": population_v,
            "elite_frac": elite_frac_v,
            "eval_episodes": eval_episodes_v,
            "holdout_episodes": int(holdout_episodes),
            "base_seed": base_seed_v,
        },
        "records": records,
    }
    (out_dir / "ablation_all_results.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _build_report(
        main_records,
        out_dir,
        {
            "run_time": run_time,
            "iterations": iterations_v,
            "population": population_v,
            "elite_frac": elite_frac_v,
            "eval_episodes": eval_episodes_v,
            "holdout_episodes": int(holdout_episodes),
        },
        note_lines=[
            "The supplemental no-resilience risk-compensated case uses w=(resilience, cost, risk)=(0.00, 0.33, 0.67) and is plotted separately under `resilience_sensitivity/`.",
            "All trade-off figures use one shared global bubble-size scale based on LR across the full ablation set.",
        ],
    )
    if weight_records:
        _build_report(
            weight_records,
            out_dir / "weight_sensitivity",
            {
                "run_time": run_time,
                "iterations": iterations_v,
                "population": population_v,
                "elite_frac": elite_frac_v,
                "eval_episodes": eval_episodes_v,
                "holdout_episodes": int(holdout_episodes),
            },
            title="Weight Sensitivity Report",
            note_lines=[
                "This supplementary plot isolates the added no-risk cost-emphasis case w=(resilience, cost, risk)=(0.33, 0.67, 0.00).",
                "Bubble sizes follow the same global LR scale used by the main trade-off figure.",
            ],
        )
    if resilience_records:
        _build_report(
            resilience_records,
            out_dir / "resilience_sensitivity",
            {
                "run_time": run_time,
                "iterations": iterations_v,
                "population": population_v,
                "elite_frac": elite_frac_v,
                "eval_episodes": eval_episodes_v,
                "holdout_episodes": int(holdout_episodes),
            },
            title="Resilience Sensitivity Report",
            note_lines=[
                "This supplementary plot compares the main no-resilience case w=(resilience, cost, risk)=(0.00, 0.50, 0.50) with the risk-compensated case w=(0.00, 0.33, 0.67).",
                "Bubble sizes follow the same global LR scale used by the main trade-off figure.",
            ],
        )
    _render_weight_preference_analysis(
        script_dir=script_dir,
        out_dir=out_dir / "weight_preference_cases",
        holdout_episodes=int(holdout_episodes),
        holdout_seed=int(base_seed_v + 900_000),
        run_time=run_time,
    )
    logger.info("Ablation completed | results=%s", str(out_dir))
    return out_dir


def redraw_only_from_results(*, out_root: Path, results_json: Path | None = None) -> Path:
    out_dir = out_root
    script_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    src = results_json if results_json is not None else (out_dir / "ablation_all_results.json")
    if not src.exists():
        raise FileNotFoundError(f"Results JSON not found: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"No records found in results JSON: {src}")
    for rec in records:
        if isinstance(rec, dict):
            _ensure_reported_return(rec)
    payload["records"] = records
    main_records = _main_ablation_records(records)
    weight_records = _weight_sensitivity_records(records)
    resilience_records = _resilience_sensitivity_records(records)
    bubble_scale = _global_lr_scale(records)

    settings = payload.get("settings", {})
    run_time = str(payload.get("run_time", "plot_only"))
    _plot_metric_panels(main_records, out_dir)
    _plot_convergence(main_records, out_dir / "ablation_convergence.png")
    _plot_tradeoff(main_records, out_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
    _copy_key_ablation_figures(out_dir)
    if weight_records:
        weight_dir = out_dir / "weight_sensitivity"
        _plot_metric_panels(weight_records, weight_dir)
        _plot_convergence(weight_records, weight_dir / "ablation_convergence.png")
        _plot_tradeoff(weight_records, weight_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
        _copy_key_ablation_figures(weight_dir, dst_subdir="weight_sensitivity")
    if resilience_records:
        resilience_dir = out_dir / "resilience_sensitivity"
        _plot_metric_panels(resilience_records, resilience_dir)
        _plot_convergence(resilience_records, resilience_dir / "ablation_convergence.png")
        _plot_tradeoff(resilience_records, resilience_dir / "ablation_tradeoff.png", bubble_scale=bubble_scale)
        _copy_key_ablation_figures(resilience_dir, dst_subdir="resilience_sensitivity")
    _build_report(
        main_records,
        out_dir,
        {
            "run_time": run_time,
            "iterations": int(settings.get("iterations", 0)),
            "population": int(settings.get("population", 0)),
            "elite_frac": float(settings.get("elite_frac", 0.0)),
            "eval_episodes": int(settings.get("eval_episodes", 0)),
            "holdout_episodes": int(settings.get("holdout_episodes", 0)),
        },
        note_lines=[
            "The supplemental no-resilience risk-compensated case uses w=(resilience, cost, risk)=(0.00, 0.33, 0.67) and is plotted separately under `resilience_sensitivity/`.",
            "All trade-off figures use one shared global bubble-size scale based on LR across the full ablation set.",
        ],
    )
    if weight_records:
        _build_report(
            weight_records,
            out_dir / "weight_sensitivity",
            {
                "run_time": run_time,
                "iterations": int(settings.get("iterations", 0)),
                "population": int(settings.get("population", 0)),
                "elite_frac": float(settings.get("elite_frac", 0.0)),
                "eval_episodes": int(settings.get("eval_episodes", 0)),
                "holdout_episodes": int(settings.get("holdout_episodes", 0)),
            },
            title="Weight Sensitivity Report",
            note_lines=[
                "This supplementary plot isolates the added no-risk cost-emphasis case w=(resilience, cost, risk)=(0.33, 0.67, 0.00).",
                "Bubble sizes follow the same global LR scale used by the main trade-off figure.",
            ],
        )
    if resilience_records:
        _build_report(
            resilience_records,
            out_dir / "resilience_sensitivity",
            {
                "run_time": run_time,
                "iterations": int(settings.get("iterations", 0)),
                "population": int(settings.get("population", 0)),
                "elite_frac": float(settings.get("elite_frac", 0.0)),
                "eval_episodes": int(settings.get("eval_episodes", 0)),
                "holdout_episodes": int(settings.get("holdout_episodes", 0)),
            },
            title="Resilience Sensitivity Report",
            note_lines=[
                "This supplementary plot compares the main no-resilience case w=(resilience, cost, risk)=(0.00, 0.50, 0.50) with the risk-compensated case w=(0.00, 0.33, 0.67).",
                "Bubble sizes follow the same global LR scale used by the main trade-off figure.",
            ],
        )
    _render_weight_preference_analysis(
        script_dir=script_dir,
        out_dir=out_dir / "weight_preference_cases",
        holdout_episodes=int(settings.get("holdout_episodes", 100)),
        holdout_seed=int(settings.get("base_seed", 20260301)) + 900_000,
        run_time=run_time,
    )
    (out_dir / "ablation_all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments for lifecycle CEM policy search.")
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Output directory for ablation results. Default: <script_dir>/lcc_rerun/ablation_results.",
    )
    parser.add_argument("--plot-only", action="store_true", help="Redraw figures/report from existing results JSON without training.")
    parser.add_argument(
        "--results-json",
        type=str,
        default="",
        help="Path to existing ablation_all_results.json (used with --plot-only).",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="",
        help="Baseline case config. Default: lcc_rerun/configs/case1_baseline.json.",
    )
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--population", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--holdout-episodes", type=int, default=100)
    parser.add_argument("--elite-frac", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_root = (
        (script_dir / "lcc_rerun" / "ablation_results").resolve()
        if not str(args.out_root).strip()
        else Path(args.out_root).resolve()
    )
    if bool(args.plot_only):
        results_json = Path(args.results_json).resolve() if str(args.results_json).strip() else None
        out_dir = redraw_only_from_results(out_root=out_root, results_json=results_json)
        print(str(out_dir))
        return

    run_dir = run_ablation(
        out_root=out_root,
        iterations=None if int(args.iterations) <= 0 else int(args.iterations),
        population=None if int(args.population) <= 0 else int(args.population),
        eval_episodes=None if int(args.eval_episodes) <= 0 else int(args.eval_episodes),
        holdout_episodes=args.holdout_episodes,
        elite_frac=None if float(args.elite_frac) < 0.0 else float(args.elite_frac),
        base_seed=None if int(args.seed) < 0 else int(args.seed),
        base_config_path=Path(args.base_config).resolve() if str(args.base_config).strip() else None,
    )
    print(str(run_dir))


if __name__ == "__main__":
    main()
