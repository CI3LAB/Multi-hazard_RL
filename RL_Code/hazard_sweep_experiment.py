from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import ablation_experiment as ae
import train_lifecycle_rl as tl


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "lcc_rerun" / "hazard_sweep_tests"
BASE_CONFIG = ROOT / "lcc_rerun" / "configs" / "case1_baseline.json"

VARIANT_ORDER = [
    "baseline_full",
    "no_maintenance",
    "no_resilience_risk_compensated",
]
ORIGINAL_VARIANT_INDEX = {
    "baseline_full": 0,
    "no_risk_term": 1,
    "no_resilience_term": 2,
    "no_resilience_risk_compensated": 3,
    "no_cost_term": 4,
    "no_pre_reinforcement": 5,
    "no_maintenance": 6,
    "no_repair": 7,
}


@dataclass(frozen=True)
class SweepCandidate:
    name: str
    desc: str
    baseline_fire: float
    fire_dominant_fire: float
    earthquake_dominant_fire: float
    baseline_eq: float
    fire_dominant_eq: float
    earthquake_dominant_eq: float
    maint_costs: tuple[float, float, float]
    det_alpha_t_levels: tuple[float, float, float]
    repair_recovery_rate_per_year: float = 0.30


def _initial_maintenance() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # det_alpha_T_levels are deterioration-rate multipliers: lower means stronger maintenance.
    # This maps effect levels [0.25, 0.50, 0.75] to rates [0.75, 0.50, 0.25].
    return (0.0, 0.8, 2.0), (0.75, 0.50, 0.25)


def _stronger_maintenance() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # This maps effect levels [0.20, 0.45, 0.80] to rates [0.80, 0.45, 0.20].
    return (0.0, 0.8, 2.0), (0.80, 0.45, 0.20)


def _candidates() -> list[SweepCandidate]:
    maint_costs, maint_det = _initial_maintenance()
    stronger_costs, stronger_det = _stronger_maintenance()
    return [
        SweepCandidate(
            name="s1_fire_5_2p5_10_eq_35_70_17p5_maint_initial",
            desc="Fire intervals 5/2.5/10 years; original earthquake rates.",
            baseline_fire=1.0 / 5.0,
            fire_dominant_fire=1.0 / 2.5,
            earthquake_dominant_fire=1.0 / 10.0,
            baseline_eq=1.0 / 35.0,
            fire_dominant_eq=1.0 / 70.0,
            earthquake_dominant_eq=2.0 / 35.0,
            maint_costs=maint_costs,
            det_alpha_t_levels=maint_det,
        ),
        SweepCandidate(
            name="s2_fire_5_2p5_10_eq_40_80_20_maint_initial",
            desc="Scenario 1 plus earthquake rates 1/40, 1/80, 2/40.",
            baseline_fire=1.0 / 5.0,
            fire_dominant_fire=1.0 / 2.5,
            earthquake_dominant_fire=1.0 / 10.0,
            baseline_eq=1.0 / 40.0,
            fire_dominant_eq=1.0 / 80.0,
            earthquake_dominant_eq=2.0 / 40.0,
            maint_costs=maint_costs,
            det_alpha_t_levels=maint_det,
        ),
        SweepCandidate(
            name="s3_fire_7p5_3p75_15_eq_40_80_20_maint_initial",
            desc="Lower fire rates: 7.5/3.75/15 years; earthquake rates from scenario 2.",
            baseline_fire=1.0 / 7.5,
            fire_dominant_fire=1.0 / 3.75,
            earthquake_dominant_fire=1.0 / 15.0,
            baseline_eq=1.0 / 40.0,
            fire_dominant_eq=1.0 / 80.0,
            earthquake_dominant_eq=2.0 / 40.0,
            maint_costs=maint_costs,
            det_alpha_t_levels=maint_det,
        ),
        SweepCandidate(
            name="s4_fire_7p5_3p75_15_eq_40_80_20_maint_stronger",
            desc="Scenario 3 plus stronger maintenance effects [0.2, 0.45, 0.8].",
            baseline_fire=1.0 / 7.5,
            fire_dominant_fire=1.0 / 3.75,
            earthquake_dominant_fire=1.0 / 15.0,
            baseline_eq=1.0 / 40.0,
            fire_dominant_eq=1.0 / 80.0,
            earthquake_dominant_eq=2.0 / 40.0,
            maint_costs=stronger_costs,
            det_alpha_t_levels=stronger_det,
        ),
        SweepCandidate(
            name="s5_fire_7p5_3p75_15_eq_50_100_25_maint_initial",
            desc="Fire intervals 7.5/3.75/15 years; earthquake rates 1/50, 1/100, 2/50; initial maintenance.",
            baseline_fire=1.0 / 7.5,
            fire_dominant_fire=1.0 / 3.75,
            earthquake_dominant_fire=1.0 / 15.0,
            baseline_eq=1.0 / 50.0,
            fire_dominant_eq=1.0 / 100.0,
            earthquake_dominant_eq=2.0 / 50.0,
            maint_costs=maint_costs,
            det_alpha_t_levels=maint_det,
        ),
        SweepCandidate(
            name="s6_s2_hazards_repair_rate_025_maint_initial",
            desc="S2 hazard rates with repair recovery rate 0.25/year and initial maintenance.",
            baseline_fire=1.0 / 5.0,
            fire_dominant_fire=1.0 / 2.5,
            earthquake_dominant_fire=1.0 / 10.0,
            baseline_eq=1.0 / 40.0,
            fire_dominant_eq=1.0 / 80.0,
            earthquake_dominant_eq=2.0 / 40.0,
            maint_costs=maint_costs,
            det_alpha_t_levels=maint_det,
            repair_recovery_rate_per_year=0.25,
        ),
        SweepCandidate(
            name="s7_s2_hazards_repair_rate_025_maint_adjusted",
            desc="S2 hazard rates with repair recovery rate 0.25/year and adjusted maintenance cost/effects.",
            baseline_fire=1.0 / 5.0,
            fire_dominant_fire=1.0 / 2.5,
            earthquake_dominant_fire=1.0 / 10.0,
            baseline_eq=1.0 / 40.0,
            fire_dominant_eq=1.0 / 80.0,
            earthquake_dominant_eq=2.0 / 40.0,
            maint_costs=(0.0, 0.25, 0.6),
            det_alpha_t_levels=(0.8, 0.4, 0.25),
            repair_recovery_rate_per_year=0.25,
        ),
    ]


def _make_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hazard_sweep")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(out_dir / "hazard_sweep.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _common_normalized_return(metrics: dict[str, Any], cfg: tl.LifecycleConfig) -> float:
    lr = float(metrics["lr"]) / max(float(cfg.lr_ref), 1.0e-12)
    risk = float(metrics["risk"]) / max(float(cfg.risk_ref), 1.0e-12)
    cost = float(metrics.get("lcc", metrics.get("npv", 0.0))) / max(float(cfg.cost_ref), 1.0e-12)
    return float(-(float(cfg.w_lr) * lr + float(cfg.w_risk) * risk + float(cfg.w_cost) * cost))


def _add_common_scores(records: list[dict[str, Any]], cfg: tl.LifecycleConfig) -> None:
    for record in records:
        for key in ("best_eval_train", "best_eval_holdout"):
            metrics = record.get(key)
            if isinstance(metrics, dict):
                metrics["common_normalized_return"] = _common_normalized_return(metrics, cfg)


def _candidate_cfg(base_cfg: tl.LifecycleConfig, candidate: SweepCandidate) -> tl.LifecycleConfig:
    return replace(
        base_cfg,
        lambda_fire_per_year=float(candidate.baseline_fire),
        lambda_eq_per_year=float(candidate.baseline_eq),
        maint_costs=tuple(map(float, candidate.maint_costs)),
        det_alpha_T_levels=tuple(map(float, candidate.det_alpha_t_levels)),
        repair_recovery_rate_per_year=float(candidate.repair_recovery_rate_per_year),
    )


def _cem_cfg(source: tl.CEMConfig, *, mode: str, seed: int) -> tl.CEMConfig:
    if mode == "light":
        return tl.CEMConfig(
            iterations=40,
            population=40,
            elite_frac=float(source.elite_frac),
            eval_episodes=15,
            seed=int(seed),
            plot_first_n=0,
            plot_interval=9999,
        )
    if mode == "full":
        return tl.CEMConfig(
            iterations=int(source.iterations),
            population=int(source.population),
            elite_frac=float(source.elite_frac),
            eval_episodes=int(source.eval_episodes),
            seed=int(seed),
            plot_first_n=0,
            plot_interval=9999,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _passes(records: list[dict[str, Any]], *, comparable_tol: float) -> tuple[bool, dict[str, float]]:
    by_name = {record["name"]: record for record in records}
    scores = {
        name: float(record["best_eval_holdout"]["common_normalized_return"])
        for name, record in by_name.items()
    }
    baseline = scores["baseline_full"]
    no_maintenance = scores["no_maintenance"]
    risk_comp = scores["no_resilience_risk_compensated"]
    return (
        baseline >= no_maintenance and baseline + float(comparable_tol) >= risk_comp,
        {
            "baseline_full": baseline,
            "no_maintenance": no_maintenance,
            "no_resilience_risk_compensated": risk_comp,
            "baseline_minus_no_maintenance": baseline - no_maintenance,
            "baseline_minus_risk_comp": baseline - risk_comp,
        },
    )


def _write_summary(
    *,
    out_dir: Path,
    candidate: SweepCandidate,
    mode: str,
    cfg: tl.LifecycleConfig,
    records: list[dict[str, Any]],
    status: str,
    comparable_tol: float,
) -> None:
    _add_common_scores(records, cfg)
    passed, scores = _passes(records, comparable_tol=comparable_tol) if len(records) == len(VARIANT_ORDER) else (False, {})
    payload = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": candidate.__dict__,
        "mode": mode,
        "status": status,
        "passed": bool(passed),
        "pass_rule": {
            "baseline_full": ">= no_maintenance",
            "risk_compensated": f"baseline_full + {comparable_tol} >= no_resilience_risk_compensated",
        },
        "scores": scores,
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mode}_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {candidate.name} ({mode})",
        "",
        f"- status: `{status}`",
        f"- passed: `{passed}`",
        f"- desc: {candidate.desc}",
        f"- baseline fire/eq: `{candidate.baseline_fire:.6f}` / `{candidate.baseline_eq:.6f}`",
        f"- fire-dominant fire/eq: `{candidate.fire_dominant_fire:.6f}` / `{candidate.fire_dominant_eq:.6f}`",
        f"- earthquake-dominant fire/eq: `{candidate.earthquake_dominant_fire:.6f}` / `{candidate.earthquake_dominant_eq:.6f}`",
        f"- maint_costs: `{list(candidate.maint_costs)}`",
        f"- det_alpha_T_levels: `{list(candidate.det_alpha_t_levels)}`",
        f"- repair_recovery_rate_per_year: `{candidate.repair_recovery_rate_per_year:.3f}`",
        "",
        "| Variant | Common Norm Return | Own Reported Return | LR | Risk | Cost | MinF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record["best_eval_holdout"]
        lines.append(
            f"| {record['name']} | {float(metrics['common_normalized_return']):.6f} | "
            f"{float(metrics['reported_return']):.6f} | {float(metrics['lr']):.4f} | "
            f"{float(metrics['risk']):.4f} | {float(metrics['lcc']):.4f} | {float(metrics['min_f']):.4f} |"
        )
    if scores:
        lines.extend(
            [
                "",
                f"- baseline - no_maintenance: `{scores['baseline_minus_no_maintenance']:.6f}`",
                f"- baseline - risk_comp: `{scores['baseline_minus_risk_comp']:.6f}`",
            ]
        )
    (out_dir / f"{mode}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_candidate_mode(
    *,
    candidate: SweepCandidate,
    base_cfg: tl.LifecycleConfig,
    source_cem_cfg: tl.CEMConfig,
    base_seed: int,
    mode: str,
    out_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    cfg = _candidate_cfg(base_cfg, candidate)
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    holdout_episodes = 200 if mode == "light" else 400
    seed_offset = 0 if mode == "light" else 100_000
    paired_holdout_seed = int(base_seed + seed_offset + 700_000)
    records: list[dict[str, Any]] = []
    all_variants = {variant["name"]: variant for variant in ae._variants(cfg)}
    cem_cfg = _cem_cfg(source_cem_cfg, mode=mode, seed=base_seed + seed_offset)

    _write_summary(
        out_dir=out_dir,
        candidate=candidate,
        mode=mode,
        cfg=cfg,
        records=records,
        status="running",
        comparable_tol=0.05 if mode == "light" else 0.02,
    )

    for name in VARIANT_ORDER:
        seed = int(base_seed + seed_offset + ORIGINAL_VARIANT_INDEX[name] * 1000)
        logger.info("Run start | candidate=%s mode=%s variant=%s seed=%d", candidate.name, mode, name, seed)
        record = ae._run_one_variant(
            variant=all_variants[name],
            cem_cfg=cem_cfg,
            seed=seed,
            holdout_episodes=holdout_episodes,
            out_dir=mode_dir,
            logger=logger,
        )
        best_params = record["best_params"]
        paired_holdout = tl._evaluate_params(  # type: ignore[attr-defined]
            all_variants[name]["cfg"],
            tl.PolicyParams(
                pre_index=int(best_params["pre_index"]),
                maint_t1=float(best_params["maint_t1"]),
                maint_t2=float(best_params["maint_t2"]),
                repair_t1=float(best_params["repair_t1"]),
                repair_t2=float(best_params["repair_t2"]),
            ),
            base_seed=paired_holdout_seed,
            episodes=holdout_episodes,
        )
        record["best_eval_holdout"] = dict(paired_holdout)
        ae._ensure_reported_return(record)
        records.append(record)
        _write_summary(
            out_dir=out_dir,
            candidate=candidate,
            mode=mode,
            cfg=cfg,
            records=records,
            status="running",
            comparable_tol=0.05 if mode == "light" else 0.02,
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run light-to-full hazard/maintenance candidate sweeps."
    )
    parser.add_argument(
        "--candidate",
        help="Run only this candidate name, instead of the full candidate sequence.",
    )
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_hazard_sweep")
    logger = _make_logger(run_dir)
    base_cfg, source_cem_cfg, source_config_path = ae._base_cfg(BASE_CONFIG)
    logger.info("Hazard sweep start | run_dir=%s base_config=%s", run_dir, source_config_path)
    logger.info("Light budget | iterations=40 population=40 eval_episodes=15 holdout=200")
    logger.info(
        "Full budget | iterations=%d population=%d eval_episodes=%d holdout=400",
        int(source_cem_cfg.iterations),
        int(source_cem_cfg.population),
        int(source_cem_cfg.eval_episodes),
    )

    candidates = _candidates()
    if args.candidate:
        candidates = [candidate for candidate in candidates if candidate.name == args.candidate]
        if not candidates:
            available = ", ".join(candidate.name for candidate in _candidates())
            parser.error(f"Unknown candidate '{args.candidate}'. Available: {available}")
    for idx, candidate in enumerate(candidates):
        candidate_dir = run_dir / candidate.name
        logger.info("Candidate start | %d/%d %s | %s", idx + 1, len(candidates), candidate.name, candidate.desc)
        light_records = _run_candidate_mode(
            candidate=candidate,
            base_cfg=base_cfg,
            source_cem_cfg=source_cem_cfg,
            base_seed=int(source_cem_cfg.seed) + idx * 20_000,
            mode="light",
            out_dir=candidate_dir,
            logger=logger,
        )
        light_passed, light_scores = _passes(light_records, comparable_tol=0.05)
        light_cfg = _candidate_cfg(base_cfg, candidate)
        _write_summary(
            out_dir=candidate_dir,
            candidate=candidate,
            mode="light",
            cfg=light_cfg,
            records=light_records,
            status="passed" if light_passed else "failed",
            comparable_tol=0.05,
        )
        logger.info("Light done | candidate=%s passed=%s scores=%s", candidate.name, light_passed, light_scores)
        if not light_passed:
            continue

        full_records = _run_candidate_mode(
            candidate=candidate,
            base_cfg=base_cfg,
            source_cem_cfg=source_cem_cfg,
            base_seed=int(source_cem_cfg.seed) + idx * 20_000,
            mode="full",
            out_dir=candidate_dir,
            logger=logger,
        )
        full_passed, full_scores = _passes(full_records, comparable_tol=0.02)
        _write_summary(
            out_dir=candidate_dir,
            candidate=candidate,
            mode="full",
            cfg=light_cfg,
            records=full_records,
            status="passed" if full_passed else "failed",
            comparable_tol=0.02,
        )
        logger.info("Full done | candidate=%s passed=%s scores=%s", candidate.name, full_passed, full_scores)
        if full_passed:
            success = {
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidate": candidate.__dict__,
                "scores": full_scores,
                "message": "Full-budget result satisfied the stopping criterion; later candidates were not run.",
            }
            (run_dir / "SUCCESS.json").write_text(
                json.dumps(success, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Hazard sweep success | stopping after candidate=%s", candidate.name)
            return 0

    (run_dir / "NO_SUCCESS.txt").write_text(
        "No candidate satisfied the full-budget stopping criterion.\n",
        encoding="utf-8",
    )
    logger.info("Hazard sweep finished without full-budget success.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
