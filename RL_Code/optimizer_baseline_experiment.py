from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import evaluate_uncertainty as unc
import train_lifecycle_rl as tl


ROOT = Path(__file__).resolve().parent
LCC_ROOT = ROOT / "lcc_rerun"
DEFAULT_CONFIG = LCC_ROOT / "configs" / "case1_baseline.json"
DEFAULT_OUT_DIR = LCC_ROOT / "optimizer_baselines"
_TIMES_NEW_ROMAN = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if _TIMES_NEW_ROMAN.exists():
    font_manager.fontManager.addfont(str(_TIMES_NEW_ROMAN))


def _plot_rc_params() -> dict[str, object]:
    return {
        "font.family": "Times New Roman",
        "font.size": 10.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
    }


@dataclass(frozen=True)
class OptimizerConfig:
    iterations: int = 200
    population: int = 80
    eval_episodes: int = 30
    holdout_episodes: int = 400
    seed: int = 20270301
    holdout_seed: int = 21060301


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("optimizer_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "optimizer_baselines.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _load_cfg(config_path: Path) -> tuple[tl.LifecycleConfig, tl.CEMConfig]:
    cfg, cem_cfg, _paths = tl._load_run_config(config_path, ROOT)  # type: ignore[attr-defined]
    return cfg, cem_cfg


def _encode(params: tl.PolicyParams) -> np.ndarray:
    return np.array(
        [
            float(params.pre_index),
            float(params.maint_t1),
            float(params.maint_t2),
            float(params.repair_t1),
            float(params.repair_t2),
        ],
        dtype=float,
    )


def _decode(x: np.ndarray) -> tl.PolicyParams:
    v = np.asarray(x, dtype=float).copy()
    pre_index = int(np.clip(np.rint(v[0]), 0, 2))
    vals = np.clip(v[1:5], 0.05, 0.95)
    return tl.PolicyParams(
        pre_index=pre_index,
        maint_t1=float(vals[0]),
        maint_t2=float(vals[1]),
        repair_t1=float(vals[2]),
        repair_t2=float(vals[3]),
    )


def _random_population(rng: np.random.Generator, n: int) -> np.ndarray:
    pop = np.empty((n, 5), dtype=float)
    pop[:, 0] = rng.integers(0, 3, size=n)
    pop[:, 1:] = rng.uniform(0.05, 0.95, size=(n, 4))
    return pop


def _clip_population(pop: np.ndarray) -> np.ndarray:
    out = np.asarray(pop, dtype=float).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, 2.0)
    out[:, 1:] = np.clip(out[:, 1:], 0.05, 0.95)
    return out


def _evaluate_vector(
    cfg: tl.LifecycleConfig,
    x: np.ndarray,
    *,
    base_seed: int,
    episodes: int,
) -> dict[str, float]:
    return dict(
        tl._evaluate_params(  # type: ignore[attr-defined]
            cfg,
            _decode(x),
            base_seed=int(base_seed),
            episodes=int(episodes),
        )
    )


def _evaluate_population(
    cfg: tl.LifecycleConfig,
    population: np.ndarray,
    *,
    seed: int,
    iteration: int,
    episodes: int,
) -> list[dict[str, float]]:
    base_seed_it = int(seed + iteration * 7919)
    params = [_decode(population[j]) for j in range(int(population.shape[0]))]
    return tl.evaluate_policy_population(
        cfg,
        params,
        base_seed_it=base_seed_it,
        episodes=int(episodes),
    )


def _score_order(scores: list[dict[str, float]]) -> list[int]:
    return np.argsort([-float(s["return"]) for s in scores]).tolist()


def _history_row(
    *,
    iteration: int,
    params: tl.PolicyParams,
    score: dict[str, float],
    best_so_far: float,
) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "best_of_iter": {"params": asdict(params), **dict(score)},
        "best_so_far_return": float(best_so_far),
    }


def run_ga(
    cfg: tl.LifecycleConfig,
    opt: OptimizerConfig,
    *,
    logger: logging.Logger,
) -> dict[str, Any]:
    rng = np.random.default_rng(opt.seed)
    pop = _random_population(rng, opt.population)
    pop[0] = _encode(tl._baseline_policy_params())  # type: ignore[attr-defined]

    elite_k = max(1, int(round(0.10 * opt.population)))
    history: list[dict[str, Any]] = []
    best_x: np.ndarray | None = None
    best_score = -float("inf")
    best_eval: dict[str, float] | None = None

    logger.info(
        "GA start | iterations=%d population=%d eval_episodes=%d seed=%d",
        opt.iterations,
        opt.population,
        opt.eval_episodes,
        opt.seed,
    )

    for it in range(opt.iterations):
        scores = _evaluate_population(
            cfg,
            pop,
            seed=opt.seed,
            iteration=it,
            episodes=opt.eval_episodes,
        )
        order = _score_order(scores)
        it_best = order[0]
        it_best_score = float(scores[it_best]["return"])
        if it_best_score > best_score:
            best_score = it_best_score
            best_x = pop[it_best].copy()
            best_eval = dict(scores[it_best])
            logger.info(
                "GA new best | iter=%d return=%.4f params=%s",
                it,
                best_score,
                json.dumps(asdict(_decode(best_x)), ensure_ascii=False),
            )
        else:
            logger.info("GA iter=%d best=%.4f best_so_far=%.4f", it, it_best_score, best_score)

        assert best_x is not None
        history.append(
            _history_row(
                iteration=it,
                params=_decode(pop[it_best]),
                score=scores[it_best],
                best_so_far=best_score,
            )
        )

        elites = pop[order[:elite_k]].copy()

        def tournament() -> np.ndarray:
            choices = rng.choice(opt.population, size=3, replace=False)
            winner = max(choices, key=lambda idx: float(scores[int(idx)]["return"]))
            return pop[int(winner)]

        new_pop = [e.copy() for e in elites]
        new_pop.append(best_x.copy())
        mutation_scale = 0.10 * (1.0 - it / max(1, opt.iterations - 1)) + 0.02
        while len(new_pop) < opt.population:
            p1 = tournament()
            p2 = tournament()
            child = p1.copy()
            if rng.random() < 0.8:
                alpha = rng.uniform(-0.25, 1.25, size=5)
                child = alpha * p1 + (1.0 - alpha) * p2
            if rng.random() < 0.20:
                child[0] = rng.integers(0, 3)
            for k in range(1, 5):
                if rng.random() < 0.20:
                    child[k] += rng.normal(0.0, mutation_scale)
            new_pop.append(child)
        pop = _clip_population(np.vstack(new_pop[: opt.population]))

    if best_x is None or best_eval is None:
        raise RuntimeError("GA did not produce a best candidate.")
    return {
        "method": "GA",
        "config": asdict(opt),
        "best_params": asdict(_decode(best_x)),
        "best_eval_train": best_eval,
        "training_best_return": float(best_score),
        "history": history,
    }


def run_pso(
    cfg: tl.LifecycleConfig,
    opt: OptimizerConfig,
    *,
    logger: logging.Logger,
) -> dict[str, Any]:
    rng = np.random.default_rng(opt.seed)
    pos = _random_population(rng, opt.population)
    pos[0] = _encode(tl._baseline_policy_params())  # type: ignore[attr-defined]
    vel = rng.normal(0.0, 0.05, size=pos.shape)
    pbest = pos.copy()
    pbest_scores = np.full(opt.population, -float("inf"), dtype=float)
    pbest_evals: list[dict[str, float] | None] = [None] * opt.population
    gbest: np.ndarray | None = None
    gbest_score = -float("inf")
    gbest_eval: dict[str, float] | None = None
    history: list[dict[str, Any]] = []

    logger.info(
        "PSO start | iterations=%d swarm=%d eval_episodes=%d seed=%d",
        opt.iterations,
        opt.population,
        opt.eval_episodes,
        opt.seed,
    )

    for it in range(opt.iterations):
        scores = _evaluate_population(
            cfg,
            pos,
            seed=opt.seed,
            iteration=it,
            episodes=opt.eval_episodes,
        )
        for j, score in enumerate(scores):
            ret = float(score["return"])
            if ret > pbest_scores[j]:
                pbest_scores[j] = ret
                pbest[j] = pos[j].copy()
                pbest_evals[j] = dict(score)
            if ret > gbest_score:
                gbest_score = ret
                gbest = pos[j].copy()
                gbest_eval = dict(score)
                logger.info(
                    "PSO new best | iter=%d return=%.4f params=%s",
                    it,
                    gbest_score,
                    json.dumps(asdict(_decode(gbest)), ensure_ascii=False),
                )

        order = _score_order(scores)
        it_best = order[0]
        assert gbest is not None
        history.append(
            _history_row(
                iteration=it,
                params=_decode(pos[it_best]),
                score=scores[it_best],
                best_so_far=gbest_score,
            )
        )
        logger.info(
            "PSO iter=%d best=%.4f best_so_far=%.4f",
            it,
            float(scores[it_best]["return"]),
            gbest_score,
        )

        inertia = 0.9 - 0.5 * (it / max(1, opt.iterations - 1))
        c1 = 1.5
        c2 = 1.5
        r1 = rng.random(size=pos.shape)
        r2 = rng.random(size=pos.shape)
        vel = inertia * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel[:, 0] = np.clip(vel[:, 0], -1.0, 1.0)
        vel[:, 1:] = np.clip(vel[:, 1:], -0.20, 0.20)
        pos = _clip_population(pos + vel)

    if gbest is None or gbest_eval is None:
        raise RuntimeError("PSO did not produce a best candidate.")
    return {
        "method": "PSO",
        "config": asdict(opt),
        "best_params": asdict(_decode(gbest)),
        "best_eval_train": gbest_eval,
        "training_best_return": float(gbest_score),
        "history": history,
    }


def _load_cem_record(
    cfg: tl.LifecycleConfig,
    opt: OptimizerConfig,
    *,
    summary_path: Path | None = None,
) -> dict[str, Any] | None:
    path = Path(summary_path) if summary_path is not None else (
        LCC_ROOT / "ablation_results" / "baseline_full" / "ablation_summary.json"
    )
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    params = tl.PolicyParams(**dict(raw["best_params"]))
    return _attach_holdout(
        cfg,
        opt,
        {
            "method": "CEM",
            "best_params": asdict(params),
            "source": str(path),
            "history": raw.get("history", []),
        },
    )


def _fixed_record(cfg: tl.LifecycleConfig, opt: OptimizerConfig) -> dict[str, Any]:
    params = tl._baseline_policy_params()  # type: ignore[attr-defined]
    return _attach_holdout(
        cfg,
        opt,
        {
            "method": "Fixed heuristic",
            "best_params": asdict(params),
            "history": [],
        },
    )


def _attach_holdout(cfg: tl.LifecycleConfig, opt: OptimizerConfig, result: dict[str, Any]) -> dict[str, Any]:
    params = tl.PolicyParams(**dict(result["best_params"]))
    holdout = unc.holdout_with_uncertainty(
        cfg,
        params,
        base_seed=int(opt.holdout_seed),
        episodes=int(opt.holdout_episodes),
    )
    out = dict(result)
    out["best_eval_holdout_common"] = holdout
    return out


def _unc_pair(holdout: dict[str, Any], key: str) -> tuple[float, float]:
    stats = dict(holdout.get("uncertainty", {})).get(key, {})
    if isinstance(stats, dict) and "mean" in stats:
        return float(stats["mean"]), float(stats.get("ci_halfwidth", 0.0))
    mapped = {"npv": "npv", "reported_return": "reported_return", "return": "return"}
    return float(holdout[mapped.get(key, key)]), 0.0


def _write_summary(records: list[dict[str, Any]], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    n_ep = 0
    for rec in records:
        h = dict(rec["best_eval_holdout_common"])
        ret_m, ret_hw = _unc_pair(h, "reported_return")
        obj_m, obj_hw = _unc_pair(h, "return")
        lr_m, lr_hw = _unc_pair(h, "lr")
        risk_m, risk_hw = _unc_pair(h, "risk")
        cost_m, cost_hw = _unc_pair(h, "npv")
        minf_m, minf_hw = _unc_pair(h, "min_f")
        n_ep = int(dict(h.get("uncertainty", {})).get("episodes", n_ep) or n_ep)
        rows.append(
            {
                "method": rec["method"],
                "objective_return": obj_m,
                "objective_return_ci": obj_hw,
                "reported_return": ret_m,
                "reported_return_ci": ret_hw,
                "lr": lr_m,
                "lr_ci": lr_hw,
                "risk": risk_m,
                "risk_ci": risk_hw,
                "lcc": float(h["lcc"]),
                "npv": cost_m,
                "npv_ci": cost_hw,
                "min_f": minf_m,
                "min_f_ci": minf_hw,
                "feasible_frac": float(h["feasible_frac"]),
                "params": json.dumps(rec["best_params"], ensure_ascii=False),
            }
        )
    csv_path = out_dir / "optimizer_baseline_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_label = n_ep if n_ep > 0 else int(rows[0].get("n", 400) if rows else 400)
    lines = [
        "Optimizer baseline holdout summary (common random seeds)",
        f"Uncertainty: mean ± 95% CI half-width (N={n_label} independent lifecycle episodes)",
        "Return is the normalized training objective used by CEM/GA/PSO.",
        "",
        "Method | Return | Resilience loss | Risk | Cost | Min F",
    ]
    for r in rows:
        lines.append(
            f"{r['method']} | {unc._fmt(r['objective_return'], r['objective_return_ci'], 4)} | "
            f"{unc._fmt(r['lr'], r['lr_ci'], 3)} | {unc._fmt(r['risk'], r['risk_ci'], 3)} | "
            f"{unc._fmt(r['npv'], r['npv_ci'], 2)} | {unc._fmt(r['min_f'], r['min_f_ci'], 3)}"
        )
    (out_dir / "optimizer_baseline_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot_convergence(records: list[dict[str, Any]], out_dir: Path) -> None:
    colors = {"GA": "#1f77b4", "PSO": "#d62728", "CEM": "#2ca02c"}
    with plt.rc_context(_plot_rc_params()):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for rec in records:
            if rec["method"] == "Fixed heuristic":
                continue
            hist = rec.get("history", [])
            if not hist:
                continue
            xs = [int(h["iteration"]) for h in hist]
            ys: list[float] = []
            best = -float("inf")
            for h in hist:
                if "best_so_far_return" in h:
                    best = max(best, float(h["best_so_far_return"]))
                else:
                    best = max(best, float(h["best_of_iter"]["return"]))
                ys.append(float(best))
            method = str(rec["method"])
            ax.plot(
                xs,
                ys,
                color=colors.get(method, "#333333"),
                linewidth=1.8,
                alpha=1.0,
                label=method,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best-so-far objective return")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "optimizer_convergence.png", bbox_inches="tight")
        plt.close(fig)


def _save_experiment(
    *,
    config_path: Path,
    out_dir: Path,
    opt: OptimizerConfig,
    records: list[dict[str, Any]],
) -> None:
    for rec in records:
        (out_dir / f"{str(rec['method']).lower().replace(' ', '_')}_result.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (out_dir / "optimizer_baseline_results.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "optimizer_config": {
                    "iterations": int(opt.iterations),
                    "population": int(opt.population),
                    "eval_episodes": int(opt.eval_episodes),
                    "holdout_episodes": int(opt.holdout_episodes),
                    "common_holdout_seed": int(opt.holdout_seed),
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary(records, out_dir)
    _plot_convergence(records, out_dir)


def run_experiment(
    *,
    config_path: Path,
    out_dir: Path,
    iterations: int,
    population: int,
    eval_episodes: int,
    holdout_episodes: int,
    cem_summary: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir)
    cfg, cem_cfg = _load_cfg(config_path)
    opt_ga = OptimizerConfig(
        iterations=iterations,
        population=population,
        eval_episodes=eval_episodes,
        holdout_episodes=holdout_episodes,
        seed=int(cem_cfg.seed + 1_000_000),
        holdout_seed=int(cem_cfg.seed + 900_000),
    )
    opt_pso = OptimizerConfig(
        iterations=iterations,
        population=population,
        eval_episodes=eval_episodes,
        holdout_episodes=holdout_episodes,
        seed=int(cem_cfg.seed + 1_010_000),
        holdout_seed=int(cem_cfg.seed + 900_000),
    )
    logger.info("Optimizer baseline setup | config=%s out_dir=%s", config_path, out_dir)

    fixed = _fixed_record(cfg, opt_ga)
    ga = _attach_holdout(cfg, opt_ga, run_ga(cfg, opt_ga, logger=logger))
    pso = _attach_holdout(cfg, opt_pso, run_pso(cfg, opt_pso, logger=logger))
    cem = _load_cem_record(cfg, opt_ga, summary_path=cem_summary)
    records = [fixed, ga, pso] + ([cem] if cem is not None else [])
    _save_experiment(config_path=config_path, out_dir=out_dir, opt=opt_ga, records=records)
    logger.info("Optimizer baseline done | out_dir=%s", out_dir)


def refresh_holdout_uncertainty(
    *,
    config_path: Path,
    out_dir: Path,
    holdout_episodes: int | None = None,
) -> None:
    result_path = out_dir / "optimizer_baseline_results.json"
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    cfg, cem_cfg = _load_cfg(config_path)
    opt_raw = dict(raw.get("optimizer_config", {}))
    opt = OptimizerConfig(
        iterations=int(opt_raw.get("iterations", 200)),
        population=int(opt_raw.get("population", 80)),
        eval_episodes=int(opt_raw.get("eval_episodes", 30)),
        holdout_episodes=int(holdout_episodes or opt_raw.get("holdout_episodes", 400)),
        seed=int(cem_cfg.seed + 1_000_000),
        holdout_seed=int(opt_raw.get("common_holdout_seed", cem_cfg.seed + 900_000)),
    )
    logger = _setup_logger(out_dir)
    logger.info(
        "Refreshing optimizer holdout uncertainty | episodes=%d seed=%d",
        opt.holdout_episodes,
        opt.holdout_seed,
    )
    records = [_attach_holdout(cfg, opt, dict(rec)) for rec in list(raw["records"])]
    _save_experiment(config_path=config_path, out_dir=out_dir, opt=opt, records=records)
    logger.info("Holdout uncertainty refresh done | out_dir=%s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA/PSO optimizer baselines for the main lifecycle case.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--holdout-episodes", type=int, default=400)
    parser.add_argument(
        "--cem-summary",
        type=str,
        default="",
        help="Optional CEM ablation_summary.json used in the optimizer comparison.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the convergence figure from saved optimizer results.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Recompute holdout mean ± 95% CI for saved GA/PSO/CEM policies without retraining.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if bool(args.plot_only):
        result_path = out_dir / "optimizer_baseline_results.json"
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        _plot_convergence(list(raw["records"]), out_dir)
        return
    if bool(args.evaluate_only):
        refresh_holdout_uncertainty(
            config_path=Path(args.config).resolve(),
            out_dir=out_dir,
            holdout_episodes=int(args.holdout_episodes),
        )
        return
    run_experiment(
        config_path=Path(args.config).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        iterations=int(args.iterations),
        population=int(args.population),
        eval_episodes=int(args.eval_episodes),
        holdout_episodes=int(args.holdout_episodes),
        cem_summary=Path(args.cem_summary).resolve() if str(args.cem_summary).strip() else None,
    )


if __name__ == "__main__":
    main()
