"""Sequential PPO on the lifecycle SMDP (case 1), same physics as CEM/GA/PSO.

The agent chooses a 3-way action at each real decision:
  pre-reinforcement (once at t=0), maintenance (every 10 years), and repair
  (when a hazard occurs). Holdout uses the optimizer common seeds.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import evaluate_uncertainty as unc
import train_lifecycle_rl as tl


ROOT = Path(__file__).resolve().parent
LCC_ROOT = ROOT / "lcc_rerun"
DEFAULT_CONFIG = LCC_ROOT / "configs" / "case1_baseline.json"
DEFAULT_OUT_DIR = LCC_ROOT / "deeprl_ppo"
OBS_DIM = 16
N_ACTIONS = 3
HID = 64


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("deeprl_ppo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "deeprl_ppo.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _worker_count() -> int:
    raw = str(os.environ.get("CEM_WORKERS", "")).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = int(os.cpu_count() or 1)
    return max(1, int(n))


def _reported_return(cfg: tl.LifecycleConfig, *, lr: float, risk: float, npv: float) -> float:
    return float(-(cfg.w_lr * lr + cfg.w_risk * risk + cfg.w_cost * npv))


def _current_objective(cfg: tl.LifecycleConfig, *, lr: float, risk: float,
                       lcc: float, npv: float, floor_violation: float) -> float:
    return float(tl._objective_return_value(
        cfg, lr=lr, risk=risk, lcc=lcc, npv=npv, floor_violation=floor_violation,
    ))


def _one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _obs_vector(
    cfg: tl.LifecycleConfig,
    *,
    t: float,
    f: float,
    pre_index: int,
    maint_level: int,
    n_procs: int,
    kind: str,
    intensity: float,
    min_f: float,
) -> np.ndarray:
    horizon = max(1.0, float(cfg.horizon_years))
    kind_oh = {
        "pre": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "maint": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "repair": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }[kind]
    pre_oh = _one_hot(int(pre_index), 3) if pre_index >= 0 else np.zeros(3, dtype=np.float32)
    return np.array(
        [
            float(t) / horizon,
            float(f),
            float(f) - float(cfg.f_crit),
            float(f) - float(cfg.f_1),
            float(f) - float(cfg.f_2),
            float(min_f),
            float(pre_oh[0]),
            float(pre_oh[1]),
            float(pre_oh[2]),
            float(maint_level) / 2.0,
            min(1.0, float(n_procs) / 4.0),
            float(kind_oh[0]),
            float(kind_oh[1]),
            float(kind_oh[2]),
            min(1.0, float(intensity)),
            max(0.0, 1.0 - float(t) / horizon),
        ],
        dtype=np.float32,
    )


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    logp: float
    value: float
    reward: float = 0.0


class ActorCritic:
    def __init__(self, rng: np.random.Generator, obs_dim: int = OBS_DIM, hid: int = HID):
        self.obs_dim = int(obs_dim)
        self.hid = int(hid)
        scale1 = math.sqrt(2.0 / (obs_dim + hid))
        scale2 = math.sqrt(2.0 / (hid + hid))
        scale_a = math.sqrt(2.0 / (hid + N_ACTIONS))
        scale_c = math.sqrt(2.0 / (hid + 1))
        self.W1 = rng.normal(0.0, scale1, size=(obs_dim, hid)).astype(np.float64)
        self.b1 = np.zeros(hid, dtype=np.float64)
        self.W2 = rng.normal(0.0, scale2, size=(hid, hid)).astype(np.float64)
        self.b2 = np.zeros(hid, dtype=np.float64)
        self.Wa = rng.normal(0.0, scale_a, size=(hid, N_ACTIONS)).astype(np.float64)
        self.ba = np.zeros(N_ACTIONS, dtype=np.float64)
        self.Wc = rng.normal(0.0, scale_c, size=(hid, 1)).astype(np.float64)
        self.bc = np.zeros(1, dtype=np.float64)

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wc, self.bc]

    def set_params(self, blobs: list[np.ndarray]) -> None:
        names = ["W1", "b1", "W2", "b2", "Wa", "ba", "Wc", "bc"]
        for name, arr in zip(names, blobs):
            setattr(self, name, np.array(arr, dtype=np.float64, copy=True))

    def snapshot(self) -> list[np.ndarray]:
        return [np.array(p, copy=True) for p in self.params()]

    def forward_batch(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        x = np.asarray(obs, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ self.Wa + self.ba
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        value = (h2 @ self.Wc + self.bc).ravel()
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "probs": probs, "value": value}
        return probs, value, cache

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
        probs, value, cache = self.forward_batch(np.asarray(obs, dtype=np.float64).reshape(1, -1))
        cache = {k: (v[0] if k != "probs" and getattr(v, "ndim", 1) == 2 and v.shape[0] == 1 else v[0] if k in {"x", "z1", "h1", "z2", "h2"} else v)
                 for k, v in cache.items()}
        cache["probs"] = probs[0]
        return probs[0], float(value[0]), cache

    def act(self, obs: np.ndarray, rng: np.random.Generator, *, greedy: bool) -> tuple[int, float, float]:
        probs, value, _cache = self.forward(obs)
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(N_ACTIONS, p=probs))
        logp = float(math.log(max(1.0e-8, float(probs[action]))))
        return action, logp, value


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 3.0e-4):
        self.params = params
        self.lr = float(lr)
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1.0e-8
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], max_norm: float = 0.5) -> None:
        flat = np.concatenate([g.ravel() for g in grads])
        nrm = float(np.linalg.norm(flat))
        scale = 1.0
        if nrm > max_norm and nrm > 0.0:
            scale = max_norm / nrm
        self.t += 1
        b1t = 1.0 - self.b1 ** self.t
        b2t = 1.0 - self.b2 ** self.t
        for i, g in enumerate(grads):
            g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            mhat = self.m[i] / b1t
            vhat = self.v[i] / b2t
            self.params[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def _softmax_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1.0e-8, 1.0)
    return float(-np.sum(p * np.log(p)))


def _backward_ppo_batch(
    net: ActorCritic,
    obs: np.ndarray,
    actions: np.ndarray,
    old_logp: np.ndarray,
    advantage: np.ndarray,
    ret: np.ndarray,
    old_value: np.ndarray,
    *,
    clip: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[list[np.ndarray], dict[str, float]]:
    probs, values, cache = net.forward_batch(obs)
    bsz = int(obs.shape[0])
    idx = np.arange(bsz)
    actions = np.asarray(actions, dtype=int)
    old_logp = np.asarray(old_logp, dtype=np.float64)
    advantage = np.asarray(advantage, dtype=np.float64)
    ret = np.asarray(ret, dtype=np.float64)
    old_value = np.asarray(old_value, dtype=np.float64)
    new_logp = np.log(np.clip(probs[idx, actions], 1.0e-8, 1.0))
    ratio = np.exp(new_logp - old_logp)
    unclipped = ratio * advantage
    clipped_ratio = np.clip(ratio, 1.0 - clip, 1.0 + clip)
    clipped = clipped_ratio * advantage
    use_unclipped = unclipped <= clipped
    in_clip = (ratio >= (1.0 - clip)) & (ratio <= (1.0 + clip))
    d_ratio = np.where(use_unclipped, -advantage, np.where(in_clip, -advantage, 0.0))
    d_logp = d_ratio * ratio
    dlogits = -probs * d_logp.reshape(-1, 1)
    dlogits[idx, actions] += d_logp
    logp_all = np.log(np.clip(probs, 1.0e-8, 1.0))
    sum_plogp = np.sum(probs * logp_all, axis=1, keepdims=True)
    dentropy = -probs * (logp_all - sum_plogp)
    dlogits = dlogits - ent_coef * dentropy
    v_clipped = old_value + np.clip(values - old_value, -clip, clip)
    vf1 = (values - ret) ** 2
    vf2 = (v_clipped - ret) ** 2
    use_v1 = vf1 <= vf2
    inside_v = np.abs(values - old_value) <= clip
    dvalue = np.where(use_v1, 2.0 * (values - ret), np.where(inside_v, 2.0 * (v_clipped - ret), 0.0))
    dvalue = dvalue * vf_coef
    inv_b = 1.0 / max(1, bsz)
    dlogits = dlogits * inv_b
    dvalue = dvalue * inv_b
    h2 = cache["h2"]
    dWa = h2.T @ dlogits
    dba = dlogits.sum(axis=0)
    dWc = h2.T @ dvalue.reshape(-1, 1)
    dbc = np.array([dvalue.sum()], dtype=np.float64)
    dh2 = dlogits @ net.Wa.T + dvalue.reshape(-1, 1) * net.Wc.T
    dz2 = dh2 * (cache["z2"] > 0.0)
    dW2 = cache["h1"].T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ net.W2.T
    dz1 = dh1 * (cache["z1"] > 0.0)
    dW1 = cache["x"].T @ dz1
    db1 = dz1.sum(axis=0)
    stats = {
        "pg": float(-np.mean(np.minimum(unclipped, clipped))),
        "vf": float(np.mean(np.minimum(vf1, vf2))),
        "ent": float(np.mean(-np.sum(probs * logp_all, axis=1))),
    }
    return [dW1, db1, dW2, db2, dWa, dba, dWc, dbc], stats


def simulate_with_policy(
    cfg: tl.LifecycleConfig,
    actor: ActorCritic,
    seed: int,
    *,
    greedy: bool,
    collect: bool,
) -> tuple[tl.EpisodeResult, list[Transition]]:
    rng = np.random.default_rng(seed)
    act_rng = np.random.default_rng(seed + 17)

    horizon_years = float(cfg.horizon_years)
    decision_dt = float(cfg.dt_years)
    record_dt = float(min(cfg.record_dt_years, decision_dt))
    if record_dt <= 0.0:
        record_dt = decision_dt
    horizon_steps = int(round(horizon_years / record_dt))

    pre_index = -1
    f_crit_eff = float(cfg.f_crit)
    f = 1.0
    f_init = 1.0
    pre_cost = 0.0
    cost = 0.0
    npv = 0.0
    det_mult = 1.0
    haz_mult = 1.0
    maint_level = 0
    lr = 0.0
    risk = 0.0
    floor_violation = 0.0
    min_f = 1.0
    feasible = True
    active_procs: list[dict[str, float]] = []
    transitions: list[Transition] = []
    j_anchor = 0.0

    eq_intensities = cfg.hazard_intensities if cfg.eq_intensities is None else cfg.eq_intensities
    eq_probs = cfg.hazard_intensity_probs if cfg.eq_intensity_probs is None else cfg.eq_intensity_probs
    fire_intensities = cfg.hazard_intensities if cfg.fire_intensities is None else cfg.fire_intensities
    fire_probs = cfg.hazard_intensity_probs if cfg.fire_intensity_probs is None else cfg.fire_intensity_probs
    maint_interval_years = float(cfg.maintenance_interval_years)

    def _apply_procs(dt: float) -> float:
        nonlocal active_procs
        delta_f = 0.0
        updated: list[dict[str, float]] = []
        for p in active_procs:
            delay = float(p["delay"])
            remaining = float(p["remaining"])
            rate = float(p["rate"])
            if remaining <= 0.0:
                continue
            if delay >= dt:
                p["delay"] = delay - dt
                updated.append(p)
                continue
            active_time = dt - max(0.0, delay)
            p["delay"] = 0.0
            used = min(active_time, remaining)
            delta_f += rate * used
            remaining -= used
            p["remaining"] = remaining
            if remaining > 1.0e-12:
                updated.append(p)
        active_procs = updated
        return float(delta_f)

    def _j() -> float:
        return _current_objective(
            cfg, lr=lr, risk=risk, lcc=cost, npv=npv, floor_violation=floor_violation,
        )

    def _close_reward() -> None:
        nonlocal j_anchor
        if not transitions:
            j_anchor = _j()
            return
        now = _j()
        transitions[-1].reward += float(now - j_anchor)
        j_anchor = now

    def _decide(kind: str, t: float, f_now: float, intensity: float) -> int:
        obs = _obs_vector(
            cfg,
            t=t,
            f=f_now,
            pre_index=pre_index,
            maint_level=maint_level,
            n_procs=len(active_procs),
            kind=kind,
            intensity=intensity,
            min_f=min_f,
        )
        _close_reward()
        action, logp, value = actor.act(obs, act_rng, greedy=greedy)
        action = int(np.clip(action, 0, 2))
        if collect:
            transitions.append(Transition(obs=obs, action=action, logp=logp, value=value))
        return action

    # t=0: pre-reinforcement, then initial maintenance level.
    pre_index = _decide("pre", 0.0, 1.0, 0.0)
    pre_index = int(np.clip(pre_index, 0, len(cfg.pre_cost_options) - 1))
    f_crit_eff = tl._effective_f_crit(cfg, pre_index)
    f = tl._initial_f(cfg, pre_index)
    f_init = float(f)
    pre_cost = float(cfg.pre_cost_options[pre_index])
    cost = pre_cost
    npv = pre_cost * tl._discount_factor(cfg.discount_rate, 0.0)
    det_mult = float(cfg.pre_deterioration_multipliers[pre_index])
    haz_mult = float(cfg.pre_hazard_damage_multipliers[pre_index])
    min_f = float(f)
    maint_level = _decide("maint", 0.0, float(f), 0.0)
    maint_level = int(np.clip(maint_level, 0, 2))

    t_years = [0.0]
    f_series = [f]
    cumulative_cost = [cost]
    t = 0.0
    for _ in range(horizon_steps):
        if t >= horizon_years - 1.0e-12:
            break
        is_year_boundary = abs(t - round(t / decision_dt) * decision_dt) <= 1.0e-9
        if is_year_boundary and t > 0.0 and maint_interval_years > 0.0:
            if abs((t % maint_interval_years)) <= 1.0e-9:
                maint_level = _decide("maint", t, float(f), 0.0)
                maint_level = int(np.clip(maint_level, 0, 2))
                maint_cost = float(cfg.maint_costs[maint_level])
                cost += maint_cost
                npv += maint_cost * tl._avg_discount_factor(
                    cfg.discount_rate, t, min(horizon_years, t + decision_dt))

        if is_year_boundary:
            dt_h = float(decision_dt)
            events: list[float] = []
            if cfg.lambda_eq_per_year > 0.0 and dt_h > 0.0:
                p = tl._hazard_prob_from_rate(float(cfg.lambda_eq_per_year), dt_h)
                if float(rng.random()) < p:
                    events.append(float(rng.choice(
                        eq_intensities, p=np.array(eq_probs, dtype=float))))
            if cfg.lambda_fire_per_year > 0.0 and dt_h > 0.0:
                p = tl._hazard_prob_from_rate(float(cfg.lambda_fire_per_year), dt_h)
                if float(rng.random()) < p:
                    events.append(float(rng.choice(
                        fire_intensities, p=np.array(fire_probs, dtype=float))))
            if len(events) > 1:
                events = [events[i] for i in rng.permutation(len(events)).tolist()]

            for intensity in events:
                intensity_eff = float(intensity) * haz_mult
                f_before_damage = float(f)
                f_after_damage = tl._clip_lo(float(f) - float(intensity_eff))
                event_drop = max(0.0, f_before_damage - f_after_damage)
                if event_drop > 0.0:
                    event_cf = tl._cf(cfg, f_after_damage, f_crit=f_crit_eff)
                    if event_cf > 0.0:
                        risk += event_cf * event_drop
                repair_level = _decide("repair", t, float(f_after_damage), float(intensity_eff))
                repair_level = int(np.clip(repair_level, 0, 2))
                f = f_after_damage
                repair_cost = float(cfg.repair_costs[repair_level]) * (
                    1.0 + 0.8 * float(intensity_eff))
                cost += repair_cost
                npv += repair_cost * tl._avg_discount_factor(
                    cfg.discount_rate, t, min(horizon_years, t + decision_dt))
                repair_duration = float(cfg.repair_durations_years[repair_level])
                repair_delta = float(cfg.repair_recovery_deltas[repair_level])
                if repair_duration > 0.0 and repair_delta > 0.0:
                    active_procs.append(
                        {
                            "delay": 0.0,
                            "remaining": repair_duration,
                            "rate": repair_delta / repair_duration,
                        }
                    )

        t_next = min(horizon_years, t + record_dt)
        dt_seg = float(max(0.0, t_next - t))
        if dt_seg <= 0.0:
            break
        f0 = float(f)
        if cfg.deterioration_model.lower() == "weibull":
            g0 = tl._weibull_g(t, tau_years=cfg.det_tau_years,
                               k=cfg.det_k, horizon_years=cfg.horizon_years)
            g1 = tl._weibull_g(t_next, tau_years=cfg.det_tau_years,
                               k=cfg.det_k, horizon_years=cfg.horizon_years)
            dg = max(0.0, g1 - g0)
            alpha_T = float(cfg.det_alpha_T_levels[int(
                np.clip(maint_level, 0, len(cfg.det_alpha_T_levels) - 1))])
            delta_alpha = alpha_T * det_mult * dg
            if cfg.det_sigma > 0.0:
                delta_alpha *= float(rng.lognormal(mean=0.0, sigma=float(cfg.det_sigma)))
            f = tl._clip_lo(f - delta_alpha)
        else:
            det_rate = cfg.base_deterioration_rate * det_mult * cfg.maint_rate_multipliers[maint_level]
            det_rate *= float(rng.lognormal(mean=0.0, sigma=0.15))
            f = tl._clip_lo(f - det_rate * dt_seg)
        repair_delta = float(_apply_procs(dt_seg))
        f = float(f) + repair_delta
        if repair_delta > 0.0:
            f = min(1.0, f)
        f = tl._clip_lo(f)
        f1 = float(f)
        f_mid = 0.5 * (f0 + f1)
        lr += tl._lr_increment_pdf(f0, f1, t0=t, t1=t_next, gamma=0.0)
        cf_val = tl._cf(cfg, f_mid, f_crit=f_crit_eff)
        if cf_val > 0.0:
            risk += cf_val * dt_seg
        floor_violation += max(0.0, cfg.f_2 - f_mid) * dt_seg
        min_f = min(min_f, f1)
        feasible = feasible and (f1 >= cfg.f_2)
        t = t_next
        t_years.append(t)
        f_series.append(f1)
        cumulative_cost.append(float(cost))

    _close_reward()
    return_value = _j()
    ep = tl.EpisodeResult(
        t_years=t_years,
        f=f_series,
        cumulative_cost=cumulative_cost,
        lr=lr,
        risk=risk,
        lcc=cost,
        npv=npv,
        min_f=min_f,
        feasible=feasible,
        return_value=return_value,
    )
    return ep, transitions


class ThresholdActor(ActorCritic):
    """Replay a frozen threshold policy through the sequential interface."""

    def __init__(self, params: tl.PolicyParams):
        super().__init__(np.random.default_rng(0))
        self.threshold = params

    def act(self, obs: np.ndarray, rng: np.random.Generator, *, greedy: bool) -> tuple[int, float, float]:
        kind = int(np.argmax(obs[11:14]))
        f = float(obs[1])
        if kind == 0:
            action = int(self.threshold.pre_index)
        elif kind == 1:
            action = int(tl._choose_maint(self.threshold, tl._maint_feature(f, 1.0)))
        else:
            action = int(tl._choose_repair(self.threshold, f))
        return action, 0.0, 0.0


def _gae(transitions: list[Transition], *, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(transitions)
    adv = np.zeros(n, dtype=np.float64)
    gae = 0.0
    next_v = 0.0
    for i in range(n - 1, -1, -1):
        delta = transitions[i].reward + gamma * next_v - transitions[i].value
        gae = delta + gamma * lam * gae
        adv[i] = gae
        next_v = transitions[i].value
    ret = adv + np.array([tr.value for tr in transitions], dtype=np.float64)
    return adv, ret


def _rollout_one(
    cfg: tl.LifecycleConfig,
    net: ActorCritic,
    seed: int,
    greedy: bool,
) -> dict[str, Any]:
    ep, trans = simulate_with_policy(cfg, net, int(seed), greedy=bool(greedy), collect=not greedy)
    packed = []
    if not greedy:
        for tr in trans:
            packed.append(
                {
                    "obs": tr.obs,
                    "action": int(tr.action),
                    "logp": float(tr.logp),
                    "value": float(tr.value),
                    "reward": float(tr.reward),
                }
            )
    return {
        "return": float(ep.return_value),
        "lr": float(ep.lr),
        "risk": float(ep.risk),
        "npv": float(ep.npv),
        "min_f": float(ep.min_f),
        "feasible": 1.0 if ep.feasible else 0.0,
        "n_decisions": len(trans),
        "transitions": packed,
    }


def _rollout_job(payload: tuple[dict[str, Any], list[np.ndarray], list[int], bool]) -> list[dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cfg_dict, weights, seeds, greedy = payload
    cfg = tl.LifecycleConfig(**tl._coerce_lifecycle_config_dict(cfg_dict))
    net = ActorCritic(np.random.default_rng(0))
    net.set_params(weights)
    return [_rollout_one(cfg, net, int(seed), bool(greedy)) for seed in seeds]


def _chunk_seeds(seeds: list[int], n_chunks: int) -> list[list[int]]:
    n_chunks = max(1, int(n_chunks))
    if not seeds:
        return []
    n_chunks = min(n_chunks, len(seeds))
    chunks: list[list[int]] = [[] for _ in range(n_chunks)]
    for i, seed in enumerate(seeds):
        chunks[i % n_chunks].append(int(seed))
    return chunks


def _map_rollouts(
    cfg_dict: dict[str, Any],
    weights: list[np.ndarray],
    seeds: list[int],
    *,
    greedy: bool,
    workers: int,
    pool: ProcessPoolExecutor | None,
) -> list[dict[str, Any]]:
    chunks = _chunk_seeds(seeds, workers if workers > 1 else 1)
    jobs = [(cfg_dict, weights, chunk, bool(greedy)) for chunk in chunks if chunk]
    if pool is None or workers <= 1:
        nested = [_rollout_job(j) for j in jobs]
    else:
        nested = list(pool.map(_rollout_job, jobs, chunksize=1))
    rows: list[dict[str, Any]] = []
    for part in nested:
        rows.extend(part)
    return rows


def _ppo_update(
    net: ActorCritic,
    opt: Adam,
    episodes_tr: list[list[Transition]],
    *,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    minibatch: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    batch: list[Transition] = []
    adv_parts: list[np.ndarray] = []
    ret_parts: list[np.ndarray] = []
    for ep_tr in episodes_tr:
        if not ep_tr:
            continue
        a, r = _gae(ep_tr, gamma=1.0, lam=0.95)
        adv_parts.append(a)
        ret_parts.append(r)
        batch.extend(ep_tr)
    if not batch:
        return {"pg": 0.0, "vf": 0.0, "ent": 0.0}
    adv = np.concatenate(adv_parts)
    ret = np.concatenate(ret_parts)
    adv = (adv - float(np.mean(adv))) / (float(np.std(adv)) + 1.0e-8)
    n = len(batch)
    idx = np.arange(n)
    acc = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "n": 0.0}
    obs_all = np.stack([tr.obs for tr in batch], axis=0)
    act_all = np.array([tr.action for tr in batch], dtype=int)
    logp_all = np.array([tr.logp for tr in batch], dtype=np.float64)
    val_all = np.array([tr.value for tr in batch], dtype=np.float64)
    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for start in range(0, n, int(minibatch)):
            sl = idx[start:start + int(minibatch)]
            g, stats = _backward_ppo_batch(
                net,
                obs_all[sl],
                act_all[sl],
                logp_all[sl],
                adv[sl],
                ret[sl],
                val_all[sl],
                clip=clip,
                vf_coef=vf_coef,
                ent_coef=ent_coef,
            )
            opt.step(g, max_norm=0.5)
            acc["pg"] += stats["pg"]
            acc["vf"] += stats["vf"]
            acc["ent"] += stats["ent"]
            acc["n"] += 1.0
    denom = max(1.0, acc["n"])
    return {"pg": acc["pg"] / denom, "vf": acc["vf"] / denom, "ent": acc["ent"] / denom}


def evaluate_holdout(
    cfg: tl.LifecycleConfig,
    net: ActorCritic,
    *,
    base_seed: int,
    episodes: int,
    workers: int,
) -> dict[str, Any]:
    cfg_dict = asdict(cfg)
    weights = net.snapshot()
    seeds = [int(base_seed + i * 10007) for i in range(int(episodes))]
    if workers <= 1 or episodes <= 4:
        rows = _map_rollouts(cfg_dict, weights, seeds, greedy=True, workers=1, pool=None)
    else:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=tl._eval_worker_init) as ex:
            rows = _map_rollouts(
                cfg_dict, weights, seeds, greedy=True, workers=int(workers), pool=ex,
            )
    obj = np.array([r["return"] for r in rows], dtype=float)
    reported = np.array(
        [_reported_return(cfg, lr=r["lr"], risk=r["risk"], npv=r["npv"]) for r in rows],
        dtype=float,
    )
    lr = np.array([r["lr"] for r in rows], dtype=float)
    risk = np.array([r["risk"] for r in rows], dtype=float)
    npv = np.array([r["npv"] for r in rows], dtype=float)
    min_f = np.array([r["min_f"] for r in rows], dtype=float)
    return {
        "objective_return": float(np.mean(obj)),
        "objective_return_ci": unc._ci_halfwidth(obj),
        "reported_return": float(np.mean(reported)),
        "reported_return_ci": unc._ci_halfwidth(reported),
        "lr": float(np.mean(lr)),
        "lr_ci": unc._ci_halfwidth(lr),
        "risk": float(np.mean(risk)),
        "risk_ci": unc._ci_halfwidth(risk),
        "npv": float(np.mean(npv)),
        "npv_ci": unc._ci_halfwidth(npv),
        "min_f": float(np.mean(min_f)),
        "min_f_ci": unc._ci_halfwidth(min_f),
        "feasible_frac": float(np.mean([r["feasible"] for r in rows])),
        "n": int(episodes),
        "eval_base_seed": int(base_seed),
    }


def train_ppo(
    cfg: tl.LifecycleConfig,
    *,
    out_dir: Path,
    seed: int,
    total_episodes: int,
    rollout_episodes: int,
    holdout_seed: int,
    holdout_episodes: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    net = ActorCritic(rng)
    opt = Adam(net.params(), lr=3.0e-4)
    workers = _worker_count()
    cfg_dict = asdict(cfg)
    logger.info(
        "PPO start | episodes=%d rollout=%d workers=%d obs_dim=%d hid=%d",
        total_episodes, rollout_episodes, workers, OBS_DIM, HID,
    )
    history: list[dict[str, Any]] = []
    best_train = -1.0e18
    best_weights = net.snapshot()
    n_done = 0
    update_i = 0
    t0 = time.time()
    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=tl._eval_worker_init)
    try:
        while n_done < int(total_episodes):
            n_batch = min(int(rollout_episodes), int(total_episodes) - n_done)
            weights = net.snapshot()
            seeds = [
                int(seed + 10_000_000 + (n_done + k) * 10007)
                for k in range(n_batch)
            ]
            rows = _map_rollouts(
                cfg_dict, weights, seeds,
                greedy=False, workers=workers, pool=pool,
            )
            episodes_tr: list[list[Transition]] = []
            rets = []
            n_trans = 0
            for row in rows:
                rets.append(float(row["return"]))
                ep_tr = [
                    Transition(
                        obs=np.asarray(tr["obs"], dtype=np.float32),
                        action=int(tr["action"]),
                        logp=float(tr["logp"]),
                        value=float(tr["value"]),
                        reward=float(tr["reward"]),
                    )
                    for tr in row["transitions"]
                ]
                n_trans += len(ep_tr)
                episodes_tr.append(ep_tr)
            stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0}
            if n_trans:
                stats = _ppo_update(
                    net, opt, episodes_tr,
                    clip=0.2, vf_coef=0.5, ent_coef=0.02,
                    epochs=4, minibatch=256, rng=rng,
                )
            n_done += n_batch
            update_i += 1
            mean_ret = float(np.mean(rets)) if rets else float("nan")
            if mean_ret > best_train:
                best_train = mean_ret
                best_weights = net.snapshot()
            rec = {
                "update": update_i,
                "episodes": n_done,
                "train_return": mean_ret,
                "best_train_return": float(best_train),
                "n_transitions": n_trans,
                **stats,
            }
            history.append(rec)
            if update_i <= 5 or update_i % 5 == 0 or n_done >= total_episodes:
                logger.info(
                    "PPO update=%d episodes=%d train_return=%.4f best=%.4f pg=%.4f ent=%.3f elapsed=%.0fs",
                    update_i, n_done, mean_ret, best_train, stats["pg"], stats["ent"],
                    time.time() - t0,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    net.set_params(best_weights)
    logger.info("Evaluating greedy policy on common holdout | N=%d seed=%d", holdout_episodes, holdout_seed)
    holdout = evaluate_holdout(
        cfg, net,
        base_seed=int(holdout_seed),
        episodes=int(holdout_episodes),
        workers=workers,
    )
    result = {
        "method": "PPO",
        "policy_class": "sequential_actor_critic",
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "hidden": HID,
        "total_episodes": int(total_episodes),
        "seed": int(seed),
        "best_train_return": float(best_train),
        "holdout": holdout,
        "history": history,
        "elapsed_sec": float(time.time() - t0),
        "weights": [p.tolist() for p in net.snapshot()],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ppo_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "Deep RL (sequential PPO) holdout vs CEM/GA/PSO common seeds",
        f"Uncertainty: mean ± 95% CI half-width (N={holdout['n']})",
        f"Training episodes: {total_episodes} (stochastic); holdout uses greedy actions.",
        "",
        "Method | Objective return | Reported return",
        f"PPO | {unc._fmt(holdout['objective_return'], holdout['objective_return_ci'], 4)} | "
        f"{unc._fmt(holdout['reported_return'], holdout['reported_return_ci'], 4)}",
    ]
    (out_dir / "ppo_holdout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "PPO holdout | objective=%.4f ± %.4f reported=%.4f ± %.4f",
        holdout["objective_return"], holdout["objective_return_ci"],
        holdout["reported_return"], holdout["reported_return_ci"],
    )
    return result


def _self_check(cfg: tl.LifecycleConfig, logger: logging.Logger) -> None:
    params = tl.PolicyParams(pre_index=1, maint_t1=0.90, maint_t2=0.95, repair_t1=0.95, repair_t2=0.75)
    seed = 21160301
    ref = tl.simulate_episode(cfg, params, seed=seed)
    actor = ThresholdActor(params)
    ep, _tr = simulate_with_policy(cfg, actor, seed, greedy=True, collect=False)
    dret = abs(float(ep.return_value) - float(ref.return_value))
    dlr = abs(float(ep.lr) - float(ref.lr))
    logger.info(
        "Physics self-check | dReturn=%.6g dLR=%.6g dRisk=%.6g dNPV=%.6g",
        dret, dlr, abs(ep.risk - ref.risk), abs(ep.npv - ref.npv),
    )
    if dret > 1.0e-8 or dlr > 1.0e-8:
        raise RuntimeError(
            f"Sequential simulator diverged from simulate_episode: dReturn={dret} dLR={dlr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequential PPO on the lifecycle SMDP.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--rollout-episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20270399)
    parser.add_argument("--holdout-seed", type=int, default=21160301)
    parser.add_argument("--holdout-episodes", type=int, default=400)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    logger = _setup_logger(out_dir)
    cfg, _cem, _paths = tl._load_run_config(args.config if args.config.is_absolute() else (ROOT / args.config), ROOT)
    if not args.skip_check:
        _self_check(cfg, logger)
    train_ppo(
        cfg,
        out_dir=out_dir,
        seed=int(args.seed),
        total_episodes=int(args.episodes),
        rollout_episodes=int(args.rollout_episodes),
        holdout_seed=int(args.holdout_seed),
        holdout_episodes=int(args.holdout_episodes),
        logger=logger,
    )


if __name__ == "__main__":
    main()
