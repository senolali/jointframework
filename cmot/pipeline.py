"""End-to-end decision pipelines (all scored against the human label).

Uncertainty-routing variant of the Joint framework (supplementary analysis):
  Phase 1  ensemble flags a conversation as suspicious;
  Phase 2  turn-level OCDD monitors the flagged conversation; on an alarm the
           LLM judge decides fraudulent vs benign_shift. A benign_shift verdict
           now *clears* the flag (false-positive correction), so the stage has a
           measurable effect. Without an alarm the Phase-1 label is kept.
           (The original 'spamming' sub-label had no ground truth and is dropped.)

Controls / improvements evaluated at matched LLM budgets:
  * Uncertainty routing: send the conversations on which the ensemble members
    disagree (non-unanimous vote) to the LLM -- recovers Phase-1 false
    negatives, which the original gate could never do;
  * Random routing at the same call count (shows whether the routing signal
    carries information);
  * LLM-only (every conversation is sent; upper bound on cost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import compute_metrics, fast_metric


def _summ(name: str, model: str, y, pred, routed: np.ndarray, llm_cost: np.ndarray, llm_lat: np.ndarray,
          ens_lat_per_conv: float, score=None) -> dict:
    m = compute_metrics(y, pred, score)
    n = len(y)
    m.update({
        "system": name, "llm": model, "llm_calls": int(routed.sum()), "llm_call_rate": float(routed.mean()),
        "llm_cost_usd": float(np.nansum(llm_cost[routed])),
        "cost_per_1k_conv_usd": float(1000 * np.nansum(llm_cost[routed]) / n),
        # serial latency per conversation = ensemble + LLM call when routed
        "latency_per_conv_s": float(ens_lat_per_conv + np.nansum(llm_lat[routed]) / n),
    })
    return m


def evaluate_pipelines(y: np.ndarray, conv_ids: np.ndarray, ens_pred: np.ndarray, ens_soft: np.ndarray,
                       ens_unanimous: np.ndarray, ens_lat: float, llm: dict[str, pd.DataFrame],
                       alarms: pd.DataFrame, judge: dict[str, pd.DataFrame], n_random: int = 200,
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [_summ("Ensemble", "-", y, ens_pred, np.zeros(len(y), bool), np.zeros(len(y)), np.zeros(len(y)),
                  ens_lat, ens_soft)]
    al = alarms.set_index("conv_id").reindex(conv_ids)
    alarm = al["alarm"].fillna(False).to_numpy(bool)
    for key, df in llm.items():
        d = df.set_index("conv_id").reindex(conv_ids)
        lp, lprob = d["pred"].to_numpy(int), d["p_malicious"].to_numpy(float)
        cost, lat = d["cost_usd"].to_numpy(float), d["latency_s"].to_numpy(float)
        rows.append(_summ("LLM-only", key, y, lp, np.ones(len(y), bool), cost, lat, 0.0, lprob))

        # Joint (OCDD-gated judge)
        if key in judge and len(judge[key]):
            j = judge[key].set_index("conv_id").reindex(conv_ids)
            routed = (ens_pred == 1) & alarm & j["judge_pred"].notna().to_numpy()
            final = ens_pred.copy()
            final[routed] = j["judge_pred"].to_numpy()[routed].astype(int)
            rows.append(_summ("Joint-OCDD", key, y, final, routed, j["cost_usd"].to_numpy(float),
                              j["latency_s"].to_numpy(float), ens_lat))
            n_r = int(routed.sum())
            rows.append(_random_route("Random-route@OCDD", key, y, ens_pred, lp, cost, lat, ens_lat, n_r,
                                      rng, n_random, pool=(ens_pred == 1)))

        # Uncertainty routing (non-unanimous vote)
        routed = ~ens_unanimous
        final = np.where(routed, lp, ens_pred)
        rows.append(_summ("Joint-Uncertainty", key, y, final, routed, cost, lat, ens_lat))
        rows.append(_random_route("Random-route@Uncertainty", key, y, ens_pred, lp, cost, lat, ens_lat,
                                  int(routed.sum()), rng, n_random))
    return pd.DataFrame(rows)


def _random_route(name, key, y, ens_pred, lp, cost, lat, ens_lat, n_r, rng, n_random, pool=None) -> dict:
    pool_idx = np.arange(len(y)) if pool is None else np.where(pool)[0]
    n_r = min(n_r, len(pool_idx))
    recs = []
    for _ in range(n_random):
        routed = np.zeros(len(y), bool)
        routed[rng.choice(pool_idx, n_r, replace=False)] = True
        recs.append(_summ(name, key, y, np.where(routed, lp, ens_pred), routed, cost, lat, ens_lat))
    df = pd.DataFrame(recs)
    out = df.iloc[0].to_dict()
    for c in df.select_dtypes("number").columns:
        out[c] = float(df[c].mean())
    out["macro_f1_sd"] = float(df["macro_f1"].std())
    return out


def routing_curve(y, ens_pred, ens_soft, llm_pred, n_random: int = 200, seed: int = 0, steps: int = 21) -> pd.DataFrame:
    """macro-F1 as a function of the fraction of conversations routed to the LLM:
    by ensemble uncertainty (|soft score - 0.5| ascending) vs at random."""
    rng = np.random.default_rng(seed)
    n = len(y)
    order = np.argsort(np.abs(ens_soft - 0.5), kind="stable")
    rows = []
    for frac in np.linspace(0, 1, steps):
        k = int(round(frac * n))
        routed = np.zeros(n, bool)
        routed[order[:k]] = True
        rows.append({"frac": frac, "strategy": "uncertainty",
                     "macro_f1": fast_metric(y, np.where(routed, llm_pred, ens_pred)),
                     "lo": np.nan, "hi": np.nan})
        vals = []
        for _ in range(n_random):
            r = np.zeros(n, bool)
            r[rng.choice(n, k, replace=False)] = True
            vals.append(fast_metric(y, np.where(r, llm_pred, ens_pred)))
        rows.append({"frac": frac, "strategy": "random", "macro_f1": float(np.mean(vals)),
                     "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))})
    return pd.DataFrame(rows)
