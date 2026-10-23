"""Metrics with explicit definitions and paired statistical tests.

Positive class = malicious (IsMalicious=True). Reported:
  accuracy, balanced accuracy, precision/recall/F1 of the malicious class,
  macro-averaged precision/recall/F1 (unweighted mean of both classes), MCC,
  AUROC (when a continuous score exists), Brier score (when probabilistic).
One averaging convention is used for every model -- the inconsistency of the
original Tables 2-3 cannot occur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

PRIMARY = "macro_f1"


def _div(a, b) -> float:
    return float(a / b) if b else 0.0


def count_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """All threshold metrics from the confusion counts (fast; used inside bootstraps).
    Same zero-division convention as scikit-learn (zero_division=0)."""
    p1, r1 = _div(tp, tp + fp), _div(tp, tp + fn)  # malicious class
    p0, r0 = _div(tn, tn + fn), _div(tn, tn + fp)  # benign class
    f1 = _div(2 * p1 * r1, p1 + r1)
    f0 = _div(2 * p0 * r0, p0 + r0)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    n = tp + fp + fn + tn
    return {
        "n": int(n), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "accuracy": _div(tp + tn, n), "balanced_accuracy": (r1 + r0) / 2,
        "precision_mal": p1, "recall_mal": r1, "f1_mal": f1,
        "macro_precision": (p1 + p0) / 2, "macro_recall": (r1 + r0) / 2, "macro_f1": (f1 + f0) / 2,
        "mcc": float((tp * tn - fp * fn) / den) if den else 0.0,
        "fpr": _div(fp, fp + tn),
    }


def fast_metric(y: np.ndarray, pred: np.ndarray, metric: str = PRIMARY) -> float:
    y, p = np.asarray(y, bool), np.asarray(pred, bool)
    tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p)); tn = int(np.sum(~y & ~p))  # noqa: E702
    return count_metrics(tp, fp, fn, tn)[metric]


def compute_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {**count_metrics(int(tp), int(fp), int(fn), int(tn)), "auroc": np.nan, "brier": np.nan}
    if score is not None:
        s = np.asarray(score, float)
        ok = ~np.isnan(s)
        if ok.sum() > 1 and len(np.unique(y[ok])) == 2:
            m["auroc"] = roc_auc_score(y[ok], s[ok])
            if np.all((s[ok] >= 0) & (s[ok] <= 1)):
                m["brier"] = brier_score_loss(y[ok], s[ok])
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def bootstrap_ci(y, pred, metric: str = PRIMARY, B: int = 2000, seed: int = 0, alpha: float = 0.05,
                 score=None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fast_metric(y[i], pred[i], metric) if score is None
                    else compute_metrics(y[i], pred[i], np.asarray(score)[i])[metric])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(y, pred_a, pred_b, metric: str = PRIMARY, B: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(fast_metric(y[i], a[i], metric) - fast_metric(y[i], b[i], metric))
    d = np.asarray(d)
    return {"diff": float(compute_metrics(y, a)[metric] - compute_metrics(y, b)[metric]),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb))  # a right, b wrong
    n10 = int(np.sum(~ca & cb))
    n = n01 + n10
    p = 1.0 if n == 0 else stats.binomtest(n01, n, 0.5).pvalue
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected repeated-CV t-test for paired score differences."""
    diffs = np.asarray(diffs, float)
    J = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(mean), "t": np.nan, "p_value": 1.0 if mean == 0 else 0.0}
    t = mean / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"mean_diff": float(mean), "t": float(t), "p_value": float(p)}


# Studentised range critical values q_0.05 / sqrt(2) for Nemenyi (Demšar, 2006), k = 2..20
_Q05 = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164, 3.219, 3.268,
        3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544]


def friedman_nemenyi(score_table: pd.DataFrame) -> dict:
    """score_table: rows = folds (blocks), columns = models; higher is better."""
    data = score_table.dropna(axis=1)
    k, N = data.shape[1], data.shape[0]
    ranks = data.rank(axis=1, ascending=False).mean(0).sort_values()
    out = {"avg_ranks": ranks.to_dict(), "k": k, "N": N}
    if k >= 3 and N >= 2:
        stat, p = stats.friedmanchisquare(*[data[c].values for c in data.columns])
        out.update({"friedman_chi2": float(stat), "friedman_p": float(p)})
    if 2 <= k <= 20:
        out["cd"] = float(_Q05[k - 2] * np.sqrt(k * (k + 1) / (6 * N)))
    return out
