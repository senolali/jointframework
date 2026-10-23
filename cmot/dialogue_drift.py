"""Within-dialogue concept drift detection with OCDD, window-reference variant (comparison baseline).

Each flagged conversation is a stream of messages between A and B. A one-class SVM models
the "normal" concept of the dialogue from a sliding window of recent messages; a new message
that falls outside that support is an outlier, and a drift is declared at the first message where
the outlier ratio over the last `m` scored messages reaches `rho` (OCDD, Gozuacik & Can 2021).

Reference definitions:
  dialogue -- OC-SVM fitted on the previous `window` messages of the same dialogue
  pooled   -- OC-SVM fitted once on 'normal' messages of training dialogues (messages before
              the annotated first PII request, and all messages of training dialogues without one)
  hybrid   -- pooled normal messages plus the dialogue's own recent window

Ground truth (evaluation only, never an input): the human annotation FirstPII gives the message
index at which the conversation first requests personal information. Conversations with
FirstPII >= 0 contain a drift; its type is 'fraudulent' for malicious and 'genuine' for benign
conversations. Conversations without a PII request contain no drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import OneClassSVM

from .data import Conversation


class MessageSpace:
    """TF-IDF + truncated SVD (TF-IDF/SVD) over individual messages."""

    def __init__(self, dims: int = 10, seed: int = 0):
        self.dims, self.seed = dims, seed

    def fit(self, messages: list[str]):
        self.vec_ = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english",
                                    strip_accents="unicode").fit(messages)
        S = self.vec_.transform(messages)
        self.svd_ = TruncatedSVD(n_components=min(self.dims, S.shape[1] - 1), random_state=self.seed).fit(S)
        return self

    def transform(self, messages: list[str]) -> np.ndarray:
        return normalize(self.svd_.transform(self.vec_.transform(messages)))


@dataclass(frozen=True)
class DriftParams:
    reference: str = "dialogue"  # dialogue | pooled | hybrid
    window: int = 2               # recent messages used as the dialogue's normal concept
    m: int = 1                    # scored messages over which the outlier ratio is computed
    rho: float = 1.0              # drift when outlier ratio >= rho
    nu: float = 0.1
    gamma: str | float = "scale"
    speakers: str = "all"         # all | initiator (messages of the conversation opener only)


def _ocsvm(X: np.ndarray, nu: float, gamma) -> OneClassSVM | None:
    if len(X) < 2:
        return None
    return OneClassSVM(nu=nu, gamma=gamma).fit(X)


def detect_drift(conv: Conversation, V: np.ndarray, p: DriftParams, pooled: np.ndarray | None) -> dict:
    """V: message vectors of this conversation. Returns the first drift message index (or None)."""
    idx = list(range(len(conv.messages))) if p.speakers == "all" else conv.initiator_turns()
    pooled_model = _ocsvm(pooled, p.nu, p.gamma) if p.reference == "pooled" and pooled is not None else None
    flags: list[bool] = []
    seen: list[int] = []
    for pos, t in enumerate(idx):
        if p.reference == "pooled":
            model = pooled_model
        else:
            recent = [V[i] for i in seen[-p.window:]]
            if len(recent) < min(2, p.window):
                seen.append(t)
                continue
            ref = np.vstack(recent)
            if p.reference == "hybrid" and pooled is not None:
                ref = np.vstack([pooled, ref])
            model = _ocsvm(ref, p.nu, p.gamma)
        if model is not None:
            flags.append(bool(model.predict(V[t][None])[0] == -1))
            last = flags[-p.m:]
            if len(last) == p.m and np.mean(last) >= p.rho:
                return {"drift": True, "drift_at": t, "n_scored": len(flags)}
        seen.append(t)
    return {"drift": False, "drift_at": None, "n_scored": len(flags)}


def pooled_normal(convs: list[Conversation], vecs: dict[int, np.ndarray], speakers: str = "all",
                  max_n: int = 600, seed: int = 0) -> np.ndarray:
    """'Normal' messages of training dialogues: before the first PII request, or all messages
    when the dialogue contains none (annotation used on TRAINING data only)."""
    parts = []
    for c in convs:
        fp = c.first_pii if (c.first_pii is not None and c.first_pii >= 0) else len(c.messages)
        idx = [i for i in (range(len(c.messages)) if speakers == "all" else c.initiator_turns()) if i < fp]
        if idx:
            parts.append(vecs[c.conv_id][idx])
    X = np.vstack(parts)
    if len(X) > max_n:
        X = X[np.random.default_rng(seed).choice(len(X), max_n, replace=False)]
    return X


def evaluate(convs: list[Conversation], vecs: dict[int, np.ndarray], p: DriftParams, pooled: np.ndarray | None,
             tolerance: int = 2) -> pd.DataFrame:
    rows = []
    for c in convs:
        r = detect_drift(c, vecs[c.conv_id], p, pooled)
        fp = c.first_pii if (c.first_pii is not None and c.first_pii >= 0) else None
        delay = (r["drift_at"] - fp) if (r["drift"] and fp is not None) else np.nan
        rows.append({"conv_id": c.conv_id, "label": c.label, "has_drift": fp is not None, "first_pii": fp,
                     "detected": r["drift"], "drift_at": r["drift_at"], "delay": delay,
                     "localized": bool(r["drift"] and fp is not None and 0 <= delay <= tolerance),
                     "early": bool(r["drift"] and fp is not None and delay < 0), "n_msgs": len(c.messages)})
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> dict:
    """Detection metrics against the FirstPII drift annotation."""
    pos, neg = df[df.has_drift], df[~df.has_drift]
    tpr = float(pos.detected.mean()) if len(pos) else np.nan
    loc = float(pos.localized.mean()) if len(pos) else np.nan
    fpr = float(neg.detected.mean()) if len(neg) else np.nan
    return {"n": len(df), "n_drift": len(pos), "n_no_drift": len(neg), "detect_rate": tpr, "localized_rate": loc,
            "early_rate": float(pos.early.mean()) if len(pos) else np.nan, "false_drift_rate": fpr,
            "balanced": np.nanmean([loc, 1 - fpr]) if len(neg) else loc,
            "median_delay": float(pos.delay.median()) if pos.detected.any() else np.nan}


def build_spaces(fit_messages: list[str], dims_list: list[int], seed: int = 0) -> dict[int, MessageSpace]:
    return {d: MessageSpace(d, seed).fit(fit_messages) for d in dims_list}


def tune_on_train(train: list[Conversation], spaces: dict[int, MessageSpace], grid: dict,
                  seed: int = 0) -> tuple[pd.DataFrame, dict[str, tuple[DriftParams, int]]]:
    """Grid search on the TRAINING conversations only. Returns the full grid and, for every
    reference definition, the configuration with the best balanced score (ties: localization)."""
    import itertools

    vecs = {d: {c.conv_id: sp.transform(c.messages) for c in train} for d, sp in spaces.items()}
    rows, seen = [], set()
    for ref, w, m, rho, nu, spk, d in itertools.product(grid["reference"], grid["window"], grid["m"], grid["rho"],
                                                        grid["nu"], grid["speakers"], list(spaces)):
        key = (ref, None if ref == "pooled" else w, m, rho, nu, spk, d)
        if key in seen:
            continue
        seen.add(key)
        p = DriftParams(reference=ref, window=w, m=m, rho=rho, nu=nu, speakers=spk)
        pooled = pooled_normal(train, vecs[d], spk, seed=seed) if ref != "dialogue" else None
        rows.append({"reference": ref, "window": w if ref != "pooled" else None, "m": m, "rho": rho, "nu": nu,
                     "speakers": spk, "dims": d,
                     **{f"train_{k}": v for k, v in score(evaluate(train, vecs[d], p, pooled)).items()}})
    tab = pd.DataFrame(rows)
    best = {}
    for ref, g in tab.groupby("reference"):
        b = g.sort_values(["train_balanced", "train_localized_rate"], ascending=False).iloc[0]
        best[ref] = (DriftParams(reference=ref, window=int(b.window) if pd.notna(b.window) else 2, m=int(b.m),
                                 rho=float(b.rho), nu=float(b.nu), speakers=b.speakers), int(b.dims))
    return tab, best
