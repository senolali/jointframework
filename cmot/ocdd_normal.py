"""Within-dialogue OCDD with a reference of NORMAL messages (primary Phase-2 drift detector).

The OCDD one-class SVM (Gozuacik & Can 2021) models the normal concept from messages that precede any
request for personal information in the TRAINING dialogues (the conversations' FirstPII annotation is used
on training data only). Each flagged dialogue is then monitored message by message: the initiator's messages
after the opening one are scored, a message is an outlier when its score exceeds a threshold calibrated on
held-out normal messages, and a drift is declared at the first message where the outlier ratio over the last
`m` scored messages reaches `rho`.

Two families are compared under the same train-only selection:
  normal   -- OC-SVM on standardized normal messages (as above)
  opening  -- cosine shift of each message from the dialogue's opening message (no reference model)
Message representations: TF-IDF+SVD or PCA-reduced te3-large embeddings, both fitted label-free.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import OneClassSVM

from .data import Conversation
from .dialogue_drift import score


def first_pii(c: Conversation) -> int | None:
    return c.first_pii if (c.first_pii is not None and c.first_pii >= 0) else None


def monitored(c: Conversation) -> list[int]:
    """Initiator messages after the opening message."""
    return [i for i in c.initiator_turns() if i >= 2]


def normal_indices(c: Conversation) -> list[int]:
    """Initiator messages before the annotated first PII request (training dialogues only)."""
    f = first_pii(c)
    return [i for i in c.initiator_turns() if f is None or i < f]


class MemoEmbedder:
    """In-memory front for the client's on-disk embedding cache (one disk read per batch of new texts)."""

    def __init__(self, client, provider: str, model: str):
        self.client, self.provider, self.model = client, provider, model
        self.memo: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        new = list(dict.fromkeys(t for t in texts if t not in self.memo))
        if new:
            for t, v in zip(new, self.client.embed(new, self.provider, self.model)):
                self.memo[t] = v
        return np.vstack([self.memo[t] for t in texts])


class EmbeddingSpace:
    """PCA-reduced sentence embeddings of individual messages (fitted without labels)."""

    def __init__(self, embed_fn, dims: int = 20, seed: int = 0):
        self.embed_fn, self.dims, self.seed = embed_fn, dims, seed

    def fit(self, messages: list[str]):
        self.pca_ = PCA(self.dims, random_state=self.seed).fit(self.embed_fn(messages))
        return self

    def transform(self, messages: list[str]) -> np.ndarray:
        return normalize(self.pca_.transform(self.embed_fn(messages)))


@dataclass(frozen=True)
class NormalOCDDConfig:
    design: str = "normal"   # normal | opening
    rep: str = "emb"         # emb | tfidf
    dims: int = 20
    nu: float = 0.5          # one-class SVM (normal design only)
    q: float = 0.7           # outlier threshold = q-quantile of held-out normal-message scores
    m: int = 2               # scored messages over which the outlier ratio is computed
    rho: float = 1.0         # drift when the outlier ratio >= rho

    @property
    def label(self) -> str:
        rep = "embedding" if self.rep == "emb" else "TF-IDF"
        base = f"{self.design} | {rep} d={self.dims} | q={self.q} m={self.m} rho={self.rho}"
        return base + (f" nu={self.nu}" if self.design == "normal" else "")


def _strata(convs: list[Conversation]) -> np.ndarray:
    """Annotated drift type: 0 = no drift, 1 = genuine drift, 2 = fraudulent drift."""
    return np.array([0 if first_pii(c) is None else (2 if c.label == 1 else 1) for c in convs])


def _folds(convs: list[Conversation], k: int, seed: int):
    k = min(k, int(np.bincount(_strata(convs)).min()))
    return StratifiedKFold(max(k, 2), shuffle=True, random_state=seed).split(np.zeros(len(convs)), _strata(convs))


class NormalOCDD:
    def __init__(self, cfg: NormalOCDDConfig, seed: int = 0, calib_folds: int = 5):
        self.cfg, self.seed, self.calib_folds = cfg, seed, calib_folds

    # ---------------------------------------------------------------- scoring
    def _fit_reference(self, convs: list[Conversation], vecs: dict[int, np.ndarray]):
        if self.cfg.design == "opening":
            return None
        X = np.vstack([vecs[c.conv_id][normal_indices(c)] for c in convs])
        sc = StandardScaler().fit(X)
        return sc, OneClassSVM(nu=self.cfg.nu, gamma="scale").fit(sc.transform(X))

    def _scores(self, ref, c: Conversation, V: np.ndarray) -> dict[int, float]:
        idx = monitored(c)
        if not idx:
            return {}
        if self.cfg.design == "opening":
            return {t: 1.0 - float(V[t] @ V[0]) for t in idx}
        sc, oc = ref
        return dict(zip(idx, -oc.decision_function(sc.transform(V[idx]))))

    # ---------------------------------------------------------------- fit / detect
    def fit(self, convs: list[Conversation], vecs: dict[int, np.ndarray]) -> "NormalOCDD":
        """Reference on all `convs`; threshold from out-of-fold scores of their normal monitored messages."""
        held = []
        for a, b in _folds(convs, self.calib_folds, self.seed):
            ref = self._fit_reference([convs[i] for i in a], vecs)
            for i in b:
                c = convs[i]
                norm = set(normal_indices(c))
                held += [s for t, s in self._scores(ref, c, vecs[c.conv_id]).items() if t in norm]
        self.threshold_ = float(np.quantile(held, self.cfg.q))
        self.ref_ = self._fit_reference(convs, vecs)
        return self

    def detect(self, c: Conversation, V: np.ndarray) -> dict:
        sc = self._scores(self.ref_, c, V)
        idx = sorted(sc)
        out = [sc[t] > self.threshold_ for t in idx]
        for k in range(len(idx)):
            last = out[max(0, k - self.cfg.m + 1):k + 1]
            if len(last) == self.cfg.m and np.mean(last) >= self.cfg.rho:
                return {"drift": True, "drift_at": idx[k], "n_scored": k + 1}
        return {"drift": False, "drift_at": None, "n_scored": len(idx)}


def evaluate_detector(det, convs: list[Conversation], vecs: dict[int, np.ndarray], tolerance: int = 2) -> pd.DataFrame:
    """Per-dialogue detections against FirstPII (same columns as dialogue_drift.evaluate)."""
    rows = []
    for c in convs:
        r = det.detect(c, vecs[c.conv_id])
        f = first_pii(c)
        delay = (r["drift_at"] - f) if (r["drift"] and f is not None) else np.nan
        rows.append({"conv_id": c.conv_id, "label": c.label, "has_drift": f is not None, "first_pii": f,
                     "detected": r["drift"], "drift_at": r["drift_at"], "delay": delay,
                     "localized": bool(r["drift"] and f is not None and 0 <= delay <= tolerance),
                     "early": bool(r["drift"] and f is not None and delay < 0), "n_msgs": len(c.messages)})
    return pd.DataFrame(rows)


class FixedMessageRule:
    """Trivial reference detector: a drift at the k-th message of every dialogue."""

    def __init__(self, k: int):
        self.k = k

    def detect(self, c: Conversation, V=None) -> dict:
        return ({"drift": True, "drift_at": self.k, "n_scored": 1} if self.k < len(c.messages)
                else {"drift": False, "drift_at": None, "n_scored": 0})


def select_on_train(train: list[Conversation], vec_sets: dict[tuple[str, int], dict[int, np.ndarray]], grid: dict,
                    seed: int = 0, outer_folds: int = 5, tolerance: int = 2) -> tuple[pd.DataFrame, NormalOCDDConfig]:
    """Nested cross-fitting on the TRAINING dialogues: every configuration is fitted on the other folds
    (its threshold calibrated inside them) and scored on the held-out fold. Selection = best pooled
    balanced score (ties: localization rate)."""
    rows = []
    outer = list(_folds(train, outer_folds, seed))
    for design, (rep, dims), q, (m, rho) in itertools.product(grid["design"], vec_sets, grid["q"], grid["m_rho"]):
        for nu in (grid["nu"] if design == "normal" else [0.0]):
            cfg = NormalOCDDConfig(design=design, rep=rep, dims=dims, nu=nu, q=q, m=m, rho=rho)
            V = vec_sets[(rep, dims)]
            parts = []
            for a, b in outer:
                det = NormalOCDD(cfg, seed).fit([train[i] for i in a], V)
                parts.append(evaluate_detector(det, [train[i] for i in b], V, tolerance))
            s = score(pd.concat(parts))
            rows.append({**asdict(cfg), "label": cfg.label, **{f"train_{k}": v for k, v in s.items()}})
    tab = pd.DataFrame(rows).sort_values(["train_balanced", "train_localized_rate"], ascending=False)
    b = tab.iloc[0]
    best = NormalOCDDConfig(design=b.design, rep=b.rep, dims=int(b.dims), nu=float(b.nu), q=float(b.q),
                            m=int(b.m), rho=float(b.rho))
    return tab, best
