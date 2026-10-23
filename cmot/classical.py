"""Phase-1 benchmark: supervised classifiers, clustering-as-classification and the
majority-vote ensemble.

Evaluation design:
  * hyper-parameters are tuned by stratified inner CV on the training partition;
  * clustering methods are fitted on the training partition and cluster ids are
    mapped to classes by training-set majority (DBSCAN noise is mapped
    explicitly; test points are assigned by nearest core sample);
  * ensemble members are either a fixed set (kNN, SVM, ImpKmeans) or the top-3
    by *inner-CV* score -- never by test score.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import View
from .impkmeans import ImpKMeans

try:
    from xgboost import XGBClassifier  # optional

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# --------------------------------------------------------------------------- clustering wrappers
class DBSCANPredictor(BaseEstimator):
    """DBSCAN with out-of-sample assignment: a new point joins the cluster of its
    nearest core sample if it lies within eps, otherwise it is noise (-1)."""

    def __init__(self, eps: float | str = "auto", min_samples: int = 4):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        ms = min(self.min_samples, len(X) - 1)
        if self.eps == "auto":
            d, _ = NearestNeighbors(n_neighbors=ms + 1).fit(X).kneighbors(X)
            self.eps_ = float(np.median(d[:, -1]))
        else:
            self.eps_ = float(self.eps)
        db = DBSCAN(eps=self.eps_, min_samples=ms).fit(X)
        self.labels_ = db.labels_
        core = db.core_sample_indices_
        self.core_X_ = X[core] if len(core) else X[:0]
        self.core_labels_ = db.labels_[core] if len(core) else np.array([], dtype=int)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if len(self.core_X_) == 0:
            return np.full(len(X), -1)
        d, idx = NearestNeighbors(n_neighbors=1).fit(self.core_X_).kneighbors(X)
        lab = self.core_labels_[idx[:, 0]]
        lab[d[:, 0] > self.eps_] = -1
        return lab


class ClusterClassifier(BaseEstimator, ClassifierMixin):
    """Turns a clusterer into a classifier via training-set majority mapping.
    predict_proba returns the training malicious rate of the assigned cluster."""

    def __init__(self, clusterer=None, scale: bool = True):
        self.clusterer = clusterer
        self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        Xs = self.scaler_.transform(X) if self.scale else X
        self.model_ = clone(self.clusterer).fit(Xs)
        cl = getattr(self.model_, "labels_", None)
        if cl is None:
            cl = self.model_.predict(Xs)
        self.prior_ = float(y.mean())
        self.rate_ = {}
        for c in np.unique(cl):
            self.rate_[int(c)] = float(y[cl == c].mean())
        if -1 not in self.rate_:  # unseen noise at test time -> global prior
            self.rate_[-1] = self.prior_
        # Cluster -> class mapping. With exactly two (non-noise) clusters a one-to-one
        # (Hungarian) assignment is used: the cluster with the higher malicious rate is
        # 'malicious', the other 'benign'. Majority mapping would send both clusters to
        # the majority class whenever the training prior is skewed (60% here).
        clusters = [c for c in self.rate_ if c != -1 and c in set(cl.tolist())]
        if len(clusters) == 2:
            hi = max(clusters, key=lambda c: self.rate_[c])
            self.map_ = {c: int(c == hi) for c in clusters}
        else:
            self.map_ = {c: int(r >= 0.5) for c, r in self.rate_.items() if c != -1}
        self.map_[-1] = int(self.prior_ >= 0.5)
        return self

    def _clusters(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X) if self.scale else X
        return self.model_.predict(Xs)

    def predict_proba(self, X):
        p1 = np.array([self.rate_.get(int(c), self.prior_) for c in self._clusters(X)])
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return np.array([self.map_.get(int(c), self.map_[-1]) for c in self._clusters(X)], dtype=int)


# --------------------------------------------------------------------------- model zoo
@dataclass
class ModelDef:
    name: str
    estimator: object
    grid: dict
    family: str  # "supervised" | "clustering"
    uses_sparse: bool = False


def model_zoo(seed: int, view: View) -> list[ModelDef]:
    sc = lambda est: Pipeline([("scaler", StandardScaler()), ("clf", est)])  # noqa: E731
    zoo = [
        ModelDef("kNN", sc(KNeighborsClassifier()),
                 {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]}, "supervised"),
        ModelDef("SVM", sc(SVC(probability=True, class_weight="balanced", random_state=seed)),
                 {"clf__kernel": ["rbf", "linear"], "clf__C": [0.1, 1.0, 10.0]}, "supervised"),
        ModelDef("Random Forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1),
                 {"max_depth": [None, 6], "max_features": ["sqrt", 0.3]}, "supervised"),
    ]
    if view.S_train is not None:
        zoo.append(ModelDef("Naive Bayes", ComplementNB(), {"alpha": [0.1, 0.5, 1.0]}, "supervised", uses_sparse=True))
    else:
        zoo.append(ModelDef("Naive Bayes", GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3]}, "supervised"))
    if HAS_XGB:
        zoo.append(ModelDef("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.9,
                                                     eval_metric="logloss", random_state=seed, n_jobs=1),
                            {"max_depth": [2, 4], "colsample_bytree": [0.5, 1.0]}, "supervised"))
    else:
        zoo.append(ModelDef("XGBoost*", HistGradientBoostingClassifier(random_state=seed),
                            {"max_depth": [2, 4], "learning_rate": [0.05, 0.1]}, "supervised"))
    zoo += [
        ModelDef("k-means", ClusterClassifier(KMeans(n_clusters=2, n_init=10, random_state=seed)),
                 {"clusterer__n_clusters": [2]}, "clustering"),
        ModelDef("DBSCAN", ClusterClassifier(DBSCANPredictor()),
                 {"clusterer__min_samples": [3, 5]}, "clustering"),
        ModelDef("GMM", ClusterClassifier(GaussianMixture(n_components=2, covariance_type="diag", random_state=seed)),
                 {"clusterer__n_components": [2]}, "clustering"),
        ModelDef("ImpKmeans", ClusterClassifier(ImpKMeans(n_clusters=2, random_state=seed)),
                 {"clusterer__n_clusters": [2], "clusterer__r": [0.3, 0.6, 1.0],
                  "clusterer__bandwidth": [0.05, 0.2]}, "clustering"),
    ]
    return zoo


@dataclass
class FitResult:
    name: str
    view: str
    family: str
    pred: np.ndarray
    score: np.ndarray  # P(malicious)
    inner_cv_score: float
    best_params: dict
    fit_s: float
    predict_s_per_conv: float


def fit_predict(md: ModelDef, view: View, y_train: np.ndarray, seed: int, inner_folds: int = 5,
                n_jobs: int = -1) -> FitResult:
    Xtr = view.S_train if md.uses_sparse else view.X_train
    Xte = view.S_test if md.uses_sparse else view.X_test
    n_min = int(np.bincount(y_train, minlength=2).min())
    cv = StratifiedKFold(n_splits=max(2, min(inner_folds, n_min)), shuffle=True, random_state=seed)
    gs = GridSearchCV(md.estimator, md.grid, scoring="f1_macro", cv=cv, n_jobs=n_jobs, refit=True, error_score=0.0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    score = gs.predict_proba(Xte)[:, 1]
    pred = gs.predict(Xte)
    pred_s = (time.perf_counter() - t1) / max(1, Xte.shape[0])
    return FitResult(md.name, view.name, md.family, np.asarray(pred, dtype=int), np.asarray(score, dtype=float),
                     float(gs.best_score_), gs.best_params_, fit_s, pred_s)


def majority_vote(results: list[FitResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred, soft score, unanimity flag). Three binary votes => no ties."""
    votes = np.vstack([r.pred for r in results])
    pred = (votes.mean(0) > 0.5).astype(int)
    soft = np.vstack([r.score for r in results]).mean(0)
    unanimous = (votes.min(0) == votes.max(0))
    return pred, soft, unanimous


def select_members(results: list[FitResult], strategy: str, fixed: list[str], k: int = 3) -> list[FitResult]:
    if strategy == "fixed":
        by_name = {r.name: r for r in results}
        missing = [m for m in fixed if m not in by_name]
        if missing:
            raise ValueError(f"Ensemble members not available: {missing}")
        return [by_name[m] for m in fixed]
    ranked = sorted(results, key=lambda r: r.inner_cv_score, reverse=True)  # inner-CV only
    return ranked[:k]
