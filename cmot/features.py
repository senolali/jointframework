"""Feature views. Every transformation is fitted on the training partition only
(inductive), or on all texts without labels (transductive)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class View:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    S_train: sparse.spmatrix | None = None  # non-negative sparse view (for multinomial-type NB)
    S_test: sparse.spmatrix | None = None
    info: dict | None = None


def tfidf_svd_view(texts_train: list[str], texts_test: list[str], cfg: dict, seed: int,
                   scope: str = "inductive") -> View:
    """TF-IDF + truncated SVD (LSA); TruncatedSVD is the sparse-safe equivalent of the PCA
    step of the pipeline.

    scope="inductive":    vocabulary, IDF and SVD are fitted on the training texts only.
    scope="transductive": they are fitted on training + test *texts* (never labels), the
                          transductive setting. It is
                          legitimate only when declared, since test documents shape the
                          representation."""
    vec = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
        min_df=cfg.get("min_df", 2) if len(texts_train) >= 20 else 1,
        max_features=cfg.get("max_features", 20000),
        stop_words="english",
        strip_accents="unicode",
    )
    fit_texts = texts_train if scope == "inductive" else texts_train + texts_test
    vec.fit(fit_texts)
    S_tr, S_te = vec.transform(texts_train), vec.transform(texts_test)
    S_fit = vec.transform(fit_texts)
    n_comp = int(min(cfg.get("svd_components", 10), S_fit.shape[0] - 1, S_fit.shape[1] - 1))
    svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=seed).fit(S_fit)
    X_tr = normalize(svd.transform(S_tr))
    X_te = normalize(svd.transform(S_te))
    return View(f"TF-IDF+SVD ({scope})", X_tr, X_te, S_tr, S_te,
                {"vocab": len(vec.vocabulary_), "svd_components": svd.n_components,
                 "explained_var": float(svd.explained_variance_ratio_.sum())})


def embedding_view(E: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, name: str) -> View:
    """Pre-computed sentence embeddings (unsupervised, per document => no label leakage)."""
    E = normalize(E)
    return View(name, E[train_idx], E[test_idx], info={"dim": E.shape[1]})


def build_views(texts: list[str], emb: np.ndarray | None, train_idx: np.ndarray, test_idx: np.ndarray,
                cfg: dict, seed: int) -> list[View]:
    views = []
    tcfg = cfg.get("tfidf", {})
    if tcfg.get("enabled", True):
        for scope in tcfg.get("fit_scope", ["inductive"]):
            views.append(tfidf_svd_view([texts[i] for i in train_idx], [texts[i] for i in test_idx], tcfg, seed, scope))
    if emb is not None:
        views.append(embedding_view(emb, train_idx, test_idx, cfg.get("embedding", {}).get("label", "Embedding")))
    return views
