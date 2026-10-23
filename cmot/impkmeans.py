"""Adapter around the official ImpKmeans package (Şenol, 2024; `pip install impkmeans`,
https://github.com/senolali/ImpKMeans).

The algorithm itself (KDE mode seeking + KD-tree radius suppression for the
initial centroids, followed by k-means with max_iter=1) is used unchanged from
the package. The adapter only adds what an out-of-sample evaluation needs:

  * the PCA projection and Min-Max scaling are fitted on the training
    partition and *re-used* for test points (the package's `predict` refits a
    new MinMaxScaler on whatever data it receives, which would place test
    points in a different coordinate system than the centroids);
  * KDE runs in a low-dimensional PCA space (default 10 dims), matching the
    dimensionality reduction of the pipeline: with bandwidth 0.05 in a 100+-dim
    unit cube every point's density is dominated by its own kernel, so the
    mode search degenerates;
  * a scikit-learn compatible interface so that `r` and `bandwidth` can be
    tuned by inner cross-validation.
"""

from __future__ import annotations

import numpy as np
from impkmeans import ImpKMeans as _PkgImpKMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class ImpKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: int = 2, r: float = 0.6, bandwidth: float = 0.05, pca_dims: int = 10,
                 random_state: int | None = 0):
        self.n_clusters = n_clusters
        self.r = r
        self.bandwidth = bandwidth
        self.pca_dims = pca_dims
        self.random_state = random_state

    def _transform(self, X) -> np.ndarray:
        Z = self.pca_.transform(np.asarray(X, dtype=float)) if self.pca_ is not None else np.asarray(X, float)
        return self.scaler_.transform(Z)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        dims = min(self.pca_dims, X.shape[1], X.shape[0] - 1)
        self.pca_ = PCA(n_components=dims, random_state=self.random_state).fit(X) if dims < X.shape[1] else None
        Z = self.pca_.transform(X) if self.pca_ is not None else X
        self.scaler_ = MinMaxScaler().fit(Z)
        Zn = self.scaler_.transform(Z)
        # Zn is already in [0, 1]; the package's internal MinMaxScaler is then an identity map.
        self.model_ = _PkgImpKMeans(k=self.n_clusters, r=self.r, bandwidth=self.bandwidth,
                                    random_state=self.random_state).fit(Zn)
        self.cluster_centers_ = self.model_.centers_
        self.init_centroids_ = self.model_.initial_points_
        self.labels_ = self.model_.labels_
        return self

    def predict(self, X):
        Zn = self._transform(X)
        d = np.linalg.norm(Zn[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(d, axis=1)
