"""Generative models of DQD parameter fluctuations.

Two ways to turn a measured disorder ensemble into an unlimited stream of
synthetic devices:

* :class:`GaussianModel` (method A) -- multivariate normal with the empirical
  mean and full covariance matrix. Keeps every pairwise correlation, needs
  ``p(p+1)/2`` numbers.
* :class:`PCAModel` (method B) -- keep only the ``k`` dominant principal
  components of the standardized data, sample independent Gaussians along them
  and map back. Keeps ``k*p`` numbers and reproduces the disorder modes of
  arXiv:2510.13578.

Both fit on a DataFrame of physical parameters, sample DataFrames back, and
round-trip through a plain ``.npz`` so a notebook can reuse a fit without
touching the raw ensemble.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DisorderModel:
    """Common interface: ``fit`` -> ``sample`` / ``cooldowns`` -> ``save``."""

    kind: str = "base"
    parameters: list[str]

    def sample(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        """Draw ``n_samples`` disorder realizations.

        A realization is a fresh placement of the trapped interface charge, so
        one draw is equally "another device from the same wafer" and "the same
        device after another cooldown" -- both re-randomize the charge from
        scratch, and the model does not distinguish them.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, kind=self.kind, parameters=np.array(self.parameters), **self._arrays())
        return path

    @staticmethod
    def load(path: str | Path) -> "DisorderModel":
        with np.load(path, allow_pickle=False) as f:
            kind = str(f["kind"])
            params = [str(p) for p in f["parameters"]]
            cls = {"gaussian": GaussianModel, "pca": PCAModel}[kind]
            return cls._from_arrays(params, dict(f))

    def _arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError


class GaussianModel(DisorderModel):
    """Method A: multivariate normal with the empirical covariance matrix."""

    kind = "gaussian"

    def __init__(self, parameters: list[str], mean: np.ndarray, cov: np.ndarray):
        self.parameters = list(parameters)
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "GaussianModel":
        return cls(list(df.columns), df.mean().to_numpy(), df.cov().to_numpy())

    def sample(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x = rng.multivariate_normal(self.mean, self.cov, size=n_samples)
        return pd.DataFrame(x, columns=self.parameters)

    @property
    def correlation(self) -> pd.DataFrame:
        sd = np.sqrt(np.diag(self.cov))
        return pd.DataFrame(self.cov / np.outer(sd, sd), index=self.parameters, columns=self.parameters)

    def _arrays(self):
        return {"mean": self.mean, "cov": self.cov}

    @classmethod
    def _from_arrays(cls, parameters, arrays):
        return cls(parameters, arrays["mean"], arrays["cov"])


class PCAModel(DisorderModel):
    """Method B: sample the ``n_components`` dominant disorder modes only.

    The data are standardized (zero mean, unit variance per parameter) before
    the PCA, so the modes are not dominated by whichever parameter happens to
    carry the largest physical units. Sub-dominant components are set to zero
    rather than sampled, which is what makes this a *reduced* model: the
    generated cloud lives on a ``n_components``-dimensional subspace.
    """

    kind = "pca"

    def __init__(
        self,
        parameters: list[str],
        mean: np.ndarray,
        scale: np.ndarray,
        components: np.ndarray,
        explained_variance: np.ndarray,
        score_mean: np.ndarray,
        score_std: np.ndarray,
        n_components: int,
    ):
        self.parameters = list(parameters)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.components = np.asarray(components, dtype=float)  # (p, p), rows are modes
        self.explained_variance = np.asarray(explained_variance, dtype=float)
        self.score_mean = np.asarray(score_mean, dtype=float)
        self.score_std = np.asarray(score_std, dtype=float)
        self.n_components = int(n_components)

    @classmethod
    def fit(cls, df: pd.DataFrame, n_components: int = 3) -> "PCAModel":
        x = df.to_numpy(dtype=float)
        n = x.shape[0]
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        z = (x - mean) / scale
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        # Sign convention: largest-magnitude loading of each mode is positive,
        # so a refitted model is comparable to a stored one.
        for i in range(vt.shape[0]):
            if vt[i, np.argmax(np.abs(vt[i]))] < 0:
                vt[i] *= -1
        scores = z @ vt.T
        return cls(
            parameters=list(df.columns),
            mean=mean,
            scale=scale,
            components=vt,
            explained_variance=s**2 / (n - 1),
            score_mean=scores.mean(axis=0),
            score_std=scores.std(axis=0),
            n_components=n_components,
        )

    def sample(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        k = self.n_components
        scores = np.zeros((n_samples, self.components.shape[0]))
        scores[:, :k] = rng.normal(self.score_mean[:k], self.score_std[:k], size=(n_samples, k))
        z = scores @ self.components
        x = z * self.scale + self.mean
        return pd.DataFrame(x, columns=self.parameters)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Project real data onto the disorder modes (scores)."""
        z = (df[self.parameters].to_numpy(dtype=float) - self.mean) / self.scale
        return z @ self.components.T

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.explained_variance / self.explained_variance.sum()

    @property
    def loadings(self) -> pd.DataFrame:
        """Mode-vs-parameter matrix; row ``i`` is disorder mode ``i+1``."""
        idx = [f"PC{i + 1}" for i in range(self.components.shape[0])]
        return pd.DataFrame(self.components, index=idx, columns=self.parameters)

    def with_n_components(self, n_components: int) -> "PCAModel":
        """Same fit, different truncation -- no refit needed."""
        return PCAModel(
            self.parameters,
            self.mean,
            self.scale,
            self.components,
            self.explained_variance,
            self.score_mean,
            self.score_std,
            n_components,
        )

    def _arrays(self):
        return {
            "mean": self.mean,
            "scale": self.scale,
            "components": self.components,
            "explained_variance": self.explained_variance,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "n_components": np.array(self.n_components),
        }

    @classmethod
    def _from_arrays(cls, parameters, arrays):
        return cls(
            parameters,
            arrays["mean"],
            arrays["scale"],
            arrays["components"],
            arrays["explained_variance"],
            arrays["score_mean"],
            arrays["score_std"],
            int(arrays["n_components"]),
        )
