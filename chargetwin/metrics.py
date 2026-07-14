"""How well does a generated ensemble reproduce the raw one?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def marginal_summary(raw: pd.DataFrame, generated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-parameter mean, std and two-sample KS distance against the raw data.

    ``ks`` is the KS statistic (0 = identical marginals); ``ks_p`` its p-value,
    so a small p-value means the generated marginal is distinguishable from the
    raw one.
    """
    rows = []
    for p in raw.columns:
        row = {"parameter": p, "raw_mean": raw[p].mean(), "raw_std": raw[p].std()}
        for name, df in generated.items():
            ks = ks_2samp(raw[p].to_numpy(), df[p].to_numpy())
            row[f"{name}_mean"] = df[p].mean()
            row[f"{name}_std"] = df[p].std()
            row[f"{name}_ks"] = ks.statistic
            row[f"{name}_ks_p"] = ks.pvalue
        rows.append(row)
    return pd.DataFrame(rows).set_index("parameter")


def covariance_error(raw: pd.DataFrame, generated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Relative Frobenius error of the generated covariance and correlation matrices."""

    def corr(df):
        c = df.cov().to_numpy()
        sd = np.sqrt(np.diag(c))
        return c / np.outer(sd, sd)

    cov_raw, corr_raw = raw.cov().to_numpy(), corr(raw)
    rows = []
    for name, df in generated.items():
        rows.append(
            {
                "model": name,
                "cov_rel_error": np.linalg.norm(df.cov().to_numpy() - cov_raw) / np.linalg.norm(cov_raw),
                "corr_rel_error": np.linalg.norm(corr(df) - corr_raw) / np.linalg.norm(corr_raw),
            }
        )
    return pd.DataFrame(rows).set_index("model")
