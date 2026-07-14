"""Plots for comparing raw and generated ensembles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import labels

PALETTE = {"Original": "tab:blue", "Covariance": "tab:red", "PCA": "goldenrod"}


def pairplot_compare(
    raw: pd.DataFrame,
    generated: dict[str, pd.DataFrame],
    height: float = 1.9,
    alpha: float = 0.45,
    with_units: bool = True,
):
    """Corner plot of raw vs generated ensembles (Fig. 5 of arXiv:2510.13578).

    ``generated`` maps a label to a sampled frame, e.g.
    ``{"Covariance": gauss.sample(500), "PCA": pca.sample(500)}``.
    """
    parameters = list(raw.columns)
    frames = [raw.assign(source="Original")]
    for name, df in generated.items():
        frames.append(df[parameters].assign(source=name))
    combined = pd.concat(frames, ignore_index=True)
    order = ["Original", *generated.keys()]

    g = sns.pairplot(
        combined,
        vars=parameters,
        hue="source",
        hue_order=order,
        palette={k: PALETTE.get(k, f"C{i}") for i, k in enumerate(order)},
        kind="scatter",
        diag_kind="kde",
        plot_kws=dict(alpha=alpha, s=12, linewidth=0),
        diag_kws=dict(fill=True, bw_adjust=1.0, linewidth=1.6, common_norm=False),
        corner=True,
        height=height,
    )

    tex = labels(parameters, with_units=with_units)
    n = len(parameters)
    for i in range(n):
        for j in range(i + 1):
            ax = g.axes[i, j]
            if ax is None:
                continue
            ax.set_xlabel(tex[j] if i == n - 1 else "")
            ax.set_ylabel(tex[i] if j == 0 else "")
    if g._legend is not None:
        g._legend.remove()
    g.axes[0, 0].legend(
        handles=[plt.Line2D([], [], color=PALETTE.get(k, f"C{i}"), lw=6, label=k) for i, k in enumerate(order)],
        title="Source",
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        frameon=False,
    )
    return g


def plot_explained_variance(pca_model, ax=None):
    """Scree plot: variance captured per disorder mode, and cumulatively."""
    ax = ax or plt.gca()
    ratio = pca_model.explained_variance_ratio
    x = np.arange(1, len(ratio) + 1)
    ax.bar(x, ratio, color="lightsteelblue", edgecolor="k", label="per mode")
    ax.plot(x, np.cumsum(ratio), "o-", color="crimson", label="cumulative")
    ax.axvline(pca_model.n_components + 0.5, ls="--", c="gray")
    ax.set_xlabel("Disorder mode")
    ax.set_ylabel("Fraction of total variance")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.02)
    ax.legend()
    return ax


def plot_mode_loadings(pca_model, n_modes: int | None = None, ax=None):
    """Heat map of which physical parameters each disorder mode moves."""
    ax = ax or plt.gca()
    n_modes = n_modes or pca_model.n_components
    load = pca_model.loadings.iloc[:n_modes]
    sns.heatmap(
        load.T,
        ax=ax,
        cmap="bwr",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws=dict(label="loading"),
    )
    ax.set_yticklabels(labels(list(load.columns), with_units=False), rotation=0)
    ax.set_xlabel("Disorder mode")
    return ax
