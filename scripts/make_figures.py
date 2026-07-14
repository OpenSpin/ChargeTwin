"""Render the figures embedded in README.md.

    python scripts/make_figures.py

Writes PNGs to docs/figures/. Re-run after changing the data or the models.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chargetwin as ct
from chargetwin.data import DATA_ROOT

FIG_ROOT = DATA_ROOT.parent / "docs" / "figures"
DATASET = "rho5e10"
SEED = 0
DPI = 130


def save(fig, name: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def corner(raw, gauss, pca) -> None:
    """Raw vs. method A vs. method B, across every pair of parameters."""
    g = ct.pairplot_compare(
        raw,
        {"Covariance": gauss.sample(500, seed=SEED), "PCA": pca.sample(500, seed=SEED)},
        height=1.7,
    )
    save(g.figure, "corner_raw_vs_models.png")


def modes(raw, pca) -> None:
    """Scree plot + what the three dominant disorder modes actually do."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ct.plot_explained_variance(pca, ax=axes[0])
    axes[0].set_title("Three modes carry 88% of the variance")
    ct.plot_mode_loadings(pca, n_modes=3, ax=axes[1])
    axes[1].set_title("What each mode does to the device")
    fig.tight_layout()
    save(fig, "disorder_modes.png")


def tunnel_coupling_log_space(raw, gauss) -> None:
    """Why the coupling is modelled as log10(2 t_c) and not t_c."""
    linear = ct.GaussianModel.fit(ct.load_dataset(DATASET, ["d", "tc"])).sample(20_000, seed=SEED)
    logged = ct.add_tunnel_coupling(gauss.sample(20_000, seed=SEED))
    truth = ct.add_tunnel_coupling(raw)["tc"]

    bins = np.linspace(-60, 300, 90)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (df, title) in zip(
        axes,
        [(linear, r"Gaussian fitted to linear $t_c$"), (logged, r"Gaussian fitted to $\log_{10} 2t_c$")],
    ):
        ax.hist(truth, bins=bins, density=True, color="tab:blue", alpha=0.55, label="raw (COMSOL)")
        ax.hist(df["tc"], bins=bins, density=True, color="tab:red", alpha=0.5, label="generated")
        ax.axvline(0, color="k", lw=1.5, ls="--")
        frac = (df["tc"] < 0).mean()
        ax.set_title(f"{title}\n{frac:.1%} of devices have $t_c < 0$")
        ax.set_xlabel(r"$t_c$   [$\mu$eV]")
        ax.legend()
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "log_tunnel_coupling.png")


def yield_curves(raw, gauss, pca) -> None:
    """Fraction of cooldowns landing inside the paper's tunability window."""
    # arXiv:2510.13578 Sec. III A: tunnel gap 2*t_c within 10-250 ueV, and the
    # plunger correction needed to re-symmetrize the dot below 20 mV, which the
    # lever arm turns into |eps| < 1.2 meV. (The paper's barrier-height cut is
    # never the binding constraint, so it is left out.)
    spec = {"tc": (5.0, 125.0), "eps": (-1.2, 1.2)}

    def in_spec(df, s=spec):
        ok = pd.Series(True, index=df.index)
        for p, (lo, hi) in s.items():
            ok &= df[p].between(lo, hi)
        return ok

    n = 20_000
    sources = {
        "raw (COMSOL, 500)": ct.add_tunnel_coupling(raw),
        "Covariance (A)": ct.add_tunnel_coupling(gauss.sample(n, seed=SEED)),
        "PCA (B)": ct.add_tunnel_coupling(pca.sample(n, seed=SEED)),
    }
    colors = ["tab:blue", "tab:red", "goldenrod"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    eps_max = np.linspace(0.2, 4.0, 60)
    for (name, df), c in zip(sources.items(), colors):
        y = [in_spec(df, {**spec, "eps": (-e, e)}).mean() for e in eps_max]
        axes[0].plot(eps_max, y, label=name, color=c, lw=2)
    axes[0].axvline(1.2, ls="--", c="gray", lw=1.2)
    axes[0].text(1.28, 0.06, "paper's cut\n(20 mV)", fontsize=9, color="gray")
    axes[0].set_xlabel(r"tolerated detuning $|\epsilon|$   [meV]")
    axes[0].set_ylabel("fraction of cooldowns in spec")
    axes[0].set_title("Yield vs. how tight the spec is")
    axes[0].legend(loc="lower right")

    rows = []
    for ds, tex in [("rho5e9", r"$5\times10^{9}$"), ("rho5e10", r"$5\times10^{10}$")]:
        model = ct.GaussianModel.fit(ct.load_dataset(ds, ct.PAPER_PARAMETERS))
        s = ct.add_tunnel_coupling(model.sample(n, seed=SEED))
        rows.append((tex, in_spec(s).mean()))
    axes[1].bar(
        [r[0] for r in rows],
        [r[1] for r in rows],
        color=["lightsteelblue", "indianred"],
        edgecolor="k",
        width=0.55,
    )
    for i, (_, v) in enumerate(rows):
        axes[1].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("fraction of cooldowns in spec")
    axes[1].set_xlabel(r"interface charge density   [cm$^{-2}$]")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Ten times the charge, a quarter as many usable cooldowns")

    fig.tight_layout()
    save(fig, "yield.png")


def main() -> None:
    raw = ct.load_dataset(DATASET, ct.PAPER_PARAMETERS)
    gauss = ct.GaussianModel.fit(raw)
    pca = ct.PCAModel.fit(raw, n_components=3)

    corner(raw, gauss, pca)
    modes(raw, pca)
    tunnel_coupling_log_space(raw, gauss)
    yield_curves(raw, gauss, pca)


if __name__ == "__main__":
    main()
