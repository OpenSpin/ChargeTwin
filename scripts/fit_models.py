"""Precompute the generative models so notebooks can skip the raw data.

    python scripts/fit_models.py

Writes one .npz per (dataset, method) into data/models/. Re-run after changing
the parameter set or adding a dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chargetwin import DATASETS, PAPER_PARAMETERS, GaussianModel, PCAModel, load_dataset
from chargetwin.data import DATA_ROOT

MODEL_ROOT = DATA_ROOT / "models"
N_COMPONENTS = 3


def main() -> None:
    for name in DATASETS:
        raw = load_dataset(name, PAPER_PARAMETERS)

        gauss = GaussianModel.fit(raw)
        gauss.save(MODEL_ROOT / f"{name}_gaussian.npz")

        pca = PCAModel.fit(raw, n_components=N_COMPONENTS)
        pca.save(MODEL_ROOT / f"{name}_pca{N_COMPONENTS}.npz")

        cum = pca.explained_variance_ratio[:N_COMPONENTS].sum()
        print(
            f"{name}: N={len(raw)}, {len(PAPER_PARAMETERS)} parameters, "
            f"top-{N_COMPONENTS} modes capture {cum:.1%} of the variance"
        )
    print(f"\nmodels written to {MODEL_ROOT}")


if __name__ == "__main__":
    main()
