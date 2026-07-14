# ChargeTwin

**Generate realistic disorder realizations of a Si/SiGe double quantum dot — as many as you want, in a second.**

Charge trapped at the semiconductor–oxide interface makes every quantum dot device different, and makes the *same* device different after every cooldown. ChargeTwin takes a finite-element ensemble of disordered devices, compresses it into a generative model, and hands you an unlimited stream of synthetic — but statistically faithful — devices.

Companion code for [*Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots*](https://arxiv.org/abs/2510.13578) (Samadi, Cywiński, Krzywda).

```python
import chargetwin as ct

raw = ct.load_dataset("rho1e10", ct.FIG5_PARAMETERS)   # 449 COMSOL realizations

gauss = ct.GaussianModel.fit(raw)                      # method A: full covariance
pca = ct.PCAModel.fit(raw, n_components=3)             # method B: 3 dominant modes

devices = gauss.sample(10_000, seed=0)                 # 10k synthetic devices
rounds = pca.cooldowns(500, seed=0)                    # 500 cooldowns of one device
```

## Install

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Pure `numpy` / `pandas` / `scipy` — no `scikit-learn` needed.

## The notebooks

Run them in order; each stands alone.

| notebook | what it does |
|---|---|
| [`01_raw_data.ipynb`](notebooks/01_raw_data.ipynb) | Look at the raw COMSOL ensembles. Pick the parameters you care about. See that they are strongly correlated. |
| [`02_models_and_validation.ipynb`](notebooks/02_models_and_validation.ipynb) | The PCA. Fit methods A and B. Reproduce Fig. 5: raw vs. A vs. B. Quantify the trade-off. |
| [`03_cooling_rounds.ipynb`](notebooks/03_cooling_rounds.ipynb) | **The point.** Thousands of cooling rounds, from a precomputed or freshly-fitted model. Yield curves. Export. |

## The two generative models

Every thermal cycle re-traps the interface charge from scratch, so a cooldown is an i.i.d. draw from the disorder distribution. Two ways to draw:

**A — Covariance (`GaussianModel`).** Multivariate normal with the empirical mean and full covariance matrix. Keeps every pairwise correlation; stores `p(p+1)/2` numbers. Reproduces the raw cloud to ~2% in covariance Frobenius norm.

**B — Dominant PCA (`PCAModel`).** Standardize, keep the `k` leading principal components, sample independent Gaussians along them, map back. Stores `k·p` numbers, and each mode is a *named physical distortion*:

| mode | variance (ρ = 10¹⁰ cm⁻²) | what it is |
|---|---|---|
| PC1 | 44% | **symmetric squeeze/stretch** — `d`, `t_c`, `⟨L_x⟩` move together |
| PC2 | 32% | **asymmetric tilt** — `ΔL_x`, `ΔF_z`, `ε` move together |
| PC3 | 10% | **common vertical shift** — dominated by `⟨F_z⟩` |

Three modes carry **85%** of all device-to-device variance. The cost of truncation is visible in the corner plot: method B collapses the residual scatter onto a 3D subspace.

Both models share one interface — `.fit()`, `.sample(n, seed)`, `.cooldowns(n)`, `.save()`, `.load()` — so switching method is a one-line change anywhere downstream.

## Data

`data/raw/` — two COMSOL ensembles, one row per disorder realization:

| dataset | interface charge density | realizations |
|---|---|---|
| `rho1e10` | 1×10¹⁰ cm⁻² | 449 |
| `rho5e9` | 5×10⁹ cm⁻² | 500 |

`load_dataset` converts to physical units and derives the dot-level parameters (sizes from confinement curvatures; sums and differences between the left and right dot). `ct.PARAMETERS` lists all 12 available; `ct.FIG5_PARAMETERS` is the paper's subset `[d, tcs, Lxavg, dLx, Favg, dF, eps]`.

`data/models/` — precomputed fits (`.npz`, a few kB each), so a notebook can generate devices without touching the raw data. Regenerate with:

```bash
python scripts/fit_models.py
```

## Layout

```
chargetwin/          data.py (load + derive), models.py (A & B), metrics.py, plots.py
data/raw/            COMSOL ensembles (.pkl)
data/models/         precomputed fits (.npz)
notebooks/           01 raw data -> 02 models -> 03 cooling rounds
scripts/fit_models.py
tests/
```

## Citation

```bibtex
@article{samadi2025statistical,
  title={Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots},
  author={Samadi, Saeed and Cywi{\'n}ski, {\L}ukasz and Krzywda, Jan A},
  journal={arXiv preprint arXiv:2510.13578},
  year={2025}
}
```

MIT licensed.
