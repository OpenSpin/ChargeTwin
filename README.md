# ChargeTwin

**Generate realistic disorder realizations of a Si/SiGe double quantum dot — as many as you want, in a second.**

Charge trapped at the semiconductor–oxide interface makes every quantum dot device different, and makes the *same* device different after every cooldown. ChargeTwin takes a finite-element ensemble of disordered devices, compresses it into a generative model, and hands you an unlimited stream of synthetic — but statistically faithful — devices.

Companion code for [*Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots*](https://arxiv.org/abs/2510.13578) (Samadi, Cywiński, Krzywda). The simulated device is specified in [`docs/device.md`](docs/device.md).

```python
import chargetwin as ct

raw = ct.load_dataset("rho5e10", ct.PAPER_PARAMETERS)   # 500 COMSOL realizations

gauss = ct.GaussianModel.fit(raw)                       # method A: full covariance
pca = ct.PCAModel.fit(raw, n_components=3)              # method B: 3 dominant modes

devices = gauss.sample(10_000, seed=0)                  # 10k synthetic realizations
devices = ct.add_tunnel_coupling(devices)               # log2tc -> t_c in ueV
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
| [`02_models_and_validation.ipynb`](notebooks/02_models_and_validation.ipynb) | The PCA. Fit methods A and B. Corner plot of raw vs. A vs. B. Quantify the trade-off. |
| [`03_cooling_rounds.ipynb`](notebooks/03_cooling_rounds.ipynb) | **The point.** Thousands of cooling rounds, from a precomputed or freshly-fitted model. Yield curves. Export. |

## One draw = one realization = one cooldown

Every thermal cycle re-traps the interface charge from scratch. So "another device off the same wafer" and "the same device after another cooldown" are the *same* draw from the *same* distribution, and there is one method for both: `model.sample(n, seed)`. The model deliberately carries no per-device memory. (If you believe some traps survive a warm-up cycle, that would be a different model — partial re-randomization between rounds — and it is not what this repo implements.)

## The two generative models

**A — Covariance (`GaussianModel`).** Multivariate normal with the empirical mean and full covariance matrix. Keeps every pairwise correlation; stores `p(p+1)/2` numbers. Reproduces the raw cloud to ~2% in covariance Frobenius norm.

**B — Dominant PCA (`PCAModel`).** Standardize, keep the `k` leading principal components, sample independent Gaussians along them, map back. Stores `k·p` numbers, and each mode is a *named physical distortion*:

| mode | variance (ρ = 5×10¹⁰ cm⁻²) | what it is |
|---|---|---|
| PC1 | 43% | **symmetric squeeze/stretch** — `d` ↑, `t_c` ↓, `⟨L_x⟩` ↓ together |
| PC2 | 33% | **asymmetric tilt** — `ε`, `ΔF_z`, `ΔL_x` together |
| PC3 | 12% | **common vertical shift** — dominated by `⟨F_z⟩` |

Three modes carry **88%** of all device-to-device variance. The cost of truncation is visible in the corner plot: method B collapses the residual scatter onto a 3D subspace.

Both models share one interface — `.fit()`, `.sample(n, seed)`, `.save()`, `.load()` — so switching method is a one-line change anywhere downstream.

### The tunnel coupling is modelled in log space

`t_c` depends exponentially on the inter-dot distance (WKB), so its distribution has a long tail. `PAPER_PARAMETERS` therefore carries `log2tc = log₁₀(2t_c)` rather than `t_c`: a Gaussian fitted to *linear* `t_c` reproduces the marginal badly and generates unphysical **negative** tunnel couplings. Call `ct.add_tunnel_coupling(df)` to get `t_c` in µeV back.

## Data

`data/raw/` — two COMSOL ensembles, one row per disorder realization:

| dataset | interface charge density | realizations |
|---|---|---|
| `rho5e9` | 5×10⁹ cm⁻² | 500 |
| `rho5e10` | 5×10¹⁰ cm⁻² | 500 |

Both are *post-tuning*: the plunger gates have been adjusted to symmetrize each disordered double well, so we study variability around a consistent operating point. `load_dataset` converts to physical units and derives the dot-level parameters; `ct.PARAMETERS` lists all 12 available. The loader reproduces Table I of the paper exactly, and a test pins it there.

`data/models/` — precomputed fits (`.npz`, a few kB each), so a notebook can generate devices without touching the raw data. Regenerate with:

```bash
python scripts/fit_models.py
```

## Layout

```
chargetwin/          data.py (load + derive), models.py (A & B), metrics.py, plots.py
data/raw/            COMSOL ensembles (.pkl)
data/models/         precomputed fits (.npz)
docs/device.md       the simulated device: geometry, gates, disorder, extraction
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
