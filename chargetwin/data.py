"""Loading of the raw COMSOL disorder ensembles and the derived DQD parameters.

Each raw ``.pkl`` holds a dict of 1D arrays; entry ``i`` of every array belongs to
disorder realization ``i`` (one simulated device / one cooldown).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_ROOT = DATA_ROOT / "raw"

HBAR = 1.0545718e-34  # J s
M_E = 9.10938356e-31  # kg
EV_PER_NM2_TO_J_PER_M2 = 1.60218e-19 * 1e18
M_EFF_SI = 0.19  # transverse effective mass in Si, units of m_e


@dataclass(frozen=True)
class Dataset:
    """A raw disorder ensemble."""

    name: str
    path: Path
    density: str  # interface charge density, cm^-2
    description: str


DATASETS: dict[str, Dataset] = {
    "rho1e10": Dataset(
        name="rho1e10",
        path=RAW_ROOT / "DQD_rho1e10_after_tuning.pkl",
        density=r"1e10 cm^-2",
        description="449 disorder realizations, after virtual-gate tuning.",
    ),
    "rho5e9": Dataset(
        name="rho5e9",
        path=RAW_ROOT / "DQD_rho5e9_after_tuning.pkl",
        density=r"5e9 cm^-2",
        description="500 disorder realizations, after virtual-gate tuning.",
    ),
}

# Physical parameters exposed to the user, with the unit they carry after
# `load_dataset` has done its conversions. These are the quantities the
# generative models (Gaussian / PCA) are fitted on. `unit_tex` is the mathtext
# spelling of `unit`, for axis labels.
PARAMETERS: dict[str, dict[str, str]] = {
    "d": dict(unit="nm", unit_tex="nm", latex=r"$d$", desc="Inter-dot distance"),
    "tcs": dict(unit="ueV", unit_tex=r"$\mu$eV", latex=r"$t_\mathrm{c}$", desc="Tunnel coupling"),
    "Lxavg": dict(unit="nm", unit_tex="nm", latex=r"$\langle L_x \rangle$", desc="Mean dot size along x"),
    "dLx": dict(unit="nm", unit_tex="nm", latex=r"$\Delta L_x$", desc="Dot-size asymmetry along x"),
    "Lyavg": dict(unit="nm", unit_tex="nm", latex=r"$\langle L_y \rangle$", desc="Mean dot size along y"),
    "dLy": dict(unit="nm", unit_tex="nm", latex=r"$\Delta L_y$", desc="Dot-size asymmetry along y"),
    "Favg": dict(unit="MV/m", unit_tex="MV/m", latex=r"$\langle F_z \rangle$", desc="Mean vertical electric field"),
    "dF": dict(unit="MV/m", unit_tex="MV/m", latex=r"$\Delta F_z$", desc="Vertical-field asymmetry"),
    "eps": dict(unit="meV", unit_tex="meV", latex=r"$\epsilon$", desc="Detuning at the tuned working point"),
    "Eavgs": dict(unit="meV", unit_tex="meV", latex=r"$\langle E \rangle$", desc="Mean orbital energy"),
    "Bhs": dict(unit="meV", unit_tex="meV", latex=r"$B_h$", desc="Tunnel barrier height"),
    "V_acs": dict(unit="a.u.", unit_tex="a.u.", latex=r"$V_\mathrm{ac}$", desc="AC drive lever arm"),
}

# The subset used for Fig. 5 of arXiv:2510.13578.
FIG5_PARAMETERS = ["d", "tcs", "Lxavg", "dLx", "Favg", "dF", "eps"]


def natural_length(a_x, a_y, m_eff: float = M_EFF_SI) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic confinement curvatures (eV/nm^2) -> dot length scales (nm)."""
    a_x = np.asarray(a_x) * EV_PER_NM2_TO_J_PER_M2
    a_y = np.asarray(a_y) * EV_PER_NM2_TO_J_PER_M2
    m = m_eff * M_E
    Lx = (HBAR**2 / (m * a_x)) ** 0.25 * 1e9
    Ly = (HBAR**2 / (m * a_y)) ** 0.25 * 1e9
    return Lx, Ly


def load_raw(name: str) -> dict[str, np.ndarray]:
    """Load a raw ensemble exactly as COMSOL post-processing wrote it."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    with open(DATASETS[name].path, "rb") as f:
        return pickle.load(f)


def load_dataset(name: str, parameters: list[str] | None = None) -> pd.DataFrame:
    """Load an ensemble and return the derived DQD parameters in physical units.

    One row per disorder realization. ``parameters`` selects columns; by default
    every entry of :data:`PARAMETERS` that the file supports is returned.
    """
    raw = load_raw(name)
    d = {k: np.asarray(v, dtype=float) for k, v in raw.items()}

    LxL, LxR = natural_length(d["axLs"], d["axRs"])
    LyL, LyR = natural_length(d["ayLs"], d["ayRs"])
    FzL, FzR = d["FzLs"] * 1e-6, d["FzRs"] * 1e-6  # V/m -> MV/m

    out = pd.DataFrame(
        {
            "d": d["d"],
            "tcs": d["tcs"],
            "Lxavg": (LxL + LxR) / 2,
            "dLx": LxL - LxR,
            "Lyavg": (LyL + LyR) / 2,
            "dLy": LyL - LyR,
            "Favg": (FzL + FzR) / 2,
            "dF": FzL - FzR,
            "eps": d["eps"] * 1e3,  # eV -> meV
            "Eavgs": d["Eavgs"] * 1e-6,
            "Bhs": d["Bhs"],
            "V_acs": d["V_acs"],
        }
    )
    if parameters is not None:
        missing = [p for p in parameters if p not in out.columns]
        if missing:
            raise KeyError(f"{name} cannot provide {missing}; available: {list(out.columns)}")
        out = out[parameters]
    return out


def available_parameters(name: str) -> list[str]:
    """Parameters that dataset ``name`` can supply."""
    return list(load_dataset(name).columns)


def labels(parameters: list[str], with_units: bool = True) -> list[str]:
    """Mathtext axis labels for a list of parameters."""
    out = []
    for p in parameters:
        meta = PARAMETERS.get(p, {})
        lab = meta.get("latex", p)
        unit = meta.get("unit_tex")
        if with_units and unit and unit != "a.u.":
            lab = f"{lab} ({unit})"
        out.append(lab)
    return out
