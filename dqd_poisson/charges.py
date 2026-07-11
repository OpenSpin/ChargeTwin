"""Trapped interface charges (disorder source), spec §6.

Charges live on the semiconductor-oxide interface plane (Si-cap/SiO2).
Nodal delta sources are explicitly disallowed by the spec (1/r singularities
pollute FV/FEM); both supported modes here produce a bounded, mesh-convergent
RHS by construction: continuum density is a plain per-cell surface charge,
and discrete defects are each smeared over ~1 interface cell as a Gaussian.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import geometry as geo
from .materials import E_CHARGE


def interface_z_index(grid: geo.GridSpec, cfg: geo.GeometryConfig) -> int:
    z_nm = grid.z_centers / geo.NM
    return int(np.argmin(np.abs(z_nm - geo.interface_z_nm(cfg))))


def rhs_continuum_sheet(grid: geo.GridSpec, cfg: geo.GeometryConfig, sigma_per_m2: float) -> np.ndarray:
    """Mode 1: uniform surface charge density sigma [C/m^2] over the whole
    interface plane -- an FV source in the interface cells (spec §6.1)."""
    k = interface_z_index(grid, cfg)
    Nx, Ny, Nz = grid.shape
    dx = np.diff(grid.x_nodes)
    dy = np.diff(grid.y_nodes)
    b = np.zeros((Nx, Ny, Nz))
    cell_area = dx[:, None] * dy[None, :]
    b[:, :, k] = sigma_per_m2 * cell_area
    return b


def sample_discrete_defects(cfg: geo.GeometryConfig, rho_per_cm2: float, rng: np.random.Generator,
                             sign: Optional[np.ndarray] = None):
    """Sample N = rho * A_interface point-defect positions uniformly over
    the 660x582 nm^2 footprint (spec §6.2, e.g. rho=5e10 cm^-2 -> ~190
    charges). Returns (x_nm, y_nm, charge_sign) arrays, charge_sign in
    {-1,+1} (occupied/unoccupied trap), uniform +1 (all occupied, i.e.
    negatively charged traps) unless `sign` distribution is supplied."""
    area_cm2 = (cfg.Lx_nm * 1e-7) * (cfg.Ly_nm * 1e-7)  # nm^2 -> cm^2
    N = int(round(rho_per_cm2 * area_cm2))
    x_nm = rng.uniform(0, cfg.Lx_nm, N) + cfg.lateral_pad_nm
    y_nm = rng.uniform(0, cfg.Ly_nm, N) + cfg.lateral_pad_nm
    if sign is None:
        sign = np.ones(N)
    return x_nm, y_nm, sign, N


def rhs_discrete_defects(grid: geo.GridSpec, cfg: geo.GeometryConfig,
                          x_nm: np.ndarray, y_nm: np.ndarray, sign: np.ndarray,
                          smear_sigma_nm: float) -> np.ndarray:
    """Mode 2: each defect is one elementary charge `e`, regularized as a
    Gaussian sheet of width `smear_sigma_nm` in-plane, smeared over ~1
    interface cell so the RHS stays bounded and mesh-convergent (spec §6.2).
    """
    k = interface_z_index(grid, cfg)
    Nx, Ny, Nz = grid.shape
    dx = np.diff(grid.x_nodes)
    dy = np.diff(grid.y_nodes)
    xc_nm = grid.x_centers / geo.NM
    yc_nm = grid.y_centers / geo.NM

    sigma_m = smear_sigma_nm * geo.NM
    b_plane = np.zeros((Nx, Ny))
    for xd, yd, s in zip(x_nm, y_nm, sign):
        gx = np.exp(-0.5 * ((xc_nm - xd) * geo.NM / sigma_m) ** 2)
        gy = np.exp(-0.5 * ((yc_nm - yd) * geo.NM / sigma_m) ** 2)
        weight = np.outer(gx, gy)
        norm = np.sum(weight * dx[:, None] * dy[None, :])
        if norm > 0:
            b_plane += s * E_CHARGE * weight / norm
    b = np.zeros((Nx, Ny, Nz))
    b[:, :, k] = b_plane
    return b
