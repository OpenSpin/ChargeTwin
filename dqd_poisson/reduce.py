"""Vertical (z) reduction and Schrodinger hand-off, spec §8.

Adiabatic separation: solve the 1D effective-mass Schrodinger equation
along z (at the dot center, or per (x,y) column), then project the 3D
potential onto that ground vertical mode to get V_2D(x,y). A fixed-z slice
is explicitly NOT sufficient (spec §8) -- it gets lever arms roughly right
but misses confinement energies.
"""

from typing import Optional, Tuple

import numpy as np

from . import geometry as geo
from .materials import E_CHARGE, M_T

HBAR = 1.054571817e-34


def u_total_along_z(phi_col: np.ndarray, u_band_col: np.ndarray) -> np.ndarray:
    """U(z) = -e*phi + U_band, in Joules. phi in volts, u_band in eV."""
    return -E_CHARGE * phi_col + u_band_col * E_CHARGE


def solve_1d_schrodinger_z(z_m: np.ndarray, u_z: np.ndarray, m_t: float = M_T,
                            n_states: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference 1D effective-mass Schrodinger eq along z on a
    (possibly non-uniform) grid, via a symmetrized tridiagonal generalized
    eigenproblem. Returns (energies[n_states], psi[n_states, Nz]),
    psi normalized so that sum(|psi|^2 * dz) = 1."""
    Nz = len(z_m)
    dz = np.diff(z_m)
    # non-uniform 3-point stencil for d/dz (1/m_t) d/dz, via FV-style
    # harmonic treatment analogous to operator.py (m_t constant here, so
    # this reduces to the standard non-uniform 3-point Laplacian).
    H = np.zeros((Nz, Nz))
    for i in range(1, Nz - 1):
        d_m, d_p = dz[i - 1], dz[i]
        h = HBAR ** 2 / (2 * m_t)
        c_m = h * 2.0 / (d_m * (d_m + d_p))
        c_p = h * 2.0 / (d_p * (d_m + d_p))
        H[i, i - 1] += -c_m
        H[i, i] += c_m + c_p
        H[i, i + 1] += -c_p
    H += np.diag(u_z)
    # hard-wall (Dirichlet, psi=0) at both ends -- hand-off boundary; the
    # true insulator hard wall is applied by the Schrodinger stage per §2.
    H = H[1:-1, 1:-1]
    evals, evecs = np.linalg.eigh(H)
    energies = evals[:n_states]
    psis = np.zeros((n_states, Nz))
    for n in range(n_states):
        psi = np.zeros(Nz)
        psi[1:-1] = evecs[:, n]
        norm = np.sqrt(np.sum(psi ** 2 * np.gradient(z_m)))
        psis[n] = psi / norm
    return energies, psis


def project_to_2d(phi: np.ndarray, u_band: np.ndarray, grid: geo.GridSpec,
                   z_window: Optional[Tuple[float, float]] = None,
                   per_column: bool = False, m_t: float = M_T):
    """V_2D(x,y) = <psi_z0| U(x,y,.) |psi_z0>_z (spec §8 eq.), plus F_z(x,y)
    and (psi_z0, U0) at the dot center for records.

    z_window restricts the 1D solve to the well+barrier neighborhood (the
    stiff-confinement region); default is the full z column, which is
    fine but slower/more sensitive to the far Dirichlet walls -- restrict
    once the well location is confirmed against the figure (spec §11)."""
    Nx, Ny, Nz = grid.shape
    z_m = grid.z_centers
    if z_window is not None:
        mask = (z_m >= z_window[0]) & (z_m <= z_window[1])
        z_idx = np.where(mask)[0]
    else:
        z_idx = np.arange(Nz)
    z_sub = z_m[z_idx]

    ic, jc = Nx // 2, Ny // 2
    u_col0 = u_total_along_z(phi[ic, jc, z_idx], u_band[ic, jc, z_idx])
    E0, psis0 = solve_1d_schrodinger_z(z_sub, u_col0, m_t=m_t, n_states=1)
    psi_z0 = psis0[0]
    U0 = E0[0]

    dz_sub = np.gradient(z_sub)
    weight = psi_z0 ** 2 * dz_sub  # sum(weight) == 1

    if per_column:
        V_2D = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                u_col = u_total_along_z(phi[i, j, z_idx], u_band[i, j, z_idx])
                E_ij, psis_ij = solve_1d_schrodinger_z(z_sub, u_col, m_t=m_t, n_states=1)
                w_ij = psis_ij[0] ** 2 * dz_sub
                V_2D[i, j] = np.sum(w_ij * u_col) / E_CHARGE
    else:
        u_field = -E_CHARGE * phi[:, :, z_idx] + u_band[:, :, z_idx] * E_CHARGE
        V_2D = np.tensordot(u_field, weight, axes=([2], [0])) / E_CHARGE  # eV

    # F_z(x,y): central difference of phi at the well index (closest to psi_z0 peak)
    k_peak = z_idx[np.argmax(psi_z0 ** 2)]
    k0 = max(1, min(Nz - 2, k_peak))
    F_z = -(phi[:, :, k0 + 1] - phi[:, :, k0 - 1]) / (z_m[k0 + 1] - z_m[k0 - 1])

    return V_2D, F_z, psi_z0, U0 / E_CHARGE, z_sub
