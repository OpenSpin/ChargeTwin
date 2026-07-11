"""Lateral 2D effective-mass Schrodinger solver (spec §8, out-of-scope stub
now implemented). First pass uses a fixed-z slice of the 3D potential (the
"2D approx" -- simpler but, per spec §8, gets lever arms roughly right while
underestimating confinement energies; project_to_2d in reduce.py is the more
faithful z-projected potential and plugs into the same solver here).

In-plane effective mass is isotropic (m_t, transverse mass) for [001]-grown
Si valleys -- no anisotropy tensor needed for this 2D lateral problem (spec
§3.3, §8). The device edges are the far Poisson Dirichlet walls, not the
oxide -- for the lateral problem we instead crop to a window around the DQD
and impose a hard wall (psi=0) at the crop boundary, which stands in for the
"U -> infinity in the insulator" confinement (spec §1 hint) since the
electron's amplitude is negligible there anyway.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import E_CHARGE, M_T

HBAR = 1.054571817e-34


def _second_derivative_1d(x_m: np.ndarray) -> np.ndarray:
    """Non-uniform 3-point second-derivative matrix, Dirichlet (psi=0) at
    both ends. Returns the (N-2)x(N-2) operator acting on interior points."""
    N = len(x_m)
    dx = np.diff(x_m)
    H = np.zeros((N, N))
    for i in range(1, N - 1):
        d_m, d_p = dx[i - 1], dx[i]
        c_m = 2.0 / (d_m * (d_m + d_p))
        c_p = 2.0 / (d_p * (d_m + d_p))
        H[i, i - 1] += c_m
        H[i, i] += -(c_m + c_p)
        H[i, i + 1] += c_p
    return H[1:-1, 1:-1]


def _hamiltonian_2d(x_m: np.ndarray, y_m: np.ndarray, U_eV: np.ndarray, m_t: float):
    """Sparse FD Hamiltonian (Joules) on the interior (Nx-2)x(Ny-2) grid,
    hard wall (psi=0) at the box edges, plus the quadrature cell-area array
    used for inner products sum(psi_a*psi_b*area) ~ integral psi_a psi_b."""
    Nx, Ny = len(x_m), len(y_m)
    assert U_eV.shape == (Nx, Ny)

    Lx = _second_derivative_1d(x_m)
    Ly = _second_derivative_1d(y_m)
    Ix = sp.identity(Nx - 2)
    Iy = sp.identity(Ny - 2)

    kinetic_coeff = HBAR ** 2 / (2 * m_t)
    T = -kinetic_coeff * (sp.kron(sp.csr_matrix(Lx), Iy) + sp.kron(Ix, sp.csr_matrix(Ly)))

    U_interior = U_eV[1:-1, 1:-1] * E_CHARGE  # eV -> J
    V = sp.diags(U_interior.flatten())

    dx = np.gradient(x_m)
    dy = np.gradient(y_m)
    cell_area = np.outer(dx[1:-1], dy[1:-1])
    return (T + V).tocsc(), cell_area, U_interior


def solve_lateral_2d(x_m: np.ndarray, y_m: np.ndarray, U_eV: np.ndarray,
                      m_t: float = M_T, n_states: int = 6,
                      tol: float = 1e-13, ncv: int = None):
    """Solve -hbar^2/(2m_t) (d2/dx2 + d2/dy2) psi + U(x,y) psi = E psi on a
    non-uniform (x,y) grid, hard wall (psi=0) at the box edges.

    x_m, y_m: 1D node coordinates (metres). U_eV: (Nx,Ny) potential energy
    (eV). Returns (energies_eV (n_states,), psis (n_states,Nx,Ny) normalized
    so sum(|psi|^2 dx dy) = 1, psi=0 on the boundary row/column).

    Note: for a near-degenerate cluster (e.g. the two lowest DQD bonding/
    antibonding states when 2*t_c is tiny), ARPACK shift-invert can return a
    duplicated Ritz vector instead of resolving the true splitting -- use
    `tunnel_coupling_two_level` instead for tunnel-coupling estimates."""
    Nx, Ny = len(x_m), len(y_m)
    H, cell_area, U_interior = _hamiltonian_2d(x_m, y_m, U_eV, m_t)

    sigma = U_interior.min() - 1e-6 * E_CHARGE
    if ncv is None:
        ncv = max(4 * n_states + 1, 40)
    evals, evecs = spla.eigsh(H, k=n_states, sigma=sigma, which="LM", tol=tol, ncv=ncv)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    psis = np.zeros((n_states, Nx, Ny))
    for n in range(n_states):
        psi_int = evecs[:, n].reshape(Nx - 2, Ny - 2)
        norm = np.sqrt(np.sum(psi_int ** 2 * cell_area))
        psis[n, 1:-1, 1:-1] = psi_int / norm

    return evals / E_CHARGE, psis


def tunnel_coupling_two_level(x_m: np.ndarray, y_m: np.ndarray, U_eV: np.ndarray,
                               x_split_m: float, m_t: float = M_T, wall_eV: float = 10.0):
    """Hund-Mulliken two-level extraction of the tunnel splitting, robust to
    the ARPACK near-degeneracy failure mode in `solve_lateral_2d`: solve the
    left/right dot in isolation (hard wall at x_split_m, imposed by adding a
    large potential on the other side), then build the 2x2 matrix of the
    *real* (unmasked) Hamiltonian in this localized-orbital basis and
    diagonalize the generalized eigenproblem.

    Returns dict with t_c_eV (= splitting/2), splitting_eV (bonding-
    antibonding gap, i.e. 2*t_c), E_L0_eV, E_R0_eV (isolated-dot ground
    energies), S_LR (orbital overlap -- should be << 1 for the two-level
    approximation to be trustworthy)."""
    x_idx_split = np.searchsorted(x_m, x_split_m)

    U_L = U_eV.copy()
    U_L[x_idx_split:, :] += wall_eV
    U_R = U_eV.copy()
    U_R[:x_idx_split, :] += wall_eV

    (E_L0,), psis_L = solve_lateral_2d(x_m, y_m, U_L, m_t=m_t, n_states=1)
    (E_R0,), psis_R = solve_lateral_2d(x_m, y_m, U_R, m_t=m_t, n_states=1)
    psi_L, psi_R = psis_L[0], psis_R[0]

    H, cell_area, _ = _hamiltonian_2d(x_m, y_m, U_eV, m_t)
    Nx, Ny = len(x_m), len(y_m)

    def apply_H(psi):
        out = np.zeros((Nx, Ny))
        Hpsi_int = (H @ psi[1:-1, 1:-1].flatten()).reshape(Nx - 2, Ny - 2)
        out[1:-1, 1:-1] = Hpsi_int
        return out

    H_psiL = apply_H(psi_L)
    H_psiR = apply_H(psi_R)

    def inner(a, b):
        return float(np.sum(a[1:-1, 1:-1] * b[1:-1, 1:-1] * cell_area))

    H_LL = inner(psi_L, H_psiL) / E_CHARGE
    H_RR = inner(psi_R, H_psiR) / E_CHARGE
    H_LR = inner(psi_L, H_psiR) / E_CHARGE
    S_LR = inner(psi_L, psi_R)

    Hmat = np.array([[H_LL, H_LR], [H_LR, H_RR]])
    Smat = np.array([[1.0, S_LR], [S_LR, 1.0]])
    import scipy.linalg as la
    evals, _ = la.eigh(Hmat, Smat)
    splitting_eV = float(evals[1] - evals[0])

    return {
        "t_c_eV": splitting_eV / 2,
        "splitting_eV": splitting_eV,
        "E_L0_eV": E_L0,
        "E_R0_eV": E_R0,
        "S_LR": S_LR,
    }


def crop_window(x_nm: np.ndarray, y_nm: np.ndarray, U_2D: np.ndarray,
                 x_center_nm: float, y_center_nm: float,
                 x_window_nm: float, y_window_nm: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the full-device (x_nm,y_nm,U_2D) to a window around the DQD, so
    the lateral solve only covers the relevant region (spec §8 interface is
    defined over the full footprint, but production-grid arrays are too
    large for a dense per-column crop -- restrict to where the electron
    actually lives)."""
    xmask = np.abs(x_nm - x_center_nm) <= x_window_nm / 2
    ymask = np.abs(y_nm - y_center_nm) <= y_window_nm / 2
    return x_nm[xmask], y_nm[ymask], U_2D[np.ix_(xmask, ymask)]
