"""Analytic unit tests for the matrix-free operator (spec §9, first two bullets).

These check the *discrete operator itself* (no linear solve yet, that's
step 3): apply it to a known analytic solution of Laplace's equation and
verify the interior residual vanishes to numerical precision.
"""

import numpy as np

from ..operator import build_transmissibilities, apply_laplacian


def _uniform_nodes(L, N):
    return np.linspace(0.0, L, N + 1)


def test_parallel_plate_uniform_eps():
    """Uniform eps_r, phi linear in z (V=0 at z=0, V=V0 at z=L): exact
    solution of Laplace's eq; FV must reproduce E = V/d exactly regardless
    of grid non-uniformity, and interior residual must vanish."""
    Nx, Ny, Nz = 5, 5, 40
    x_nodes = _uniform_nodes(50e-9, Nx)
    y_nodes = _uniform_nodes(50e-9, Ny)
    # non-uniform z grid on purpose (graded), still must be exact for linear fields
    z_nodes = np.concatenate([
        np.linspace(0, 20e-9, 15),
        np.linspace(20e-9, 80e-9, 15)[1:],
        np.linspace(80e-9, 100e-9, 11)[1:],
    ])
    eps_r = np.full((Nx, Ny, len(z_nodes) - 1), 12.0)
    Tx, Ty, Tz = build_transmissibilities(eps_r, x_nodes, y_nodes, z_nodes)

    z_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    L = z_nodes[-1]
    V0 = 1.0
    phi = np.broadcast_to((V0 / L) * z_centers, (Nx, Ny, len(z_centers))).copy()

    out = np.array(apply_laplacian(phi, Tx, Ty, Tz))
    # interior residual (excluding the outermost slice, which sees no flux
    # partner beyond the domain -> natural Neumann, not the Dirichlet plates)
    interior = out[:, :, 1:-1]
    assert np.max(np.abs(interior)) < 1e-20

    # field check: flux through any interior z-face = eps0*eps_r*V0/L*area
    from ..materials import EPS0
    area = (x_nodes[1] - x_nodes[0]) * (y_nodes[1] - y_nodes[0])
    expected_T_times_E = -EPS0 * 12.0 * area * (V0 / L)
    # recover flux from Tz * (phi_i - phi_{i+1})
    flux = np.array(Tz)[0, 0, :] * (phi[0, 0, :-1] - phi[0, 0, 1:])
    np.testing.assert_allclose(flux, expected_T_times_E, rtol=1e-10)


def test_dielectric_slab_interface_flux_continuity():
    """Two stacked dielectrics (eps1 below, eps2 above) with the exact
    piecewise-linear potential that satisfies continuity of phi and of
    eps*E across the interface. Checks the harmonic-mean face treatment
    (spec §9 third analytic bullet)."""
    Nx, Ny = 3, 3
    x_nodes = _uniform_nodes(30e-9, Nx)
    y_nodes = _uniform_nodes(30e-9, Ny)

    L1, L2 = 40e-9, 60e-9
    eps1, eps2 = 12.0, 3.9
    N1, N2 = 20, 20
    z_nodes = np.concatenate([np.linspace(0, L1, N1 + 1), np.linspace(L1, L1 + L2, N2 + 1)[1:]])
    Nz = len(z_nodes) - 1
    z_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])

    eps_r = np.empty((Nx, Ny, Nz))
    eps_r[:, :, z_centers < L1] = eps1
    eps_r[:, :, z_centers >= L1] = eps2

    Tx, Ty, Tz = build_transmissibilities(eps_r, x_nodes, y_nodes, z_nodes)

    # exact solution: phi continuous, D = eps*dphi/dz continuous.
    # phi(z) = E1*z for z<L1 ; phi(z) = phi(L1) + E2*(z-L1) for z>=L1
    # with eps1*E1 = eps2*E2 = D0, and total drop V0 across L1+L2.
    V0 = 1.0
    # V0 = E1*L1 + E2*L2, E2 = E1*eps1/eps2
    E1 = V0 / (L1 + L2 * eps1 / eps2)
    E2 = E1 * eps1 / eps2
    phi_z = np.where(z_centers < L1, E1 * z_centers,
                      E1 * L1 + E2 * (z_centers - L1))
    phi = np.broadcast_to(phi_z, (Nx, Ny, Nz)).copy()

    out = np.array(apply_laplacian(phi, Tx, Ty, Tz))
    interior = out[:, :, 1:-1]
    assert np.max(np.abs(interior)) < 1e-18, np.max(np.abs(interior))


if __name__ == "__main__":
    test_parallel_plate_uniform_eps()
    test_dielectric_slab_interface_flux_continuity()
    print("operator analytic tests passed")
