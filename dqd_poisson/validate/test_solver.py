"""Validate PCG (Jacobi and MG preconditioners) against the analytic
parallel-plate solution, and cross-check against pyamg on a small grid
(spec §5.2, §9)."""

import numpy as np
import jax.numpy as jnp

from ..operator import build_transmissibilities
from ..solver import (Operator, pcg, jacobi_preconditioner,
                       build_mg_hierarchy, make_mg_preconditioner, solve_pyamg)


def _parallel_plate_problem(Nx=6, Ny=6, Nz=30, V0=1.0, eps_r_val=12.0):
    x_nodes = np.linspace(0, 60e-9, Nx + 1)
    y_nodes = np.linspace(0, 60e-9, Ny + 1)
    z_nodes = np.linspace(0, 100e-9, Nz + 1)
    eps_r = np.full((Nx, Ny, Nz), eps_r_val)
    Tx, Ty, Tz = build_transmissibilities(eps_r, x_nodes, y_nodes, z_nodes)

    dmask = np.zeros((Nx, Ny, Nz), dtype=bool)
    dmask[:, :, 0] = True
    dmask[:, :, -1] = True

    b = np.zeros((Nx, Ny, Nz))
    b[:, :, -1] = V0

    op = Operator(Tx, Ty, Tz, jnp.asarray(dmask))
    # Dirichlet is imposed at the first/last CELL CENTERS (not the domain
    # edges), so the fair analytic reference is linear between those two
    # points, not between z=0 and z=z_nodes[-1].
    z_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    z0, z1 = z_centers[0], z_centers[-1]
    phi_line = V0 * (z_centers - z0) / (z1 - z0)
    phi_exact = np.broadcast_to(phi_line, (Nx, Ny, Nz))
    return op, jnp.asarray(b), phi_exact, (x_nodes, y_nodes, z_nodes, eps_r)


def test_pcg_jacobi_matches_analytic():
    op, b, phi_exact, _ = _parallel_plate_problem()
    phi, info = pcg(op, b, precond=jacobi_preconditioner(op), tol=1e-12, maxiter=2000)
    err = np.max(np.abs(np.array(phi) - phi_exact))
    assert err < 1e-9, (err, info)


def test_pcg_mg_matches_analytic_and_is_competitive():
    op, b, phi_exact, (xn, yn, zn, eps_r) = _parallel_plate_problem()
    dmask = np.array(op.dirichlet_mask)
    levels = build_mg_hierarchy(eps_r, xn, yn, zn, dmask)
    precond = make_mg_preconditioner(levels)

    phi, info_mg = pcg(op, b, precond=precond, tol=1e-12, maxiter=2000)
    err = np.max(np.abs(np.array(phi) - phi_exact))
    assert err < 1e-9, (err, info_mg)

    _, info_jac = pcg(op, b, precond=jacobi_preconditioner(op), tol=1e-12, maxiter=2000)
    print("MG iters:", info_mg["iters"], "Jacobi iters:", info_jac["iters"])
    assert info_mg["iters"] <= info_jac["iters"]


def test_pcg_matches_pyamg_small_grid():
    op, b, phi_exact, _ = _parallel_plate_problem(Nx=3, Ny=3, Nz=12)
    phi_pcg, _ = pcg(op, b, precond=jacobi_preconditioner(op), tol=1e-12, maxiter=2000)
    phi_amg = solve_pyamg(op, np.array(b))
    np.testing.assert_allclose(np.array(phi_pcg), phi_amg, atol=1e-4)


if __name__ == "__main__":
    test_pcg_jacobi_matches_analytic()
    print("jacobi-pcg ok")
    test_pcg_mg_matches_analytic_and_is_competitive()
    print("mg-pcg ok")
    test_pcg_matches_pyamg_small_grid()
    print("pyamg cross-check ok")
