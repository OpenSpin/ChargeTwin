"""DotPotential hand-off object, gate-response superposition, and the
ensemble driver (spec §5.1, §8, §10 api.py)."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as geo
from . import charges as ch
from . import reduce as rd
from .operator import build_transmissibilities
from .solver import Operator, build_mg_hierarchy, make_mg_preconditioner, pcg, solve_pyamg, solve_direct

GATE_NAMES = ("PL", "PR", "barrier", "screening")


@dataclass
class Problem:
    """Everything that depends only on geometry/permittivity -- assembled
    once and reused for every gate-voltage combination and every disorder
    realization (spec §5.1)."""
    cfg: geo.GeometryConfig
    grid: geo.GridSpec
    eps_r: np.ndarray
    gate_masks: Dict[str, np.ndarray]
    dirichlet_mask: np.ndarray
    operator: Operator
    mg_levels: list = field(repr=False)
    precond: Callable = field(repr=False)


def build_problem(cfg: geo.GeometryConfig) -> Problem:
    grid = geo.build_grid(cfg)
    eps_r = geo.eps_r_field(grid, cfg)
    gate_masks = geo.gate_masks(grid, cfg)
    dirichlet_mask = np.zeros(grid.shape, dtype=bool)
    for m in gate_masks.values():
        dirichlet_mask |= m
    # far-field ground (spec §2): pins the deep substrate, removes the
    # pure-Neumann near-nullspace. Value 0, so it does not enter any gate
    # unit response beyond enforcing phi=0 at the far boundary.
    ground = geo.ground_mask(grid, cfg)
    ground = ground & ~dirichlet_mask  # gates win where they overlap the edge
    dirichlet_mask |= ground

    Tx, Ty, Tz = build_transmissibilities(eps_r, grid.x_nodes, grid.y_nodes, grid.z_nodes)
    operator = Operator(Tx, Ty, Tz, jnp.asarray(dirichlet_mask))

    levels = build_mg_hierarchy(eps_r, grid.x_nodes, grid.y_nodes, grid.z_nodes, dirichlet_mask)
    precond = make_mg_preconditioner(levels)

    return Problem(cfg, grid, eps_r, gate_masks, dirichlet_mask, operator, levels, precond)


def solve_gate_unit_responses(problem: Problem, tol=1e-10, maxiter=2000,
                              backend="jax") -> Dict[str, np.ndarray]:
    """One solve per gate (only 4), operator reused (spec §5.1: 'Precompute
    the unit gate responses... one solve per gate').

    backend='jax'   -> matrix-free PCG + geometric z-line/semicoarsening MG
                       (fast on uniform grids; can stall on strongly graded
                       production grids).
    backend='pyamg' -> assembled-matrix algebraic MG (spec §5.2 reliable CPU
                       baseline; robust on graded/anisotropic grids)."""
    responses = {}
    for g in GATE_NAMES:
        if g not in problem.gate_masks:
            continue
        mask = np.asarray(problem.gate_masks[g])
        b = np.where(mask, 1.0, 0.0)
        if backend == "pyamg":
            phi, info = solve_pyamg(problem.operator, b, tol=tol, return_info=True)
        elif backend == "direct":
            phi = solve_direct(problem.operator, b)
            info = {"iters": 0, "res": 0.0}
        else:
            phi, info = pcg(problem.operator, jnp.asarray(b), precond=problem.precond,
                            tol=tol, maxiter=maxiter)
        responses[g] = np.array(phi)
        print(f"gate {g}: iters={info['iters']} rel_res={info['res']:.2e}", flush=True)
    return responses


def superpose(gate_response: Dict[str, np.ndarray], voltages: Dict[str, float]) -> np.ndarray:
    """phi(V) = sum_g V_g * phi_g_unit -- no new Poisson solve (spec §5.1)."""
    phi = None
    for g, v in voltages.items():
        if g not in gate_response:
            continue
        term = v * gate_response[g]
        phi = term if phi is None else phi + term
    return phi


def lever_arm_slice(problem: Problem, gate_response: Dict[str, np.ndarray],
                     dot_xy_nm, gate: str) -> float:
    """d(phi_dot)/dV_gate, evaluated at a fixed z (well mid-plane) -- an
    interim electrostatics-only diagnostic. Spec §8 notes a fixed-z slice
    gets lever arms roughly right but should be replaced by the z-projected
    V_2D once reduce.py exists (this is not yet the final lever arm)."""
    grid = problem.grid
    x_nm = grid.x_centers / geo.NM
    y_nm = grid.y_centers / geo.NM
    z_nm = grid.z_centers / geo.NM
    i = np.argmin(np.abs(x_nm - dot_xy_nm[0]))
    j = np.argmin(np.abs(y_nm - dot_xy_nm[1]))
    k = np.argmin(np.abs(z_nm - geo.well_z_center_nm(problem.cfg)))
    return float(gate_response[gate][i, j, k])


def lever_arm_projected(problem: Problem, gate_response: Dict[str, np.ndarray],
                         voltages: Dict[str, float], dot_xy_nm, gate: str) -> float:
    """Proper energy lever arm alpha_g = dU_dot/dV_gate [eV/V] using the
    z-projected potential (spec §8): with the vertical mode psi_z0 fixed at
    the operating point (Hellmann-Feynman, first order), alpha_g =
    -<psi_z0| phi_g^unit(x_dot,y_dot,.) |psi_z0>_z. This is the value spec §8
    says a fixed-z slice only approximates."""
    grid = problem.grid
    x_nm = grid.x_centers / geo.NM
    y_nm = grid.y_centers / geo.NM
    i = np.argmin(np.abs(x_nm - dot_xy_nm[0]))
    j = np.argmin(np.abs(y_nm - dot_xy_nm[1]))

    # vertical mode at operating point, restricted to the well neighborhood
    phi_op = superpose(gate_response, voltages)
    u_band = geo.u_band_field(grid, problem.cfg)
    z_m = grid.z_centers
    zc_nm = z_m / geo.NM
    # window around the well (well +/- ~20 nm) to avoid the far Dirichlet walls
    wc = geo.well_z_center_nm(problem.cfg)
    zmask = (zc_nm > wc - 25) & (zc_nm < wc + 25)
    zk = np.where(zmask)[0]
    u_col = rd.u_total_along_z(phi_op[i, j, zk], u_band[i, j, zk])
    _, psis = rd.solve_1d_schrodinger_z(z_m[zk], u_col, n_states=1)
    w = psis[0] ** 2 * np.gradient(z_m[zk])
    w = w / w.sum()
    return -float(np.sum(w * gate_response[gate][i, j, zk]))


def locate_dot(problem: Problem, gate_response: Dict[str, np.ndarray],
               voltages: Dict[str, float], side: str = "left",
               z_window_nm: float = 25.0):
    """Find the lateral position of a dot as the minimum of the z-projected
    confinement potential V_2D at the operating point (the electron localizes
    at the potential minimum). `side` picks the left/right search half. Returns
    (x_nm, y_nm) of the minimum within the finger y-band.

    This is the physically meaningful place to read the lever arm: alpha =
    phi_g^unit(x_dot). Because x_dot itself shifts with gate voltage, the
    energy lever arm alpha=dE_L/dV is *not* strictly linear even though phi is
    -- sampling at the true dot rather than the fixed PL-finger centre is what
    accounts for most of that (spec §8)."""
    grid = problem.grid
    cfg = problem.cfg
    phi_op = superpose(gate_response, voltages)
    u_band = geo.u_band_field(grid, cfg)
    wc = geo.well_z_center_nm(cfg)
    V_2D, _, _, _, _ = rd.project_to_2d(
        phi_op, u_band, grid,
        z_window=((wc - z_window_nm) * geo.NM, (wc + z_window_nm) * geo.NM))

    x_nm = grid.x_centers / geo.NM
    y_nm = grid.y_centers / geo.NM
    xc = cfg.lateral_pad_nm + cfg.Lx_nm / 2.0
    yc = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0
    y0 = yc - cfg.d_channel_nm / 2.0
    y1 = yc + cfg.d_channel_nm / 2.0
    xmask = (x_nm < xc) if side == "left" else (x_nm > xc)
    ymask = (y_nm >= y0) & (y_nm <= y1)
    region = np.full(V_2D.shape, np.inf)
    ii, jj = np.where(xmask[:, None] & ymask[None, :])
    region[ii, jj] = V_2D[ii, jj]
    i, j = np.unravel_index(np.argmin(region), region.shape)
    return float(x_nm[i]), float(y_nm[j])


def vertical_field_well(problem: Problem, phi: np.ndarray, xy_nm) -> float:
    """F_z = -dphi/dz at the well, at lateral position xy_nm (spec §3.4, §8)."""
    grid = problem.grid
    x_nm = grid.x_centers / geo.NM
    y_nm = grid.y_centers / geo.NM
    z_nm = grid.z_centers
    i = np.argmin(np.abs(x_nm - xy_nm[0]))
    j = np.argmin(np.abs(y_nm - xy_nm[1]))
    k = np.argmin(np.abs(grid.z_centers / geo.NM - geo.well_z_center_nm(problem.cfg)))
    dphidz = (phi[i, j, k + 1] - phi[i, j, k - 1]) / (z_nm[k + 1] - z_nm[k - 1])
    return -dphidz


@dataclass
class DotPotential:
    xy_grid: tuple
    V_2D: np.ndarray
    psi_z0: Optional[np.ndarray]
    U0: Optional[float]
    F_z: np.ndarray
    m_t: float
    gate_response: Dict[str, np.ndarray]
    meta: dict


# ---------------------------------------------------------------------------
# Ensemble driver (spec §5.1, §7): each disorder realization is one solve
# with the *same* operator -- batch over the leading axis and vmap the
# (matrix-free, jit-able) CG solve. Uses jax's built-in CG rather than the
# custom PCG in solver.py because jax.lax-based control flow is required to
# vmap/jit through a data-dependent iteration count; the Dirichlet-identity
# rows keep gate cells pinned automatically (residual/search direction are
# exactly zero there from the first iteration), so no extra masking needed.
# ---------------------------------------------------------------------------

def _make_batched_solver(problem: Problem, tol=1e-8, maxiter=1000):
    op = problem.operator
    diag = op.diag()

    def M(r):
        return r / diag

    def solve_one(b):
        x0 = jnp.where(op.dirichlet_mask, b, 0.0)
        x, _ = jax.scipy.sparse.linalg.cg(op.apply, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        return x

    return jax.jit(jax.vmap(solve_one))


def solve_disorder_ensemble(problem: Problem, b_batch: np.ndarray, tol=1e-8, maxiter=1000) -> np.ndarray:
    """b_batch: (n_realizations, Nx,Ny,Nz) charge RHS (spec §6). Returns
    phi_charge for each realization, one batched jit+vmap solve reusing the
    same operator/preconditioner diagonal for all of them (spec §5.1)."""
    solver = _make_batched_solver(problem, tol=tol, maxiter=maxiter)
    return np.array(solver(jnp.asarray(b_batch)))


def generate_disorder_ensemble(problem: Problem, n_realizations: int, rho_per_cm2: float,
                                smear_sigma_nm: float = 2.0, seed: int = 0) -> np.ndarray:
    """Sample n_realizations independent discrete-defect charge sheets
    (spec §6.2) and return the stacked RHS batch, (n_realizations,Nx,Ny,Nz)."""
    rng = np.random.default_rng(seed)
    batch = []
    for _ in range(n_realizations):
        x_nm, y_nm, sign, _ = ch.sample_discrete_defects(problem.cfg, rho_per_cm2, rng)
        b = ch.rhs_discrete_defects(problem.grid, problem.cfg, x_nm, y_nm, sign, smear_sigma_nm)
        batch.append(b)
    return np.stack(batch, axis=0)


def build_dot_potential(problem: Problem, gate_response: Dict[str, np.ndarray],
                         voltages: Dict[str, float], phi_charge: Optional[np.ndarray] = None,
                         disorder_seed: Optional[int] = None, rho_per_cm2: Optional[float] = None,
                         per_column: bool = False) -> DotPotential:
    """Assemble the hand-off object (spec §8): phi = superposition of gate
    unit responses + (optional) disorder potential, then z-project."""
    phi = superpose(gate_response, voltages)
    if phi_charge is not None:
        phi = phi + phi_charge

    u_band = geo.u_band_field(problem.grid, problem.cfg)
    V_2D, F_z, psi_z0, U0, z_sub = rd.project_to_2d(phi, u_band, problem.grid, per_column=per_column)

    grid = problem.grid
    xy_grid = (grid.x_centers / geo.NM, grid.y_centers / geo.NM)
    meta = {"voltages": voltages, "disorder_seed": disorder_seed, "rho_per_cm2": rho_per_cm2,
            "grid_shape": grid.shape}
    from .materials import M_T
    return DotPotential(xy_grid, V_2D, psi_z0, U0, F_z, M_T, gate_response, meta)


def save_dot_potential(path: str, dot: DotPotential):
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("x_nm", data=dot.xy_grid[0])
        f.create_dataset("y_nm", data=dot.xy_grid[1])
        f.create_dataset("V_2D", data=dot.V_2D)
        f.create_dataset("F_z", data=dot.F_z)
        if dot.psi_z0 is not None:
            f.create_dataset("psi_z0", data=dot.psi_z0)
        for g, resp in dot.gate_response.items():
            f.create_dataset(f"gate_response/{g}", data=resp)
        f.attrs["U0_eV"] = dot.U0 if dot.U0 is not None else float("nan")
        f.attrs["m_t"] = dot.m_t
        for k, v in dot.meta.items():
            if v is None:
                continue
            try:
                f.attrs[f"meta_{k}"] = v if not isinstance(v, dict) else str(v)
            except TypeError:
                f.attrs[f"meta_{k}"] = str(v)
