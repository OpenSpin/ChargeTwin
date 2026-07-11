"""Solve-once linear solver: matrix-free PCG with a geometric-multigrid (MG)
V-cycle preconditioner, all JAX/`jit`-able (spec §5.2). `A` (operator +
Dirichlet mask) is fixed per study; only `b` changes across gate voltages
and disorder realizations (spec §5.1).

Dirichlet handling: cells under `dirichlet_mask` are frozen at their initial
value in `phi0` for the whole solve -- the operator gives them an identity
row (operator.apply_dirichlet) and the residual/search directions are
masked to zero there every iteration, so CG only ever updates the free
(non-Dirichlet) unknowns. This is the standard matrix-free way to impose
Dirichlet BCs without assembling/eliminating rows.
"""

from functools import partial
from typing import NamedTuple, Optional

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .operator import apply_operator, apply_laplacian


class Operator(NamedTuple):
    Tx: jnp.ndarray
    Ty: jnp.ndarray
    Tz: jnp.ndarray
    dirichlet_mask: jnp.ndarray  # bool (Nx,Ny,Nz)

    @property
    def shape(self):
        return self.dirichlet_mask.shape

    def apply(self, phi):
        return apply_operator(phi, self.Tx, self.Ty, self.Tz, self.dirichlet_mask)

    def diag(self):
        """Diagonal of A: sum of face transmissibilities touching each cell
        (identity, i.e. 1, at Dirichlet cells)."""
        Nx, Ny, Nz = self.shape
        d = jnp.zeros((Nx, Ny, Nz))
        d = d.at[:-1, :, :].add(self.Tx)
        d = d.at[1:, :, :].add(self.Tx)
        d = d.at[:, :-1, :].add(self.Ty)
        d = d.at[:, 1:, :].add(self.Ty)
        d = d.at[:, :, :-1].add(self.Tz)
        d = d.at[:, :, 1:].add(self.Tz)
        return jnp.where(self.dirichlet_mask, 1.0, d)


def _mask(field, free_mask):
    return field * free_mask


# ---------------------------------------------------------------------------
# Geometric multigrid V-cycle preconditioner
#
# The grid is strongly anisotropic and layered: thin high-contrast dielectric
# layers with small dz make the vertical face transmissibility Tz dominate the
# lateral ones (Tz/Tx ~ 10). Point smoothers + full coarsening are known to
# fail on such problems. We therefore use the textbook robust combination:
#   * z-LINE smoothing -- solve each vertical column's tridiagonal system
#     exactly (batched Thomas), which handles the strong z-coupling directly;
#   * xy-SEMICOARSENING -- coarsen only the lateral (weakly-coupled) x,y axes,
#     keeping z fixed at every level.
# This yields (near) grid-independent PCG iteration counts (spec §5.2).
# ---------------------------------------------------------------------------

def _pad_to_even(a, axis):
    n = a.shape[axis]
    if n % 2 == 0:
        return a
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (0, 1)
    return jnp.pad(a, pad_width, mode="edge")


def restrict_xy(field):
    """Full-weighting restriction in x,y only (z untouched -- semicoarsening)."""
    f = _pad_to_even(_pad_to_even(field, 0), 1)
    Nx, Ny, Nz = f.shape
    f = f.reshape(Nx // 2, 2, Ny // 2, 2, Nz)
    return f.mean(axis=(1, 3))


def prolong_xy(coarse, fine_shape):
    """Bilinear prolongation in x,y only (z untouched). Bilinear -- not
    piecewise-constant -- is essential here: the z-line smoother does not damp
    lateral high-frequency error, so a constant prolongation would inject xy
    high-freq modes every cycle and iteration counts would grow with lateral
    refinement. Linear interpolation injects negligible high-freq error."""
    fx, fy, Nz = fine_shape
    return jax.image.resize(coarse, (fx, fy, Nz), method="linear")


# retained under the old names for backward compatibility with callers/tests
restrict = restrict_xy
prolong = prolong_xy


def coarsen_nodes(nodes: np.ndarray) -> np.ndarray:
    idx = list(range(0, len(nodes) - 1, 2)) + [len(nodes) - 1]
    return nodes[idx]


def build_mg_hierarchy(eps_r: np.ndarray, x_nodes, y_nodes, z_nodes,
                        dirichlet_mask: np.ndarray, min_size: int = 4,
                        max_levels: int = 7):
    """Build the semicoarsening hierarchy: coarsen x,y by 2 each level, keep
    z fixed. Coarse transmissibilities are rebuilt from block-averaged eps_r
    on the coarsened lateral node set (geometric MG; adequate for a
    preconditioner, spec §5.2)."""
    from .operator import build_transmissibilities

    levels = []
    eps_cur, xn, yn, dmask = eps_r, x_nodes, y_nodes, dirichlet_mask
    for _ in range(max_levels):
        Tx, Ty, Tz = build_transmissibilities(eps_cur, xn, yn, z_nodes)
        levels.append(Operator(Tx, Ty, Tz, jnp.asarray(dmask)))
        if min(eps_cur.shape[0], eps_cur.shape[1]) <= min_size:
            break
        eps_next = np.array(restrict_xy(jnp.asarray(eps_cur)))
        dmask_next = np.array(restrict_xy(jnp.asarray(dmask, dtype=jnp.float64))) > 0.5
        xn, yn = coarsen_nodes(xn), coarsen_nodes(yn)
        if eps_next.shape[:2] == eps_cur.shape[:2]:
            break
        eps_cur, dmask = eps_next, dmask_next
    return levels


def _tridiag_solve_z(lower, diag, upper, rhs):
    """Solve, for every (i,j) column independently, the tridiagonal system
    along z via the Thomas algorithm (vectorized over x,y using lax.scan with
    z as the scan axis). lower[...,0] and upper[...,-1] are ignored."""
    a = jnp.moveaxis(lower, -1, 0)
    b = jnp.moveaxis(diag, -1, 0)
    c = jnp.moveaxis(upper, -1, 0)
    d = jnp.moveaxis(rhs, -1, 0)

    cp0 = c[0] / b[0]
    dp0 = d[0] / b[0]

    def fwd(carry, xs):
        cp_prev, dp_prev = carry
        ak, bk, ck, dk = xs
        m = bk - ak * cp_prev
        cp = ck / m
        dp = (dk - ak * dp_prev) / m
        return (cp, dp), (cp, dp)

    (_, _), (cps, dps) = jax.lax.scan(fwd, (cp0, dp0), (a[1:], b[1:], c[1:], d[1:]))
    cp = jnp.concatenate([cp0[None], cps], axis=0)
    dp = jnp.concatenate([dp0[None], dps], axis=0)

    xN = dp[-1]

    def bwd(x_next, xs):
        cpk, dpk = xs
        xk = dpk - cpk * x_next
        return xk, xk

    _, x_rev = jax.lax.scan(bwd, xN, (cp[:-1][::-1], dp[:-1][::-1]))
    x = jnp.concatenate([x_rev[::-1], xN[None]], axis=0)
    return jnp.moveaxis(x, 0, -1)


def zline_smooth(op: Operator, phi, b, free_mask, n_iter=1, omega=0.8):
    """Damped z-line Jacobi: each sweep solves the vertical tridiagonal
    (diagonal of A + z-face couplings) exactly per column. Dirichlet rows are
    identity so their correction is zero."""
    Nx, Ny, Nz = op.shape
    diag = op.diag()
    lower = jnp.zeros((Nx, Ny, Nz)).at[:, :, 1:].set(-op.Tz)
    upper = jnp.zeros((Nx, Ny, Nz)).at[:, :, :-1].set(-op.Tz)
    dmask = op.dirichlet_mask
    lower = jnp.where(dmask, 0.0, lower)
    upper = jnp.where(dmask, 0.0, upper)
    bdiag = jnp.where(dmask, 1.0, diag)

    for _ in range(n_iter):
        r = _mask(b - op.apply(phi), free_mask)
        delta = _tridiag_solve_z(lower, bdiag, upper, r)
        phi = phi + omega * _mask(delta, free_mask)
    return phi


def v_cycle(levels, level_idx, phi, b, free_masks, n_pre=1, n_post=1):
    op = levels[level_idx]
    fm = free_masks[level_idx]
    phi = zline_smooth(op, phi, b, fm, n_pre)

    if level_idx == len(levels) - 1:
        # coarsest level: a few extra line sweeps in lieu of an exact solve
        return zline_smooth(op, phi, b, fm, n_iter=4)

    r = _mask(b - op.apply(phi), fm)
    r_c = restrict_xy(r)
    e_c0 = jnp.zeros_like(r_c)
    e_c = v_cycle(levels, level_idx + 1, e_c0, r_c, free_masks, n_pre, n_post)
    phi = phi + _mask(prolong_xy(e_c, op.shape), fm)

    phi = zline_smooth(op, phi, b, fm, n_post)
    return phi


def make_mg_preconditioner(levels):
    free_masks = [jnp.asarray(~np.array(lv.dirichlet_mask), dtype=jnp.float64) for lv in levels]

    def precond(r):
        return v_cycle(levels, 0, jnp.zeros_like(r), r, free_masks)

    return precond


def jacobi_preconditioner(op: Operator):
    diag = op.diag()

    def precond(r):
        return r / diag

    return precond


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------

def pcg(op: Operator, b, phi0=None, precond=None, tol=1e-10, maxiter=500):
    """Matrix-free PCG, fully jit-compiled via lax.while_loop. `b` and `phi0`
    must already carry the Dirichlet values at gate cells (phi0 there = target
    voltage, b there = target voltage, since the operator is identity there).

    The whole iteration is compiled once (operator stencil + MG preconditioner
    + CG recurrence), so the lax.scan inside the z-line smoother is traced a
    single time rather than per iteration -- essential for performance."""
    free_mask = jnp.asarray(~np.array(op.dirichlet_mask), dtype=jnp.float64)
    if phi0 is None:
        phi0 = jnp.where(op.dirichlet_mask, b, 0.0)
    if precond is None:
        precond = jacobi_preconditioner(op)

    b = jnp.asarray(b)

    @jax.jit
    def run(phi0, b):
        r0 = _mask(b - op.apply(phi0), free_mask)
        z0 = _mask(precond(r0), free_mask)
        rz0 = jnp.sum(r0 * z0)
        # Converge on residual reduction relative to the INITIAL residual, not
        # to ||b||: with Dirichlet-lifted sources ||b|| at free cells is ~0, so
        # a ||r||/||b|| test is ill-posed. ||r||/||r0|| is well-posed.
        r0_norm = jnp.linalg.norm(r0) + 1e-300

        def cond(state):
            phi, r, p, rz, i = state
            return jnp.logical_and(jnp.linalg.norm(r) / r0_norm > tol, i < maxiter)

        def body(state):
            phi, r, p, rz, i = state
            Ap = _mask(op.apply(p), free_mask)
            alpha = rz / (jnp.sum(p * Ap) + 1e-300)
            phi = phi + alpha * p
            r = r - alpha * Ap
            z = _mask(precond(r), free_mask)
            rz_new = jnp.sum(r * z)
            beta = rz_new / (rz + 1e-300)
            p = z + beta * p
            return phi, r, p, rz_new, i + 1

        state = (phi0, r0, z0, rz0, jnp.array(0))
        phi, r, p, rz, i = jax.lax.while_loop(cond, body, state)
        return phi, jnp.linalg.norm(r) / r0_norm, i

    phi, res, i = run(phi0, b)
    return phi, {"iters": int(i), "res": float(res)}


# ---------------------------------------------------------------------------
# pyamg cross-check (small grids only -- assembles a sparse matrix)
# ---------------------------------------------------------------------------

def assemble_sparse(op: Operator):
    """Assemble A as a scipy sparse CSR matrix (vectorized -- scales to
    millions of cells). Off-diagonal -T entries for each x/y/z face plus the
    row-sum diagonal; Dirichlet rows are replaced by the identity."""
    import scipy.sparse as sp

    Nx, Ny, Nz = op.shape
    n = Nx * Ny * Nz
    Tx, Ty, Tz = np.array(op.Tx), np.array(op.Ty), np.array(op.Tz)
    flat_mask = np.array(op.dirichlet_mask).reshape(-1)

    lin = np.arange(n).reshape(Nx, Ny, Nz)

    def face_edges(idx_a, idx_b, T):
        a = idx_a.reshape(-1)
        b = idx_b.reshape(-1)
        t = T.reshape(-1)
        # symmetric off-diagonal pair
        rows = np.concatenate([a, b])
        cols = np.concatenate([b, a])
        vals = np.concatenate([-t, -t])
        return rows, cols, vals, a, b, t

    R, C, V = [], [], []
    diag = np.zeros(n)
    for idx_a, idx_b, T in (
        (lin[:-1, :, :], lin[1:, :, :], Tx),
        (lin[:, :-1, :], lin[:, 1:, :], Ty),
        (lin[:, :, :-1], lin[:, :, 1:], Tz),
    ):
        rows, cols, vals, a, b, t = face_edges(idx_a, idx_b, T)
        R.append(rows); C.append(cols); V.append(vals)
        np.add.at(diag, a, t)
        np.add.at(diag, b, t)

    R.append(np.arange(n)); C.append(np.arange(n)); V.append(diag)
    A = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                      shape=(n, n)).tocsr()

    # Dirichlet rows -> identity, vectorized: zero those rows, add 1 on diagonal
    keep = (~flat_mask).astype(np.float64)
    A = sp.diags(keep) @ A + sp.diags(flat_mask.astype(np.float64))
    return A.tocsr()


def solve_pyamg(op: Operator, b: np.ndarray, tol=1e-10, return_info=False, maxiter=500):
    """Robust algebraic-multigrid solve (spec §5.2 reliable CPU baseline).

    The matrix mixes O(1) Dirichlet identity rows with O(1e-19) free-cell rows
    (eps0 scale). We symmetrically Jacobi-rescale (D^-1/2 A D^-1/2) to unit
    diagonal, then run scipy-CG preconditioned by the AMG V-cycle, stopping on
    the TRUE relative residual of the scaled (unit-diagonal) system -- which,
    unlike pyamg's own internal test, faithfully reflects solution accuracy.
    """
    import pyamg
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A_ff, b_f, free, phi = _free_block_system(op, b)
    # uniform O(1e-19) scale -> normalize by median diagonal to O(1), which is
    # all that's needed now that the Dirichlet DOFs are eliminated (the free
    # block is well-conditioned; no near-singular Dirichlet coupling remains).
    scale = float(np.median(A_ff.diagonal()))
    A_ff = (A_ff / scale).tocsr()
    b_f = b_f / scale

    ml = pyamg.smoothed_aggregation_solver(A_ff, max_coarse=500)
    M = ml.aspreconditioner(cycle="V")
    it = {"n": 0}

    def cb(xk):
        it["n"] += 1

    x_f, _ = spla.cg(A_ff, b_f, rtol=tol, maxiter=maxiter, M=M, callback=cb)
    phi.reshape(-1)[free] = x_f
    out = phi.reshape(op.shape)
    if return_info:
        rel = float(np.linalg.norm(b_f - A_ff @ x_f) / (np.linalg.norm(b_f) + 1e-300))
        return out, {"iters": it["n"], "res": rel}
    return out


def solve_direct(op: Operator, b: np.ndarray):
    """Exact sparse LU solve (scipy spsolve) on the Dirichlet-eliminated free
    block. Reference/ground-truth; 3D LU fill-in is heavy so not for production
    (spec §5.2)."""
    import scipy.sparse.linalg as spla

    A_ff, b_f, free, phi = _free_block_system(op, b)
    x_f = spla.spsolve(A_ff.tocsc(), b_f)
    phi.reshape(-1)[free] = x_f
    return phi.reshape(op.shape)


def _free_block_system(op: Operator, b: np.ndarray):
    """Eliminate Dirichlet DOFs: given the full assembled A (identity rows at
    Dirichlet cells) and RHS b (carrying Dirichlet values at those cells),
    return the free-free block A_ff, the reduced RHS b_f = b_free - A_fd phi_d,
    the free-cell index array, and a full-size phi array pre-filled with the
    Dirichlet values. Solving A_ff x_f = b_f and scattering x_f into phi[free]
    gives the full solution. This removes the ill-conditioning caused by mixing
    O(1) Dirichlet identity rows with O(1e-19) free rows."""
    A = assemble_sparse(op)
    b = np.asarray(b).reshape(-1)
    dmask = np.asarray(op.dirichlet_mask).reshape(-1)
    free = np.nonzero(~dmask)[0]
    dir_ = np.nonzero(dmask)[0]

    A = A.tocsr()
    A_ff = A[free][:, free]
    A_fd = A[free][:, dir_]
    phi_d = b[dir_]                      # Dirichlet values (from RHS lifting)
    b_f = b[free] - A_fd @ phi_d

    phi = np.zeros(A.shape[0])
    phi[dir_] = phi_d
    return A_ff.tocsr(), b_f, free, phi
