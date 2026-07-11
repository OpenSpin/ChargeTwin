"""Matrix-free -div(eps0 eps_r grad phi) operator on a non-uniform Cartesian
FV grid, with harmonic-mean face permittivities (spec §4, §7).

Convention: (A phi)_i = sum_faces T_face * (phi_i - phi_neighbor). This is
the symmetric FV discretization; faces with no neighbor (domain edges)
contribute nothing, i.e. the untouched operator is naturally homogeneous
Neumann at the outer box (spec §2). Dirichlet gate cells are imposed by
`apply_dirichlet` which overrides those rows with the identity.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .materials import EPS0


def harmonic_face(eps_a, eps_b, d_a, d_b):
    """Distance-weighted harmonic mean face permittivity for two cells of
    half-widths d_a, d_b either side of the face (spec §4: 'harmonic-mean
    face permittivities... guarantees flux continuity')."""
    return (d_a + d_b) / (d_a / eps_a + d_b / eps_b)


def build_transmissibilities(eps_r: np.ndarray, x_nodes, y_nodes, z_nodes):
    """eps_r: (Nx,Ny,Nz) cell-centered. Returns (Tx,Ty,Tz) face
    transmissibilities [F], shapes (Nx-1,Ny,Nz), (Nx,Ny-1,Nz), (Nx,Ny,Nz-1)."""
    dx = np.diff(x_nodes)
    dy = np.diff(y_nodes)
    dz = np.diff(z_nodes)
    Nx, Ny, Nz = eps_r.shape

    # x-faces
    eps_face_x = harmonic_face(eps_r[:-1, :, :], eps_r[1:, :, :],
                                dx[:-1, None, None] / 2, dx[1:, None, None] / 2)
    area_x = dy[None, :, None] * dz[None, None, :]
    dist_x = 0.5 * (dx[:-1, None, None] + dx[1:, None, None])
    Tx = EPS0 * eps_face_x * area_x / dist_x

    # y-faces
    eps_face_y = harmonic_face(eps_r[:, :-1, :], eps_r[:, 1:, :],
                                dy[None, :-1, None] / 2, dy[None, 1:, None] / 2)
    area_y = dx[:, None, None] * dz[None, None, :]
    dist_y = 0.5 * (dy[None, :-1, None] + dy[None, 1:, None])
    Ty = EPS0 * eps_face_y * area_y / dist_y

    # z-faces
    eps_face_z = harmonic_face(eps_r[:, :, :-1], eps_r[:, :, 1:],
                                dz[None, None, :-1] / 2, dz[None, None, 1:] / 2)
    area_z = dx[:, None, None] * dy[None, :, None]
    dist_z = 0.5 * (dz[None, None, :-1] + dz[None, None, 1:])
    Tz = EPS0 * eps_face_z * area_z / dist_z

    return jnp.asarray(Tx), jnp.asarray(Ty), jnp.asarray(Tz)


def apply_laplacian(phi, Tx, Ty, Tz):
    """Raw FV flux-balance operator (natural homogeneous-Neumann outer BC)."""
    out = jnp.zeros_like(phi)

    dxi = phi[:-1, :, :] - phi[1:, :, :]
    term = Tx * dxi
    out = out.at[:-1, :, :].add(term)
    out = out.at[1:, :, :].add(-term)

    dyi = phi[:, :-1, :] - phi[:, 1:, :]
    term = Ty * dyi
    out = out.at[:, :-1, :].add(term)
    out = out.at[:, 1:, :].add(-term)

    dzi = phi[:, :, :-1] - phi[:, :, 1:]
    term = Tz * dzi
    out = out.at[:, :, :-1].add(term)
    out = out.at[:, :, 1:].add(-term)

    return out


def apply_dirichlet(raw_out, phi, dirichlet_mask):
    """Override Dirichlet-gate rows with the identity: (A phi)_i = phi_i."""
    return jnp.where(dirichlet_mask, phi, raw_out)


def apply_operator(phi, Tx, Ty, Tz, dirichlet_mask):
    raw = apply_laplacian(phi, Tx, Ty, Tz)
    return apply_dirichlet(raw, phi, dirichlet_mask)
