"""First Schrodinger-stage demo (spec §8): fixed-z slice of the confinement
potential at the well, cropped to a window around the DQD, and the lowest
lateral eigenstates on it (2D approx: isotropic in-plane mass, no z-mode
projection yet -- see reduce.py/project_to_2d for the more faithful version).

Produces dqd_poisson/output/lateral_2d.png: potential + |psi|^2 panels.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import geometry as geo
from ..api import build_problem, solve_gate_unit_responses, superpose
from ..schrodinger import solve_lateral_2d, crop_window, tunnel_coupling_two_level
from ..materials import E_CHARGE


def main():
    cfg = geo.GeometryConfig(dxy_fine_nm=3.0, dxy_coarse_nm=25.0, dz_fine_nm=2.0,
                              dz_coarse_nm=20.0, lateral_pad_nm=80.0, bottom_pad_nm=0.0,
                              v_barrier=-0.02)
    problem = build_problem(cfg)
    print("grid shape", problem.grid.shape, flush=True)

    gate_response = solve_gate_unit_responses(problem, backend="pyamg")
    voltages = geo.gate_voltages(cfg)  # nominal operating point, spec §3.2
    phi = superpose(gate_response, voltages)

    z_well_nm = geo.well_z_center_nm(cfg)
    print(f"well z-centre = {z_well_nm:.1f} nm "
          f"(post-fix; previously z=35nm before the SiGe-layer-order fix)")
    z_nm = problem.grid.z_centers / geo.NM
    k = np.argmin(np.abs(z_nm - z_well_nm))
    z_used = z_nm[k]

    u_band = geo.u_band_field(problem.grid, cfg)  # 0 in the well everywhere
    U_2D_full = (-phi[:, :, k] + u_band[:, :, k])  # eV (phi already in volts, e=1 -> U=-phi in eV)

    x_nm = problem.grid.x_centers / geo.NM
    y_nm = problem.grid.y_centers / geo.NM
    xc = cfg.lateral_pad_nm + cfg.Lx_nm / 2.0
    yc = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0

    x_win, y_win, U_2D = crop_window(x_nm, y_nm, U_2D_full, xc, yc,
                                      x_window_nm=400.0, y_window_nm=200.0)
    print("cropped window shape", U_2D.shape, flush=True)

    energies_eV, psis = solve_lateral_2d(x_win * geo.NM, y_win * geo.NM, U_2D, n_states=6,
                                          tol=1e-13)
    print("lateral eigenstates (eV, full precision):")
    for n, E in enumerate(energies_eV):
        print(f"  E{n} = {E:.10f} eV")
    two_tc_ueV = (energies_eV[1] - energies_eV[0]) * 1e6
    print(f"E1-E0 (direct ARPACK splitting -- unreliable if E0,E1 are a tight "
          f"near-degenerate cluster, see tunnel_coupling_two_level below) = "
          f"{two_tc_ueV:.4f} ueV   (target 2t_c,0 ~ 24 ueV)")
    print(f"E2-E0 (~orbital excitation E_orb if E1 is the other dot's ground state) "
          f"= {(energies_eV[2]-energies_eV[0])*1e3:.4f} meV")

    result = tunnel_coupling_two_level(x_win * geo.NM, y_win * geo.NM, U_2D, xc * geo.NM)
    print("--- Hund-Mulliken two-level tunnel coupling ---")
    print(f"E_L0 = {result['E_L0_eV']*1e3:.6f} meV   E_R0 = {result['E_R0_eV']*1e3:.6f} meV")
    print(f"S_LR (orbital overlap) = {result['S_LR']:.4e}")
    print(f"2*t_c = {result['splitting_eV']*1e6:.4f} ueV   (target 2t_c,0 ~ 24 ueV)")
    print(f"t_c   = {result['t_c_eV']*1e6:.4f} ueV")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    ax = axes[0, 0]
    im = ax.pcolormesh(x_win, y_win, U_2D.T, shading="auto", cmap="viridis")
    ax.set_title(f"V(x,y) at z={z_used:.1f} nm (well)")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.colorbar(im, ax=ax, label="eV")

    for n in range(6):
        r, c = divmod(n + 1, 4)
        ax = axes[r, c]
        im = ax.pcolormesh(x_win, y_win, (psis[n] ** 2).T, shading="auto", cmap="magma")
        ax.set_title(f"|psi_{n}|^2, E={energies_eV[n]*1e3:.3f} meV")
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lateral_2d.png")
    plt.savefig(out_path, dpi=130)
    print("saved", out_path)


if __name__ == "__main__":
    main()
