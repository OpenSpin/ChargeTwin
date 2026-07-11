"""Sweep barrier gate voltage and report the tunnel coupling 2*t_c at each
point, reusing a single Poisson solve (spec §5.1: gate sweeps are free once
the four unit responses are known -- only superposition, no new solve).

t_c is exponentially sensitive to the barrier height/width (spec §9), so this
sweep is the cheap way to find where 2*t_c lands near the paper's target
(~24 ueV at the nominal V_barrier=-0.1V) without re-solving Poisson per point.
"""

import numpy as np

from .. import geometry as geo
from ..api import build_problem, solve_gate_unit_responses, superpose
from ..schrodinger import crop_window, tunnel_coupling_two_level

BARRIER_VOLTAGES = [-0.10, -0.06, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]


def main():
    cfg = geo.GeometryConfig(dxy_fine_nm=3.0, dxy_coarse_nm=25.0, dz_fine_nm=2.0,
                              dz_coarse_nm=20.0, lateral_pad_nm=80.0, bottom_pad_nm=0.0)
    problem = build_problem(cfg)
    print("grid shape", problem.grid.shape, flush=True)

    gate_response = solve_gate_unit_responses(problem, backend="pyamg")

    z_well_nm = geo.well_z_center_nm(cfg)
    z_nm = problem.grid.z_centers / geo.NM
    k = np.argmin(np.abs(z_nm - z_well_nm))

    u_band = geo.u_band_field(problem.grid, cfg)
    x_nm = problem.grid.x_centers / geo.NM
    y_nm = problem.grid.y_centers / geo.NM
    xc = cfg.lateral_pad_nm + cfg.Lx_nm / 2.0
    yc = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0

    print(f"{'V_barrier':>10} {'2t_c (ueV)':>14} {'S_LR':>12}")
    for vb in BARRIER_VOLTAGES:
        voltages = {"PL": cfg.v_PL, "PR": cfg.v_PR, "barrier": vb, "screening": cfg.v_screening}
        phi = superpose(gate_response, voltages)
        U_2D_full = -phi[:, :, k] + u_band[:, :, k]
        x_win, y_win, U_2D = crop_window(x_nm, y_nm, U_2D_full, xc, yc,
                                          x_window_nm=400.0, y_window_nm=200.0)
        result = tunnel_coupling_two_level(x_win * geo.NM, y_win * geo.NM, U_2D, xc * geo.NM)
        print(f"{vb:>10.3f} {result['splitting_eV']*1e6:>14.6f} {result['S_LR']:>12.3e}", flush=True)


if __name__ == "__main__":
    main()
