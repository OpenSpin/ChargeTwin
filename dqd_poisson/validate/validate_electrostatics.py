"""Clean-device (no disorder) physics validation against spec §3.4/§9:
lever arms alpha_LL, alpha_LR, |alpha|, and F_z,0 in the well.

Runs on a deliberately coarsened grid (dxy/dz relaxed vs. the production
defaults in GeometryConfig) to keep the solve tractable on CPU for this
validation pass; see spec §9 convergence-study note -- refine the grid to
tighten these numbers, this script is a sanity check, not the final number.
"""

import numpy as np

from .. import geometry as geo
from ..api import (build_problem, solve_gate_unit_responses, superpose,
                   lever_arm_slice, lever_arm_projected, vertical_field_well)


def main():
    cfg = geo.GeometryConfig(dxy_fine_nm=10, dxy_coarse_nm=25, dz_fine_nm=3,
                              dz_coarse_nm=20, lateral_pad_nm=60, bottom_pad_nm=40)
    problem = build_problem(cfg)
    print("grid shape", problem.grid.shape, "cells", np.prod(problem.grid.shape), flush=True)

    gate_response = solve_gate_unit_responses(problem, backend="pyamg")

    gates = geo.build_gates(cfg)
    pl = next(g for g in gates if g.name == "PL")
    pr = next(g for g in gates if g.name == "PR")
    y_mid = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0
    dotL_xy = ((pl.x0_nm + pl.x1_nm) / 2, y_mid)
    dotR_xy = ((pr.x0_nm + pr.x1_nm) / 2, y_mid)

    # energy lever arm: U = -e*phi, so dU/dV [eV/V] = -(phi response) [V/V]
    alpha_LL = -lever_arm_slice(problem, gate_response, dotL_xy, "PL")
    alpha_LR = -lever_arm_slice(problem, gate_response, dotL_xy, "PR")
    alpha_RR = -lever_arm_slice(problem, gate_response, dotR_xy, "PR")
    alpha_RL = -lever_arm_slice(problem, gate_response, dotR_xy, "PL")

    print("--- fixed-z slice lever arms ---")
    print(f"alpha_LL = {alpha_LL:+.4f} eV/V   (target -0.11)")
    print(f"alpha_LR = {alpha_LR:+.4f} eV/V   (target -0.044)")
    print(f"alpha_RR = {alpha_RR:+.4f} eV/V")
    print(f"alpha_RL = {alpha_RL:+.4f} eV/V")
    print(f"|alpha| = {abs(alpha_LL) + abs(alpha_LR):.4f} eV/V  (target 0.12)")

    voltages = geo.gate_voltages(cfg)
    aLL_p = lever_arm_projected(problem, gate_response, voltages, dotL_xy, "PL")
    aLR_p = lever_arm_projected(problem, gate_response, voltages, dotL_xy, "PR")
    print("--- z-projected lever arms (spec 8) ---")
    print(f"alpha_LL(proj) = {aLL_p:+.4f} eV/V   (target -0.11)")
    print(f"alpha_LR(proj) = {aLR_p:+.4f} eV/V   (target -0.044)")

    # F_z,0 target (spec §3.4) is defined at the nominal operating point
    # (spec §3.2: V_PL=V_PR=0.35 V), which differs from the Fig.1(f)
    # lever-arm sweep point (cfg default 0.19 V) used above.
    nominal_voltages = dict(voltages, PL=0.35, PR=0.35)
    phi_nom = superpose(gate_response, nominal_voltages)
    Fz = vertical_field_well(problem, phi_nom, dotL_xy)
    print(f"|F_z| at dotL (nominal op point, V_PL=V_PR=0.35) = {abs(Fz):.3e} V/m  (target 5.34e6 V/m)  [sign {'down' if Fz<0 else 'up'}]")


if __name__ == "__main__":
    main()
