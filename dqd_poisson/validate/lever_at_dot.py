"""Lever arm sampled at the TRUE dot (potential minimum) vs at the fixed
PL-finger centre, at the Fig.1(f) operating point. Shows how much the dot-
position choice changes alpha (the source of alpha's apparent nonlinearity,
spec §8). Usage: python -m dqd_poisson.validate.lever_at_dot
     (or directly: python3 dqd_poisson/validate/lever_at_dot.py from repo root)
"""

import sys, os

# allow direct execution from the validate/ dir or repo root
if __name__ == "__main__" and __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    from dqd_poisson import geometry as geo
    from dqd_poisson.api import (build_problem, solve_gate_unit_responses, superpose,
                                 lever_arm_slice, lever_arm_projected, locate_dot,
                                 vertical_field_well)
else:
    from .. import geometry as geo
    from ..api import (build_problem, solve_gate_unit_responses, superpose,
                       lever_arm_slice, lever_arm_projected, locate_dot,
                       vertical_field_well)

import numpy as np


def main(dxy=4.0, dz=1.5):
    cfg = geo.GeometryConfig(dxy_fine_nm=dxy, dxy_coarse_nm=2.5 * dxy,
                              dz_fine_nm=dz, dz_coarse_nm=max(20.0, 6 * dz),
                              lateral_pad_nm=60, bottom_pad_nm=40)
    problem = build_problem(cfg)
    print(f"grid {problem.grid.shape}  op point PL=PR={cfg.v_PL} B={cfg.v_barrier} scr={cfg.v_screening}",
          flush=True)
    gr = solve_gate_unit_responses(problem, tol=1e-10, backend="pyamg")
    volt = geo.gate_voltages(cfg)

    gates = geo.build_gates(cfg)
    pl = next(g for g in gates if g.name == "PL")
    y_mid = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0
    pl_center = ((pl.x0_nm + pl.x1_nm) / 2, y_mid)

    dotL = locate_dot(problem, gr, volt, side="left")
    print(f"PL-finger centre xy = ({pl_center[0]:.0f}, {pl_center[1]:.0f}) nm")
    print(f"dot-L minimum   xy = ({dotL[0]:.0f}, {dotL[1]:.0f}) nm")

    print("\n            alpha_LL   alpha_LR   (target -0.110 / -0.044)")
    aLL_c = -lever_arm_slice(problem, gr, pl_center, "PL")
    aLR_c = -lever_arm_slice(problem, gr, pl_center, "PR")
    print(f"at PL centre  {aLL_c:+.3f}    {aLR_c:+.3f}   (slice)")
    aLL_d = -lever_arm_slice(problem, gr, dotL, "PL")
    aLR_d = -lever_arm_slice(problem, gr, dotL, "PR")
    print(f"at dot min    {aLL_d:+.3f}    {aLR_d:+.3f}   (slice)")
    aLL_dp = lever_arm_projected(problem, gr, volt, dotL, "PL")
    aLR_dp = lever_arm_projected(problem, gr, volt, dotL, "PR")
    print(f"at dot min    {aLL_dp:+.3f}    {aLR_dp:+.3f}   (projected)")

    phi_op = superpose(gr, volt)
    Fz = abs(vertical_field_well(problem, phi_op, dotL))
    print(f"\n|F_z| at dot-L = {Fz/1e6:.3f} MV/m  (target 5.34)")


if __name__ == "__main__":
    main()
