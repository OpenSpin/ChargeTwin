"""Grid-refinement convergence study (spec §9): refine dxy_fine/dz_fine and
track alpha_LL, alpha_LR, F_z,0 until stable. Uses the robust pyamg backend
(assembled algebraic MG) -- the geometric JAX MG stalls on strongly graded
production grids, whereas AMG converges in ~1 CG iteration (spec §5.2).

`2t_c`/`d` require the Schrodinger stage (out of scope here) -- this covers
the electrostatics-only observables.

Usage: python -m dqd_poisson.validate.convergence_study
"""

import numpy as np

from .. import geometry as geo
from ..api import (build_problem, solve_gate_unit_responses, superpose,
                   lever_arm_slice, lever_arm_projected, vertical_field_well)


def run_one(dxy_fine, dz_fine, lateral_pad=60.0, bottom_pad=40.0, backend="pyamg"):
    cfg = geo.GeometryConfig(dxy_fine_nm=dxy_fine, dxy_coarse_nm=max(25.0, 2.5 * dxy_fine),
                              dz_fine_nm=dz_fine, dz_coarse_nm=max(20.0, 6 * dz_fine),
                              lateral_pad_nm=lateral_pad, bottom_pad_nm=bottom_pad)
    problem = build_problem(cfg)
    n_cells = int(np.prod(problem.grid.shape))
    gate_response = solve_gate_unit_responses(problem, tol=1e-10, backend=backend)

    gates = geo.build_gates(cfg)
    pl = next(g for g in gates if g.name == "PL")
    y_mid = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0
    dotL_xy = ((pl.x0_nm + pl.x1_nm) / 2, y_mid)

    voltages = geo.gate_voltages(cfg)
    aLL = -lever_arm_slice(problem, gate_response, dotL_xy, "PL")
    aLR = -lever_arm_slice(problem, gate_response, dotL_xy, "PR")
    aLL_p = lever_arm_projected(problem, gate_response, voltages, dotL_xy, "PL")
    aLR_p = lever_arm_projected(problem, gate_response, voltages, dotL_xy, "PR")
    phi_op = superpose(gate_response, voltages)
    Fz = vertical_field_well(problem, phi_op, dotL_xy)
    return dict(n_cells=n_cells, dxy_fine=dxy_fine, dz_fine=dz_fine,
                aLL=aLL, aLR=aLR, aLL_p=aLL_p, aLR_p=aLR_p, Fz=abs(Fz))


def main():
    grid_levels = [(8.0, 3.0), (5.0, 2.0), (3.0, 1.5), (2.0, 1.0)]
    print(f"{'cells':>10} {'dxy':>5} {'dz':>4} {'aLL':>9} {'aLL_proj':>9} "
          f"{'aLR':>9} {'aLR_proj':>9} {'|Fz| MV/m':>10}", flush=True)
    print(f"{'target':>10} {'':>5} {'':>4} {-0.11:>9.3f} {-0.11:>9.3f} "
          f"{-0.044:>9.3f} {-0.044:>9.3f} {5.34:>10.2f}", flush=True)
    for dxy, dz in grid_levels:
        r = run_one(dxy, dz)
        print(f"{r['n_cells']:>10} {r['dxy_fine']:>5.1f} {r['dz_fine']:>4.1f} "
              f"{r['aLL']:>9.3f} {r['aLL_p']:>9.3f} {r['aLR']:>9.3f} {r['aLR_p']:>9.3f} "
              f"{r['Fz'] / 1e6:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
