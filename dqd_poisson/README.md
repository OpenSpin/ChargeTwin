# dqd_poisson

Open-source, JAX-accelerated Poisson solver for the Si/SiGe DQD electrostatics
stage (replaces COMSOL). See `../poisson_simulator_spec.md` for the full spec.

## Layout

- `geometry.py` — layer stack, gate rectangles, graded Cartesian grid. Several
  exact positions (gate pitch, screening-layer eps_r) are ASSUMPTIONs pending
  Fig. 1(e) of the paper (spec §11) — see `GeometryConfig` docstring.
- `materials.py` — eps_r, band offsets, effective mass constants.
- `operator.py` — matrix-free `-div(eps grad phi)` stencil, harmonic-mean faces.
- `solver.py` — three solve backends sharing one assembled/matrix-free operator:
  `jax` (matrix-free PCG + geometric **z-line-smoothed, xy-semicoarsening** MG;
  fast on uniform grids, stalls on strongly graded grids), `pyamg` (assembled
  algebraic-MG + CG), and `direct` (exact sparse LU). Convergence tests use
  `||r||/||r0||` (the `||r||/||b||` test is ill-posed here — sources enter only
  via Dirichlet lifting so `||b||` at free cells is ~0).
  **Important caveat:** the graded-grid operator is ill-conditioned (cond ~1e10
  from cell-size grading + eps contrast). Iterative solvers reduce the *residual*
  to 1e-12 but that still leaves ~1e-2 *solution* error — so for trustworthy
  absolute numbers on graded grids use `backend='direct'` (exact, but 3D LU
  fill-in limits it to moderate grids). Getting an iterative solver to
  production scale with good relative accuracy is a genuine open task (spec §9
  calls the quantitative regime "demanding").

- Far-field BC (`geometry.ground_mask`, spec §2): the outer box is natural
  Neumann except the **far bottom face is grounded (Dirichlet phi=0)**. Without
  it the deep-substrate potential is under-determined (a near-nullspace) and
  different solvers return different fields at the dot. Toggle via
  `GeometryConfig.far_bottom_ground` / `far_lateral_ground`.
- `charges.py` — interface charge sources (continuum sheet, discrete defects).
- `reduce.py` — 1D z-Schrodinger + projection to `V_2D(x,y)` (spec §8).
- `api.py` — `Problem`/`DotPotential`, gate-response superposition, ensemble
  driver (batched vmap+jit CG over disorder realizations), HDF5 storage.
- `validate/` — analytic unit tests, solver cross-checks, electrostatics
  validation against paper targets, convergence study.

## Running

```
python -m venv .venv && source .venv/bin/activate
pip install jax jaxlib numpy scipy matplotlib h5py pyamg pytest
python -m dqd_poisson.validate.test_operator
python -m dqd_poisson.validate.test_solver
python -m dqd_poisson.validate.test_reduce
python -m dqd_poisson.validate.validate_electrostatics   # lever arms, F_z,0
```

Note: on Apple Silicon, use an arm64 Python (not Rosetta/x86_64 Anaconda) —
jaxlib's AVX-optimized wheel crashes under Rosetta emulation.

## Status vs. build order (spec §10)

1. Geometry + grid — done
2. Matrix-free operator + analytic tests — done
3. PCG + MG preconditioner vs. pyamg — done
4. Gate BC lifting + lever-arm/F_z validation — implemented, converged solver.
   Geometry corrected per user-confirmed Fig.1(e) read: the screening
   electrode is a real 10 nm METAL slab sandwiched between two 10 nm SiO2
   layers, sitting immediately lateral to the channel (not a distant grounded
   plane as first modeled); PL/PR/barrier fingers sit only in the top 27 nm
   metal layer, directly above the upper SiO2 (they do not dip down into the
   screening slab).

   **z-stack bug fix:** the well was wrongly sandwiched between two SiGe
   layers (30 nm below + 50 nm above), i.e. ~66.5 nm below the gate stack.
   Spec §3.1's row order is substrate(30)+barrier(60) both *below* the well,
   well directly under the Si cap -- only ~26.5 nm below the gate stack.
   Fixed (`SiGe_lower` 30 + `SiGe_barrier` 60 below `Si_well`, no SiGe above
   it); restored `t_oxide_lower_nm`/`t_screening_slab_nm` to the spec's 10/10
   nm (the 5/5 values above were a since-obsolete compensating hack for the
   wrong well position). Also applied user-confirmed values: `w_metal_nm`
   45 nm (was 40), `plunger_pitch_nm` 95 nm to match the clean-device
   interdot distance target `d0=95nm` (was 90).

   Effect at the Fig.1(f) alpha point (plungers 0.19 V) / nominal F_z point
   (plungers 0.35 V, spec §3.2):
     alpha_LL = -0.26 eV/V  (target -0.11)   now ~2.3x OVER (was -0.11, spot on
                                              but with the wrong well depth)
     alpha_LR = -0.03 eV/V  (target -0.044)  still ~1.5x under
     |F_z|    = 2.36 MV/m  (target 5.34)     ~2.3x under (was ~4-11x under)
   Confirmed stable across grid resolution (coarse validate_electrostatics.py
   grid vs. finer lever_at_dot.py grid agree to ~5%) -- this is a geometry
   mismatch, not a discretization error.

   The remaining gap can't be closed by further vertical-distance tuning
   alone: alpha_LL overshooting means the gate looks too CLOSE, while F_z
   undershooting means it looks too FAR -- moving the well/gate distance
   either way pushes one metric further from target while helping the other.
   The real device geometry must differ from the literal spec-table stack in
   a way that decouples the two (e.g. actual finger lateral width/isolation
   gap, or a via reaching partway into the screening slab) -- still needs
   the real Fig.1(e) dimensions to resolve; not solvable by guessing further.
   See `validate/lever_at_dot.py`.
5. Interface charges (continuum + discrete) — done
6. z-reduction + `DotPotential` hand-off — done (fixed-slice F_z + full 1D
   projection); `per_column=True` for higher fidelity, `per_column=False` for
   speed
7. Ensemble driver (vmap) + storage — done
8. Convergence studies — script in place, needs runs at production grid size
