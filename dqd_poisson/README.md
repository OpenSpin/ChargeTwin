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
   screening slab). With this fix, at the Fig.1(f) alpha operating point
   (plungers 0.19 V):
     alpha_LL = -0.13 eV/V  (target -0.11)   close -- was -0.41 before the fix
     alpha_LR = -0.02 eV/V  (target -0.044)  now undershoots (was -0.03, also low)
   At the nominal operating point (spec §3.2, plungers 0.35 V):
     |F_z| = 1.28 MV/m  (target 5.34)  ~4x low
   alpha_LL is now close to the paper. F_z is still off by about the same
   factor as before the screening fix -- plausible cause: the extra ~30 nm of
   oxide/screening-slab/oxide stack now between the top gate and the well
   (vs. ~15 nm when fingers wrongly dipped down) pushes the plunger too far
   from the dot vertically. This suggests the fingers DO reach further down
   than "only in the top 27 nm slab" in the real device (e.g. through a via/
   extension not visible at this cross-section), or the well sits closer to
   the interface than modeled. Needs the actual finger vertical extent from
   Fig.1(e) to resolve -- open item. See `validate/lever_at_dot.py`.
5. Interface charges (continuum + discrete) — done
6. z-reduction + `DotPotential` hand-off — done (fixed-slice F_z + full 1D
   projection); `per_column=True` for higher fidelity, `per_column=False` for
   speed
7. Ensemble driver (vmap) + storage — done
8. Convergence studies — script in place, needs runs at production grid size
