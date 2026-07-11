# Specification — Open-Source Poisson Simulator for a Si/SiGe Double Quantum Dot

**Target paper:** *Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots* (Samadi, Krzywda, Cywiński), arXiv:2510.13578v2.
**Purpose:** Replace the COMSOL electrostatics stage with an open-source, JAX-accelerated solver that produces the confinement potential `V(x,y)` for the single-electron Schrödinger step (solved separately, later).
**Scope of this document:** the **electrostatics (Poisson) module only**, plus the vertical (z) reduction and the hand-off API to the Schrödinger solver. The lateral Schrödinger eigensolver, tuning loop, and parameter extraction are *out of scope* but their interface is defined in §8.

---

## 1. Physical model and key simplification

We work in the **single-electron approximation**. The one electron's charge density has negligible feedback on the electrostatic potential, so **Poisson is fully decoupled from Schrödinger** — no self-consistent Schrödinger–Poisson loop is needed. This is the single most important simplification and it drives the whole design:

- Poisson is a **linear** elliptic PDE whose only inputs are gate voltages (Dirichlet) and fixed trapped interface charges (source).
- The discretized operator depends **only on geometry and permittivity**, which are *fixed for the entire study*. Only the right-hand side (gate voltages + disorder charges) changes across gate sweeps and across disorder realizations.
- Therefore: **assemble/factorize/precondition the operator once, reuse it for every configuration.** Ensembles of thousands of devices and gate sweeps become cheap. See §5.

Governing equation, on domain Ω (the full 3D heterostructure + gate stack + oxide):

```
-∇·( ε₀ εr(r) ∇φ(r) ) = ρ_free(r)
```

with `ρ_free` the trapped interface charge (the only free charge in the single-electron model). The electron confinement potential is `U(r) = -e·φ(r) + U_band(r)`, where `U_band` is the piecewise conduction-band offset of the heterostructure.

---

## 2. Boundary conditions

| Boundary / region | Condition | Notes |
|---|---|---|
| Metal gate surfaces (PL, PR, barrier, screening) | **Dirichlet** `φ = V_gate` | Voltages are the tunable inputs; enter the linear system via RHS lifting (see §5), so they do not change the operator. |
| Trapped charges at the semiconductor–oxide interface (Si-cap/SiO₂) | **Interface (sheet) source** | Modeled as a surface charge density on the interface plane → a source term, **not** a BC. See §6. |
| Insulating region / hard-wall confinement (`U → ∞`) | Handled in the **Schrödinger** stage as a Dirichlet wall on the wavefunction, **not** in Poisson. In Poisson, the oxide is just a low-εr dielectric. | The paper imposes `U→∞` in the insulator for the electron; that is a wavefunction BC, applied later. |
| Outer domain box (far lateral/bottom faces) | **Homogeneous Neumann** `∂φ/∂n = 0` (or Dirichlet `φ=0` on the far bottom, whichever is closer to COMSOL) | Place the box far enough that the choice is negligible near the DQD. Make it a config flag; validate both. |

The operator is **symmetric positive definite** (self-adjoint elliptic with these BCs) — exploited by the solver in §5.

---

## 3. Geometry and material parameters (from the paper)

Overall lateral device footprint: **660 × 582 nm²**. Growth axis ẑ = [001]. Values below are taken from the paper text and Fig. 1(e); confirm against the figure when building the mesh.

### 3.1 Layer stack (z, bottom → top)

| Layer | Thickness | εr | Band offset role |
|---|---|---|---|
| SiGe substrate (Si₀.₇Ge₀.₃) | 30 nm | 13.2 | barrier |
| SiGe barrier (lower) | 60 nm | 13.2 | barrier, `U₀ = 150 meV` above Si well |
| Si quantum well | 10 nm | 12 | well (electron localized here) |
| Si cap | 1.5 nm | 12 | — |
| SiO₂ (oxide) | 10 nm | 3.9 | insulator |
| screening dielectric/layer | 10 nm | (as in figure) | below gate metal |
| metal gates | 27 nm height | conductor (Dirichlet) | gate width `w_metal = 45 nm` |

- Channel gap in the screening layer: `d_channel = 142 nm`.
- Trapped-charge interface: the **semiconductor–oxide interface** (Si cap / SiO₂). Charge sheet density `ρ` swept over **5×10⁹ – 5×10¹⁰ cm⁻²**.

> Note: the quantum-well thickness value is largely irrelevant to the electrostatics of the confinement (electron localizes within ≈5 nm of the top Si/SiGe interface under the vertical field); `h_Si = 5–10 nm` both fine.

### 3.2 Gates

Four electrodes: **PL** (left plunger), **PR** (right plunger), **barrier**, **screening**. Nominal operating point:

| Gate | Voltage |
|---|---|
| PL, PR (plungers) | 0.35 V |
| Barrier | −0.1 V |
| Screening | (as in model; ground or fixed bias) |

### 3.3 Materials & electron parameters

| Quantity | Symbol | Value |
|---|---|---|
| εr SiO₂ | | 3.9 |
| εr Si | | 12 |
| εr SiGe | | 13.2 |
| Conduction-band offset (SiGe barrier) | U₀ | 150 meV |
| Transverse effective mass | m_t | 0.19 mₑ |
| (Schrödinger stage) | | |

### 3.4 Defect-free validation targets (must reproduce before disorder)

| Quantity | Target |
|---|---|
| Plunger lever arm magnitude | `|α| = 0.12 eV/V` |
| Individual lever arms | `α_LL ≈ -0.11, α_LR ≈ -0.044 kV/V` (Fig. 1f) |
| Vertical field in well | `F_z,0 = 5.34 MV/m` |
| Orbital excitation energy | `E_orb ≈ 1.5 meV` per dot |
| Lateral confinement lengths | `L_x,0 = 20 nm`, `L_y,0 = 18 nm` (ratio ≈1.08) |
| Interdot distance (clean) | `d₀ = 95 nm` |
| Tunnel gap (clean) | `2t_c,0 ≈ 24 μeV` |

`|α|`, `F_z,0`, and the lever arms are **pure electrostatics** → they validate *this module*. `E_orb`, `L_x`, `d₀`, `2t_c` require the Schrödinger stage but are listed for the end-to-end check.

---

## 4. Discretization

**Recommendation: finite-volume (FV) on a structured, non-uniform Cartesian grid.** Rationale:

- The geometry is axis-aligned: horizontal dielectric slabs + rectangular gates. A structured grid captures it exactly at interfaces without unstructured meshing.
- Structured stencils are trivially **vectorizable and GPU-friendly in JAX** (matrix-free `∇·ε∇` as a 7-point stencil).
- Piecewise-constant εr handled by **harmonic-mean face permittivities** (correct FV treatment of dielectric jumps, guarantees flux continuity).

Grid guidance:
- **Graded resolution**: ~1 nm near the top Si/SiGe interface, the well, the oxide, and under the gates; stretch to 5–10 nm in the deep substrate and lateral far field.
- Ensure grid planes coincide with each material interface and each gate edge (snap the mesh to geometry).
- Rough size: ~200 × 200 × 120 ≈ 5×10⁶ cells worst case; a graded grid can cut this substantially. Keep it a config so convergence studies (§9) can refine.

*Alternative / cross-check:* an unstructured FEM reference in **`scikit-fem`** or **FEniCSx (dolfinx)** on the same geometry, used only to validate the FV solver on a few configs (§9). Do not build the production path on unstructured FEM — it fights JAX.

---

## 5. Linear solver and the "solve-once" strategy

### 5.1 Operator is constant; only the RHS changes

Discretized system: `A φ = b`. `A` (SPD) is fixed for the whole study. Contributions to `b`:

```
b = b_charge(disorder)  +  Σ_g  V_g · b_g^(unit)
```

- `b_g^(unit)`: RHS produced by lifting Dirichlet BC of gate `g` set to 1 V (others 0), i.e. `A φ_g^unit = b_g^unit`. Precompute the **unit gate responses** `φ_g^unit` once (one solve per gate — only 4).
- `b_charge`: source from the interface charge sheet for a given disorder realization (§6).

Then **by linearity**:
```
φ(V_PL,V_PR,V_B,V_scr; disorder) = Σ_g V_g φ_g^unit  +  φ_charge(disorder)
```
- **Gate sweeps and the tuning loop are essentially free** once the four `φ_g^unit` are known — no new solves, just weighted sums. This is exactly what the Schrödinger tuning loop needs (§8).
- **Disorder realizations** each need one solve for `φ_charge` — but with the *same* `A`, so reuse the factorization/preconditioner.

### 5.2 Recommended solver

Primary (performant, JAX-native):
- **Matrix-free preconditioned CG (PCG)** with a **geometric-multigrid V-cycle preconditioner**, all in JAX (`jit`ed). SPD ⇒ CG is optimal; MG preconditioning gives grid-independent iteration counts. No large factorization to store (avoids 3D Cholesky fill-in, which can be tens of GB at 5M unknowns).
- Warm-start across the ensemble (successive disorder RHS are similar) to cut iterations.
- Enable **float64** (`jax.config.update("jax_enable_x64", True)`) — required; `t_c` is exponentially sensitive (see §9), single precision is insufficient.

Fallbacks / references:
- **`pyamg`** (algebraic multigrid): build the hierarchy once on `A`, reuse `.solve()` for all RHS. Very reliable, CPU. Good baseline to validate the JAX MG against.
- **Sparse Cholesky** (`scikit-sparse` / CHOLMOD): only if memory allows; factor once, back-substitute per RHS. Fast per-solve but heavy memory in 3D — treat as optional.

### 5.3 Differentiability (optional but cheap to get)

Because the map `V_gates → φ` is linear, gradients (e.g. `∂(observable)/∂V_gate` for gradient-based tuning or lever-arm extraction) come from a single adjoint solve with the same operator. Use `jax.lax.custom_linear_solve` or the **`lineax`** library so `A⁻¹` differentiates correctly without unrolling CG. This makes lever arms `α` computable analytically rather than by finite differences.

---

## 6. Trapped interface charges (disorder source)

Charges live on the semiconductor–oxide interface plane. **Do not** use nodal delta sources — they create 1/r singularities that FEM/FV resolves poorly and that pollute the extracted potential near the dot.

Two supported modes:

1. **Continuum surface density** (matches "density ρ"): add a surface charge density `σ` on the interface plane as an FV source in the interface cells (a flux-jump term in the weak form). Use for smooth-field studies.
2. **Discrete defects** (matches the disorder ensemble): sample `N = ρ · A_interface` point charges at random in-plane positions (`ρ = 5×10¹⁰ cm⁻²` over 660×582 nm² ⇒ ~190 charges). **Regularize** each as a Gaussian sheet smeared over ~1 interface cell so the RHS is bounded and mesh-convergent. Charge sign/occupation per the paper's disorder model.

Ensemble generation: draw defect positions (and signs/occupations), assemble `b_charge`, solve with the reused operator. This loop is embarrassingly parallel and a natural `jax.vmap`/`pmap` target (§7).

*Fast approximate accelerator (optional):* since all defects share one interface plane and the stack is layered, a **layered-dielectric Green's function** `G(x−x', y−y', z_well, z_int)` computed once (in-plane FFT) turns each realization's disorder potential into a **convolution** of the charge sheet with the kernel — O(N log N) via `jnp.fft`, no linear solve. It approximates the lateral gate structure as a uniform ground plane, so use it for rapid scans / sanity checks and validate against the exact FV solve. Keep both paths.

---

## 7. JAX design notes

- **x64 on** globally (see §5.2).
- **Matrix-free operator**: implement `A @ φ` as a stencil (`jax.numpy` roll/slice or `lax.conv`) using face-averaged εr arrays. No assembled sparse matrix needed for the JAX path.
- **Batch the ensemble**: stack disorder RHS as a leading batch axis; `vmap` the PCG solve over it. Use `pmap` across GPUs if available. Solves within a batch share the MG preconditioner (function of `A` only).
- **`jit`** the full solve (operator + preconditioner + CG). Keep grid sizes static so JIT caches.
- **Memory**: one field at 5×10⁶ float64 ≈ 40 MB; batching 64 realizations ≈ 2.6 GB of fields + MG work arrays — sits comfortably on a single modern GPU. Reduce batch or grid if needed.
- **I/O**: store `φ_g^unit` (4 fields) and per-realization `φ_charge` or, better, the already-reduced 2D potentials (§8) to keep disk small.

Dependencies (all open source): `jax`, `jaxlib`, `numpy`, `scipy`, `lineax` (optional, diff/solve), `pyamg` (fallback), `meshio`/`gmsh` (optional geometry I/O), `scikit-fem` or `dolfinx` (optional reference), `matplotlib` (diagnostics), `h5py` (storage).

---

## 8. Vertical (z) reduction and hand-off to Schrödinger

This is the subtle physics step and the **most likely source of divergence from COMSOL** — specify it carefully.

The electron sees stiff vertical confinement (10 nm well + `F_z ≈ 5.34 MV/m` ⇒ localization ≈5 nm) and soft lateral confinement (`L ≈ 20 nm`, `E_orb ≈ 1.5 meV`). Adiabatic separation of z from (x,y):

1. From `φ`, form the total vertical potential `U(x,y,z) = -eφ + U_band` along z.
2. At the dot region, solve the **1D effective-mass Schrödinger equation in z** (longitudinal mass along [001]) to get the ground vertical mode `Ψ_z,0(z)` and energy `U_0`. (Do this at the dot center, or per (x,y) column for higher fidelity.)
3. **Project** the 3D potential onto that vertical mode to get the 2D lateral potential:
   ```
   V_2D(x,y) = ⟨Ψ_z,0 | U(x,y,·) | Ψ_z,0⟩_z
   ```
   Do **not** simply slice at a fixed z — that gets lever arms roughly right but misses the confinement energies. The projection is what reproduces `E_orb`, `L_x`, `d`.
4. Extract `F_z(x,y)` from the z-gradient of `U` at the well for the vertical-field diagnostics.

**Hand-off object (the module's output):**

```
DotPotential:
  xy_grid:      (Nx, Ny) coordinates over the DQD footprint
  V_2D:         (Nx, Ny) lateral confinement potential  [J or eV]
  psi_z0, U0:   vertical mode + energy (for records)
  F_z:          (Nx, Ny) vertical field at the well
  m_t:          transverse effective mass (0.19 mₑ), anisotropy tensor
  gate_response: callable/precomputed φ_g^unit so the Schrödinger
                 tuning loop can add arbitrary gate-voltage combos
                 with NO new Poisson solve (linearity, §5.1)
  meta:         gate voltages, disorder seed, ρ, grid config
```

The Schrödinger stage then builds `V_2D` for any `(V_PL, V_PR, V_B)` by superposition, solves the **2D anisotropic effective-mass eigenproblem** (`m_t`, ellipticity `L_x/L_y`), and runs the tuning root-find (find ΔV_LR that symmetrizes the wells; read `2t_c` as the symmetric–antisymmetric gap). Keeping `gate_response` in the hand-off is what makes that loop cheap.

---

## 9. Validation and convergence plan

**Analytic unit tests (must pass first):**
- Parallel-plate capacitor (uniform εr): recover linear φ and `E = V/d`.
- Point charge in uniform dielectric vs. regularized source: recover Coulomb tail away from the smear radius.
- Dielectric slab interface: flux continuity `ε₁E₁ = ε₂E₂` across a horizontal interface (checks harmonic-mean faces).
- Layered Green's function vs. exact FV for a single charge (validates the §6 accelerator).

**Physics validation against the paper (clean device, no disorder):**
- Lever arms `α_LL ≈ -0.11`, `α_LR ≈ -0.044 kV/V`, `|α| = 0.12 eV/V` — from `∂φ_dot/∂V_gate` (a byproduct of `φ_g^unit`; ideally via the adjoint of §5.3).
- `F_z,0 = 5.34 MV/m` in the well.
- End-to-end (with Schrödinger): `E_orb ≈ 1.5 meV`, `L_x=20 nm`, `L_y=18 nm`, `d₀=95 nm`, `2t_c,0 ≈ 24 μeV`.

**Convergence studies (critical because `t_c ∝ exp(-β d √(h_B))`, Eq. 3 of the paper):**
- Refine the grid until `2t_c` and `d` are stable to the μeV / sub-nm level. `t_c` is exponentially sensitive to `d` and barrier height `h_B`, both of which feed the exponent — resolve the interdot barrier region finely.
- Test sensitivity to (a) the z-projection choice (fixed-slice vs. projected vs. per-column), (b) the far-boundary BC (Neumann vs. Dirichlet), (c) the charge smear radius. Report the resulting spread in `2t_c`.
- Cross-check a handful of configs against the `scikit-fem`/FEniCSx reference and, if available, the original COMSOL numbers.

**Honest expectation:** a working Poisson solver reproducing `|α|` and `F_z` is straightforward. Reproducing the paper's **`t_c` distributions quantitatively** is demanding and gated almost entirely by (i) mesh resolution in the interdot barrier and (ii) the z-projection fidelity. Budget convergence effort there.

---

## 10. Suggested repo structure and implementation order

```
dqd_poisson/
  geometry.py       # layer stack, gate rectangles, grid generation (graded Cartesian)
  materials.py      # εr map, band offsets, effective masses
  operator.py       # matrix-free ∇·ε∇ stencil, harmonic-mean faces (JAX)
  solver.py         # PCG + geometric-MG preconditioner; pyamg fallback; adjoint
  charges.py        # interface charge sheet / discrete defects, smearing, sampling
  greens.py         # optional layered-dielectric FFT accelerator
  reduce.py         # 1D z-Schrödinger, projection to V_2D, F_z extraction
  api.py            # DotPotential dataclass, superposition, ensemble driver
  validate/         # analytic tests, convergence scripts, paper-target checks
  reference/        # scikit-fem / dolfinx cross-check (optional)
```

**Build order for the coding agent:**
1. Geometry + εr map + graded grid; visualize slices.
2. Matrix-free operator + harmonic-mean faces; analytic unit tests (§9).
3. PCG + MG preconditioner; verify against `pyamg` on a few RHS.
4. Gate BC lifting + `φ_g^unit` precompute; **validate `|α|` and `F_z,0`**.
5. Interface charges (both modes) + smearing; disorder RHS.
6. z-reduction + `DotPotential` hand-off (§8).
7. Ensemble driver (`vmap`) + storage; optional FFT accelerator.
8. Convergence studies; hand off to the Schrödinger module.

---

## 11. Open decisions to confirm before/while implementing

- Exact **screening-gate** bias and geometry, and the far-field bottom BC used in COMSOL (affects `F_z` and `α` at the few-% level).
- Whether to model **discrete defects** (ensemble) or **continuum σ** (smooth studies) as the default disorder mode — spec supports both.
- Whether gradients (`lineax`/adjoint) are wanted now (for analytic lever arms and gradient-based tuning) or deferred.
- Whether to keep the **layered-Green's-function accelerator** in v1 or add it after the exact FV path is validated.
```
