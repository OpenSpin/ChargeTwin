"""Layer stack, gate rectangles, and graded Cartesian grid generation.

Spec: poisson_simulator_spec.md §3 (geometry/materials) and §4 (discretization).

z=0 is the top of the bottom ground plane (thin metal, V=0): a Dirichlet
phi=0 boundary condition at z=0 stands in for that slab, no bulk metal layer
needed below it (optional cfg.bottom_pad_nm can still pad the SiGe below,
for far-BC convergence studies).

Layer stack bottom -> top (z in nm, from z=0), per spec §3.1 -- both SiGe
layers sit BELOW the well, the well is directly under the Si cap:
  SiGe_lower    ( 0.0, 30.0)  eps=13.2   <- substrate
  SiGe_barrier  (30.0, 90.0)  eps=13.2   <- barrier, U0=150 meV above well
  Si_well       (90.0,100.0)  eps=12.0   <- quantum well
  Si_cap        (100.0,101.5) eps=12.0   <- trapped-charge interface (top face)
  SiO2_lower    (101.5,111.5) eps=3.9
  screening_slab(111.5,121.5) metal outside channel (V=0) / SiO2 inside channel
  top branch    (121.5,148.5) outside channel: SiO2 filler; inside channel:
                               three metallic gates (PL, barrier, PR) with
                               3 nm SiO2 isolation on both sides of the 142 nm
                               channel (136 nm gate span in y).

Channel runs along x-axis, width d_channel=142 nm in y. Gates (PL, PR, barrier)
are short bars spanning 136 nm of the channel in y (3 nm SiO2 isolation each
side), sitting in the top branch directly above the screening_slab. Screening
metal exists only in the screening_slab layer, outside the channel in y.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import materials as mat

NM = 1e-9  # nm -> m


@dataclass(frozen=True)
class LayerSpec:
    name: str
    thickness_nm: float
    eps_r: float
    u_band_eV: float  # conduction-band offset relative to Si well


@dataclass(frozen=True)
class Gate:
    name: str
    x0_nm: float
    x1_nm: float
    y0_nm: float
    y1_nm: float
    voltage: float  # volts, nominal operating point (spec §3.2)


@dataclass
class GeometryConfig:
    # Lateral footprint (spec §3): 660 x 582 nm^2
    Lx_nm: float = 660.0
    Ly_nm: float = 582.0

    # z layer stack, bottom -> top (spec §3.1). Well sits directly under the
    # Si cap -- both SiGe layers (substrate + barrier) are BELOW the well,
    # there is no SiGe layer above it.
    t_sige_lower_nm: float = 30.0    # SiGe substrate, below the barrier
    t_sige_barrier_nm: float = 60.0  # SiGe barrier, directly below the well
    t_well_nm: float = 10.0          # Si quantum well
    t_cap_nm: float = 1.5            # Si cap; charge-trap interface is Si_cap/SiO2_lower
    t_oxide_lower_nm: float = 10.0   # SiO2 below screening slab (spec §3.1)
    t_screening_slab_nm: float = 10.0  # screening METAL slab (outside channel); SiO2 inside channel
    t_gate_metal_nm: float = 27.0    # top branch height: PL/PR/barrier fingers, inside channel only

    # Gate geometry (spec §3.2/3.3)
    w_metal_nm: float = 45.0  # plunger/barrier finger width in x
    plunger_pitch_nm: float = 95.0  # PL-PR center-to-center in x (clean-device d0=95nm, spec §3.4); barrier centered between
    d_channel_nm: float = 142.0  # channel width in y (gap in screening layer)
    t_gate_iso_nm: float = 3.0  # SiO2 isolation between gate fingers and channel edge (each side, in y)

    # Operating point. Fig.1(f) lever-arm sweep taken around V_L,V_R ~ 0.19 V
    v_PL: float = 0.19
    v_PR: float = 0.19
    v_barrier: float = -0.10
    v_screening: float = 0.0  # grounded

    # Grid resolution targets (nm), graded per spec §4
    dz_fine_nm: float = 1.0
    dz_coarse_nm: float = 8.0
    dxy_fine_nm: float = 2.0
    dxy_coarse_nm: float = 10.0

    # Far-field BC placement: extra padding beyond the 660x582 footprint
    lateral_pad_nm: float = 150.0
    bottom_pad_nm: float = 0.0  # optional extra SiGe padding below z=0 ground plane (far-BC studies)

    # Far-field boundary condition (spec §2): z=0 is the top of the bottom
    # ground metal (thin slab, V=0) -- Dirichlet phi=0 at z=0 stands in for it.
    far_bottom_ground: bool = True
    far_lateral_ground: bool = False


def build_z_stack(cfg: GeometryConfig) -> List[LayerSpec]:
    """Dielectric layer stack, bottom -> top, from z=0 (ground plane) to the
    top of the screening_slab. The top branch (gates/oxide, height
    t_gate_metal_nm) is handled separately in z_interfaces_nm/eps_r_field/
    gate_masks since its content forks by lateral position: SiO2 filler
    everywhere except the gate footprint inside the channel (Dirichlet)."""
    layers = []
    if cfg.bottom_pad_nm > 0:
        layers.append(LayerSpec("bottom_pad", cfg.bottom_pad_nm, mat.EPS_R_SIGE, mat.U0_SIGE_BARRIER))
    layers += [
        LayerSpec("SiGe_lower", cfg.t_sige_lower_nm, mat.EPS_R_SIGE, mat.U0_SIGE_BARRIER),
        LayerSpec("SiGe_barrier", cfg.t_sige_barrier_nm, mat.EPS_R_SIGE, mat.U0_SIGE_BARRIER),
        LayerSpec("Si_well", cfg.t_well_nm, mat.EPS_R_SI, 0.0),
        LayerSpec("Si_cap", cfg.t_cap_nm, mat.EPS_R_SI, 0.0),
        LayerSpec("SiO2_lower", cfg.t_oxide_lower_nm, mat.EPS_R_SIO2, 3.0),
        LayerSpec("screening_slab", cfg.t_screening_slab_nm, mat.EPS_R_SIO2, 3.0),
    ]
    return layers


def z_interfaces_nm(cfg: GeometryConfig) -> List[float]:
    """Cumulative z-coordinates (nm) of every layer boundary, z=0 at the
    ground plane (bottom of optional padding), increasing upward. The final
    entry is the top of the top branch (gates/oxide, height t_gate_metal_nm
    above the screening_slab)."""
    zs = [0.0]
    for layer in build_z_stack(cfg):
        zs.append(zs[-1] + layer.thickness_nm)
    zs.append(zs[-1] + cfg.t_gate_metal_nm)
    return zs


def well_z_center_nm(cfg: GeometryConfig) -> float:
    """z-coordinate (nm) of the Si quantum-well mid-plane, for diagnostics."""
    z_break = z_interfaces_nm(cfg)
    names = [l.name for l in build_z_stack(cfg)]
    i = names.index("Si_well")
    return 0.5 * (z_break[i] + z_break[i + 1])


def interface_z_nm(cfg: GeometryConfig) -> float:
    """z of the Si-cap / SiO2 interface -- where trapped charges live (spec §6)."""
    z_break = z_interfaces_nm(cfg)
    names = [l.name for l in build_z_stack(cfg)]
    i = names.index("Si_cap")
    return z_break[i + 1]


def build_gates(cfg: GeometryConfig) -> List[Gate]:
    """Three finger gates (PL, PR, barrier) spanning 136 nm of the 142 nm
    channel in y (3 nm SiO2 isolation each side), plus two screening-gate
    halves outside the channel in y.

    Channel runs along x. Gates are short bars in x (width w_metal) that span
    d_channel - 2*t_gate_iso in y. Screening metal exists only in the
    screening_slab layer, outside the channel (y outside the d_channel gap)."""
    xc = cfg.lateral_pad_nm + cfg.Lx_nm / 2.0
    half_pitch = cfg.plunger_pitch_nm / 2.0
    hw = cfg.w_metal_nm / 2.0

    x_pl_c = xc - half_pitch
    x_pr_c = xc + half_pitch
    x_b_c = xc

    # Channel centered on domain in y
    yc = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0
    hc = cfg.d_channel_nm / 2.0
    y_ch0 = yc - hc
    y_ch1 = yc + hc

    # Finger gates: inset by t_gate_iso_nm from the channel edge in y (SiO2
    # isolation strip), width w_metal in x
    y_gate0 = y_ch0 + cfg.t_gate_iso_nm
    y_gate1 = y_ch1 - cfg.t_gate_iso_nm
    gates = [
        Gate("PL", x_pl_c - hw, x_pl_c + hw, y_gate0, y_gate1, cfg.v_PL),
        Gate("PR", x_pr_c - hw, x_pr_c + hw, y_gate0, y_gate1, cfg.v_PR),
        Gate("barrier", x_b_c - hw, x_b_c + hw, y_gate0, y_gate1, cfg.v_barrier),
    ]

    # Screening gate: two halves outside the channel in y, only in
    # screening_slab layer (not in top metal). Full x-extent.
    x_full0 = 0.0
    x_full1 = cfg.lateral_pad_nm * 2 + cfg.Lx_nm
    y_full0 = 0.0
    y_full1 = cfg.lateral_pad_nm * 2 + cfg.Ly_nm
    gates.append(Gate("screening_bottom", x_full0, x_full1, y_full0, y_ch0, cfg.v_screening))
    gates.append(Gate("screening_top", x_full0, x_full1, y_ch1, y_full1, cfg.v_screening))

    return gates


def graded_axis(breakpoints_nm: List[float], spacings_nm: List[float]) -> np.ndarray:
    """Build 1D node coordinates (meters) that exactly hit every breakpoint,
    using the target spacing within each segment (spec §4: 'snap the mesh to
    geometry'). len(spacings_nm) == len(breakpoints_nm) - 1."""
    assert len(spacings_nm) == len(breakpoints_nm) - 1
    nodes = [breakpoints_nm[0]]
    for i in range(len(spacings_nm)):
        a, b, h = breakpoints_nm[i], breakpoints_nm[i + 1], spacings_nm[i]
        n = max(1, round((b - a) / h))
        seg = np.linspace(a, b, n + 1)[1:]
        nodes.extend(seg.tolist())
    return np.array(nodes) * NM


@dataclass
class GridSpec:
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    z_nodes: np.ndarray
    x_centers: np.ndarray = field(init=False)
    y_centers: np.ndarray = field(init=False)
    z_centers: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x_centers = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        self.y_centers = 0.5 * (self.y_nodes[:-1] + self.y_nodes[1:])
        self.z_centers = 0.5 * (self.z_nodes[:-1] + self.z_nodes[1:])

    @property
    def shape(self):
        return (len(self.x_centers), len(self.y_centers), len(self.z_centers))


def build_grid(cfg: GeometryConfig) -> GridSpec:
    Lx_total = cfg.Lx_nm + 2 * cfg.lateral_pad_nm
    Ly_total = cfg.Ly_nm + 2 * cfg.lateral_pad_nm

    x_break = [0.0, cfg.lateral_pad_nm, cfg.lateral_pad_nm + cfg.Lx_nm, Lx_total]
    x_space = [cfg.dxy_coarse_nm, cfg.dxy_fine_nm, cfg.dxy_coarse_nm]
    y_break = [0.0, cfg.lateral_pad_nm, cfg.lateral_pad_nm + cfg.Ly_nm, Ly_total]
    y_space = [cfg.dxy_coarse_nm, cfg.dxy_fine_nm, cfg.dxy_coarse_nm]

    z_break = z_interfaces_nm(cfg)
    n_seg = len(z_break) - 1
    z_space = [cfg.dz_coarse_nm] * n_seg
    fine_from = max(0, n_seg - 6)
    for i in range(fine_from, n_seg):
        z_space[i] = cfg.dz_fine_nm

    x_nodes = graded_axis(x_break, x_space)
    y_nodes = graded_axis(y_break, y_space)
    z_nodes = graded_axis(z_break, z_space)
    return GridSpec(x_nodes, y_nodes, z_nodes)


def eps_r_field(grid: GridSpec, cfg: GeometryConfig) -> np.ndarray:
    """Cell-centered relative permittivity, (Nx,Ny,Nz)."""
    z_stack = build_z_stack(cfg)
    z_break = z_interfaces_nm(cfg)
    zc_nm = grid.z_centers / NM

    eps_col = np.empty_like(zc_nm)
    for i in range(len(z_stack)):
        lo, hi = z_break[i], z_break[i + 1]
        mask = (zc_nm >= lo) & (zc_nm < hi)
        eps_col[mask] = z_stack[i].eps_r
    # top branch (gates/oxide): SiO2 filler (irrelevant under Dirichlet gate cells)
    lo, hi = z_break[-2], z_break[-1]
    mask = (zc_nm >= lo) & (zc_nm <= hi)
    eps_col[mask] = mat.EPS_R_SIO2

    Nx, Ny, Nz = grid.shape
    return np.broadcast_to(eps_col, (Nx, Ny, Nz)).copy()


def u_band_field(grid: GridSpec, cfg: GeometryConfig) -> np.ndarray:
    """Cell-centered conduction-band offset U_band (eV), (Nx,Ny,Nz)."""
    z_stack = build_z_stack(cfg)
    z_break = z_interfaces_nm(cfg)
    zc_nm = grid.z_centers / NM

    u_col = np.empty_like(zc_nm)
    for i in range(len(z_stack)):
        lo, hi = z_break[i], z_break[i + 1]
        mask = (zc_nm >= lo) & (zc_nm < hi)
        u_col[mask] = z_stack[i].u_band_eV
    lo, hi = z_break[-2], z_break[-1]
    mask = (zc_nm >= lo) & (zc_nm <= hi)
    u_col[mask] = z_stack[i].u_band_eV  # top branch: same as layer below (unused where metal)

    Nx, Ny, Nz = grid.shape
    return np.broadcast_to(u_col, (Nx, Ny, Nz)).copy()


def gate_masks(grid: GridSpec, cfg: GeometryConfig) -> "dict[str, np.ndarray]":
    """Boolean (Nx,Ny,Nz) mask per gate.

    PL/PR/barrier fingers: only in the top branch (height t_gate_metal_nm,
    directly above screening_slab), spanning 136 nm of the channel in y
    (d_channel=142 nm minus 3 nm SiO2 isolation each side), width
    w_metal=45 nm in x.

    Screening: only in the screening_slab layer, outside the channel in y
    (y < y_center - d_channel/2 or y > y_center + d_channel/2). No screening
    metal in the top branch."""
    z_break = z_interfaces_nm(cfg)
    z_names = [l.name for l in build_z_stack(cfg)]
    i_screen_slab = z_names.index("screening_slab")
    screen_slab_bot = z_break[i_screen_slab]
    screen_slab_top = z_break[i_screen_slab + 1]
    metal_top_bot = z_break[-2]   # bottom of top branch (= top of screening_slab)
    metal_top_top = z_break[-1]

    zc_nm = grid.z_centers / NM
    # Fingers: only in the top branch, directly above screening_slab
    z_mask_finger = (zc_nm >= metal_top_bot) & (zc_nm <= metal_top_top)
    # Screening: only in screening_slab layer
    z_mask_screen = (zc_nm >= screen_slab_bot) & (zc_nm <= screen_slab_top)

    xc_nm = grid.x_centers / NM
    yc_nm = grid.y_centers / NM

    masks = {}
    for g in build_gates(cfg):
        is_finger = g.name in ("PL", "PR", "barrier")
        z_mask = z_mask_finger if is_finger else z_mask_screen
        x_mask = (xc_nm >= g.x0_nm) & (xc_nm < g.x1_nm)
        y_mask = (yc_nm >= g.y0_nm) & (yc_nm < g.y1_nm)
        m3 = x_mask[:, None, None] & y_mask[None, :, None] & z_mask[None, None, :]
        if g.name in masks:
            masks[g.name] = masks[g.name] | m3
        else:
            masks[g.name] = m3
    # merge the two screening halves into one logical gate
    if "screening_bottom" in masks or "screening_top" in masks:
        masks["screening"] = masks.pop("screening_bottom", np.zeros(grid.shape, bool)) | \
                              masks.pop("screening_top", np.zeros(grid.shape, bool))
    return masks


def gate_voltages(cfg: GeometryConfig) -> "dict[str, float]":
    return {"PL": cfg.v_PL, "PR": cfg.v_PR, "barrier": cfg.v_barrier, "screening": cfg.v_screening}


def ground_mask(grid: GridSpec, cfg: GeometryConfig) -> np.ndarray:
    """Boolean (Nx,Ny,Nz) mask of far-field cells held at Dirichlet phi=0
    (spec §2). Removes the pure-Neumann near-nullspace."""
    Nx, Ny, Nz = grid.shape
    m = np.zeros((Nx, Ny, Nz), dtype=bool)
    if cfg.far_bottom_ground:
        m[:, :, 0] = True
    if cfg.far_lateral_ground:
        m[0, :, :] = True
        m[-1, :, :] = True
        m[:, 0, :] = True
        m[:, -1, :] = True
    return m