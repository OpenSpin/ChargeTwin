"""Sanity-check slices of the geometry: eps_r(x,z) at y=dot-center, eps_r(y,z)
at x=dot-center, and the gate-metal mask in-plane (x,y). Also shows the
gate-metal mask overlaid on the cross-sections so gate locations are visible.
Run: python -m dqd_poisson.validate.plot_geometry
     (or directly: python3 dqd_poisson/validate/plot_geometry.py from repo root)
"""

import sys, os

# allow direct execution: "python3 plot_geometry.py" from validate/ dir
if __name__ == "__main__" and __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    from dqd_poisson import geometry as geo
else:
    from .. import geometry as geo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_MAT_CAT = {"SiGe": 0, "Si": 1, "SiO2": 2, "screening_metal": 3, "gate_metal": 4}
_MAT_NAMES = sorted(_MAT_CAT, key=_MAT_CAT.get)
_MAT_COLORS = ["gold", "yellowgreen", "indigo", "dimgray", "red"]
_LAYER_CAT = {
    "bottom_pad": "SiGe", "SiGe_lower": "SiGe", "Si_well": "Si", "SiGe_upper": "SiGe",
    "Si_cap": "Si", "SiO2_lower": "SiO2", "screening_slab": "SiO2", "top_branch": "SiO2",
}


def build_material_labels(grid, cfg, masks):
    """Categorical (Nx,Ny,Nz) material code per cell: SiGe/Si/SiO2 from the
    layer stack, with Dirichlet screening/gate cells overriding to metal."""
    z_break = geo.z_interfaces_nm(cfg)
    z_names = [l.name for l in geo.build_z_stack(cfg)] + ["top_branch"]
    zc_nm = grid.z_centers / geo.NM

    col = np.empty_like(zc_nm, dtype=int)
    for i, name in enumerate(z_names):
        lo, hi = z_break[i], z_break[i + 1]
        m = (zc_nm >= lo) & (zc_nm <= hi)
        col[m] = _MAT_CAT[_LAYER_CAT[name]]

    Nx, Ny, Nz = grid.shape
    labels = np.broadcast_to(col, (Nx, Ny, Nz)).copy()
    labels[masks.get("screening", np.zeros(grid.shape, bool))] = _MAT_CAT["screening_metal"]
    for g in ("PL", "PR", "barrier"):
        if g in masks:
            labels[masks[g]] = _MAT_CAT["gate_metal"]
    return labels


def _plot_material_slice(ax, fig, u_nm, z_nm, labels_2d, title, xlabel):
    cmap = plt.matplotlib.colors.ListedColormap(_MAT_COLORS)
    bounds = np.arange(-0.5, len(_MAT_NAMES) + 0.5)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.pcolormesh(u_nm, z_nm, labels_2d.T, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("z (nm)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(_MAT_NAMES)))
    cbar.set_ticklabels(_MAT_NAMES)


def main(outdir="output"):
    cfg = geo.GeometryConfig()
    grid = geo.build_grid(cfg)
    eps = geo.eps_r_field(grid, cfg)
    masks = geo.gate_masks(grid, cfg)
    labels = build_material_labels(grid, cfg, masks)

    x_nm = grid.x_centers / geo.NM
    y_nm = grid.y_centers / geo.NM
    z_nm = grid.z_centers / geo.NM

    # channel center in y
    xc = cfg.lateral_pad_nm + cfg.Lx_nm / 2.0
    yc = cfg.lateral_pad_nm + cfg.Ly_nm / 2.0
    y_ch0 = yc - cfg.d_channel_nm / 2.0
    y_ch1 = yc + cfg.d_channel_nm / 2.0

    # --- 1. Top view (x,y): gate footprint ---
    metal_any = np.zeros(grid.shape[:2], dtype=int)
    for i, (name, m) in enumerate(sorted(masks.items()), start=1):
        metal_any[m.any(axis=2)] = i
    fig0, ax0 = plt.subplots(figsize=(6, 6))
    cmap = plt.matplotlib.colors.ListedColormap(["white", "red", "blue", "green", "orange"])
    bounds = np.arange(-0.5, len(masks) + 1.5)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im0 = ax0.pcolormesh(x_nm, y_nm, metal_any.T, shading="auto", cmap=cmap, norm=norm)
    ax0.set_xlabel("x (nm)")
    ax0.set_ylabel("y (nm)")
    ax0.set_title("gate footprint (top view)")
    ax0.set_aspect("equal")
    cbar = fig0.colorbar(im0, ax=ax0, ticks=np.arange(1, len(masks) + 1))
    cbar.set_ticklabels(sorted(masks.keys()))
    # mark the xz and yz cut lines
    ax0.axvline(xc, color="k", ls="--", lw=0.5, label="xz cut (x center)")
    ax0.axhline(yc, color="gray", ls="--", lw=0.5, label="yz cut (y channel-center)")
    # channel boundaries
    ax0.axhline(y_ch0, color="gray", ls=":", lw=0.3)
    ax0.axhline(y_ch1, color="gray", ls=":", lw=0.3)
    fig0.savefig(f"{outdir}/gates_xy.png", dpi=120)
    print("wrote", f"{outdir}/gates_xy.png")

    # --- 2. xz cut through channel-center y (along channel) ---
    jy = np.argmin(np.abs(y_nm - yc))
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # eps_r
    im1 = ax1a.pcolormesh(x_nm, z_nm, eps[:, jy, :].T, shading="auto")
    ax1a.set_ylabel("z (nm)")
    ax1a.set_title("eps_r(x,z) at dot-center y")
    fig1.colorbar(im1, ax=ax1a, label="eps_r")

    # material breakdown (SiGe/Si/SiO2/screening metal/gate metal)
    _plot_material_slice(ax1b, fig1, x_nm, z_nm, labels[:, jy, :], "materials at xz cut", "x (nm)")
    fig1.savefig(f"{outdir}/eps_xz.png", dpi=120)
    print("wrote", f"{outdir}/eps_xz.png")

    # --- 3. yz cut through x=center (perpendicular to channel) ---
    ix = np.argmin(np.abs(x_nm - xc))
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # eps_r
    im2 = ax2a.pcolormesh(y_nm, z_nm, eps[ix, :, :].T, shading="auto")
    ax2a.set_ylabel("z (nm)")
    ax2a.set_title("eps_r(y,z) at x=center")
    fig2.colorbar(im2, ax=ax2a, label="eps_r")

    # material breakdown (SiGe/Si/SiO2/screening metal/gate metal)
    _plot_material_slice(ax2b, fig2, y_nm, z_nm, labels[ix, :, :], "materials at yz cut", "y (nm)")
    fig2.savefig(f"{outdir}/eps_yz.png", dpi=120)
    print("wrote", f"{outdir}/eps_yz.png")

    # --- Print geometry summary ---
    print("\n--- Geometry summary ---")
    z_break = geo.z_interfaces_nm(cfg)
    z_names = [l.name for l in geo.build_z_stack(cfg)] + ["gate_metal"]
    print("z interfaces (nm):")
    for i in range(len(z_names)):
        print(f"  {z_names[i]:20s}  z={z_break[i]:6.1f} .. {z_break[i+1]:6.1f}  dz={z_break[i+1]-z_break[i]:.1f}")
    print(f"well center z = {geo.well_z_center_nm(cfg):.1f} nm")
    # actual finger z range from gate_masks logic: top branch sits directly
    # above screening_slab
    finger_bot_nm = z_break[-2]  # top of screening_slab
    finger_top_nm = z_break[-1]
    print(f"finger z range = {finger_bot_nm:.1f} .. {finger_top_nm:.1f} nm")
    print(f"finger bottom to well center distance = {geo.well_z_center_nm(cfg) - finger_bot_nm:.1f} nm")
    print(f"finger bottom to Si-cap/SiO2 interface = {geo.interface_z_nm(cfg) - finger_bot_nm:.1f} nm")


if __name__ == "__main__":
    main()
