#!/usr/bin/env python
"""Figure 1A: MCAT's absorbing Markov chain drawn on a subject's T1, zoomed on a patch of cortex.

    python fig1a_chain_t1.py --t1 sub-X_T1w.nii.gz --pve sub-X_desc-pve_probseg.nii.gz \\
        --odf models/sub-X_model-msmtcsd_param-wm_dwimap.nii.gz -o fig1a_chain

Left: the slice, with the zoom window outlined.  Right: the window on the diffusion grid,
over the T1, with MCAT's states as its tissue rule makes them (mcat.afq.tissue_labels):
white matter (WM PVE > 0.4) is transient -- arrows show each voxel's hop probabilities to its
in-plane neighbours, from the fODF towards each neighbour (normalised over the neighbours a
walker may enter); gray matter absorbs (a valid ending); CSF is never entered.  Third panel:
the cascade from one seed, the in-plane chain solved exactly (states: voxel x heading, turns
within --cone): each white-matter voxel's expected visits, each gray-matter voxel's
probability of being where the walk ends.  All of it is an in-plane illustration of the chain
(MCAT's own walk is in 3D, on 26 neighbours and 26 headings, with a 35 degree cone).

--center / --size choose the window (world mm); by default it is the patch of the slice's
upper half with the most white matter touching gray matter.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
WM_THRESHOLD = 0.4                 # mcat.afq.WM_THRESHOLD (pyAFQ's tracker)
GM_COLOUR = "#e8743b"              # absorbing
CSF_COLOUR = "#4a90c2"             # never entered
ARROW_COLOUR = "#ffd84d"


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--t1", required=True, help="T1 (same world space as the diffusion data, e.g. HCP T1w)")
    p.add_argument("--pve", required=True, help="pyAFQ's desc-pve_probseg (CSF, GM, WM), on any grid (T1's)")
    p.add_argument("--odf", required=True, help="fODF SH (DIPY descoteaux07 legacy; pyAFQ's model dwimap)")
    p.add_argument("--view", choices=AXIS, default="coronal")
    p.add_argument("--slice-mm", type=float, default=-18.0,
                   help="slice position, world mm along the view's axis (default -18: coronal through "
                        "the central sulcus' region)")
    p.add_argument("--center", type=float, nargs=2, metavar=("U", "V"),
                   help="window centre in the slice's world coordinates, mm (default: automatic)")
    p.add_argument("--size", type=float, default=17.0, help="window side, mm (default 17)")
    p.add_argument("--seed-mm", type=float, nargs=2, metavar=("U", "V"),
                   help="the seed voxel of the cascade panel, world mm in the slice (default: among the window's "
                        "deepest white matter -- within 80%% of the largest distance to any non-white-matter voxel "
                        "-- the one whose walks end most in the window's gray matter)")
    p.add_argument("--cone", type=float, default=45.0,
                   help="cascade: largest turn per hop, degrees (in plane, 8 neighbours: 45)")
    p.add_argument("--min-p", type=float, default=0.06, help="hop probabilities below this are not drawn")
    p.add_argument("--fontsize", type=float, default=13)
    p.add_argument("-o", "--out", default="fig1a_chain")
    return p.parse_args()


# ----------------------------------------------------------------------- geometry
def plane_axes(view):
    """(the slice axis, the two in-plane world axes drawn as horizontal / vertical)."""
    k = AXIS[view]
    return k, [d for d in range(3) if d != k]


def slice_index(img, k, mm):
    """Voxel index along the voxel axis closest to world axis ``k`` for world ``mm``."""
    ax = int(np.argmax(np.abs(img.affine[k, :3])))
    centre = np.array(img.shape[:3]) / 2.0
    w = img.affine[:3, :3] @ centre + img.affine[:3, 3]
    w[k] = mm
    ijk = np.linalg.solve(img.affine[:3, :3], w - img.affine[:3, 3])
    return ax, int(np.clip(round(ijk[ax]), 0, img.shape[ax] - 1))


def slab(img, ax, i, data=None):
    """The voxel slice ``i`` along voxel axis ``ax`` (all trailing dims)."""
    d = img.dataobj if data is None else data
    sl = [slice(None)] * 3
    sl[ax] = i
    return np.asarray(d[tuple(sl)])


def voxel_world(img, ax, i):
    """World coordinates (n0, n1, 3) of the voxel centres of slice ``i``."""
    shp = [img.shape[d] for d in range(3) if d != ax]
    g = np.indices(shp).reshape(2, -1).T
    ijk = np.insert(g, ax, i, axis=1)
    return nib.affines.apply_affine(img.affine, ijk).reshape(*shp, 3)


def t1_plane(t1, k, mm, dims, extent, res=0.25):
    """The T1 resampled (linear) on a fine world grid of the plane: (image, extent)."""
    u = np.arange(extent[0], extent[1] + res / 2, res)
    v = np.arange(extent[2], extent[3] + res / 2, res)
    U, V = np.meshgrid(u, v)
    w = np.zeros(U.shape + (3,))
    w[..., dims[0]], w[..., dims[1]], w[..., k] = U, V, mm
    ijk = nib.affines.apply_affine(np.linalg.inv(t1.affine), w.reshape(-1, 3)).T
    data = np.asarray(t1.dataobj, np.float32)
    img = ndimage.map_coordinates(data, ijk, order=1, mode="constant").reshape(U.shape)
    return img, (u[0], u[-1], v[0], v[-1])


def pve_on_slice(pve, world):
    """CSF, GM, WM fractions at the diffusion-grid voxel centres ``world`` (n0, n1, 3), sampled
    linearly from the PVE on its own grid (pyAFQ writes it on the T1 grid), as MCAT resamples it."""
    ijk = nib.affines.apply_affine(np.linalg.inv(pve.affine), world.reshape(-1, 3)).T
    data = np.asarray(pve.dataobj, np.float32)
    return [ndimage.map_coordinates(data[..., c], ijk, order=1, mode="nearest").reshape(world.shape[:2])
            for c in range(3)]


# ----------------------------------------------------------------------- the chain, in plane
def fodf_basis(n_coef, dirs):
    """(n_coef, len(dirs)): sf = sh @ B (descoteaux07 legacy, symmetric or full; mcat.geometry)."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sym = (-3.0 + np.sqrt(1.0 + 8.0 * n_coef)) / 2.0
    full = not float(sym).is_integer()
    order = int(round(np.sqrt(n_coef) - 1)) if full else int(round(sym))
    return np.asarray(sh_to_sf_matrix(Sphere(xyz=dirs), sh_order_max=order, full_basis=full,
                                      return_inv=False), np.float64)


def in_plane_steps(img, ax):
    """The 8 in-plane neighbour offsets (voxel), and their world unit directions."""
    offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    plane = [d for d in range(3) if d != ax]
    dirs = []
    for a, b in offs:
        o = np.zeros(3)
        o[plane[0]], o[plane[1]] = a, b
        w = img.affine[:3, :3] @ o
        dirs.append(w / np.linalg.norm(w))
    return np.array(offs), np.array(dirs)


def hop_probabilities(sh, state, offs, B):
    """P[i, j, n]: in-plane hop probabilities of transient voxel (i, j) to neighbour n, from the
    fODF towards each neighbour; CSF and outside are never entered (probability 0)."""
    sf = np.clip(np.einsum("ijc,cn->ijn", sh, B), 0, None)
    n0, n1 = state.shape
    P = np.zeros((n0, n1, len(offs)))
    for n, (a, b) in enumerate(offs):
        tgt = np.zeros((n0, n1), np.int8)
        tgt[max(-a, 0):n0 - max(a, 0), max(-b, 0):n1 - max(b, 0)] = \
            state[max(a, 0):n0 + min(a, 0), max(b, 0):n1 + min(b, 0)]
        P[..., n] = sf[..., n] * np.isin(tgt, (1, 2))           # enter WM or GM only
    s = P.sum(-1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-30), 0)
    P[state != 1] = 0
    return P


def cascade(state, sf, offs, seed, cone=45.0, tol=1e-5, max_steps=2000):
    """The in-plane chain solved exactly from one seed: states (voxel, heading), the first hop
    from the seed to every enterable neighbour by the fODF towards it (both ways along the
    fibre), then hops within ``cone`` of the heading, by the fODF, renormalised over the
    enterable ones; entering gray matter absorbs, CSF / outside are never entered, and a
    state with nowhere to go loses its mass (an invalid ending).  Returns (visits (n0, n1):
    expected visits per voxel, the seed's included; absorbed (n0, n1): probability of ending
    in each gray-matter voxel; lost: the probability of no valid ending)."""
    n0, n1 = state.shape
    H = len(offs)
    unit = offs / np.linalg.norm(offs, axis=1, keepdims=True)
    turn_ok = unit @ unit.T >= np.cos(np.radians(cone)) - 1e-9          # [heading, next hop]
    ii, jj = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    tgt_i = ii[..., None] + offs[:, 0]
    tgt_j = jj[..., None] + offs[:, 1]
    inside = (tgt_i >= 0) & (tgt_i < n0) & (tgt_j >= 0) & (tgt_j < n1)
    tstate = np.where(inside, state[np.clip(tgt_i, 0, n0 - 1), np.clip(tgt_j, 0, n1 - 1)], 0)
    enter = inside & np.isin(tstate, (1, 2))                          # (n0, n1, 8)
    w = sf * enter
    visits = np.zeros((n0, n1))
    absorbed = np.zeros((n0, n1))
    visits[tuple(seed)] = 1.0
    # launch: one unit split over the seed's enterable neighbours by the fODF
    x = np.zeros((n0, n1, H))                    # mass arriving at (voxel) with heading h
    w0 = w[tuple(seed)]
    if w0.sum() <= 0:
        return visits, absorbed, 1.0
    for n in range(H):
        if w0[n] > 0:
            ti, tj = seed[0] + offs[n, 0], seed[1] + offs[n, 1]
            if state[ti, tj] == 2:
                absorbed[ti, tj] += w0[n] / w0.sum()
            else:
                x[ti, tj, n] += w0[n] / w0.sum()
    lost = 0.0
    for _ in range(max_steps):
        mass = x.sum()
        if mass < tol:
            break
        visits += x.sum(-1)
        nxt = np.zeros_like(x)
        for h in range(H):
            xh = x[..., h]
            if not xh.any():
                continue
            wh = w * turn_ok[h]                          # hops allowed from heading h
            tot = wh.sum(-1)
            stuck = (tot <= 0) & (xh > 0)
            lost += xh[stuck].sum()
            p = np.where(tot[..., None] > 0, wh / np.maximum(tot[..., None], 1e-300), 0) * xh[..., None]
            for n in range(H):
                pn = p[..., n]
                if not pn.any():
                    continue
                src = np.nonzero(pn)
                ti, tj = src[0] + offs[n, 0], src[1] + offs[n, 1]
                gm = state[ti, tj] == 2
                np.add.at(absorbed, (ti[gm], tj[gm]), pn[src][gm])
                np.add.at(nxt[..., n], (ti[~gm], tj[~gm]), pn[src][~gm])
        x = nxt
    return visits, absorbed, lost + x.sum()


# ----------------------------------------------------------------------- window
def auto_center(state, world, dims, size, zoom_vox):
    """The window (centre, world mm) in the slice's upper half with the most white-matter voxels
    touching gray matter."""
    wm, gm = state == 1, state == 2
    touch = wm & ndimage.binary_dilation(gm, np.ones((3, 3), bool))
    score = ndimage.uniform_filter(touch.astype(float), zoom_vox) * \
        np.minimum(ndimage.uniform_filter(wm.astype(float), zoom_vox),
                   ndimage.uniform_filter(gm.astype(float), zoom_vox))
    up = world[..., dims[1]]
    score *= 1 - ndimage.uniform_filter(((state == 3) | (state == 0)).astype(float), zoom_vox)
    score[up < np.percentile(up[state > 0], 55)] = 0           # the upper part of the brain
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return world[i, j, dims[0]], world[i, j, dims[1]]


def main():
    a = parse()
    t1, pve, odf = nib.load(a.t1), nib.load(a.pve), nib.load(a.odf)
    k, dims = plane_axes(a.view)
    ax, i = slice_index(odf, k, a.slice_mm)
    world = voxel_world(odf, ax, i)                            # (n0, n1, 3)
    mm = float(world[0, 0, k])
    sh = slab(odf, ax, i).astype(np.float64)
    csf, gm, wm = pve_on_slice(pve, world)
    inside = np.any(sh != 0, axis=-1)
    state = np.zeros(inside.shape, np.int8)                     # 0 outside, 1 WM, 2 GM, 3 CSF
    state[inside & (wm > WM_THRESHOLD)] = 1
    state[inside & (state == 0) & (gm >= csf)] = 2
    state[inside & (state == 0)] = 3

    zooms = np.sqrt((odf.affine[:3, :3] ** 2).sum(0))
    vox = float(np.mean([zooms[d] for d in range(3) if d != ax]))
    zoom_vox = max(3, int(round(a.size / vox)))
    cu, cv = a.center if a.center else auto_center(state, world, dims, a.size, zoom_vox)

    offs, dirs = in_plane_steps(odf, ax)
    B = fodf_basis(sh.shape[-1], dirs)
    sf = np.clip(np.einsum("ijc,cn->ijn", sh, B), 0, None)
    P = hop_probabilities(sh, state, offs, B)

    half = a.size / 2
    win = (np.abs(world[..., dims[0]] - cu) <= half) & (np.abs(world[..., dims[1]] - cv) <= half)
    depth = ndimage.distance_transform_edt(state == 1)       # white matter's distance to anything else
    if a.seed_mm:
        dist = (world[..., dims[0]] - a.seed_mm[0]) ** 2 + (world[..., dims[1]] - a.seed_mm[1]) ** 2
        seed = np.array(np.unravel_index(int(np.argmin(np.where(state == 1, dist, np.inf))), state.shape))
    else:   # the window's deepest white matter; of it, the seed whose walks end most in the window's GM
        deep = np.where(win & (state == 1), depth, 0)
        cand = np.argwhere(deep >= 0.8 * deep.max()) if deep.max() > 0 else np.argwhere(win & (state == 1))
        cand = cand[np.random.default_rng(0).permutation(len(cand))[:40]]
        rows, cols = np.nonzero(win)                # score on a crop around the window (fast)
        m = zoom_vox
        r0, r1 = max(rows.min() - m, 0), min(rows.max() + m + 1, state.shape[0])
        c0, c1 = max(cols.min() - m, 0), min(cols.max() + m + 1, state.shape[1])
        crop = (slice(r0, r1), slice(c0, c1))
        score = [cascade(state[crop], sf[crop], offs, c - [r0, c0], a.cone, tol=1e-3)[1][win[crop]].sum()
                 for c in cand]
        seed = cand[int(np.argmax(score))]
    visits, absorbed, lost = cascade(state, sf, offs, seed, a.cone)
    print(f"cascade from ({world[tuple(seed)][dims[0]]:.1f}, {world[tuple(seed)][dims[1]]:.1f}) mm: "
          f"{absorbed.sum():.3f} absorbed in gray matter ({absorbed[win].sum():.3f} inside the window), "
          f"{lost:.3f} no valid ending")

    fs = a.fontsize
    fig = plt.figure(figsize=(19, 6.6))
    gsp = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)

    # left: the whole slice with the window
    ax0 = fig.add_subplot(gsp[0])
    allw = world[state > 0]
    ext0 = [allw[:, dims[0]].min() - 4, allw[:, dims[0]].max() + 4,
            allw[:, dims[1]].min() - 4, allw[:, dims[1]].max() + 4]
    side = max(ext0[1] - ext0[0], ext0[3] - ext0[2])           # square, as the other two panels
    for j in (0, 2):
        mid = (ext0[j] + ext0[j + 1]) / 2
        ext0[j], ext0[j + 1] = mid - side / 2, mid + side / 2
    img0, e0 = t1_plane(t1, k, mm, dims, ext0, res=0.7)
    ax0.imshow(img0, cmap="gray", origin="lower", extent=e0, vmax=np.percentile(img0, 99.5))
    ax0.add_patch(Rectangle((cu - half, cv - half), a.size, a.size, fill=False, ec=ARROW_COLOUR, lw=2))

    # right: the window
    ax1 = fig.add_subplot(gsp[1])
    e1 = (cu - half, cu + half, cv - half, cv + half)
    img1, e1 = t1_plane(t1, k, mm, dims, e1, res=0.2)
    ax1.imshow(img1, cmap="gray", origin="lower", extent=e1, vmin=0, vmax=1.8 * np.percentile(img0, 99.5))
    patches_gm, patches_csf, grid = [], [], []
    for (p, q) in np.argwhere(win):
        u, v = world[p, q, dims[0]], world[p, q, dims[1]]
        r = Rectangle((u - vox / 2, v - vox / 2), vox, vox)
        grid.append(Rectangle((u - vox / 2, v - vox / 2), vox, vox))
        if state[p, q] == 2:
            patches_gm.append(r)
        elif state[p, q] == 3:
            patches_csf.append(r)
    ax1.add_collection(PatchCollection(grid, fc="none", ec="white", lw=0.3, alpha=0.35))
    ax1.add_collection(PatchCollection(patches_gm, fc=GM_COLOUR, ec=GM_COLOUR, lw=0.6, alpha=0.55))
    ax1.add_collection(PatchCollection(patches_csf, fc=CSF_COLOUR, ec=CSF_COLOUR, lw=0.6, alpha=0.5))

    step = [np.array([dirs[n][dims[0]], dirs[n][dims[1]]]) for n in range(len(offs))]
    for (p, q) in np.argwhere(win & (state == 1)):
        u, v = world[p, q, dims[0]], world[p, q, dims[1]]
        ax1.plot(u, v, "o", ms=2.6, color="white", mec="black", mew=0.3, zorder=3)
        for n in range(len(offs)):
            pr = P[p, q, n]
            if pr < a.min_p:
                continue
            length = 0.47 * vox * (1.0 if abs(offs[n]).sum() == 1 else np.sqrt(2))
            d = step[n] / np.linalg.norm(step[n])
            ax1.add_patch(FancyArrowPatch((u + 0.12 * vox * d[0], v + 0.12 * vox * d[1]),
                                          (u + length * d[0], v + length * d[1]),
                                          arrowstyle="-|>", mutation_scale=5 + 12 * pr, lw=0.5 + 3.0 * pr,
                                          color=ARROW_COLOUR, alpha=0.45 + 0.55 * min(1, 2 * pr), zorder=4))
    ax1.set_xlim(e1[0], e1[1])
    ax1.set_ylim(e1[2], e1[3])
    ax1.legend(handles=[
        plt.Line2D([], [], color=ARROW_COLOUR, lw=2, marker=">", ms=7, mfc=ARROW_COLOUR,
                   label="white matter: transient"),
        Patch(fc=GM_COLOUR, alpha=0.7, label="gray matter: absorbing"),
        Patch(fc=CSF_COLOUR, alpha=0.7, label="CSF: never entered")],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, frameon=False, fontsize=fs - 3)

    # third: the cascade from one seed
    from matplotlib.colors import LogNorm

    ax2 = fig.add_subplot(gsp[2])
    ax2.imshow(img1, cmap="gray", origin="lower", extent=e1, vmin=0, vmax=1.8 * np.percentile(img0, 99.5))
    vis_in = visits[win & (state == 1)]
    vmax = float(vis_in.max()) if vis_in.size else 1.0
    vnorm = LogNorm(vmax * 1e-3, vmax)
    anorm = LogNorm(1e-3, max(float(absorbed[win].max()), 2e-3))
    cells_v, val_v, cells_a, val_a, cells_c = [], [], [], [], []
    for (p, q) in np.argwhere(win):
        r = Rectangle((world[p, q, dims[0]] - vox / 2, world[p, q, dims[1]] - vox / 2), vox, vox)
        if state[p, q] == 1 and visits[p, q] >= vmax * 1e-3:
            cells_v.append(r)
            val_v.append(visits[p, q])
        elif state[p, q] == 2:
            cells_a.append(r)
            val_a.append(max(absorbed[p, q], 1e-12))
        elif state[p, q] == 3:
            cells_c.append(r)
    ax2.add_collection(PatchCollection(grid, fc="none", ec="white", lw=0.3, alpha=0.35))
    pv = PatchCollection(cells_v, cmap="magma", norm=vnorm, ec="none", alpha=0.9, zorder=2)
    pv.set_array(np.asarray(val_v))
    ax2.add_collection(pv)
    pa = PatchCollection(cells_a, cmap="Greens", norm=anorm, ec=GM_COLOUR, lw=0.8, alpha=0.95, zorder=2)
    pa.set_array(np.asarray(val_a))
    ax2.add_collection(pa)
    ax2.add_collection(PatchCollection(cells_c, fc=CSF_COLOUR, ec=CSF_COLOUR, lw=0.6, alpha=0.5))
    su, sv = world[tuple(seed)][dims[0]], world[tuple(seed)][dims[1]]
    ax2.plot(su, sv, "*", ms=18, color="white", mec="black", mew=0.9, zorder=6)
    ax1.plot(su, sv, "*", ms=18, color="white", mec="black", mew=0.9, zorder=6)
    ax0.plot(su, sv, "*", ms=7, color="white", mec="black", mew=0.5, zorder=6)
    ax2.set_xlim(e1[0], e1[1])
    ax2.set_ylim(e1[2], e1[3])
    for coll, lab, loc in ((pv, "expected visits", 0.0),
                           (pa, "P(ending here)", 0.58)):
        cax = ax2.inset_axes([loc, -0.09, 0.42, 0.035])
        cb = fig.colorbar(coll, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fs - 4)
        cb.ax.tick_params(labelsize=fs - 5)
    for axx, letter in ((ax0, "a"), (ax1, "b"), (ax2, "c")):
        axx.text(0.0, 1.015, letter, transform=axx.transAxes, ha="left", va="bottom", fontsize=fs + 7,
                 fontweight="bold", fontfamily=["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"])
        axx.set_xticks([])
        axx.set_yticks([])
        axx.set_aspect("equal")
        for s in axx.spines.values():
            s.set_visible(False)
    fig.savefig(f"{a.out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{a.out}.pdf", bbox_inches="tight", facecolor="white")
    print(f"window centre ({cu:.1f}, {cv:.1f}) mm, {win.sum()} voxels; wrote {a.out}.png / .pdf")


if __name__ == "__main__":
    main()