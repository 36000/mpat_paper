#!/usr/bin/env python
"""Figure of the RGVP on the one oblique slice that holds as much of it as possible: all four
bundles (each eye to each LGN) as one density in one colour, their visitation ODFs, every
bundle's core, and the endpoint and exclusion ROIs the run used.

    python mpat_rgvp_figure.py out/mpat-rgvp --t1 sub-01_T1w.nii.gz -o rgvp

Writes <o>.pdf (vector), <o>.png (300 dpi) and <o>_plane.json (the plane, for the methods).

* Bundles: all of the run's eye -> LGN / thalamus bundles (--bundles crossed: only the two
  through the chiasm).
* Slice: the plane through the bundles that fits them best (weighted PCA of every voxel at or
  above --floor of its bundle's peak, weights rising with log density); its normal is the
  direction they vary least.  Shown from above, anterior up, subject left on the left
  (neurological), sampled at 0.5 mm.
* Background: the T1 (--t1), or the b0 (--b0, on the DWI grid so it cannot be misaligned with
  the densities); default the run's tissue labels; on the slice, in grey.
* Density: each bundle divided by its own peak, and the bundles merged (the largest of these
  at each point), as a maximum-intensity projection onto the slice along its normal (through
  all of the bundles, or a slab of +-(--mip-mm) around the slice), so the whole pathway shows
  even where it leaves the slice.  In one colour (--color), cut at --floor (5e-2) of the
  peak; its opacity rises linearly on the log scale from 0 at the floor to --density-alpha
  (0.8) at the peak.  Only the density is projected: the background, ROIs and ODFs are the
  slice itself.
* ODFs: the bundles' visitation ODFs (the directions their walkers take), each divided by its
  bundle's peak and summed, on the voxels the slice passes through, cut by the slice: the ODF
  along each in-plane direction as a closed outline, scaled to each voxel's maximum.
* Cores: every bundle's core (mpat.core) projected onto the slice; crossed bundles solid,
  uncrossed dashed.
* ROIs: the endpoint ROIs (eyes, LGN / thalamus: the bundles' start and end masks) outlined,
  and the exclusion ROIs (the run's invalid-ending tissue: cortex, ventricles, the rest of
  the thalamus, the image edge) tinted; --no-rois leaves them out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os.path as op
import re

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

INVALID, EDGE = 3, 4                                    # the RGVP run's dseg labels


def parse():
    p = argparse.ArgumentParser(description="figure of the RGVP on a fitted oblique slice")
    p.add_argument("folder", help="the RGVP run's output folder (mpat-rgvp)")
    p.add_argument("--t1", help="background (default: the run's _desc-mpat_dseg)")
    p.add_argument("--b0", nargs="?", const="auto", metavar="PATH",
                   help="b0 background instead (already on the DWI grid, so aligned with the densities); "
                        "bare --b0 finds pyAFQ's *_b0ref.nii.gz next to the run")
    p.add_argument("-o", "--out", default="rgvp", help="output prefix")
    p.add_argument("--bundles", choices=["all", "crossed"], default="all")
    p.add_argument("--floor", type=float, default=5e-2, help="faintest density shown, x each bundle's peak")
    p.add_argument("--color", default="#ff8a1f", help="the density's colour")
    p.add_argument("--mip-mm", type=float, default=None,
                   help="project the density over +- this many mm of the slice (default: all of it)")
    p.add_argument("--density-alpha", type=float, default=0.8,
                   help="density opacity at the peak; it falls linearly (on the log scale) to 0 at --floor")
    p.add_argument("--glyph-color", default="#3a1500")
    p.add_argument("--core-color", default="#0b2a6b")
    p.add_argument("--roi-color", default="#7dd3fc", help="endpoint ROI outlines")
    p.add_argument("--exclude-color", default="#4b5d8a", help="exclusion ROI tint")
    p.add_argument("--no-rois", action="store_true")
    p.add_argument("--res", type=float, default=0.5, help="slice sampling (mm)")
    p.add_argument("--margin", type=float, default=12.0, help="mm around the bundles shown")
    p.add_argument("--glyph-scale", type=float, default=1.35, help="glyph radius, x half the voxel spacing")
    p.add_argument("--aspect", type=float, default=None,
                   help="image height / width, by widening the shorter side (default: the bundles' own)")
    p.add_argument("--fontsize", type=float, default=15.0, help="base font size (pt); the rest scale with it")
    p.add_argument("--width-mm", type=float, default=180.0, help="figure width (Nature: 89 or 183)")
    return p.parse_args()


def rgvp_bundles(folder, which):
    """[(key, label, crossed, files)] for the RGVP run's bundles."""
    from mpat.core import bundle_files

    files = bundle_files(folder)
    out = []
    for key, f in sorted(files.items()):
        m = re.fullmatch(r"(Left|Right)eye(Left|Right)(LGN|thalamus)", key)
        if not m:
            continue
        crossed = m.group(1) != m.group(2)
        if crossed or which == "all":
            out.append((key, f"{m.group(1)} eye → {m.group(2).lower()} {m.group(3)}", crossed, f))
    want = 4 if which == "all" else 2
    if len(out) != want:
        raise SystemExit(f"expected {want} RGVP bundles under {folder}, found {[k for k, *_ in out]} "
                         f"among {sorted(files)}")
    return out


def fit_plane(points, weights):
    """Weighted PCA: centre, and an orthonormal (right, anterior, normal) basis with the normal
    the least-variance axis, pointing up; 'right' is world +x projected into the plane."""
    c = np.average(points, axis=0, weights=weights)
    X = (points - c) * np.sqrt(weights)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2] * np.sign(vt[2][2] or 1.0)             # up
    r = np.array([1.0, 0, 0]) - n * n[0]
    r /= np.linalg.norm(r)
    a = np.cross(n, r)                               # anterior, in the plane
    return c, r, a, n


def sample(img, data, world, order=1):
    """``data`` (on ``img``'s grid) at world points (..., 3), 0 outside."""
    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), world.reshape(-1, 3)).T
    return ndi.map_coordinates(np.asarray(data, np.float32), ijk, order=order, mode="constant",
                               cval=0.0).reshape(world.shape[:-1])


def background(args, world):
    t1 = args.t1
    if args.b0 == "auto":
        up = op.dirname(op.abspath(args.folder).rstrip("/"))
        cand = sorted(set(glob.glob(op.join(args.folder, "**", "*_b0ref.nii.gz"), recursive=True)
                          + glob.glob(op.join(up, "**", "*_b0ref.nii.gz"), recursive=True)),
                      key=lambda x: (x.count("/"), x))
        if not cand:
            raise SystemExit(f"--b0: no *_b0ref.nii.gz under {args.folder} or {up}; pass its path")
        t1 = cand[0]
        print(f"background: {t1}")
    elif args.b0:
        t1 = args.b0
    if t1 is None:
        cand = sorted(glob.glob(op.join(args.folder, "*_desc-mpat_dseg.nii.gz")))
        t1 = cand[0] if cand else None
    if t1 is None:
        return np.zeros(world.shape[:-1])
    bimg = nib.load(t1)
    bdat = np.asarray(bimg.dataobj, np.float32)
    bdat = bdat[..., 0] if bdat.ndim == 4 else bdat
    back = sample(bimg, bdat, world, order=0 if "dseg" in t1 else 1)
    lo, hi = np.percentile(back[back > 0], [1, 99.5]) if (back > 0).any() else (0, 1)
    return np.clip((back - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.9 * 0.85


def main():
    args = parse()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mpat.geometry import _sh_basis

    floor = args.floor
    bundles = rgvp_bundles(args.folder, args.bundles)
    dimg = nib.load(bundles[0][3]["density"])
    A = dimg.affine
    zooms = np.linalg.norm(A[:3, :3], axis=0)
    dens, peaks = [], []
    for *_, f in bundles:
        d = np.clip(np.nan_to_num(np.asarray(nib.load(f["density"]).dataobj, np.float32)), 0, None)
        peaks.append(float(d.max()))
        dens.append(d / d.max())
    merged = np.max(dens, axis=0)

    # ---- the plane
    pts, wts = [], []
    for d in dens:
        ijk = np.argwhere(d >= floor)
        pts.append(nib.affines.apply_affine(A, ijk))
        wts.append(np.log10(d[tuple(ijk.T)]) - np.log10(floor) + 0.1)
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    c, er, ea, en = fit_plane(pts, wts)
    tilt = float(np.degrees(np.arccos(abs(en[2]))))
    uv = np.c_[(pts - c) @ er, (pts - c) @ ea]
    lo, hi = uv.min(0) - args.margin, uv.max(0) + args.margin
    if args.aspect:                                     # widen the shorter side to --aspect (h / w)
        size = hi - lo
        want = np.array([max(size[0], size[1] / args.aspect), max(size[1], size[0] * args.aspect)])
        lo, hi = lo - (want - size) / 2, hi + (want - size) / 2
    us = np.arange(lo[0], hi[0], args.res)
    vs = np.arange(lo[1], hi[1], args.res)
    U, V = np.meshgrid(us, vs)
    world = c + U[..., None] * er + V[..., None] * ea
    extent = [us[0] - args.res / 2, us[-1] + args.res / 2, vs[0] - args.res / 2, vs[-1] + args.res / 2]

    # ---- background, exclusion tint, density
    grey = background(args, world)
    rgb = np.repeat(grey[..., None], 3, -1)
    dseg_f = sorted(glob.glob(op.join(args.folder, "*_desc-mpat_dseg.nii.gz")))
    excl = None
    if not args.no_rois and dseg_f:
        dseg = nib.load(dseg_f[0])
        lab = np.asarray(dseg.dataobj)
        excl = sample(dseg, np.isin(lab, [INVALID, EDGE]), world, order=1)
        a = 0.45 * np.clip(excl, 0, 1)[..., None]
        rgb = (1 - a) * rgb + a * np.array(to_rgb(args.exclude_color))
    # maximum-intensity projection of the density along the slice normal
    h = (pts - c) @ en
    if args.mip_mm is not None:
        h0, h1 = -args.mip_mm, args.mip_mm
    else:
        h0, h1 = h.min() - 1.0, h.max() + 1.0
    f = np.zeros(U.shape, np.float32)
    for o in np.arange(h0, h1 + 1e-6, args.res):
        np.maximum(f, sample(dimg, merged, world + o * en, order=1), out=f)
    print(f"density MIP over {h0:+.1f} .. {h1:+.1f} mm along the slice normal")
    t = np.clip((np.log10(np.maximum(f, 1e-30)) - np.log10(floor)) / -np.log10(floor), 0, 1)
    t[f < floor] = 0
    alpha = (args.density_alpha * t)[..., None]
    rgb = (1 - alpha) * rgb + alpha * np.array(to_rgb(args.color))

    # ---- figure
    fs = args.fontsize
    w_in = args.width_mm / 25.4
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    band = 0.12 + 0.05 * fs                             # colour bar and legend, under the image
    H = w_in * aspect + band
    fig = plt.figure(figsize=(w_in, H), facecolor="black")
    ax = fig.add_axes([0, band / H, 1, w_in * aspect / H])
    ax.set_facecolor("black")
    ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=extent, interpolation="bilinear")

    # ---- endpoint ROI outlines, exclusion edges
    handles = []
    if not args.no_rois:
        ends = np.zeros(dimg.shape[:3], bool)
        for *_, fb in bundles:
            for k in ("start", "end"):
                if fb[k]:
                    ends |= np.asarray(nib.load(fb[k]).dataobj) > 0
        if ends.any():
            e = sample(dimg, ends, world, order=1)
            if e.max() >= 0.5:
                ax.contour(U, V, e, levels=[0.5], colors=[args.roi_color], linewidths=0.9, zorder=4)
                handles.append(Line2D([], [], color=args.roi_color, lw=0.9, label="endpoint ROI"))
        if excl is not None and excl.max() >= 0.5:
            ax.contour(U, V, excl, levels=[0.5], colors=[args.exclude_color], linewidths=0.5, linestyles=":",
                       zorder=2)
            handles.append(Patch(facecolor=args.exclude_color, alpha=0.6, edgecolor="none",
                                 label="exclusion ROI"))

    # ---- ODF outlines: the bundles' visitation ODFs, each / its peak, summed, on the slab
    R = A[:3, :3] / zooms                               # voxel axes -> world directions
    th = np.linspace(0, 2 * np.pi, 97)
    dirs_vox = (np.cos(th)[:, None] * er + np.sin(th)[:, None] * ea) @ R   # the SH frame is the voxel axes
    spacing = float(np.mean(zooms))
    ijk = np.argwhere(merged >= floor)
    p = nib.affines.apply_affine(A, ijk)
    on = np.abs((p - c) @ en) <= 0.5 * spacing          # a one-voxel slab
    ijk, p = ijk[on], p[on]
    coef = None
    for (*_, fb), pk in zip(bundles, peaks):
        if fb["odf"] is None:
            continue
        cb = np.asarray(nib.load(fb["odf"]).dataobj, np.float32)[tuple(ijk.T)] / pk
        coef = cb if coef is None else coef + cb
    if coef is not None and len(p):
        r = np.clip(coef @ _sh_basis(dirs_vox, coef.shape[1]), 0, None)
        m = r.max(1)
        keep = m > 0
        r, p = r[keep] / m[keep, None], p[keep]
        cu, cv = (p - c) @ er, (p - c) @ ea
        rad = 0.5 * args.glyph_scale * spacing
        polys = np.stack([cu[:, None] + rad * r * np.cos(th), cv[:, None] + rad * r * np.sin(th)], -1)
        ax.add_collection(PolyCollection(polys, facecolors=args.glyph_color, edgecolors="white",
                                         linewidths=0.35, zorder=3))

    # ---- cores: every bundle; crossed solid, uncrossed dashed
    for key, label, crossed, fb in bundles:
        xyz = None
        try:
            xyz = np.asarray(nib.streamlines.load(fb["prefix"] + "_desc-core.tck").streamlines[0])
        except Exception:  # noqa: BLE001 -- no core written
            print(f"{key}: no core file")
        if xyz is None:
            continue
        ls = "-" if crossed else (0, (3.5, 2.0))
        ax.plot((xyz - c) @ er, (xyz - c) @ ea, color=args.core_color, lw=2.0, ls=ls, solid_capstyle="round",
                dash_capstyle="round", zorder=6, path_effects=[pe.Stroke(linewidth=3.6, foreground="white"),
                                                                pe.Normal()])
    handles[:0] = [Line2D([], [], color=args.core_color, lw=2.0, ls="-", label="core, crossed",
                          path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])]
    if args.bundles == "all":
        handles.insert(1, Line2D([], [], color=args.core_color, lw=2.0, ls=(0, (3.5, 2.0)), label="core, uncrossed",
                                 path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()]))

    # ---- scale bar
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    outline = [pe.withStroke(linewidth=2.2, foreground="black")]
    x0, y0 = us[0] + 4, vs[0] + 4
    ax.plot([x0, x0 + 10], [y0, y0], color="white", lw=2.5, solid_capstyle="butt", zorder=7)
    ax.text(x0 + 5, y0 + 1.6, "10 mm", color="white", fontsize=0.85 * fs, ha="center", va="bottom",
            path_effects=outline, zorder=7)

    # ---- colour bar (the colour at its opacity over mid-grey tissue) and legend
    from matplotlib.cm import ScalarMappable

    g, col = np.array([0.42] * 3), np.array(to_rgb(args.color))
    ramp = [(1 - a) * g + a * col for a in np.linspace(0, args.density_alpha, 256)]
    cax = fig.add_axes([0.04, 0.4 * band / H, 0.38, 0.12 / H])
    cb = fig.colorbar(ScalarMappable(norm=LogNorm(floor, 1), cmap=LinearSegmentedColormap.from_list("ramp", ramp)),
                      cax=cax, orientation="horizontal")
    cb.set_label("density / each bundle's peak (MIP)", color="white", fontsize=0.85 * fs, labelpad=0.35 * fs)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(colors="white", labelsize=0.75 * fs, length=2)
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.5)
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.46, band / 2 / H), ncol=2,
                     frameon=False, fontsize=0.8 * fs, labelcolor="white", handlelength=2.4, columnspacing=1.0,
                     labelspacing=0.4)
    for h in leg.legend_handles:
        if isinstance(h, Line2D):
            h.set_path_effects([pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()]
                               if h.get_color() == args.core_color else [])

    fig.savefig(args.out + ".pdf", facecolor="black")
    fig.savefig(args.out + ".png", dpi=300, facecolor="black")
    with open(args.out + "_plane.json", "w") as fh:
        json.dump(dict(centre_mm=c.tolist(), normal=en.tolist(), right=er.tolist(), anterior=ea.tolist(),
                       tilt_from_axial_deg=tilt, bundles=[k for k, *_ in bundles], floor=floor,
                       density_mip_mm=[float(h0), float(h1)]), fh, indent=1)
    print(f"{args.out}.pdf / .png: plane through {c.round(1)} mm, tilted {tilt:.0f} deg from axial")


if __name__ == "__main__":
    main()