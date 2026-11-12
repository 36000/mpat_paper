#!/usr/bin/env python3
"""Figure 1B: one bundle's density, pyAFQ at several seed counts next to MPAT (magma, log
scale, each panel divided by its own peak).

Example (HCP 627549 CST)
------------------------
python fig1b_mips.py --bundle "Left Corticospinal" \
    --markov mcat/sub-627549_bundle-LeftCorticospinal_desc-density_map.nii.gz \
    --background sub-627549_desc-brain_mask.nii.gz --view coronal --out fig1b.png

The pyAFQ runs are hardcoded (PYAFQ_RUNS: the segmented tractography of the 1e6 / 1e7 / 1e8
seed runs and how many whole-brain streamlines each tracked).  Each is subsampled to what
it would have recognized from exactly 1e6 / 1e7 / 1e8 whole-brain streamlines: a random
target / tracked fraction of its recognized bundle streamlines (e.g. 1e6 / 2554530 of the
1e6-seed run's), without replacement (--rng-seed), then counted per voxel on the MPAT grid.
--pyafq / --labels override the runs with other files (tractography, all streamlines kept;
or 3-D density maps).

Preview mode (no text at all, exactly 600x400 px):
python fig1b_mips.py ... --preview --out preview.png
python fig1b_mips.py ... --preview --preview-size 800 300 --out preview.png

Every map is resampled onto the Markov grid, reoriented to RAS, and shown as a
maximum-intensity projection (or a single slice with --slice-mm).

Panel titles: "<label> streamlines total" over "<N> in the CST" (N: the bundle's
streamlines drawn; "<label> seeds" with --pyafq / --labels), then "MPAT" over "no streamlines
" (--markov-sub); --fontsize.

Normalization: each panel is divided by its own peak (default).  --norm-upper 0.5 (a check):
by its peak within the upper half of the bundle (the superior 50% of MPAT's extent, the same region for every panel), so a
bright spot low in the bundle (the brainstem, for the CST) does not set the scale; anything
brighter saturates.

By default (--floor auto) every panel shows everything above its own noise level:
  pyAFQ   : every voxel touched by at least one streamline (counts are integers;
            maps not on the Markov grid are resampled nearest-neighbour so no
            fractional halo is invented);
  Markov  : everything above --markov-floor (default 1e-6 of peak, the solver's
            relative tolerance; values below it are within iteration error).
The shared colour bar starts at the lowest panel floor.
"""
import argparse
import os.path as op

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
# (panel label, segmented tractography, whole-brain streamlines tracked, as if this many)
PYAFQ_RUNS = [
    ("$10^6$", "sub-627549_desc-bundles_tractography.trx", 127580246, int(1e6)),
    ("$10^7$", "sub-627549_desc-bundles_tractography.trx", 127580246, int(1e7)),
    ("$10^8$", "sub-627549_desc-bundles_tractography.trx", 127580246, int(1e8)),
]


def density_from_tractography(path, bundle, ref, keep=1.0, rng=None):
    """Streamlines of ``bundle`` per voxel of ``ref`` (each streamline counted once per voxel),
    of a random ``keep`` fraction of them (without replacement) when keep < 1."""
    import AFQ.utils.streamlines as aus
    from dipy.tracking.utils import density_map

    sft = aus.SegmentedSFT.fromfile(path, "same").get_bundle(bundle)
    sft.to_rasmm()
    sls = sft.streamlines
    n = len(sls)
    if keep < 1:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=int(round(keep * n)), replace=False))
        sls = sls[idx]
    d = density_map(sls, ref.affine, ref.shape[:3])
    print(f"{op.basename(path)}: {n} {bundle} streamlines recognized, {len(sls)} kept ({keep:.4g})")
    img = nib.Nifti1Image(d.astype(np.float32), ref.affine)
    img.n_streamlines = len(sls)
    return img


def load_on(path, ref, order=0, bundle=None, keep=1.0, rng=None):
    if path.endswith((".trx", ".trk")):
        if not bundle:
            raise SystemExit("--bundle is required with tractography files")
        img = density_from_tractography(path, bundle, ref, keep, rng)
        out = nib.as_closest_canonical(img)
        out.n_streamlines = img.n_streamlines     # the bundle's streamlines drawn (for the title)
        return out
    img = nib.load(path)
    if img.ndim == 4:                          # multi-bundle file: sum the bundles
        img = nib.Nifti1Image(np.asarray(img.dataobj, float).sum(-1), img.affine)
    if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-3):
        img = resample_from_to(img, (ref.shape[:3], ref.affine), order=order)
    return nib.as_closest_canonical(img)      # RAS: i = left->right, j = posterior->anterior, k = inferior->superior


def project(vol, axis, slice_idx=None):
    v = np.take(vol, slice_idx, axis) if slice_idx is not None else vol.max(axis)
    return v.T                                 # rows = the remaining second axis, for origin='lower'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pyafq", nargs="+",
                   help="instead of PYAFQ_RUNS: pyAFQ segmented tractography (.trx / .trk, with --bundle; "
                        "not subsampled) or 3-D density maps")
    p.add_argument("--bundle", default="Left Corticospinal",
                   help="pyAFQ bundle name in the tractography files (default 'Left Corticospinal')")
    p.add_argument("--rng-seed", type=int, default=0, help="seed of the subsampling")
    p.add_argument("--labels", nargs="+", help="seed counts for the pyAFQ maps (used in panel titles)")
    p.add_argument("--markov", required=True, help="MPAT *_desc-density_map.nii.gz")
    p.add_argument("--markov-label", default="MPAT")
    p.add_argument("--markov-sub", default="no streamlines",
                   help="second line under MPAT's title (\"\" for none)")
    p.add_argument("--bundle-short", default="CST", help="the bundle's name in the panel titles")
    p.add_argument("--background", help="brain mask or any anatomical image for a grey silhouette")
    p.add_argument("--view", choices=AXIS, default="coronal")
    p.add_argument("--slice-mm", type=float, help="show one slice at this world coordinate instead of a MIP")
    p.add_argument("--floor", default="auto",
                   help="'auto' (each panel down to its own noise level) or a fraction of each panel's peak")
    p.add_argument("--markov-floor", type=float, default=1e-6,
                   help="with --floor auto: Markov values below this fraction of peak are treated as "
                        "solver noise (match the run's --tol)")
    p.add_argument("--norm-upper", type=float, default=1.0,
                   help="normalize each panel by its peak in this superior fraction of MPAT's extent "
                        "(same region for all panels; 1 = whole bundle)")
    p.add_argument("--fontsize", type=float, default=26,
                   help="panel titles and colour-bar label, points (tick labels 2/3 of it, figure title 1.25x)")
    p.add_argument("--title", default="", help="figure title (default: none, for a panel figure)")
    p.add_argument("--out", default="density_panels.png")
    p.add_argument("--dpi", type=int, default=300,
                   help="output dpi; default 'auto' = enough pixels that every voxel of the Markov grid "
                        "is drawn >= --px-per-voxel pixels wide (never below 200)")
    p.add_argument("--px-per-voxel", type=int, default=1,
                   help="with auto dpi: output pixels per native voxel (1 = native resolution)")
    p.add_argument("--preview", action="store_true",
                   help="text-free preview: panels only (no titles, suptitle or colour bar), "
                        "saved at exactly --preview-size pixels")
    p.add_argument("--preview-size", type=int, nargs=2, default=(600, 400), metavar=("W", "H"),
                   help="preview image size in pixels (default 600 400)")
    a = p.parse_args()

    ref = nib.load(a.markov)
    rng = np.random.default_rng(a.rng_seed)
    if a.pyafq:
        maps = [(lab, load_on(f, ref, bundle=a.bundle)) for lab, f in zip(a.labels or [None] * len(a.pyafq), a.pyafq)]
    else:     # the hardcoded runs, each subsampled as if from exactly `target` whole-brain streamlines
        maps = [(lab, load_on(f, ref, bundle=a.bundle, keep=min(1.0, target / tracked), rng=rng))
                for lab, f, tracked, target in PYAFQ_RUNS]
    def pyafq_title(i, lab, img):
        head = (f"pyAFQ, {lab} seeds" if a.pyafq else f"{lab} streamlines total") if lab \
            else f"pyAFQ, run {i + 1}"
        n_b = getattr(img, "n_streamlines", None)
        return head + (f"\n{n_b:,} in the {a.bundle_short}" if n_b is not None else "")

    maps = [(pyafq_title(i, lab, img), img) for i, (lab, img) in enumerate(maps)]
    lines = max(t.count("\n") for t, _ in maps)
    mtitle = a.markov_label + (f"\n{a.markov_sub}" if a.markov_sub else "")
    mtitle += "\n" * max(0, lines - mtitle.count("\n"))         # level with the pyAFQ titles
    maps.append((mtitle, load_on(a.markov, ref)))   # same grid as ref: never resampled
    ax_id = AXIS[a.view]

    canon = maps[-1][1]
    sl = None
    if a.slice_mm is not None:
        ijk = np.linalg.inv(canon.affine) @ np.r_[[a.slice_mm if d == ax_id else 0 for d in range(3)], 1]
        sl = int(np.clip(round(ijk[ax_id]), 0, canon.shape[ax_id] - 1))

    bg = None
    if a.background:
        b = np.asarray(load_on(a.background, ref).dataobj, float)
        bg = project(b > 0, ax_id, sl).astype(float)

    # world extent of the displayed plane
    dims = [d for d in range(3) if d != ax_id]
    corners = canon.affine[:3, :3] @ np.array([[0, 0, 0], np.array(canon.shape[:3]) - 1]).T + canon.affine[:3, 3:]
    ext = (corners[dims[0], 0], corners[dims[0], 1], corners[dims[1], 0], corners[dims[1], 1])

    # the normalization region: the superior --norm-upper of MPAT's extent (RAS: axis 2 is superior)
    md = np.clip(np.asarray(canon.dataobj, float), 0, None)
    zs = np.nonzero((md >= a.markov_floor * md.max()).any((0, 1)))[0]
    z_cut = zs.max() - a.norm_upper * (zs.max() - zs.min()) if (zs.size and a.norm_upper < 1) else -1

    def norm_peak(d):
        sub = d[:, :, int(np.ceil(z_cut)):] if z_cut >= 0 else d
        return float(sub.max()) or float(d.max()) or 1.0

    if z_cut >= 0:
        zw = nib.affines.apply_affine(canon.affine, [[0, 0, z_cut]])[0, 2]
        print(f"normalizing by each panel's peak above z = {zw:.1f} mm (superior {a.norm_upper:.0%} of MPAT's extent)")

    def panel_floor(d, peak, is_markov):
        if a.floor != "auto":
            return float(a.floor)
        if is_markov:
            return a.markov_floor
        nz = d[d > 0]
        return float(nz.min() / peak) * 0.999 if nz.size else 1.0   # one streamline

    floors = []
    for i, (lab, img) in enumerate(maps):
        d = np.clip(np.asarray(img.dataobj, float), 0, None)
        floors.append(panel_floor(d, norm_peak(d), i == len(maps) - 1))
    norm = LogNorm(min(floors), 1)

    # tight crop: bounding box of all displayed content across every panel
    projs = []
    for lab, img in maps:
        d = np.clip(np.asarray(img.dataobj, float), 0, None)
        projs.append(project(d / norm_peak(d), ax_id, sl))   # brighter than the peak: saturates
    mask = np.zeros_like(projs[0], bool)
    for r, f in zip(projs, floors):
        mask |= r >= f
    if not mask.any():
        mask[:] = True
    rows, cols = np.where(mask)
    pad = 2
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, mask.shape[0] - 1)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, mask.shape[1] - 1)
    nr, nc = mask.shape
    x0, x1, y0, y1 = ext
    # pixel centres map linearly onto the extent (origin='lower')
    def to_x(c): return x0 + (x1 - x0) * c / (nc - 1)
    def to_y(r): return y0 + (y1 - y0) * r / (nr - 1)
    xlim = (to_x(c0), to_x(c1))
    ylim = (to_y(r0), to_y(r1))
    aspect = abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) if ylim[1] != ylim[0] else 1.0

    n = len(maps)
    if a.preview:
        # exact pixel size: figsize in inches at 100 dpi, axes filling the whole canvas
        W, H = a.preview_size
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="black")
        axs = fig.subplots(1, n)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    else:
        pw = 4.6                            # fixed, so --fontsize changes the text relative to the panels
        ph = pw / aspect
        fig, axs = plt.subplots(1, n, figsize=(pw * n + 1.2, ph + 1.8), constrained_layout=True)

    for ax, (lab, img), r, floor in zip(np.atleast_1d(axs), maps, projs, floors):
        if bg is not None:
            ax.imshow(np.where(bg > 0, 0.13, 0), cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext)
        else:
            ax.set_facecolor("black")
        im = ax.imshow(np.ma.masked_less(r, floor), cmap="magma", norm=norm, origin="lower",
                       extent=ext, interpolation="nearest")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if not a.preview:
            ax.set_title(lab, fontsize=a.fontsize, linespacing=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    if a.preview:
        # no bbox_inches='tight' here: it would change the pixel size
        fig.savefig(a.out, dpi=100, facecolor="black")
        print(f"wrote {a.out} ({a.preview_size[0]}x{a.preview_size[1]} px preview)")
        return

    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01)
    cb.set_label("density / own peak (log)" if a.norm_upper >= 1 else "density / own upper-bundle peak (log)",
                 fontsize=a.fontsize)
    cb.ax.tick_params(labelsize=round(a.fontsize * 2 / 3))
    what = f"slice {a.view} = {a.slice_mm:g} mm" if sl is not None else f"{a.view} maximum-intensity projection"
    if a.title:
        fig.suptitle(f"{a.title} — {what}", fontsize=1.25 * a.fontsize)
    dpi = a.dpi
    if dpi is None:
        fig.canvas.draw()                                # settle constrained_layout before measuring
        vox_across = (c1 - c0 + 1)                       # native voxels across one panel
        panel_in = ax.get_position().width * fig.get_figwidth()
        dpi = int(np.ceil(a.px_per_voxel * vox_across / panel_in))
        dpi = max(dpi, 200)
    fig.savefig(a.out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    print(f"wrote {a.out} (dpi={dpi})")


if __name__ == "__main__":
    main()
