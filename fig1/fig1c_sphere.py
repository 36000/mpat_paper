#!/usr/bin/env python
"""Figure 1C: the cortical endpoints of one bundle on the subject's cortex (folded, a little
inflated), pyAFQ (left) next to MCAT (right) (neuropythy + k3d, rendered headless as in the
visual-cortex script).

    python fig1c_sphere.py 627549 mcat/sub-627549_bundle-LeftCorticospinal_desc-endpoints_map.nii.gz \\
        --hemi left

Writes fig1c_pyafq.png and fig1c_mcat.png (and .html), both cropped to the same box, and
fig1c.png, the two side by side with "10^8 streamlines total; <N> in the CST" (as fig1b's
titles, N: the CST streamlines drawn) and "MCAT" underneath (-o changes the "fig1c" prefix;
--no-labels, --pyafq-label, --bundle-short, --label-size).  pyAFQ: the
bundle's streamline endpoints (first and last points) counted per voxel of MCAT's grid, from
the same subsample fig1b_mips.py draws for its 1e8 panel (PYAFQ_TRX, TOTAL_STREAMLINES,
SUBSAMPLE_TARGETS and --rng-seed as there, the draws repeated in the same order, so the same
streamlines).  Both are painted onto the surface the same way and seen from the same camera
(set by MCAT's endpoints); each on a log scale over its own peak, down to --floor.

MCAT writes each bundle's expected endpoints per voxel (``*_desc-endpoints_map.nii.gz``,
absorbing voxels: the gray matter where its walkers end).  Each endpoint voxel is assigned to
its nearest white-surface vertex (and the white vertices within --spread beyond it), which is
the same vertex on the pial and drawn surfaces; facing banks of a sulcus are far apart on the
white surface, so nothing crosses one (--assign white, default; voxel / column: the earlier
vertex-samples-the-map rules).  Colours: magma, log scale over each map's own peak (as MCAT's figures),
composited over a curvature underlay; the camera is centred on the endpoints'
weighted geometric median (found on the sphere) and looks from --elevation above the
horizontal on their side (--azimuth), superior up.  --surface / --inflate: what is drawn
(default the white surface 0.3 of the way to the inflated one; --surface sphere as before).

HCP (neuropythy hcp_subject) surfaces are in the same T1w/ACPC space as pyAFQ's outputs
on HCP, so no transform is needed (--affine for other data: a 4x4 text file taking the
map's world coordinates to the surfaces').
"""
import argparse
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402


# pyAFQ's run and fig1b_mips.py's subsampling (keep these in step with PYAFQ_RUNS there)
PYAFQ_TRX = "sub-627549_desc-bundles_tractography.trx"
TOTAL_STREAMLINES = 110584180
SUBSAMPLE_TARGETS = (int(1e6), int(1e7), int(1e8))   # fig1b's panels, in its order
PYAFQ_TARGET = int(1e8)                               # the one drawn here


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("subject", help="HCP subject ID (neuropythy hcp_subject)")
    p.add_argument("endpoints", help="MCAT *_desc-endpoints_map.nii.gz")
    p.add_argument("--hemi", choices=["left", "right"], required=True)
    p.add_argument("-o", "--out", default="fig1c")
    p.add_argument("--floor", type=float, default=1e-10, help="faintest endpoint density shown, x the peak")
    p.add_argument("--assign", choices=["white", "voxel", "column"], default="white",
                   help="white (default): each endpoint voxel goes to its nearest white-surface vertex and the "
                        "white vertices within --spread mm beyond it (the same vertices on the pial surface; "
                        "facing banks of a sulcus are far apart on the white surface, so nothing crosses one); "
                        "voxel: each vertex takes the voxel its mid-thickness point lies in; column: the largest "
                        "interpolated value along its white -> pial column (--depths; smears across sulci)")
    p.add_argument("--spread", type=float, default=None,
                   help="with --assign white: mm past the nearest white vertex a voxel also reaches "
                        "(default: half the voxel size)")
    p.add_argument("--max-dist", type=float, default=5.0,
                   help="with --assign white: endpoint voxels farther than this from the white surface are left "
                        "out (mm; the brainstem end, etc.)")
    p.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5],
                   help="with --assign column: fractions of the white-to-pial distance sampled per vertex")
    p.add_argument("--affine", help="4x4 text file: endpoint-map world -> surface coordinates (default identity)")
    p.add_argument("--hcp-profile", default="hcp", help="boto3 profile with HCP S3 credentials")
    p.add_argument("--cache", default="~/AFQ_data/npythy_cache/")
    p.add_argument("--size", type=int, default=2048, help="PNG size (pixels)")
    p.add_argument("--distance", type=float, default=2.6, help="camera distance, in surface radii")
    p.add_argument("--elevation", type=float, default=45.0,
                   help="camera elevation above the horizontal, deg (90 top down, 0 from the side); "
                        "superior stays up")
    p.add_argument("--azimuth", type=float, default=None,
                   help="camera side, deg from straight lateral towards anterior (negative: posterior); "
                        "default: the side the endpoints face")
    p.add_argument("--surface", default="white", choices=["white", "midgray", "pial", "inflated", "sphere"],
                   help="the folded surface drawn (default white), or inflated / sphere")
    p.add_argument("--pyafq", default=PYAFQ_TRX, help="pyAFQ's segmented tractography (.trx)")
    p.add_argument("--bundle", default="Left Corticospinal", help="the bundle's name in --pyafq")
    p.add_argument("--rng-seed", type=int, default=0, help="as fig1b_mips.py's --rng-seed")
    p.add_argument("--no-pyafq", action="store_true", help="MCAT only")
    p.add_argument("--pyafq-label", default="$10^8$", help="pyAFQ's streamline count in its label")
    p.add_argument("--bundle-short", default="CST", help="the bundle's name in pyAFQ's label")
    p.add_argument("--no-labels", action="store_true", help="no labels under the side-by-side renders")
    p.add_argument("--label-size", type=float, help="label font size, points (default ~5%% of the height)")
    p.add_argument("--inflate", type=float, default=0.3,
                   help="with a folded --surface: how far towards the inflated surface it is drawn, "
                        "0 (as folded) .. 1 (inflated); default 0.3, a little")
    return p.parse_args()


def pyafq_endpoints(path, bundle, ref, seed=0):
    """The bundle's streamline endpoints per voxel of ``ref``, from the streamlines fig1b_mips.py
    keeps for its PYAFQ_TARGET panel: its draws (one rng, SUBSAMPLE_TARGETS in order, each a
    target / TOTAL_STREAMLINES fraction without replacement) repeated, the PYAFQ_TARGET one used."""
    import AFQ.utils.streamlines as aus

    sft = aus.SegmentedSFT.fromfile(path, "same").get_bundle(bundle)
    sft.to_rasmm()
    sls = sft.streamlines
    n = len(sls)
    rng = np.random.default_rng(seed)
    idx = None
    for target in SUBSAMPLE_TARGETS:
        keep = min(1.0, target / TOTAL_STREAMLINES)
        pick = np.sort(rng.choice(n, size=int(round(keep * n)), replace=False)) if keep < 1 else np.arange(n)
        if target == PYAFQ_TARGET:
            idx = pick
    sls = sls[idx]
    ends = np.concatenate([np.array([s[0] for s in sls]), np.array([s[-1] for s in sls])])
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine), ends)).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(ref.shape[:3])), axis=1)
    counts = np.zeros(ref.shape[:3], np.float32)
    np.add.at(counts, tuple(ijk[ok].T), 1.0)
    print(f"{op.basename(path)}: {n} {bundle} streamlines, {len(sls)} kept (as fig1b's "
          f"{PYAFQ_TARGET:.0e} panel), {int(counts.sum())} endpoints")
    img = nib.Nifti1Image(counts, ref.affine)
    img.n_streamlines = len(sls)          # for the label
    return img


def white_values(img, surf_white, world_to_surf, spread=None, max_dist=5.0):
    """Each vertex's endpoint value from the voxels assigned to it: every voxel with endpoints
    goes to its nearest white-surface vertex and to the white vertices within ``spread`` mm
    beyond that distance (default half a voxel); a vertex takes the largest value it receives.
    Voxels farther than ``max_dist`` from the white surface are left out."""
    from scipy.spatial import cKDTree

    data = np.asarray(img.dataobj, np.float32)
    ijk = np.argwhere(data > 0)
    vals = data[tuple(ijk.T)]
    pts = nib.affines.apply_affine(world_to_surf @ img.affine, ijk)
    w = np.asarray(surf_white.coordinates, float).T
    tree = cKDTree(w)
    d0, _ = tree.query(pts)
    keep = d0 <= max_dist
    if spread is None:
        spread = 0.5 * float(np.mean(nib.affines.voxel_sizes(img.affine)))
    out = np.zeros(len(w), np.float32)
    for nb, v in zip(tree.query_ball_point(pts[keep], d0[keep] + spread), vals[keep]):
        np.maximum.at(out, nb, v)
    print(f"{keep.sum()} of {len(vals)} endpoint voxels within {max_dist:g} mm of the white surface "
          f"({vals[keep].sum() / max(vals.sum(), 1e-30):.1%} of the endpoint mass)")
    return out


def vertex_values(img, surf_white, surf_pial, depths, world_to_surf, assign="voxel"):
    """Each vertex's endpoint value: the voxel its mid-thickness point lies in (assign "voxel"),
    or the largest linearly interpolated value along its white -> pial column ("column")."""
    data = np.asarray(img.dataobj, np.float32)
    w = np.asarray(surf_white.coordinates, float).T
    pial = np.asarray(surf_pial.coordinates, float).T
    to_vox = np.linalg.inv(world_to_surf @ img.affine)          # surface coords -> map voxels
    best = np.zeros(len(w), np.float32)
    if assign == "voxel":
        ijk = np.rint(nib.affines.apply_affine(to_vox, (w + pial) / 2)).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(data.shape[:3])), axis=1)
        best[ok] = data[tuple(ijk[ok].T)]
        return best
    for f in depths:
        pts = w + f * (pial - w)
        ijk = nib.affines.apply_affine(to_vox, pts).T
        np.maximum(best, ndimage.map_coordinates(data, ijk, order=1, mode="constant", cval=0.0), out=best)
    return best


def median_direction(verts, values, floor=0.0, iters=200):
    """Unit direction (from the sphere's centre) of the endpoints' weighted geometric median:
    the point minimising the density-weighted sum of distances to the vertices shown (values
    >= ``floor`` x the peak) on the unit sphere (Weiszfeld), projected back onto the sphere.
    Unlike the mean it is not dragged by a far, faint tail."""
    u = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    w = np.clip(np.asarray(values, float), 0, None)
    if not (w > 0).any():
        return np.array([1.0, 0.0, 0.0])
    keep = w >= floor * w.max()
    u, w = u[keep], w[keep]
    m = (u * w[:, None]).sum(0) / w.sum()               # start at the mean
    for _ in range(iters):
        dist = np.maximum(np.linalg.norm(u - m, axis=1), 1e-9)
        nxt = (u * (w / dist)[:, None]).sum(0) / (w / dist).sum()
        if np.linalg.norm(nxt - m) < 1e-9:
            m = nxt
            break
        m = nxt
    return m / np.linalg.norm(m)


def view_direction(d, hemi, elevation=45.0, azimuth=None):
    """The direction the camera looks from (unit, from the target out): ``elevation`` degrees
    above the horizontal, on the side given by ``azimuth`` (degrees from straight lateral,
    towards anterior; None: the side the endpoints' median ``d`` faces, or straight lateral when
    it faces up / down).  Below 90 degrees of elevation superior stays up on screen."""
    lateral = np.array([-1.0, 0.0, 0.0]) if hemi == "left" else np.array([1.0, 0.0, 0.0])
    if azimuth is not None:
        a = np.radians(azimuth)
        h = np.cos(a) * lateral + np.sin(a) * np.array([0.0, 1.0, 0.0])
    else:
        h = np.array([d[0], d[1], 0.0])
        h = h / np.linalg.norm(h) if np.linalg.norm(h) > 0.2 else lateral
        if h @ lateral < 0:                 # medial side: look from laterally instead
            h = h - 2 * (h @ lateral) * lateral
    e = np.radians(np.clip(elevation, -89.0, 89.0))
    v = np.cos(e) * h + np.sin(e) * np.array([0.0, 0.0, 1.0])
    return v / np.linalg.norm(v)


def display_coords(cortex, surface="white", inflate=0.3):
    """The vertices drawn, centred: a folded surface moved ``inflate`` of the way towards the
    inflated one (each centred first; same vertices), or the inflated surface / sphere."""
    def xyz(name):
        v = np.asarray(cortex.surface(name).coordinates, float).T
        return v - v.mean(0)

    if surface in ("inflated", "sphere") or inflate <= 0:
        return xyz(surface).astype(np.float32)
    return ((1 - inflate) * xyz(surface) + inflate * xyz("inflated")).astype(np.float32)


def colours(values, floor, underlay):
    """Magma on a log scale over the peak, opaque above ``floor``, over ``underlay``."""
    v = values / max(values.max(), 1e-30)
    t = np.clip((np.log10(np.maximum(v, 1e-30)) - np.log10(floor)) / -np.log10(floor), 0, 1)
    rgb = plt.get_cmap("magma")(0.15 + 0.85 * t)[:, :3]
    on = (v >= floor)[:, None]
    return np.where(on, rgb, underlay)


def render(verts, faces, rgb, camera, out, size, name):
    """One k3d render of the surface, written as <out>.html and <out>.png (headless)."""
    import k3d
    from k3d.headless import k3d_remote
    from pyvirtualdisplay import Display
    from selenium import webdriver

    c = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint32)
    plot = k3d.plot(grid_visible=False, camera_auto_fit=False, axes_helper=0, background_color=0xFFFFFF)
    plot += k3d.mesh(verts, faces, colors=(c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2], side="double",
                     flat_shading=False, name=name)
    plot.camera = camera
    with open(f"{out}.html", "w") as fh:
        fh.write(plot.get_snapshot())
    display = Display(visible=False, size=(size, size))
    display.start()
    opts = webdriver.ChromeOptions()
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(size, size)
    remote = k3d_remote(plot, driver, width=size, height=size)
    try:
        remote.sync(hold_until_refreshed=True)
        with open(f"{out}.png", "wb") as fh:
            fh.write(remote.get_screenshot())
    finally:
        remote.close()
        display.stop()
    print(f"wrote {out}.html / .png")


def _sans():
    """Helvetica if installed, else Arial, else matplotlib's default sans (no findfont warnings)."""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Helvetica", "Arial") if f in have), "sans-serif")


def side_by_side(pngs, out, labels=None, fontsize=None):
    """Each render cropped to the same box (what any of them draws, plus a margin; the files are
    overwritten cropped), then the crops next to each other in <out>.png, each with its label
    (``labels``) centred underneath."""
    imgs = [plt.imread(f)[..., :3] for f in pngs]
    lit = np.zeros(imgs[0].shape[:2], bool)
    for im in imgs:
        lit |= im.min(2) < 0.98
    rows, cols = np.nonzero(lit)
    pad = 10
    r0, r1 = max(rows.min() - pad, 0), rows.max() + pad
    c0, c1 = max(cols.min() - pad, 0), cols.max() + pad
    crop = [im[r0:r1, c0:c1] for im in imgs]
    for f, im in zip(pngs, crop):
        plt.imsave(f, np.clip(im, 0, 1))
    gap_px = 20
    gap = np.ones((crop[0].shape[0], gap_px, 3))
    row = np.clip(np.concatenate(sum(([im, gap] for im in crop), [])[:-1], axis=1), 0, 1)
    if not labels:
        plt.imsave(f"{out}.png", row)
    else:
        h, w = row.shape[:2]
        dpi = 100
        fs = fontsize or max(10.0, 0.05 * h * 72 / dpi)          # points: ~5% of the image height
        band = int(round(2.2 * fs * dpi / 72))                    # pixels under the images for the text
        fig = plt.figure(figsize=(w / dpi, (h + band) / dpi), dpi=dpi, facecolor="white")
        ax = fig.add_axes([0, band / (h + band), 1, h / (h + band)])
        ax.imshow(row, interpolation="nearest")
        ax.set_axis_off()
        cw = crop[0].shape[1]
        for i, lab in enumerate(labels):
            x = (i * (cw + gap_px) + cw / 2) / w
            fig.text(x, 0.5 * band / (h + band), lab, ha="center", va="center", fontsize=fs,
                     fontfamily=_sans())
        fig.savefig(f"{out}.png", dpi=dpi, facecolor="white")
        plt.close(fig)
    print(f"wrote {out}.png ({' | '.join(op.basename(f) for f in pngs)})")


def main():
    a = parse()
    import boto3
    import neuropythy as ny

    creds = boto3.Session(profile_name=a.hcp_profile).get_credentials().get_frozen_credentials()
    ny.config["data_cache_root"] = a.cache
    ny.config["hcp_credentials"] = (creds.access_key, creds.secret_key)
    sub = ny.hcp_subject(a.subject)
    cortex = sub.hemis[{"left": "lh", "right": "rh"}[a.hemi]]
    white, pial, sphere = cortex.surface("white"), cortex.surface("pial"), cortex.surface("sphere")
    world_to_surf = np.loadtxt(a.affine) if a.affine else np.eye(4)

    def surface_values(img):
        if a.assign == "white":
            return white_values(img, white, world_to_surf, a.spread, a.max_dist)
        return vertex_values(img, white, pial, a.depths, world_to_surf, a.assign)

    mcat_img = nib.load(a.endpoints)
    vals = {"mcat": surface_values(mcat_img)}
    if not a.no_pyafq:
        pyafq_img = pyafq_endpoints(a.pyafq, a.bundle, mcat_img, a.rng_seed)
        vals["pyafq"] = surface_values(pyafq_img)
    for k, v in vals.items():
        print(f"{k}: {(v > 0).sum()} of {len(v)} vertices with endpoints; peak {v.max():.3g}")

    curv = np.asarray(cortex.prop("curvature"), float)
    underlay = np.repeat(np.where(curv > 0, 0.35, 0.65)[:, None], 3, axis=1)   # sulci dark
    verts = display_coords(cortex, a.surface, a.inflate)
    faces = getattr(sphere.tess, "indexed_faces", None)
    faces = np.asarray(faces if faces is not None else sphere.tess.faces, np.uint32).T

    # camera, from MCAT's endpoints and shared by both renders: at their median (weighted
    # geometric median on the sphere, the same vertex on the surface drawn), from --elevation
    # above the horizontal on their side, superior up
    sph = np.asarray(sphere.coordinates, float).T
    sph = sph - sph.mean(0)
    d = median_direction(sph, vals["mcat"], a.floor)
    target = verts[int(np.argmax(sph @ d / np.linalg.norm(sph, axis=1)))].astype(float)
    radius = np.linalg.norm(verts, axis=1).max()
    view = view_direction(d, a.hemi, a.elevation, a.azimuth)
    up = np.array([0.0, 0.0, 1.0]) - view[2] * view
    up /= np.linalg.norm(up)
    eye = target + view * radius * (a.distance - 1)
    camera = [float(v) for v in (*eye, *target, *up)]   # k3d: plain floats
    print(f"camera towards {np.round(d, 3)} (MCAT's endpoints' median); elevation {a.elevation:g} deg, azimuth "
          f"{np.degrees(np.arctan2(view[1], view[0])):.0f} deg (x axis 0, anterior 90)")

    order = [k for k in ("pyafq", "mcat") if k in vals]          # pyAFQ left, MCAT right
    for k in order:
        render(verts, faces, colours(vals[k], a.floor, underlay), camera, f"{a.out}_{k}", a.size, f"{a.hemi} {k}")
    if len(order) > 1:
        names = {"pyafq": f"{a.pyafq_label} streamlines total; {pyafq_img.n_streamlines:,} in the {a.bundle_short}",
                 "mcat": "MCAT"}
        side_by_side([f"{a.out}_{k}.png" for k in order], a.out,
                     None if a.no_labels else [names[k] for k in order], a.label_size)


if __name__ == "__main__":
    main()