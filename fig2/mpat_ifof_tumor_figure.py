#!/usr/bin/env python
"""Four-panel figure of MPAT's IFOFs around a tumour, in the style of MPAT's collages.

    python mpat_ifof_tumor_figure.py ./tumor_out \\
        --tumor /home/john/AFQ_data/tumor/sub-0393/anat_in_diff/sub-0393_space-diff_desc-tumor_label-1_mask.nii.gz

Panels:
  1  both IFOFs, axial    the slice whose slab holds the most tumour and, second
                          (--ifof-weight), the most right IFOF
  2  both IFOFs, coronal through the tumour's centroid
  3  right IFOF, sagittal the slice whose slab holds the most right IFOF
  4  3-D                  the right IFOF (surface at --iso of its peak) and the tumour, by
                          NiiVue in a headless browser (playwright), seen from the tumour's
                          side and centred on it; also written as <out>_3d.html (interactive).
                          --no-3d leaves it out.
A slab is the slice and --slab slices (one voxel apart) on either side.  Densities are each
bundle's own peak-normalised maximum over the slab (log, magma, as in the collages), over
the T1 slice; the tumour is green (dashed where the slice misses it: its outline projected).
No text on the panels; the colour bar below keeps its label.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy import ndimage  # noqa: E402

RIGHT, LEFT, BOTH = "Right Inferior Fronto-occipital", "Left Inferior Fronto-occipital", "both"
TUMOR_RGBA = (0.35, 1.0, 0.45, 0.30)       # faint green fill
TUMOR_EDGE = (0.45, 1.0, 0.55, 0.85)       # its outline
TUMOR_RGB = (90, 230, 110)                  # 3-D surface
X, Y, Z = np.eye(3)                         # world (RAS+) axes
VIEWS = {   # normal, image right, image up (neurological: right on the right, anterior up / right)
    "axial": (Z, X, Y), "coronal": (Y, X, Z), "sagittal": (X, Y, Z)}


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("afq_out", help="pyAFQ output folder of the run (holds mpat/ and pyAFQ's T1 / brain mask)")
    p.add_argument("--tumor", required=True, help="tumour mask on the DWI grid")
    p.add_argument("--mpat-dir", help="MPAT outputs (default: <afq_out>/mpat)")
    p.add_argument("--t1", help="T1 on the DWI grid (default: pyAFQ's masked T1 in afq_out)")
    p.add_argument("--brain-mask", help="brain mask (default: pyAFQ's in afq_out)")
    p.add_argument("--out", help="PNG to write (default: <mpat-dir>/<subject>_desc-ifoftumor_figure.png)")
    p.add_argument("--slab", type=int, default=2, help="slices on either side of each slice (default 2)")
    p.add_argument("--ifof-weight", type=float, default=0.5,
                   help="axial slice score: tumour fraction in the slab + this x right IFOF fraction")
    p.add_argument("--floor", type=float, default=1e-3, help="faintest density shown, fraction of peak")
    p.add_argument("--fit-floor", type=float, default=1e-2, help="IFOF voxels above this fraction of peak count")
    p.add_argument("--no-3d", action="store_true", help="leave out the 3-D panel")
    p.add_argument("--iso", type=float, default=1e-2, help="3-D: IFOF surface at this fraction of its peak")
    p.add_argument("--zoom", type=float, default=1.3, help="3-D: zoom (1: everything fits)")
    p.add_argument("--azimuth", type=float, help="3-D camera azimuth, NiiVue's degrees (0 posterior, 90 left, "
                   "180 anterior, 270 right; default: from the tumour's side)")
    p.add_argument("--elevation", type=float, help="3-D camera elevation (default: the tumour's, at least 30)")
    p.add_argument("--niivue-js", help="3-D: a local niivue.umd.js instead of the CDN (offline)")
    p.add_argument("--cols", type=int, default=4, help="panels per row (4: one row, 2: two by two)")
    p.add_argument("--tile", type=float, default=3.2, help="inches for the largest panel side")
    p.add_argument("--fontsize", type=float, default=9.0)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args(argv)


# ----------------------------------------------------------------------- inputs
def _one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {what}: {pattern}")
    return hits[0]


def density_file(mpat_dir, name):
    desc = "".join(ch for ch in name if ch.isalnum())
    return _one(op.join(mpat_dir, f"*_bundle-{desc}_desc-density_map.nii.gz"), f"{name} density")


def load(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, np.float32), img.affine


def out_png(a, mpat_dir):
    subject = op.basename(density_file(mpat_dir, RIGHT)).split("_bundle-")[0]
    return a.out or op.join(mpat_dir, f"{subject}_desc-ifoftumor_figure.png")


# ----------------------------------------------------------------------- slice positions
def best_offset(n, pts_t, pts_b, w_b, h, res, lam):
    """Position d along ``n`` whose slab |x . n - d| <= h holds the most
    (tumour points) / (all) + lam (bundle weight) / (all).  Returns (d, tumour, bundle fraction)."""
    pt, pb = pts_t @ n, pts_b @ n
    w_b = w_b / w_b.sum()
    lo = min(pt.min() if len(pt) else np.inf, pb.min()) - h - res
    nbin = int(np.ceil((max(pt.max() if len(pt) else -np.inf, pb.max()) + h + res - lo) / res)) + 1
    ct = np.bincount(((pt - lo) / res).astype(int), minlength=nbin) / max(len(pt), 1) if len(pt) else np.zeros(nbin)
    cb = np.bincount(((pb - lo) / res).astype(int), weights=w_b, minlength=nbin)
    Ct, Cb = np.concatenate([[0], np.cumsum(ct)]), np.concatenate([[0], np.cumsum(cb)])
    k, i = int(round(h / res)), np.arange(nbin)
    a0, a1 = np.clip(i - k, 0, nbin), np.clip(i + k + 1, 0, nbin)
    ft, fb = Ct[a1] - Ct[a0], Cb[a1] - Cb[a0]
    m = int(np.argmax(ft + lam * fb))
    return lo + (m + 0.5) * res, float(ft[m]), float(fb[m])


# ----------------------------------------------------------------------- resampling
class Sampler:
    """Samples volumes on a square grid in the plane through ``centre`` spanned by (u, v)."""

    def __init__(self, n, u, v, centre, half, step):
        self.n, self.u, self.v, self.c = n, u, v, np.asarray(centre, float)
        self.s = np.arange(-half, half + step / 2, step)
        self.step = step
        uu, vv = np.meshgrid(self.s, self.s)                   # rows: v, columns: u
        self.grid = self.c + uu[..., None] * u + vv[..., None] * v

    def _at(self, vol, aff, offset=0.0):
        ijk = nib.affines.apply_affine(np.linalg.inv(aff), (self.grid + offset * self.n).reshape(-1, 3)).T
        return ndimage.map_coordinates(vol, ijk, order=1, mode="constant", cval=0.0).reshape(self.grid.shape[:2])

    def slice(self, vol, aff):
        return self._at(vol, aff)

    def slab_max(self, vol, aff, k, spacing):
        return np.max([self._at(vol, aff, j * spacing) for j in range(-k, k + 1)], axis=0)

    def projection(self, mask, aff, lo, hi):
        return np.any([self._at(mask, aff, t) > 0.5 for t in np.arange(lo, hi + self.step / 2, self.step)], axis=0)


# ----------------------------------------------------------------------- 3-D
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>IFOF 3-D</title>
<style>html,body{margin:0;background:#000;height:100%}#box{width:__W__px;height:__W__px}</style>
__NIIVUE__</head><body><div id="box"><canvas id="gl"></canvas></div><script>
const MESHES = __MESHES__;
(async () => {
  const { Niivue, NVMesh, SLICE_TYPE } = window.niivue;
  const nv = new Niivue({ backColor: [0, 0, 0, 1], isOrientCube: false, show3Dcrosshair: false,
                          isColorbar: false, crosshairWidth: 0 });
  await nv.attachTo("gl");
  const bytes = (b64) => { const s = atob(b64), u = new Uint8Array(s.length);
                           for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; };
  for (const m of MESHES)
    nv.addMesh(await NVMesh.readMesh(bytes(m.data), m.name, nv.gl, m.opacity, new Uint8Array(m.rgba), true));
  nv.setSliceType(SLICE_TYPE.RENDER);
  nv.scene.volScaleMultiplier = __ZOOM__;
  nv.setRenderAzimuthElevation(__AZ__, __EL__);
  nv.drawScene();
  window.__done = true;
})().catch((e) => { window.__err = String(e); });
</script></body></html>"""


def render_3d(a, dR, aff, tumor, tum_aff, png, size_px=900):
    """The right IFOF's surface (--iso of its peak) and the tumour's, by NiiVue: written to
    <png>_3d.html and returned as an image from a headless Chromium (None, with a note,
    without playwright / WebGL / the NiiVue script).  NiiVue turns and zooms about the centre
    of what it draws; two invisible points put that centre on the tumour."""
    from mpat.viewer import NIIVUE_CDN, _mz3, isosurface_mz3

    col = [int(round(255 * c)) for c in plt.get_cmap("magma")(0.78)[:3]]
    meshes, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
    for name, vol, af, level, rgb in (("right_ifof", dR, aff, a.iso, col),
                                      ("tumour", tumor.astype(np.float32), tum_aff, 0.5, list(TUMOR_RGB))):
        v = ndimage.gaussian_filter(vol.astype(np.float32), 0.6)
        if (v >= level).any():
            ijk = np.argwhere(v >= level)
            xyz = nib.affines.apply_affine(af, ijk)
            lo, hi = np.minimum(lo, xyz.min(0)), np.maximum(hi, xyz.max(0))
            meshes.append(dict(name=f"{name}.mz3", data=base64.b64encode(isosurface_mz3(v, level, af, rgb)).decode(),
                               opacity=1.0, rgba=[*rgb, 255]))
    t = nib.affines.apply_affine(tum_aff, np.argwhere(tumor)).mean(0)
    p0, p1 = 2 * t - hi, 2 * t - lo                           # the extents' centre becomes the tumour
    pts = np.array([p0, p1, (p0 + p1) / 2 + 0.01])
    tri = np.array([[0, 1, 2]])
    meshes.append(dict(name="centre.mz3", data=base64.b64encode(_mz3(tri, pts, np.zeros((3, 4)))).decode(),
                       opacity=0.0, rgba=[0, 0, 0, 0]))
    d = t - (lo + hi) / 2                                     # look from the tumour's side of the bundle
    az = a.azimuth if a.azimuth is not None else float(np.degrees(np.arctan2(-d[0], -d[1])) % 360)
    el = a.elevation if a.elevation is not None else float(max(30.0, np.degrees(np.arctan2(d[2], np.hypot(*d[:2])))))
    if a.niivue_js:
        with open(a.niivue_js) as fh:
            lib = f"<script>{fh.read()}</script>"
    else:
        lib = f'<script src="{NIIVUE_CDN}"></script>'
    page = (PAGE.replace("__NIIVUE__", lib).replace("__MESHES__", json.dumps(meshes)).replace("__W__", str(size_px))
            .replace("__ZOOM__", str(a.zoom)).replace("__AZ__", f"{az:.1f}").replace("__EL__", f"{el:.1f}"))
    html = op.splitext(png)[0] + "_3d.html"
    with open(html, "w") as fh:
        fh.write(page)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"3-D panel left out: no playwright (pip install playwright; playwright install chromium); see {html}")
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                               "--ignore-gpu-blocklist"])
            page_ = browser.new_page(viewport={"width": size_px, "height": size_px}, device_scale_factor=2)
            page_.goto("file://" + op.abspath(html))
            page_.wait_for_function("window.__done || window.__err", timeout=120000)
            err = page_.evaluate("window.__err || null")
            if err:
                raise RuntimeError(err)
            page_.wait_for_timeout(300)
            shot = page_.locator("#box").screenshot()
            browser.close()
    except Exception as e:  # no WebGL, no network for the CDN, ...
        print(f"3-D panel left out ({e}); see {html}")
        return None
    img = plt.imread(io.BytesIO(shot))[..., :3]
    lit = np.argwhere(img.max(2) > 0.02)                      # crop the black around the render
    if len(lit):
        (r0, c0), (r1, c1) = lit.min(0), lit.max(0) + 1
        pad = int(0.03 * max(img.shape[:2]))
        img = img[max(r0 - pad, 0):r1 + pad, max(c0 - pad, 0):c1 + pad]
    print(f"  3-D camera: azimuth {az:.0f}, elevation {el:.0f}")
    return img


# ----------------------------------------------------------------------- figure
def main(a):
    mpat_dir = a.mpat_dir or op.join(a.afq_out, "mpat")
    dR, aff = load(density_file(mpat_dir, RIGHT))
    dL, affL = load(density_file(mpat_dir, LEFT))
    if not np.allclose(aff, affL, atol=1e-3) or dR.shape != dL.shape:
        raise ValueError("the two IFOF maps are on different grids")
    dR, dL = dR / max(dR.max(), 1e-30), dL / max(dL.max(), 1e-30)
    t1, t1_aff = load(a.t1 or _one(op.join(a.afq_out, "*_desc-masked_T1w.nii.gz"), "pyAFQ masked T1 (give --t1)"))
    bm, bm_aff = load(a.brain_mask or _one(op.join(a.afq_out, "*_desc-brain_mask.nii.gz"),
                                           "brain mask (give --brain-mask)"))
    tumor, tum_aff = load(a.tumor)
    tumor = tumor > 0
    if not tumor.any():
        raise ValueError(f"tumour mask is empty: {a.tumor}")
    tum_xyz = nib.affines.apply_affine(tum_aff, np.argwhere(tumor))
    centre = tum_xyz.mean(0)

    # slice positions
    step = float(np.abs(nib.affines.voxel_sizes(aff)).min()) / 2
    h = (a.slab + 0.5) * 2 * step                               # the slab's half thickness
    m = dR >= a.fit_floor
    R_xyz, R_w = nib.affines.apply_affine(aff, np.argwhere(m)), dR[m].astype(float)
    pos = {"coronal": centre @ Y}
    pos["axial"], ft_ax, fb_ax = best_offset(Z, tum_xyz, R_xyz, R_w, h, step, a.ifof_weight)
    pos["sagittal"], _, fb_sag = best_offset(X, np.empty((0, 3)), R_xyz, R_w, h, step, 1.0)
    print(f"  axial slab: {100 * ft_ax:.0f}% of the tumour, {100 * fb_ax:.0f}% of the right IFOF; "
          f"sagittal slab: {100 * fb_sag:.0f}% of the right IFOF")

    # T1 grey levels as in the collages: 1-99.5 percentiles inside the brain, gamma 0.9
    same = t1.shape == bm.shape and np.allclose(t1_aff, bm_aff, atol=1e-3)
    inside = t1[(bm > 0) & (t1 > 0)] if same else t1[t1 > 0]
    vlo, vhi = np.percentile(inside, [1, 99.5])
    bxyz = nib.affines.apply_affine(bm_aff, np.argwhere(bm > 0))
    half = float(np.ceil(np.linalg.norm(bxyz - centre, axis=1).max())) + 2

    drawn = []
    for name, view, dens in ((BOTH, "axial", np.maximum(dR, dL)), (BOTH, "coronal", np.maximum(dR, dL)), (RIGHT, "sagittal", dR)):
        n, u, v = VIEWS[view]
        S = Sampler(n, u, v, centre + (pos[view] - centre @ n) * n, half, step)
        brain = S.slice((bm > 0).astype(np.float32), bm_aff) > 0.5
        grey = np.clip((S.slice(t1, t1_aff) - vlo) / max(vhi - vlo, 1e-9), 0, 1) ** 0.9 * brain
        tum = S.slice(tumor.astype(np.float32), tum_aff) > 0.5
        shadow = None
        if not tum.any():                     # the slice misses the tumour: its outline, projected
            dd = (tum_xyz - S.c) @ n
            shadow = S.projection(tumor.astype(np.float32), tum_aff, dd.min() - step, dd.max() + step)
        drawn.append(dict(name=name, view=view, S=S, grey=grey, brain=brain, tumor=tum, shadow=shadow,
                          dens=S.slab_max(dens, aff, a.slab, 2 * step)))

    # crop every panel to what it draws, one mm scale
    boxes = []
    for p in drawn:
        m = p["brain"] | (p["dens"] >= a.floor) | p["tumor"] | (p["shadow"] if p["shadow"] is not None else False)
        rc = np.argwhere(m) if m.any() else np.argwhere(np.ones_like(m))
        (r0, c0), (r1, c1) = rc.min(0), rc.max(0) + 1
        s, hh = p["S"].s, p["S"].step / 2
        boxes.append(dict(rows=slice(r0, r1), cols=slice(c0, c1), ext=[s[c0] - hh, s[c1 - 1] + hh, s[r0] - hh,
                                                                        s[r1 - 1] + hh]))
    scale = a.tile / max(max(b["ext"][1] - b["ext"][0], b["ext"][3] - b["ext"][2]) for b in boxes)  # in/mm
    sizes = [((b["ext"][1] - b["ext"][0]) * scale, (b["ext"][3] - b["ext"][2]) * scale) for b in boxes]
    out = out_png(a, mpat_dir)
    if not a.no_3d:
        img = render_3d(a, dR, aff, tumor, tum_aff, out)
        if img is not None:
            hh = max(sz[1] for sz in sizes)
            drawn.append(dict(view="3d", img=img))
            boxes.append(None)
            sizes.append((hh * img.shape[1] / img.shape[0], hh))

    # layout: panels side by side at their own widths, rows as tall as their tallest
    cols = max(1, min(a.cols, len(drawn)))
    rows = [list(range(r, min(len(drawn), r + cols))) for r in range(0, len(drawn), cols)]
    rw = [sum(sizes[i][0] for i in r) for r in rows]
    rh = [max(sizes[i][1] for i in r) for r in rows]
    fs = a.fontsize
    bottom = 0.3 + 0.04 * fs
    W, H = max(rw), sum(rh) + bottom
    fig = plt.figure(figsize=(W, H), facecolor="black")
    norm, cmap = LogNorm(a.floor, 1.0), plt.get_cmap("magma")
    tumor_cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), TUMOR_RGBA])
    im, y_top = None, H
    for r, idx in enumerate(rows):
        x = (W - rw[r]) / 2
        for i in idx:
            p, b, (w, hgt) = drawn[i], boxes[i], sizes[i]
            ax = fig.add_axes([x / W, (y_top - rh[r] + (rh[r] - hgt) / 2) / H, w / W, hgt / H])
            x += w
            ax.set_facecolor("black")
            ax.set_axis_off()
            if p["view"] == "3d":
                ax.imshow(p["img"], interpolation="lanczos")
            else:
                e = b["ext"]
                cut = lambda img: img[b["rows"], b["cols"]]  # noqa: E731
                ax.imshow(cut(p["grey"]), cmap="gray", vmin=0, vmax=1.05, origin="lower", extent=e,
                          interpolation="bilinear")
                ax.imshow(cut(p["tumor"]).astype(np.uint8), cmap=tumor_cmap, vmin=0, vmax=1, origin="lower",
                          extent=e, interpolation="nearest")
                im = ax.imshow(np.ma.masked_less(cut(p["dens"]), a.floor), cmap=cmap, norm=norm, origin="lower",
                               extent=e, interpolation="bilinear")
                ss = p["S"].s
                for mask, style in ((p["tumor"], "-"), (p["shadow"], "--")):
                    if mask is not None and cut(mask).any():
                        ax.contour(ss[b["cols"]], ss[b["rows"]], cut(mask).astype(float), levels=[0.5],
                                   colors=[TUMOR_EDGE], linewidths=0.8, linestyles=style, zorder=4)
                        break
                ax.set_xlim(e[0], e[1])
                ax.set_ylim(e[2], e[3])
                ax.set_aspect("equal")
        y_top -= rh[r]
    cax = fig.add_axes([0.04, (bottom - 0.22) / H, 0.3, 0.12 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors="white", labelsize=0.85 * fs)
    cax.text(1.06, 0.5, "density / peak (log); tumour in green", transform=cax.transAxes, color="white",
             fontsize=fs, ha="left", va="center")
    fig.savefig(out, dpi=a.dpi, facecolor="black")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(_parse())