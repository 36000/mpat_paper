#!/usr/bin/env python
"""RGVP with MCAT: eyes -> LGN, four bundles, on any subject pyAFQ can process.

The retinogeniculate visual pathway (optic nerve, chiasm, optic tract, LGN) found by the
Markov chain rather than by waypoints: streamlines with one end in an eye and the other in
a lateral geniculate nucleus.  Four bundles, solved together: each eye to each LGN.  The
chiasm sends the nasal fibres across, so both crossed bundles are expected; the chiasm
table reports each eye's crossed fraction (about 0.5 in humans).

    python mcat_rgvp.py dwi.nii.gz dwi.bval dwi.bvec T1w.nii.gz out/ \
        --n-proc 8 --max-mem 16GB

Steps (the ROI technique of the earlier HBN optic runs):

  1. pyAFQ (ParticipantAFQ) on the subject: registration to MNI, SynthSeg on the T1, its
     brain mask and fODF model.
  2. Eyes.  Each eye's MNI centre (--eye-mni-left / --eye-mni-right; defaults: the centres
     of the earlier MNI eye box) is warped into the subject with pyAFQ's own mapping (as a
     4 mm sphere, through its BundleDict ROI machinery).  Within --eye-search mm (20) of it
     the eye is the dark (fluid) blob outside SynthSeg's brain on the T1 nearest the seed.  The threshold
     is scanned from near this subject's ventricle T1 towards its white-matter T1, and the
     roundest blob of plausible globe volume wins (too generous a threshold lets the optic-
     nerve sheath join the globe as a tail).  Parts thinner than ~6 mm (the nerve and its
     sheath) are cut off first (opening with a 3 mm ball), and the eye is then a fitted
     sphere: centred where the blob is deepest, radius the median distance to its surface,
     clipped only by the field of view and the brain.  --eyes arc instead fits the sphere
     to the arc of the globe's back wall (dark vitreous against bright orbital fat; a
     gradient-directed Hough transform, then a least-squares refit): it needs only part of
     the wall, so blurred or de-faced T1s (HBN) do not stop it.  The T1 is required and must be UNMASKED (the
     orbits are cut out of pyAFQ's masked T1).  Each eye is
     dilated by --dilate-mm (5 mm) so it takes in the back of the globe and the start of
     the nerve, never into the brain.
  3. Thalami: SynthSeg labels 7 and 25, cut down to the octant that holds the lateral
     geniculate nucleus (--target lgn, the default): the posterior half, then its inferior
     half, then that block's lateral half, each cut at the midpoint along the ROI's own
     principal axis closest to the anatomical one (split_along_anatomical_axis).  The rest
     of the thalamus ends walkers invalidly (--thalamus-rest passable lets them through).
     --target thalamus uses the whole thalamus.
  4. fODF over the orbits.  pyAFQ fits its fODF inside the brain mask, and the optic nerve
     lies outside it.  The fODF is refit with pyAFQ (same model, second ParticipantAFQ in
     <out>/afq_orbits) everywhere inside the convex hull of the brain mask and the dilated
     eyes, with no signal cut (fat-suppressed orbital fat is dark on b0, and the optic nerve
     runs through it); only voxels outside the acquired field of view are left out.  If the
     mask changes, the old refit is removed and redone.
  5. Tissue.  Eyes and thalami are the only valid endpoints (grey matter).  Cerebral and
     cerebellar cortex and the lateral ventricles end walkers invalidly, so nothing reaches
     the thalamus through the frontal lobe.
     Every other voxel with an fODF is passable: no PVE, no brain mask.  The passable
     region is cropped to the convex hull of the eyes and targets, grown by --margin mm
     (10; saved as the crop mask).  A band of --thalamus-band mm (5) around the thalami is
     always passable, as white matter, whatever SynthSeg calls it and even outside the crop,
     so walkers can wrap around a thalamus to reach its LGN.  A walker may end in a target
     only stepping in from its lateral side (--entry: from the 9 neighbours one voxel lateral
     of it; lateral+plane also allows the 8 in its sagittal plane), as the optic tract
     reaches the LGN; steps in from elsewhere are invalid endings.  Walkers leaving
     the crop are removed (invalid), and the edge of the fODF field is a wall.  The endpoints'
     own fODFs (vitreous, grey matter) carry no fibre direction, so the look-ahead into
     them is isotropic.
     Image edge: the outer voxel layer of the acquired field of view (the box of
     slices with any b0 signal, so zero-padded slabs count as outside) end walkers
     invalidly, eyes and LGNs included, so streamlines that run off the image are
     excluded rather than reflected.  The log warns when an eye loses voxels to it (the
     field of view clips the orbit).
  6. Seeds: every passable voxel touching a target (LGN), not the eyes, whose surroundings
     (muscles, fat) launch walkers into the orbit (--seed both: the old behaviour).  Four bundles (start = one
     eye, end = one target), solved together.  Outputs as mcat.run writes them, a chiasm
     table, the cores, and a viewer page (slices with glyphs and the fODF, 3-D surfaces);
     in a notebook: mcat.view(<out>/mcat-rgvp, glyphs=True, fodf=...).

Written against MCAT (main, 2026-11-05) and pyAFQ 3.4; not run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as op
import shutil
import time
import warnings

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

log = logging.getLogger("mcat")

THALAMUS = {"left": 7, "right": 25}                     # SynthSeg labels (AFQ.nn.synthseg)
VENTRICLES = [3, 21]                                    # left / right lateral ventricle
CORTEX = [2, 20, 6, 24] + list(range(33, 102))          # cerebral, cerebellar cortex, parcels
WM = [1, 19]                                            # cerebral white matter
EDGE_VOX = 1                                            # image-edge band that ends walkers
EYE_ML = (2.0, 12.0)                                    # plausible globe volume before dilation
EYE_SEED_MM = 4.0                                       # radius of the MNI eye-centre sphere warped
OPEN_MM = 3.0                                           # opening radius: removes parts < ~6 mm thick
ARC_RADII = (9.0, 13.0)                                 # globe radius range searched by --eyes arc (mm)
RAS_AXES = {"AP": np.array([0.0, 1.0, 0.0]),            # positive side: anterior,
            "SI": np.array([0.0, 0.0, 1.0]),            # superior,
            "RL": np.array([1.0, 0.0, 0.0])}            # right (nibabel world = RAS+)


def parse():
    p = argparse.ArgumentParser(description="MAT RGVP: eyes -> thalamus, one bundle")
    p.add_argument("dwi"), p.add_argument("bval"), p.add_argument("bvec")
    p.add_argument("t1", help="UNMASKED T1w (orbits visible); pyAFQ runs SynthSeg on it")
    p.add_argument("output_dir")
    p.add_argument("--eye-mni-left", nargs=3, type=float, default=[-30.5, 56.4, -39.7], metavar=("X", "Y", "Z"),
                   help="left eye centre in MNI mm (default: the centre of the earlier MNI eye box)")
    p.add_argument("--eye-mni-right", nargs=3, type=float, default=[29.3, 56.4, -40.3], metavar=("X", "Y", "Z"))
    p.add_argument("--eye-search", type=float, default=20.0,
                   help="mm around each warped eye centre searched for the globe")
    p.add_argument("--n-proc", type=int, default=os.cpu_count())
    p.add_argument("--max-mem", default="16GB")
    p.add_argument("--moments", action="store_true", help="spatial moments (narrower, ~5x slower)")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--angle", type=float, default=35.0)
    p.add_argument("--eyes", choices=["threshold", "arc"], default="threshold",
                   help="threshold: the dark-blob search (threshold scan, opening, sphere fit); arc: a sphere "
                        "fitted to the arc of the globe's back wall (dark vitreous against bright orbital "
                        "fat), which needs only part of the wall, so face blurring and de-facing (HBN) do "
                        "not stop it")
    p.add_argument("--dilate-mm", type=float, default=5.0)
    p.add_argument("--margin", type=float, default=10.0,
                   help="walkers stay within the convex hull of the eyes and targets grown by this "
                        "many mm (0: the hull itself; negative: no crop)")
    p.add_argument("--entry", choices=["lateral", "lateral+plane", "any"], default="lateral",
                   help="streamlines may end in a target only stepping in from its lateral side: the 9 "
                        "neighbours one voxel lateral, or also the 8 in its sagittal plane, or any")
    p.add_argument("--thalamus-band", type=float, default=5.0,
                   help="mm around the thalami that are always passable, as white matter, so walkers "
                        "can wrap around a thalamus to its LGN (0: none)")
    p.add_argument("--seed", choices=["target", "both"], default="target",
                   help="seed next to the LGNs (thalami) only, or next to the eyes too")
    p.add_argument("--target", choices=["lgn", "thalamus"], default="lgn",
                   help="end ROI: the LGN octant of each thalamus (posterior, inferior, lateral), "
                        "or the whole thalamus")
    p.add_argument("--thalamus-rest", choices=["invalid", "passable"], default="invalid",
                   help="with --target lgn: the rest of the thalamus ends walkers (invalid) or "
                        "lets them through")
    p.add_argument("--afq-kwargs", default="{}", help="JSON, passed to ParticipantAFQ")
    return p.parse_args()


# ----------------------------------------------------------------------- helpers
def split_along_anatomical_axis(mask, affine, axis="AP", ambiguity_tol=0.1):
    """Split a binary ROI in two along the principal axis closest to an anatomical axis.

    1. Voxel indices -> world (mm) coordinates via the affine, so voxel size,
       orientation, and obliquity are all handled.
    2. PCA (SVD of the centered coordinates) gives the ROI's own axes,
       independent of the image grid and intensities.
    3. Pick the PC with the largest |cosine| to the requested anatomical axis,
       and flip its sign so it points toward A / S / R.
    4. Project every voxel onto that PC and cut at the midpoint of the extent
       (equal lengths along the axis).

    Returns (positive_side, negative_side), e.g. (anterior, posterior) for "AP".
    """
    mask = np.asarray(mask).astype(bool)
    ijk = np.argwhere(mask)
    if len(ijk) < 3:
        raise ValueError("ROI needs at least 3 voxels for a 3D PCA.")

    affine = np.asarray(affine, dtype=float)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    centered = xyz - xyz.mean(axis=0)

    # Rows of vt are the principal axes, ordered by explained variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    target = RAS_AXES[axis]
    cos = vt @ target
    order = np.argsort(-np.abs(cos))
    k = int(order[0])
    pc = vt[k] * np.sign(cos[k])  # orient toward the positive anatomical side

    # With 3 orthonormal PCs the best match is never worse than ~54.7 deg,
    # but two PCs can be nearly tied, which makes the choice unstable.
    if np.abs(cos[order[0]]) - np.abs(cos[order[1]]) < ambiguity_tol:
        warnings.warn(
            f"{axis}: PC{k + 1} (|cos|={abs(cos[k]):.2f}) and "
            f"PC{order[1] + 1} (|cos|={abs(cos[order[1]]):.2f}) are nearly "
            "equally aligned with the target axis; the selected split may be "
            "unstable across subjects.")

    proj = centered @ pc
    thr = 0.5 * (proj.min() + proj.max())  # midpoint

    pos = np.zeros_like(mask)
    pos[tuple(ijk[proj > thr].T)] = True
    neg = mask & ~pos

    return pos, neg


def restrict_entry(chain, launches, targets, affine, mode):
    """Accept an ending in a target (LGN) only when the walker steps in from its lateral side:
    from one of the 9 neighbours one voxel lateral of it ("lateral"), or also from the 8 in
    the same sagittal plane ("lateral+plane").  Steps in from elsewhere are rerouted to an
    invalid ending, in the chain's absorption (R) and in the launches' first steps, so every
    row still sums to 1.  ``targets``: {"left": mask, "right": mask} on the chain's grid."""
    side = np.zeros(chain.pshape, np.int8)
    inner = side[1:-1, 1:-1, 1:-1]
    inner[targets["left"]], inner[targets["right"]] = 1, 2
    side = side.ravel()
    k = int(np.argmax(np.abs(affine[0, :3])))       # the voxel axis closest to left-right
    sx = np.sign(affine[0, k])

    def ok(v, w):          # v: the walker's voxel, w: the target voxel (padded linear indices)
        dv = np.unravel_index(v, chain.pshape)[k] - np.unravel_index(w, chain.pshape)[k]
        lateral = np.where(side[w] == 1, -1.0, 1.0)   # world x is negative on the left
        along = dv * sx * lateral                     # > 0: the walker is lateral of the target
        return along > 0 if mode == "lateral" else along >= 0

    invalid = np.flatnonzero(chain.a_class == 3)
    if not len(invalid):
        raise SystemExit("no invalid ending next to the walkable region to reroute medial entries to")
    rows = np.repeat(np.arange(len(chain.R_ptr) - 1), np.diff(chain.R_ptr))
    w = chain.a_lin[chain.R_ind]
    hit = np.flatnonzero(side[w] > 0)
    bad = hit[~ok(chain.t_lin[rows[hit] // chain.nb], w[hit])]
    chain.R_ind[bad] = invalid[0]
    h = np.repeat(np.arange(len(launches.lptr) - 1), np.diff(launches.lptr))
    wl = launches.hseed[h] + chain.noff[launches.lkey.astype(np.int64) // chain.nb]
    lhit = np.flatnonzero(side[wl] > 0)
    lbad = lhit[~ok(launches.hseed[h][lhit], wl[lhit])]
    launches.lval[lbad] = 0.0          # a first step into the target from the wrong side: lost
    log.info("target entry %s: %d of %d steps into a target rerouted to an invalid ending, %d of %d "
             "first steps dropped", mode, len(bad), len(hit), len(lbad), len(lhit))


def lgn_octant(thalamus, affine, side):
    """The thalamus's posterior, inferior, lateral eighth, where the LGN lies (beneath the
    pulvinar, on the thalamus's inferolateral surface).  Each cut is made on the block left
    by the previous one, so each PCA follows that block's own shape."""
    _, posterior = split_along_anatomical_axis(thalamus, affine, "AP")
    _, inferior = split_along_anatomical_axis(posterior, affine, "SI")
    right, left = split_along_anatomical_axis(inferior, affine, "RL")
    return left if side == "left" else right            # lateral: away from the midline


def to_grid(mask, src_aff, shape, aff):
    """Nearest-neighbour resampling of a boolean mask onto another grid."""
    if tuple(mask.shape) == tuple(shape) and np.allclose(src_aff, aff, atol=1e-3):
        return np.asarray(mask, bool)
    r = resample_from_to(nib.Nifti1Image(mask.astype(np.float32), src_aff), (tuple(shape), aff), order=0)
    return np.asarray(r.dataobj) > 0.5


def convex_hull_mask(mask):
    """Voxels inside the convex hull of ``mask`` (same grid)."""
    from scipy.spatial import ConvexHull, Delaunay

    pts = np.argwhere(mask ^ ndi.binary_erosion(mask))       # surface voxels are enough
    tri = Delaunay(pts[ConvexHull(pts).vertices])
    lo, hi = pts.min(0), pts.max(0) + 1
    sub = np.indices(hi - lo).reshape(3, -1).T + lo
    inside = np.zeros(mask.shape, bool)
    inside[tuple(sub.T)] = tri.find_simplex(sub) >= 0
    return inside | mask


def warp_eye_points(afq, args, dwi_img, stem):
    """{side: eye centre (mm, subject)}: each MNI eye centre (--eye-mni-left / -right) as a
    small sphere on pyAFQ's template, warped into the subject as pyAFQ warps its bundle ROIs
    (with its own mapping), then its centroid."""
    from AFQ.api.bundle_dict import BundleDict

    tmpl = afq.export("reg_template")
    tmpl = nib.load(tmpl) if isinstance(tmpl, str) else tmpl
    xyz = nib.affines.apply_affine(tmpl.affine, np.indices(tmpl.shape[:3]).reshape(3, -1).T)
    points = {"left": args.eye_mni_left, "right": args.eye_mni_right}
    defs = {}
    for side, p in points.items():
        ball = (np.linalg.norm(xyz - np.asarray(p, float), axis=1) <= EYE_SEED_MM).reshape(tmpl.shape[:3])
        if not ball.any():
            raise SystemExit(f"MNI eye centre {p} ({side}) is outside pyAFQ's template")
        f = f"{stem}_space-template_desc-eyeseed{side}_mask.nii.gz"
        nib.save(nib.Nifti1Image(ball.astype(np.uint8), tmpl.affine), f)
        defs[f"Eye {side}"] = {"include": [f], "space": "template"}
    bd = BundleDict(defs, resample_to=tmpl)
    centres = {}
    for side in points:
        inc = bd.transform_rois(f"Eye {side}", afq.export("mapping"), dwi_img)["include"]
        img = inc[0] if isinstance(inc, list) else inc
        m = np.asarray(img.dataobj) > 0.5
        if not m.any():
            raise SystemExit(f"the {side} MNI eye centre warped to nothing: check pyAFQ's registration")
        centres[side] = nib.affines.apply_affine(dwi_img.affine, np.argwhere(m).mean(0))
        log.info("%s eye seed: MNI %s -> subject (%.0f, %.0f, %.0f) mm", side, points[side], *centres[side])
    return centres


def roundness(mask, affine):
    """Smallest over largest principal extent of a blob (1: a sphere, ~0: a line), in mm."""
    xyz = nib.affines.apply_affine(affine, np.argwhere(mask))
    if len(xyz) < 4:
        return 0.0
    sv = np.linalg.svd(xyz - xyz.mean(0), compute_uv=False)
    return float(sv[-1] / sv[0])


def find_eyes(args, seg, seg_aff, centres, dwi_img, save):
    """{side: dilated eye mask on the DWI grid}, info.  On the (unmasked) T1, within
    --eye-search mm of each warped MNI eye centre and outside the brain, the eye is the dark
    (fluid) blob nearest that centre.  The threshold is scanned from near the ventricles' T1
    towards white matter's, and the roundest blob of a plausible globe volume wins: a
    threshold too generous lets the optic-nerve sheath and orbital fluid join the globe as a
    tail, which makes it oblong."""
    t1 = nib.load(args.t1)
    data = np.asarray(t1.dataobj, np.float32)
    data = data[..., 0] if data.ndim == 4 else data
    shape, aff = data.shape, t1.affine
    xyz = nib.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T).reshape(shape + (3,))
    near = {s: np.linalg.norm(xyz - centres[s], axis=-1) <= args.eye_search for s in centres}
    zero = float((data[near["left"] | near["right"]] == 0).mean())
    if zero > 0.5:
        raise SystemExit(f"{args.t1} is zero in {100 * zero:.0f}% of the eye search regions: it is masked; "
                         "give the unmasked T1 (the orbits must be visible)")
    on = lambda labels: to_grid(np.isin(seg, labels), seg_aff, shape, aff)  # noqa: E731
    csf_med, wm_med = float(np.median(data[on(VENTRICLES)])), float(np.median(data[on(WM)]))
    if csf_med >= wm_med:
        raise SystemExit(f"{args.t1}: ventricles ({csf_med:.0f}) are not darker than white matter "
                         f"({wm_med:.0f}); is it a T1?")
    brain = to_grid(seg > 0, seg_aff, shape, aff)
    zooms = np.linalg.norm(aff[:3, :3], axis=0)
    r = np.ceil(OPEN_MM / zooms).astype(int)             # the opening ball, in voxels
    g = np.indices(2 * r + 1).reshape(3, -1).T - r
    ball = np.zeros(2 * r + 1, bool)
    ball[tuple((g + r)[np.linalg.norm(g * zooms, axis=1) <= OPEN_MM].T)] = True
    vox_ml = abs(np.linalg.det(aff[:3, :3])) / 1000
    eyes, info = {}, dict(ventricles_t1=csf_med, wm_t1=wm_med)
    for side in ("left", "right"):
        region = near[side] & ~brain
        best = None                                      # (plausible, roundness, frac, eye)
        for frac in np.linspace(0.1, 0.6, 26):           # threshold: ventricles -> towards WM
            thr = csf_med + frac * (wm_med - csf_med)
            lab, n = ndi.label((data < thr) & (data != 0) & region)   # 0: outside the field of view
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            big = np.flatnonzero(sizes * vox_ml >= EYE_ML[0])
            if not len(big):
                continue
            # the eye-sized blob whose centroid is nearest the seed
            cents = ndi.center_of_mass(np.ones(shape), lab, big + 1)
            dist = [np.linalg.norm(nib.affines.apply_affine(aff, c) - centres[side]) for c in cents]
            k = int(big[int(np.argmin(dist))])
            # cut off anything thinner than the opening ball (the nerve, its sheath)
            eye = ndi.binary_opening(lab == k + 1, ball)
            lab2, n2 = ndi.label(eye)
            if n2 == 0:
                continue
            eye = lab2 == 1 + int(np.argmax(np.bincount(lab2.ravel())[1:]))
            ml = eye.sum() * vox_ml
            cand = (EYE_ML[0] <= ml <= EYE_ML[1], roundness(eye, aff), frac, eye)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            raise SystemExit(f"no eye-sized fluid blob within {args.eye_search:.0f} mm of the {side} eye seed; "
                             "check the seed (log) and the T1, or raise --eye-search")
        plausible, rnd, frac, blob = best
        # the eye is a sphere: centred where the blob is deepest (a tail cannot move that), with
        # the median distance from there to the blob's surface as radius; clipped only where
        # the field of view cuts it (T1 exactly 0) and kept out of the brain
        depth = ndi.distance_transform_edt(blob, sampling=zooms)
        centre = nib.affines.apply_affine(aff, np.unravel_index(np.argmax(depth), shape))
        surface = blob & ~ndi.binary_erosion(blob)
        surface &= ~ndi.binary_dilation(data == 0)       # not the faces where the field of view cuts it
        radius = float(np.median(np.linalg.norm(nib.affines.apply_affine(aff, np.argwhere(surface)) - centre, axis=1)))
        radius += 0.5 * float(zooms.mean())              # surface voxel centres lie half a voxel inside
        eye = (np.linalg.norm(xyz - centre, axis=-1) <= radius) & (data != 0) & ~brain
        ml = float(eye.sum() * vox_ml)
        c = centre
        log.info("%s eye: sphere of radius %.1f mm (%.1f mL) at (%.0f, %.0f, %.0f) mm, fitted to the blob at "
                 "threshold %.0f%% of the way from ventricles to WM (roundness %.2f after removing thin parts)",
                 side, radius, ml, *c, 100 * frac, rnd)
        if not plausible:
            log.warning("%s eye is %.1f mL, outside the plausible %s mL: check the eye%s mask", side, ml,
                        EYE_ML, side)
        if not 9.0 <= radius <= 14.0:
            log.warning("%s eye radius %.1f mm is outside the adult globe's ~11-13 mm: check the eye%s mask",
                        side, radius, side)
        dist = ndi.distance_transform_edt(~eye, sampling=np.linalg.norm(aff[:3, :3], axis=0))
        grown = (dist <= args.dilate_mm) & ~brain        # back of the globe, never into the brain
        eyes[side] = to_grid(grown, aff, dwi_img.shape[:3], dwi_img.affine)
        save(eyes[side], f"eye{side}")
        info[side] = dict(ml=ml, radius_mm=radius, roundness_before_fit=rnd, threshold_fraction=float(frac),
                          centre_mm=[float(v) for v in c])
    if (eyes["left"] & eyes["right"]).any():
        raise SystemExit("the dilated eyes overlap; lower --dilate-mm")
    return eyes, info


def load_t1(args):
    t1 = nib.load(args.t1)
    data = np.asarray(t1.dataobj, np.float32)
    return t1, (data[..., 0] if data.ndim == 4 else data)


def ball_mask(shape, aff, centre, radius):
    """Voxels of a grid whose centres lie within ``radius`` mm of ``centre`` (computed in its box)."""
    zooms = np.linalg.norm(aff[:3, :3], axis=0)
    c = nib.affines.apply_affine(np.linalg.inv(aff), centre)
    lo = np.maximum(np.floor(c - radius / zooms - 1).astype(int), 0)
    hi = np.minimum(np.ceil(c + radius / zooms + 2).astype(int), shape[:3])
    out = np.zeros(shape[:3], bool)
    if np.any(hi <= lo):
        return out
    ijk = np.indices(hi - lo).reshape(3, -1).T + lo
    inside = np.linalg.norm(nib.affines.apply_affine(aff, ijk) - centre, axis=1) <= radius
    out[tuple(ijk[inside].T)] = True
    return out


def grow_eye(eye, aff, brain, mm):
    """``eye`` grown by ``mm`` (same grid), never into the brain."""
    if mm <= 0:
        return eye & ~brain
    dist = ndi.distance_transform_edt(~eye, sampling=np.linalg.norm(aff[:3, :3], axis=0))
    return (dist <= mm) & ~brain


def fit_eye_arc(data, affine, prior_mm, search_mm=20.0, radii=ARC_RADII, sigma_mm=1.0, cell_mm=0.5,
                edge_pct=75.0, min_polarity=0.5):
    """(centre_mm, radius_mm, info) of the globe near ``prior_mm`` on a T1.

    On a T1 the globe is dark and the orbital fat around its back is bright, so the back wall
    is a strong dark-to-bright edge whose gradient points away from the eye's centre.  Every
    such edge voxel votes for the centres a radius r back along its gradient (gradient-directed
    Hough, r over ``radii``); a sphere's votes pile up at its centre however little of it is
    left (de-facing, face blurring, a cut field of view), while the blur boundary (a plane),
    the smooth blurred face and the nerve spread theirs out.  The peak is refined by a
    weighted least-squares sphere fit to the edges that face it."""
    data = np.asarray(data, np.float32)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    inv = np.linalg.inv(affine)
    prior_mm = np.asarray(prior_mm, float)
    reach = search_mm + radii[1] + 3 * sigma_mm + 2
    c_vox = np.floor(nib.affines.apply_affine(inv, prior_mm)).astype(int)
    half = np.ceil(reach / zooms).astype(int)
    lo, hi = np.maximum(c_vox - half, 0), np.minimum(c_vox + half + 1, data.shape[:3])
    sub = data[tuple(slice(a, b) for a, b in zip(lo, hi))]
    # gradient in world mm: d I / d world = (d I / d voxel) M^-1
    g_vox = np.stack([ndi.gaussian_filter(sub, sigma_mm / zooms, order=tuple(int(i == k) for i in range(3)))
                      for k in range(3)], -1).reshape(-1, 3)
    M = affine[:3, :3]
    g = g_vox @ np.linalg.inv(M)
    p = (np.indices(sub.shape).reshape(3, -1).T + lo) @ M.T + affine[:3, 3]
    mag = np.linalg.norm(g, axis=1)
    off = p - prior_mm
    dist = np.linalg.norm(off, axis=1)
    near = (dist <= search_mm + radii[1]) & (dist >= 2.0)
    pol = np.einsum("ij,ij->i", off, g) / np.maximum(dist * mag, 1e-9)
    e = near & (mag >= np.percentile(mag[near], edge_pct)) & (pol >= min_polarity)
    pe, ge, me = p[e], g[e] / mag[e, None], mag[e]
    w = np.minimum(me / np.median(me), 3.0)

    n = int(np.ceil(2 * search_mm / cell_mm)) + 1
    acc = np.zeros((n, n, n), np.float32)
    for r in np.arange(radii[0], radii[1] + 1e-6, cell_mm):
        idx = np.round((pe - r * ge - prior_mm + search_mm) / cell_mm).astype(int)
        ok = np.all((idx >= 0) & (idx < n), axis=1)
        np.add.at(acc, tuple(idx[ok].T), w[ok])
    acc = ndi.gaussian_filter(acc, 1.0 / cell_mm)
    grid = np.indices(acc.shape).transpose(1, 2, 3, 0) * cell_mm - search_mm
    acc *= np.exp(-0.5 * (np.linalg.norm(grid, axis=-1) / (1.5 * search_mm)) ** 2)   # ties: nearer the prior
    c = prior_mm - search_mm + np.array(np.unravel_index(np.argmax(acc), acc.shape)) * cell_mm

    def inliers(c, r=None, tol=1.5):
        o = pe - c
        d = np.linalg.norm(o, axis=1)
        ok = (np.einsum("ij,ij->i", o, ge) / np.maximum(d, 1e-9) >= 0.8) & (d >= radii[0] - 2) & (d <= radii[1] + 2)
        if r is not None:
            ok &= np.abs(d - r) <= tol
        return ok, d

    ok, d = inliers(c)
    hist, edges = np.histogram(d[ok], bins=np.arange(radii[0] - 2, radii[1] + 2.01, 0.5), weights=w[ok])
    r = float(np.clip(edges[np.argmax(np.convolve(hist, [0.25, 0.5, 0.25], "same"))] + 0.25, *radii))
    for tol in (2.0, 1.5, 1.0, 1.0):
        ok, d = inliers(c, r, tol)
        if ok.sum() < 20:
            break
        P, sw = pe[ok], np.sqrt(w[ok])
        sol, *_ = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))] * sw[:, None], (P ** 2).sum(1) * sw, rcond=None)
        c_new, r_new = sol[:3], float(np.sqrt(max(sol[3] + sol[:3] @ sol[:3], 1e-6)))
        if not (radii[0] - 1 <= r_new <= radii[1] + 1) or np.linalg.norm(c_new - c) > 4:
            break                                        # the free fit ran off: keep the last answer
        c, r = c_new, float(np.clip(r_new, *radii))
    ok, d = inliers(c, r, 1.0)
    # coverage: the fraction of 100 directions around the centre with an edge on the sphere
    k = np.arange(100) + 0.5
    phi, th = np.arccos(1 - 2 * k / 100), np.pi * (1 + 5 ** 0.5) * k
    dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    u = (pe[ok] - c) / np.maximum(d[ok, None], 1e-9)
    cover = len(np.unique(np.argmax(u @ dirs.T, axis=1))) / 100 if len(u) else 0.0
    return c, r, dict(n_edges=int(e.sum()), n_inliers=int(ok.sum()), coverage=float(cover))


def arc_eyes(args, seg, seg_aff, centres, dwi_img, save):
    """{side: eye mask on the DWI grid}, info, with each eye the sphere fitted to its wall's arc."""
    t1, data = load_t1(args)
    aff, shape = t1.affine, data.shape
    brain = to_grid(seg > 0, seg_aff, shape, aff)
    eyes, info = {}, {}
    for side in ("left", "right"):
        c, r, fit = fit_eye_arc(data, aff, centres[side], search_mm=args.eye_search)
        log.info("%s eye (arc): sphere of radius %.1f mm at (%.1f, %.1f, %.1f) mm, %.1f mm from the seed; "
                 "%d edge voxels on it, covering %.0f%% of directions", side, r, *c,
                 np.linalg.norm(c - centres[side]), fit["n_inliers"], 100 * fit["coverage"])
        if fit["coverage"] < 0.2 or fit["n_inliers"] < 100:
            log.warning("%s eye (arc): little of the globe's wall was found: check the eye%s mask", side, side)
        if min(abs(r - ARC_RADII[0]), abs(r - ARC_RADII[1])) < 0.3:
            log.warning("%s eye (arc): radius %.1f mm is at the edge of the range searched %s", side, r, ARC_RADII)
        eye = ball_mask(shape, aff, c, r) & (data != 0) & ~brain
        eyes[side] = to_grid(grow_eye(eye, aff, brain, args.dilate_mm), aff, dwi_img.shape[:3], dwi_img.affine)
        save(eyes[side], f"eye{side}")
        info[side] = dict(method="arc", centre_mm=c.tolist(), radius_mm=r, **fit)
    if (eyes["left"] & eyes["right"]).any():
        raise SystemExit("the eyes overlap; lower --dilate-mm")
    return eyes, info


# ----------------------------------------------------------------------- run
def main():
    args = parse()
    from AFQ.api.participant import ParticipantAFQ
    from AFQ.definitions.image import ImageFile

    import numba

    from mcat.afq import ODF_FLOOR, _bytes
    from mcat.chain import Chain, ChainConfig
    from mcat.core import write_cores
    from mcat.geometry import visitation_odf
    from mcat.solve import prepare_launches, segment_bundles
    from mcat.viewer import save_html

    numba.set_num_threads(max(1, min(args.n_proc, numba.config.NUMBA_NUM_THREADS)))
    t0 = time.time()
    kw = json.loads(args.afq_kwargs)
    afq = ParticipantAFQ(args.dwi, args.bval, args.bvec, args.t1, args.output_dir, tracking_params={"odf_model": "CSD"}, **kw)
    out = op.join(args.output_dir, "mcat-rgvp-moments" if args.moments else "mcat-rgvp")
    os.makedirs(out, exist_ok=True)
    stem = op.join(out, op.basename(afq.kwargs["base_fname"]))
    fh = logging.FileHandler(f"{stem}_desc-rgvp_log.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    model = f"{afq.export('tracking_params')['odf_model'].lower()}_params"
    sh_path = afq.export(model)
    sh_img = nib.load(sh_path)
    shape, aff = sh_img.shape[:3], sh_img.affine

    def save(mask, desc):
        nib.save(nib.Nifti1Image(np.asarray(mask, np.uint8), aff), f"{stem}_desc-{desc}_mask.nii.gz")

    # ---- endpoints on the DWI grid
    seg_img = nib.load(afq.export("synthseg_model"))
    seg = np.rint(np.asarray(seg_img.dataobj)).astype(np.int32)
    seg = seg[..., 0] if seg.ndim == 4 else seg
    centres = warp_eye_points(afq, args, sh_img, stem)
    if args.eyes == "arc":
        eyes, eye_info = arc_eyes(args, seg, seg_img.affine, centres, sh_img, save)
    else:
        eyes, eye_info = find_eyes(args, seg, seg_img.affine, centres, sh_img, save)
    # thalamus (or its LGN octant), cut on SynthSeg's finer grid, then put on the DWI grid
    thal, rest = {}, np.zeros(shape, bool)
    vox_ml = abs(np.linalg.det(seg_img.affine[:3, :3])) / 1000
    for s, lab in THALAMUS.items():
        whole = seg == lab
        target = lgn_octant(whole, seg_img.affine, s) if args.target == "lgn" else whole
        thal[s] = to_grid(target, seg_img.affine, shape, aff)
        rest |= to_grid(whole, seg_img.affine, shape, aff) & ~thal[s]
        c = nib.affines.apply_affine(seg_img.affine, np.argwhere(target).mean(0))
        log.info("%s %s: %.2f mL of a %.2f mL thalamus, centre (%.0f, %.0f, %.0f) mm, %d DWI voxels",
                 s, args.target, target.sum() * vox_ml, whole.sum() * vox_ml, *c, thal[s].sum())
        save(thal[s], f"{args.target}{s}")
    start, end = eyes["left"] | eyes["right"], thal["left"] | thal["right"]
    gm = start | end
    rest &= ~gm

    # ---- fODF that covers the orbits
    # fitted everywhere in the hull of brain + eyes, with no signal cut: fat-suppressed orbital
    # fat is dark on b0, and the optic nerve runs through it.  Only voxels outside the
    # acquired field of view (b0 exactly 0) are left out
    refit_dir = op.join(args.output_dir, "afq_orbits")
    mask_f = op.join(refit_dir, "brainandorbits_mask.nii.gz")
    brain = to_grid(np.asarray(nib.load(afq.export("brain_mask")).dataobj) > 0,
                    nib.load(afq.export("brain_mask")).affine, shape, aff)
    b0 = np.asarray(nib.load(afq.export("b0")).dataobj, np.float32)
    fit = convex_hull_mask(brain | start) & (b0 != 0)
    if op.exists(mask_f) and not np.array_equal(np.asarray(nib.load(mask_f).dataobj) > 0, fit):
        # pyAFQ would reuse the fODF it fitted in the old mask: start the refit over
        log.warning("the refit mask changed: removing %s and refitting the fODF", refit_dir)
        shutil.rmtree(refit_dir)
    os.makedirs(refit_dir, exist_ok=True)
    if not op.exists(mask_f):
        nib.save(nib.Nifti1Image(fit.astype(np.uint8), aff), mask_f)
    log.info("refit mask: brain %d -> brain + orbits %d voxels", brain.sum(), fit.sum())
    kw2 = dict(kw, brain_mask_definition=ImageFile(path=mask_f))
    afq2 = ParticipantAFQ(args.dwi, args.bval, args.bvec, args.t1, refit_dir, tracking_params={"odf_model": "CSD"}, **kw2)
    sh_path = afq2.export(model)
    sh_img = nib.load(sh_path)
    if sh_img.shape[:3] != tuple(shape) or not np.allclose(sh_img.affine, aff, atol=1e-3):
        raise SystemExit("the refit fODF is on another grid")
    sh = np.nan_to_num(np.asarray(sh_img.dataobj, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    has_odf = np.any(sh != 0, axis=-1)

    # ---- edge of the image: the outer voxels of the acquired field of view (the box of
    # slices with any b0 signal; zero-padded slabs are outside it) end walkers invalidly
    edge = np.zeros(shape, bool)
    b0 = np.asarray(nib.load(afq.export("b0")).dataobj, np.float32)
    sig = np.argwhere(b0 > 0)
    lo, hi = sig.min(0) + EDGE_VOX, sig.max(0) - EDGE_VOX + 1
    edge[:] = True
    edge[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
    for side in ("left", "right"):
        lost = (eyes[side] & edge).sum()
        if lost:
            log.warning("%s eye: %d of %d voxels lie on the image edge (the field of view clips "
                        "the orbit); they are not valid endpoints", side, lost, eyes[side].sum())
        eyes[side] &= ~edge
        thal[side] &= ~edge
    gm = eyes["left"] | eyes["right"] | thal["left"] | thal["right"]
    log.info("image edge: %d voxels (%d with an fODF) end walkers invalidly", edge.sum(), (edge & has_odf).sum())

    # ---- tissue: eyes + thalami valid ends, cortex / ventricles invalid, the rest passable
    transient = has_odf & ~gm
    if args.margin >= 0:   # the convex hull of the eyes and targets, grown by --margin mm
        crop = convex_hull_mask(gm)
        if args.margin > 0:
            crop = ndi.distance_transform_edt(~crop, sampling=np.linalg.norm(aff[:3, :3], axis=0)) <= args.margin
        log.info("crop: hull of eyes + %ss, %.0f mm margin: %d of %d fODF voxels passable", args.target,
                 args.margin, (transient & crop).sum(), transient.sum())
        transient &= crop
        save(crop, "crop")
    kill_labels = CORTEX + VENTRICLES
    kill = to_grid(np.isin(seg, kill_labels), seg_img.affine, shape, aff) & ~gm & has_odf
    if args.target == "lgn" and args.thalamus_rest == "invalid":
        kill |= rest & has_odf          # walkers reaching another part of the thalamus end there
    transient &= ~kill
    # a band around the thalami that walkers can always move in (as white matter, whatever
    # SynthSeg calls it and wherever the crop ends), so they can wrap around a thalamus to
    # reach its LGN
    if args.thalamus_band > 0:
        whole = rest | thal["left"] | thal["right"]
        band = (ndi.distance_transform_edt(~whole, sampling=np.linalg.norm(aff[:3, :3], axis=0))
                <= args.thalamus_band) & ~whole & has_odf & ~gm
        log.info("thalamus band: %d voxels within %.0f mm of the thalami are passable (%d were not)",
                 band.sum(), args.thalamus_band, (band & ~transient).sum())
        transient |= band
        kill &= ~band
        save(band, "thalamusband")
    absorb = np.zeros(shape, np.int8)
    transient &= ~edge
    absorb[has_odf & ~transient] = 3   # invalid ending: cortex, ventricles, leaving the crop
    absorb[edge] = 3                   # invalid ending: the image edge
    absorb[gm] = 1                     # GM: valid ending (only eyes and targets)
    sh_chain = sh.copy()               # isotropic look-ahead into the endpoints and the image
    iso = gm | edge                    # edge (which may have no fODF, and would be a wall)
    sh_chain[iso] = 0.0
    sh_chain[iso, 0] = 1.0
    dseg = np.zeros(shape, np.uint8)
    dseg[transient], dseg[gm], dseg[kill], dseg[edge] = 1, 2, 3, 4
    nib.save(nib.Nifti1Image(dseg, aff), f"{stem}_desc-mcat_dseg.nii.gz")
    for side in ("left", "right"):
        shell = ndi.binary_dilation(eyes[side], np.ones((3, 3, 3), bool)) & ~gm
        frac = (shell & transient).sum() / max(shell.sum(), 1)
        log.info("%s eye: %.0f%% of its surroundings passable", side, 100 * frac)
        if frac == 0:
            raise SystemExit(f"no fODF around the {side} eye (check afq_orbits/brainandorbits_mask)")

    # ---- chain and the four bundles
    cfg = ChainConfig(angle=args.angle, moments=args.moments)
    chain = Chain.build(sh_chain, sh_img.header.get_zooms()[:3], transient, absorb, cfg,
                        cache=f"{stem}_desc-rgvpchain_cache.npz")
    del sh_chain
    label = "LGN" if args.target == "lgn" else "thalamus"
    # seeds: passable voxels touching a target (LGN / thalamus) only -- seeding around the eyes
    # launches walkers into the orbit (muscles, fat) under and beside the globe
    near = thal["left"] | thal["right"] | ((eyes["left"] | eyes["right"]) if args.seed == "both" else False)
    seeds = np.argwhere(transient & ndi.binary_dilation(near, np.ones((3, 3, 3), bool)))
    log.info("seeding %d passable voxels touching %s", len(seeds),
             f"a {label}" if args.seed == "target" else f"an eye or a {label}")
    launches = prepare_launches(chain, seeds)
    chain.release_pmfs()
    if args.entry != "any":
        restrict_entry(chain, launches, thal, aff, args.entry)
    specs, sides = [], {}
    for e in ("left", "right"):
        for t in ("left", "right"):
            name = f"{e.title()} eye - {t.title()} {label}"
            specs.append(dict(name=name, include=[], exclude=None, start=eyes[e], end=thal[t]))
            sides[name] = (e, t)
    counts = {}

    def write(res):
        e, t = sides[res.name]
        b = f"{stem}_bundle-{''.join(ch for ch in res.name if ch.isalnum())}"
        d = res.density_t
        keep = np.flatnonzero(d >= ODF_FLOOR * d.max()) if d.max() > 0 else np.zeros(0, np.int64)
        ijk = tuple((np.stack(np.unravel_index(chain.t_lin[keep], chain.pshape), 1) - 1).T)
        coef = visitation_odf(res.density_class[keep], sh[ijk], chain.geo)
        odf = np.zeros(shape + coef.shape[1:], np.float32)
        odf[ijk] = coef
        sp = np.zeros(shape, np.float32)
        np.add.at(sp, tuple(res.seed_vox.T), res.seed_prob)
        ends = chain.to_volume(res.endpoints_a, "absorbing")
        for desc, vol, suffix in (("density", chain.to_volume(d), "map"), ("endpoints", ends, "map"),
                                  ("seedprob", sp, "probmap"), ("visitodf", odf, "dwimap")):
            nib.save(nib.Nifti1Image(vol, aff), f"{b}_desc-{desc}_{suffix}.nii.gz")
        for desc, m in (("start", eyes[e]), ("end", thal[t])):
            nib.save(nib.Nifti1Image(m.astype(np.uint8), aff), f"{b}_desc-{desc}_mask.nii.gz")
        counts[res.name] = n = res.info["expected_bundle_streamlines"]
        with open(f"{b}_desc-summary.json", "w") as fh:
            json.dump(dict(bundle=res.name, eye=e, target=f"{t} {args.target}", expected_streamlines=n,
                           thalamus_rest=args.thalamus_rest, thalamus_band_mm=args.thalamus_band, entry=args.entry,
                           eyes=eye_info,
                           n_seeds=int(len(seeds)), config=cfg.key(), result=res.info), fh, indent=1, default=str)
        log.info("%s: %.3g expected streamlines", res.name, n)

    segment_bundles(chain, specs, launches, tol=args.tol, max_mem=_bytes(args.max_mem), valid_classes=(1,),
                    on_result=write)

    # ---- chiasm: each eye's crossed fraction
    rows = []
    for e in ("left", "right"):
        other = "right" if e == "left" else "left"
        same = counts[f"{e.title()} eye - {e.title()} {label}"]
        cross = counts[f"{e.title()} eye - {other.title()} {label}"]
        frac = cross / (same + cross) if same + cross > 0 else float("nan")
        rows.append(f"{e}\t{same:.6g}\t{cross:.6g}\t{frac:.4f}")
        log.info("%s eye: %.3g uncrossed, %.3g crossed -> crossed fraction %.2f (human chiasm: about 0.5)",
                 e, same, cross, frac)
    with open(f"{stem}_desc-rgvp_chiasm.tsv", "w") as fh:
        fh.write("eye\tuncrossed\tcrossed\tcrossed_fraction\n" + "\n".join(rows) + "\n")
    write_cores(out)
    page = save_html(f"{stem}_desc-rgvp_view.html", out, glyphs=True, fodf=sh_path)
    log.info("done in %.0f min -> %s (viewer: %s)", (time.time() - t0) / 60, out, page)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
