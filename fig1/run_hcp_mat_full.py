#!/usr/bin/env python
"""HCP test-retest of pyAFQ's default bundles with MPAT (and, with --pyafq, pyAFQ's own
tractography): fuzzy Dice only.

    python run_hcp_mat_full.py 111312 [more subjects ...] [--pyafq]
    python run_hcp_mat_full.py --combine          # concatenate every subject's CSV

For each subject and session (1200, Retest) this reuses the pyAFQ derivatives that
run_hcp_reward.py wrote to <writeable>/reward/pyafq_out/sub-<id>/ses-<ses>/ (msmt CSD fit,
SynthSeg, PVE, brain mask, WM/GM interface, SyN mapping) and segments pyAFQ's default
bundles (default_bd + slf_bd + callosal_bd, what pyAFQ uses when bundle_info is unset)
with MPAT.  pyAFQ's output_dir is the existing session folder, so it finds its
derivatives there and recomputes nothing (it only computes a file that is missing);
MPAT writes only to its own tree (mpat.afq.run's ``out_dir``):

    <out-root>/sub-<id>/ses-<ses>/mpat/       MPAT outputs (densities, endpoints, ...)
    <out-root>/sub-<id>/sub-<id>_desc-fuzzydice.csv
    <out-root>/sub-<id>/sub-<id>_log.txt
    <out-root>/all_desc-fuzzydice.csv          (--combine)

--pyafq: pyAFQ's own pipeline as well (tractography + recognition, the same settings as the
    derivatives: msmt CSD, trx), in its own folder <out-root>/sub-<id>/ses-<ses>/pyafq/.
    That folder starts as hard links to the pyAFQ session's derivatives (no extra disk, not
    symlinks; copies if the trees are on different file systems), without any tractography,
    so pyAFQ recomputes only the tractography and the bundles and never writes into the
    original folder.  Each bundle's streamlines become a density map (streamline count per
    voxel, dipy density_map) on MPAT's grid, pyafq/densities/, and their fuzzy Dice goes in the same
    CSV with technique "pyAFQ" (expected_streamlines_* are then the streamline counts).
    Bundles: pyAFQ's defaults only (pyAFQ rejects MPAT's "mpat" rules on the HCC and fornix
    definitions).  Resumes: a session whose densities are all written is skipped.

--cst-fornix-hcc (for a tree written before MPAT's midline side fix):
    reruns both corticospinal tracts (the Left CST came out empty: MPAT kept it to the right
    hemisphere) and adds the fornix and the hippocampal cingulum (MPAT's bundle dictionary,
    mpat.bundles.mpat_bd); the other bundles' outputs are reused.  The old CST outputs move to
    <out-root>/sub-<id>/ses-<ses>/mpat_old_cst/ the first time (once per session, so a rerun
    resumes instead of starting over).  The fornix's hypothalamic end needs SynthSeg's
    hypothalamus, which pyAFQ's SynthSeg run lacks: MPAT reruns SynthSeg for it on its own,
    once per session, into the MPAT folder (sub-<id>_model-synthseg2hypo_dseg.nii.gz).

fuzzy_dice = sum_v min(p_v, q_v), p and q the two sessions' density maps each scaled to
sum 1 (Crum et al. 2006): threshold-free, 1 for identical maps.  A bundle missing or
empty in either session gets 0 (as wdsc_common.dice_or_zero); sessions on different
grids get NaN.  expected_streamlines_* are MPAT's expected numbers of bundle
streamlines (one streamline per fODF peak of each WM/GM-interface seed voxel).
"""
from __future__ import annotations

import argparse
import glob
import shutil
import json
import logging
import multiprocessing
import os
import os.path as op
import re
import sys

# ----------------------------------------------------------------------- settings
SESSIONS = ["1200", "Retest"]
COHORT = "HCP"
TECHNIQUE = "MPAT"
PYAFQ_TECHNIQUE = "pyAFQ"
# pyAFQ outputs that --pyafq recomputes: never linked into its folder
NOT_LINKED = re.compile(r"(^|/)(tractography|bundles|clean_bundles|viz_bundles|stats|tract_profiles)(/|$)"
                        r"|_tractography|_profiles|\.html$")
TOL = 1e-6              # relative tolerance of the solves (MPAT's default)
REDO_BUNDLES = ("Left Corticospinal", "Right Corticospinal")          # --cst-fornix-hcc
NEW_BUNDLES = ("Left Fornix", "Right Fornix", "Left Cingulum Hippocampus", "Right Cingulum Hippocampus")
MAX_MEM = "auto"        # MPAT memory budget: "auto" = min(80%, limit - 10 GB) of the job's limit
OUT_NAME = "hcp_default_mpat_msmtcsd"   # a fresh tree: the earlier one (msmt aODF) is not resumed
# pyAFQ derivatives MPAT needs; a missing one would be computed into the pyAFQ folder
NEEDED = ("models/sub-{sub}_model-msmtcsd_param-wm_dwimap.nii.gz",
          "nn/sub-{sub}_model-synthseg2_probseg.nii.gz", "sub-{sub}_desc-pve_probseg.nii.gz",
          "sub-{sub}_desc-brain_mask.nii.gz", "sub-{sub}_desc-wmgmi_mask.nii.gz",
          "sub-{sub}_desc-mapping_from-mni_to-subject_xform.nii.gz")

try:
    from paths import read_only_path, writeable_path
except ImportError:  # outside the cluster layout: give every root on the command line
    read_only_path = writeable_path = None


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("subjects", nargs="*", help="subject IDs without 'sub-'")
    p.add_argument("--combine", action="store_true",
                   help="concatenate every sub-*/sub-*_desc-fuzzydice.csv under --out-root")
    p.add_argument("--pyafq-root", default=writeable_path and writeable_path + "reward/pyafq_out",
                   help="pyAFQ outputs of run_hcp_reward.py (sub-*/ses-*)")
    p.add_argument("--hcp-root", default=read_only_path and read_only_path + "data",
                   help="holds HCP_<session>/derivatives/dmriprep/ (the DWI pyAFQ was run on)")
    p.add_argument("--out-root", default=writeable_path and writeable_path + OUT_NAME)
    p.add_argument("--sessions", nargs="+", default=SESSIONS)
    p.add_argument("--bundles", nargs="+", help="only these pyAFQ bundle names (default: all defaults)")
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="threads (default: all cores)")
    p.add_argument("--max-mem", default=MAX_MEM, help=f"MPAT memory budget (default {MAX_MEM})")
    p.add_argument("--tol", type=float, default=TOL)
    p.add_argument("--moments", action="store_true", help="MPAT spatial first moments (~5x slower)")
    p.add_argument("--verbose", action="store_true",
                   help="MPAT timing inside the solves (ms per power step)")
    p.add_argument("--cores", action="store_true", help="also write MPAT's bundle cores")
    p.add_argument("--viz", action="store_true", help="also write MPAT's collage and viewer pages")
    p.add_argument("--force", action="store_true", help="rerun MPAT even if its outputs exist")
    p.add_argument("--cst-fornix-hcc", action="store_true",
                   help="rerun the corticospinal tracts (midline side fix) and add the fornix and the "
                        "hippocampal cingulum; the other bundles' outputs are reused")
    p.add_argument("--pyafq", action="store_true",
                   help="also run pyAFQ's own tractography and recognition (its own folder) and compute "
                        "its fuzzy Dice (technique pyAFQ in the CSV)")
    p.add_argument("--allow-pyafq", action="store_true",
                   help="let pyAFQ compute missing derivatives (into the pyAFQ folder); default: stop")
    a = p.parse_args(argv)
    if not a.subjects and not a.combine:
        p.error("give subject IDs and/or --combine")
    for k in ("out_root",) + (("pyafq_root", "hcp_root") if a.subjects else ()):
        if not getattr(a, k):
            p.error(f"--{k.replace('_', '-')} is required when paths.py is not importable")
    return a


A = _parse() if __name__ == "__main__" else None
if A is not None:
    # as run_hcp_reward.py: the same templates and caches
    if read_only_path:
        os.environ.setdefault("TEMPLATEFLOW_HOME", read_only_path + "data/templateflow/")
        os.environ.setdefault("DIPY_HOME", read_only_path + "data/")
        os.environ.setdefault("AFQ_HOME", read_only_path + "data/")
    if writeable_path:
        os.environ.setdefault("NUMBA_CACHE_DIR", writeable_path + "cache/")
        os.environ.setdefault("TRX_TMPDIR", writeable_path + "TRX_TMP/")
        os.makedirs(os.environ["TRX_TMPDIR"], exist_ok=True)
    if A.threads:
        os.environ["NUMBA_NUM_THREADS"] = str(A.threads)

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("hcp_mpat")


# ----------------------------------------------------------------------- names
def desc_of(name):
    """MPAT's file name for a bundle (mpat.afq: every non-alphanumeric dropped)."""
    return re.sub(r"[^A-Za-z0-9]", "", name)


def split_desc(desc):
    """wdsc_common.split_desc: 'LeftPosteriorArcuate' -> ('Left', 'Posterior Arcuate')."""
    hemi, name = "", desc
    for h in ("Left", "Right"):
        if desc.startswith(h) and len(desc) > len(h):
            hemi, name = h, desc[len(h):]
            break
    return hemi, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


# ----------------------------------------------------------------------- pyAFQ, reused
def participant(a, sub, ses, pyafq_dir, mpat_bundles=True):
    """The ParticipantAFQ of run_hcp_reward.py on its own output folder (same inputs and
    the settings that name or shape the derivatives), with pyAFQ's default bundles (MPAT's,
    with --cst-fornix-hcc and ``mpat_bundles``)."""
    from AFQ.api.participant import ParticipantAFQ
    import AFQ.definitions.image as afm

    hcp = op.join(a.hcp_root, f"HCP_{ses}", "derivatives", "dmriprep", f"sub-{sub}", "ses-01")
    dwi = dict(dwi_data_file=op.join(hcp, "dwi", f"sub-{sub}_dwi.nii.gz"),
               bval_file=op.join(hcp, "dwi", f"sub-{sub}_dwi.bval"),
               bvec_file=op.join(hcp, "dwi", f"sub-{sub}_dwi.bvec"),
               t1_file=op.join(hcp, "anat", f"sub-{sub}_T1w.nii.gz"))
    missing = [f for f in dwi.values() if not op.exists(f)]
    if missing:
        raise FileNotFoundError(f"sub-{sub} ses-{ses}: missing {missing}")
    synthseg = op.join(pyafq_dir, "nn", f"sub-{sub}_model-synthseg2_probseg.nii.gz")
    extra = {}
    if a.cst_fornix_hcc and mpat_bundles:     # MPAT's bundles: pyAFQ's defaults + the HCC + the fornix
        from mpat.bundles import mpat_bd

        extra["bundle_info"] = mpat_bd()
    return ParticipantAFQ(
        **dwi, output_dir=pyafq_dir, numba_n_threads=a.threads, **extra,
        brain_mask_definition=afm.LabelledImageFile(path=synthseg, exclusive_labels=[0]),
        tracking_params={"odf_model": "msmtcsd", "trx": True})   # msmt CSD (symmetric) fODF, read by MPAT too


# ----------------------------------------------------------------------- one session
def bundle_done(mdir, sub, name):
    stem = op.join(mdir, f"sub-{sub}_bundle-{desc_of(name)}")
    return op.exists(f"{stem}_desc-density_map.nii.gz") and op.exists(f"{stem}_desc-summary.json")


def names_file(mdir, sub):
    """The session's bundle names (pyAFQ's bundle dictionary), saved by run_session."""
    return op.join(mdir, f"sub-{sub}_desc-bundlenames.json")


def retire_old_cst(mdir, sub):
    """--cst-fornix-hcc: move the corticospinal outputs written before the midline side fix out
    of the MPAT folder (to mpat_old_cst/ beside it), once per session (a marker file remembers),
    so they are recomputed and a resumed run does not recompute them again."""
    marker = op.join(mdir, f"sub-{sub}_desc-cstredone.json")
    if op.exists(marker) or not op.isdir(mdir):
        return
    old = op.join(op.dirname(mdir), "mpat_old_cst")
    moved = []
    for name in REDO_BUNDLES:
        for f in glob.glob(op.join(mdir, f"sub-{sub}_bundle-{desc_of(name)}_*")):
            os.makedirs(old, exist_ok=True)
            shutil.move(f, op.join(old, op.basename(f)))
            moved.append(op.basename(f))
    with open(marker, "w") as fh:
        json.dump(dict(reason="midline side fix: corticospinal tracts recomputed", moved_to=old,
                       moved=moved), fh, indent=1)
    log.info("%s: %d old corticospinal files moved to %s; they will be recomputed", mdir, len(moved), old)


def add_names(mdir, sub, names, new):
    """The session's bundle names with ``new`` appended (saved)."""
    names = list(names) + [n for n in new if n not in names]
    with open(names_file(mdir, sub), "w") as fh:
        json.dump(names, fh, indent=1)
    return names


def run_session(a, sub, ses):
    """MPAT on the bundles of this session that have no outputs yet (all with --force)."""
    from mpat.afq import run

    src = op.join(a.pyafq_root, f"sub-{sub}", f"ses-{ses}")
    mdir = mpat_dir(a, sub, ses)
    if a.cst_fornix_hcc:
        retire_old_cst(mdir, sub)
    names = None
    if op.exists(names_file(mdir, sub)):
        with open(names_file(mdir, sub)) as fh:
            names = json.load(fh)
        if a.cst_fornix_hcc:
            names = add_names(mdir, sub, names, NEW_BUNDLES)
        names = [n for n in names if not a.bundles or n in a.bundles]
        if not a.force and all(bundle_done(mdir, sub, n) for n in names):
            log.info("sub-%s ses-%s: all %d bundles written; reusing them (--force to rerun)", sub, ses, len(names))
            return
    if not op.isdir(src):
        raise FileNotFoundError(f"pyAFQ session folder not found: {src}")
    missing = [n.format(sub=sub) for n in NEEDED if not op.exists(op.join(src, n.format(sub=sub)))]
    if missing and not a.allow_pyafq:
        raise FileNotFoundError(f"sub-{sub} ses-{ses}: pyAFQ derivatives missing from {src}: {missing} "
                                "(--allow-pyafq lets pyAFQ compute them there)")
    afq = participant(a, sub, ses, src)
    if names is None:
        names = list(afq.export("bundle_dict").bundle_names)
        os.makedirs(mdir, exist_ok=True)
        if a.cst_fornix_hcc:
            open(op.join(mdir, f"sub-{sub}_desc-cstredone.json"), "w").write("{}")   # nothing old to redo
        with open(names_file(mdir, sub), "w") as fh:
            json.dump(names, fh, indent=1)
        names = [n for n in names if not a.bundles or n in a.bundles]
    todo = [n for n in names if a.force or not bundle_done(mdir, sub, n)]
    if not todo:
        log.info("sub-%s ses-%s: all %d bundles written; reusing them (--force to rerun)", sub, ses, len(names))
        return
    log.info("sub-%s ses-%s: %d of %d bundles to compute (the rest are written)", sub, ses, len(todo), len(names))
    run(afq, a.threads, a.max_mem, bundles=todo, moments=a.moments, tol=a.tol,
        cores=a.cores, viz=a.viz, out_dir=mdir, verbose=a.verbose)


# ----------------------------------------------------------------------- pyAFQ's own run (--pyafq)
def pyafq_dir(a, sub, ses):
    return op.join(a.out_root, f"sub-{sub}", f"ses-{ses}", "pyafq")


def density_dir(a, sub, ses):
    return op.join(pyafq_dir(a, sub, ses), "densities")


def link_derivatives(src, dst):
    """dst mirrors src's pyAFQ derivatives as hard links (copies across file systems),
    leaving out tractography, bundles and profiles (NOT_LINKED).  Existing files are kept."""
    n_link = n_copy = 0
    for root, dirs, files in os.walk(src):
        rel = op.relpath(root, src)
        rel = "" if rel == "." else rel
        dirs[:] = [d for d in dirs if not NOT_LINKED.search(op.join(rel, d) + "/")]
        for f in files:
            r = op.join(rel, f)
            if NOT_LINKED.search(r):
                continue
            out = op.join(dst, r)
            if op.exists(out):
                continue
            os.makedirs(op.dirname(out), exist_ok=True)
            try:
                os.link(op.join(src, r), out)
                n_link += 1
            except OSError:
                shutil.copy2(op.join(src, r), out)
                n_copy += 1
    if n_link or n_copy:
        log.info("%s: %d derivatives hard-linked, %d copied from %s", dst, n_link, n_copy, src)


def run_pyafq_session(a, sub, ses):
    """pyAFQ's tractography and recognition in its own folder, then each bundle's streamline
    density on MPAT's grid (pyafq/densities/), in the MPAT file layout so compare() reads both."""
    from dipy.tracking.utils import density_map
    import AFQ.utils.streamlines as aus

    src = op.join(a.pyafq_root, f"sub-{sub}", f"ses-{ses}")
    pdir, ddir = pyafq_dir(a, sub, ses), density_dir(a, sub, ses)
    if op.exists(names_file(ddir, sub)):
        with open(names_file(ddir, sub)) as fh:
            names = json.load(fh)
        if not a.force and all(bundle_done(ddir, sub, n) for n in names):
            log.info("sub-%s ses-%s: pyAFQ densities written; reusing them", sub, ses)
            return
    if not op.isdir(src):
        raise FileNotFoundError(f"pyAFQ session folder not found: {src}")
    missing = [n.format(sub=sub) for n in NEEDED if not op.exists(op.join(src, n.format(sub=sub)))]
    if missing and not a.allow_pyafq:
        raise FileNotFoundError(f"sub-{sub} ses-{ses}: pyAFQ derivatives missing from {src}: {missing} "
                                "(--allow-pyafq lets pyAFQ compute them)")
    link_derivatives(src, pdir)
    afq = participant(a, sub, ses, pdir, mpat_bundles=False)   # pyAFQ's own defaults
    names = list(afq.export("bundle_dict").bundle_names)
    log.info("sub-%s ses-%s: pyAFQ tractography and recognition in %s", sub, ses, pdir)
    seg_file = afq.export("bundles")
    ref = nib.load(op.join(pdir, NEEDED[0].format(sub=sub)))       # the msmt CSD fODF: MPAT's grid
    seg = aus.SegmentedSFT.fromfile(seg_file, "same")
    seg.to_rasmm()
    os.makedirs(ddir, exist_ok=True)
    for name in names:
        stem = op.join(ddir, f"sub-{sub}_bundle-{desc_of(name)}")
        try:
            sls = seg.get_bundle(name).streamlines
        except KeyError:              # no streamlines recognized
            sls = []
        n = len(sls)
        dens = (density_map(sls, ref.affine, ref.shape[:3]) if n else np.zeros(ref.shape[:3])).astype(np.float32)
        nib.save(nib.Nifti1Image(dens, ref.affine), f"{stem}_desc-density_map.nii.gz")
        with open(f"{stem}_desc-summary.json", "w") as fh:
            json.dump(dict(bundle=name, technique=PYAFQ_TECHNIQUE, source=seg_file,
                           result=dict(expected_bundle_streamlines=float(n), n_streamlines=n),
                           density="streamline count per voxel (dipy density_map)"), fh, indent=1)
        log.info("sub-%s ses-%s pyAFQ %-36s %d streamlines", sub, ses, name, n)
    with open(names_file(ddir, sub), "w") as fh:      # last: marks the session done
        json.dump(names, fh, indent=1)


def _session_child(a, sub, ses, logfile, what="mpat"):
    """run_session (or run_pyafq_session) in its own process: everything it held goes back
    to the system after."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    for name in ("hcp_mpat", "mpat", "AFQ"):
        logging.getLogger(name).addHandler(fh)
    try:
        (run_pyafq_session if what == "pyafq" else run_session)(a, sub, ses)
    except Exception:
        log.exception("sub-%s ses-%s %s failed", sub, ses, what)
        sys.exit(1)


# ----------------------------------------------------------------------- comparison
def load_density(mdir, sub, desc):
    """(density or None if missing / empty, affine, expected streamlines)."""
    stem = op.join(mdir, f"sub-{sub}_bundle-{desc}")
    f = f"{stem}_desc-density_map.nii.gz"
    if not op.exists(f):
        return None, None, 0.0
    img = nib.load(f)
    d = np.clip(np.asarray(img.dataobj, np.float64), 0, None)
    n = 0.0
    if op.exists(f"{stem}_desc-summary.json"):
        with open(f"{stem}_desc-summary.json") as fh:
            n = float(json.load(fh)["result"]["expected_bundle_streamlines"])
    return (d if d.sum() > 0 else None), img.affine, n


def fuzzy_dice(p, q):
    """sum_v min(p_v, q_v) of the two maps scaled to sum 1 (Crum et al. 2006)."""
    return float(np.minimum(p / p.sum(), q / q.sum()).sum())


def bundle_descs(mdirs, sub, bundles):
    """The bundles asked for, or else every bundle of either session's bundle dictionary and
    every density map written."""
    if bundles:
        return sorted({desc_of(b) for b in bundles})
    descs = set()
    for m in mdirs:
        if op.exists(names_file(m, sub)):
            with open(names_file(m, sub)) as fh:
                descs |= {desc_of(n) for n in json.load(fh)}
        for f in glob.glob(op.join(m, f"sub-{sub}_bundle-*_desc-density_map.nii.gz")):
            descs.add(op.basename(f).split("_bundle-")[1].split("_desc-")[0])
    return sorted(descs)


def mpat_dir(a, sub, ses):
    return op.join(a.out_root, f"sub-{sub}", f"ses-{ses}", "mpat-moments" if a.moments else "mpat")


def compare(a, sub, technique=TECHNIQUE):
    s1, s2 = a.sessions[:2]
    where = density_dir if technique == PYAFQ_TECHNIQUE else mpat_dir
    mdirs = [where(a, sub, s) for s in (s1, s2)]
    rows = []
    for desc in bundle_descs(mdirs, sub, a.bundles):
        (P, aP, n1), (Q, aQ, n2) = [load_density(m, sub, desc) for m in mdirs]
        hemi, name = split_desc(desc)
        if P is None or Q is None:
            log.warning("sub-%s %s: %s missing or empty in %s; fuzzy Dice 0", sub, technique, desc,
                        ", ".join(s for s, x in ((s1, P), (s2, Q)) if x is None))
            fd = 0.0
        elif P.shape != Q.shape or not np.allclose(aP, aQ, atol=1e-3):
            log.warning("sub-%s %s: %s sessions are on different grids; NaN", sub, technique, desc)
            fd = np.nan
        else:
            fd = fuzzy_dice(P, Q)
        rows.append(dict(cohort=COHORT, subject=sub, hemisphere=hemi, bundle=name, technique=technique,
                         session_1=s1, session_2=s2, expected_streamlines_session_1=n1,
                         expected_streamlines_session_2=n2, fuzzy_dice=fd,
                         moments=bool(a.moments)))
        log.info("sub-%s %-6s %-36s fuzzy Dice %.3f", sub, technique, desc, fd)
    return pd.DataFrame(rows)


def combine(a):
    files = sorted(glob.glob(op.join(a.out_root, "sub-*", "sub-*_desc-fuzzydice.csv")))
    if not files:
        log.error("no sub-*/sub-*_desc-fuzzydice.csv under %s", a.out_root)
        return 1
    df = pd.concat([pd.read_csv(f, dtype={"subject": str}) for f in files], ignore_index=True)
    out = op.join(a.out_root, "all_desc-fuzzydice.csv")
    df.to_csv(out, index=False)
    log.info("wrote %s (%d subjects, %d rows)", out, df["subject"].nunique(), len(df))
    return 0


# ----------------------------------------------------------------------- main
def main(a):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    status = 0
    for subject in a.subjects:
        sub = subject.replace("sub-", "")
        sub_out = op.join(a.out_root, f"sub-{sub}")
        os.makedirs(sub_out, exist_ok=True)
        fh = logging.FileHandler(op.join(sub_out, f"sub-{sub}_log.txt"))
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        for name in ("hcp_mpat", "mpat", "AFQ"):
            logging.getLogger(name).addHandler(fh)
        try:
            ctx = multiprocessing.get_context("spawn")
            frames = []
            for what, tech in [("mpat", TECHNIQUE)] + ([("pyafq", PYAFQ_TECHNIQUE)] if a.pyafq else []):
                done = []
                for ses in a.sessions:       # each session in a fresh process
                    proc = ctx.Process(target=_session_child, args=(a, sub, ses, fh.baseFilename, what))
                    proc.start()
                    proc.join()
                    if proc.exitcode == 0:
                        done.append(ses)
                    else:
                        log.error("sub-%s ses-%s %s: exit code %s%s", sub, ses, what, proc.exitcode,
                                  " (killed: out of memory?)" if proc.exitcode == -9 else "")
                if len(done) < 2:
                    log.error("sub-%s %s: need two finished sessions (have %s)", sub, what, done)
                    status = 1
                    continue
                frames.append(compare(a, sub, tech))
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            csv = op.join(sub_out, f"sub-{sub}_desc-fuzzydice.csv")
            df.to_csv(csv, index=False)
            log.info("wrote %s (%d bundles)", csv, len(df))
        finally:
            for name in ("hcp_mpat", "mpat", "AFQ"):
                logging.getLogger(name).removeHandler(fh)
            fh.close()
    if a.combine:
        status = combine(a) or status
    return status


if __name__ == "__main__":
    sys.exit(main(A))

