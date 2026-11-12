#!/usr/bin/env python
"""HCP 627549 seed-count run for Figure 1B, locally: one big pyAFQ run (5e7 random seeds,
probabilistic, msmt CSD; about 1.3e8 streamlines) whose recognized left CST fig1b_mips.py
subsamples to 1e6 / 1e7 / 1e8 whole-brain streamlines.

    python seed_count_local.py

* Inputs: fetch_hcp.  Tissue and brain mask: pyAFQ 3.4's defaults, from SynthSeg (as MPAT's
  runs); fODF: msmt CSD (no aODF), as MPAT reads it.  Nothing is linked from the regular local
  run: that one used FreeSurfer's aparc+aseg, and its brain mask shaped the model fits and the
  registration as well as the tissue.
* Output: <hcp>/derivatives/afq_seed_count/sub-<id>/ses-01/nseeds-1e8-synthseg/ (a new folder: the
  earlier aparc+aseg run's nseeds-1e8/ is left as it is).  LINK_FROM: a folder of pyAFQ
  derivatives made with these same settings (SynthSeg, msmt CSD) to hard-link in first, so
  only the tractography and bundles are computed; None (default): everything computed here.
* Bundles: the left CST only, as MCAT's --synthseg-cst defines it (mcat.bundles.synthseg_cst_bd):
  precentral, paracentral and postcentral cortex to the brainstem, SynthSeg's labels used
  exactly (exact_endpoints: no 4 mm growth), no waypoints, kept to the left hemisphere; no
  cleaning (as MCAT), minimum length 40.  The masks come from MCAT
  (mcat.afq.synthseg_cst_for_pyafq: SynthSeg with its cortical parcellation, run once if it is
  not there yet, the ends written on the b0 grid); MCAT_DIR: where MCAT keeps (or finds) that
  parcellation, e.g. the folder of an MCAT --synthseg-cst run on this subject.
* Recognition: if this folder already has recognition from an earlier definition, it is
  clobbered once (clobber(dependent_on="recog"); the tractography is kept) and redone; a
  marker file stops a rerun from redoing it.  Only the recognized bundles are exported
  (fig1b reads <...>_desc-bundles_tractography.trx), not profiles or viz.

Memory: whole-brain tractography with 5e7 seeds is large (trx, TRX_TMPDIR on a disk with
room); lower N_SEEDS if it does not fit, fig1b's subsampling only needs the run to have at
least 1e8 streamlines for the 1e8 panel (fewer: that panel keeps all of them).
"""
import os
import os.path as op
import shutil

os.environ.setdefault("TRX_TMPDIR", op.expanduser("~/AFQ_data/TRX_TMP/"))
os.makedirs(os.environ["TRX_TMPDIR"], exist_ok=True)

import AFQ.api.bundle_dict as abd  # noqa: E402
import AFQ.data.fetch as afd  # noqa: E402
from AFQ.api.participant import ParticipantAFQ  # noqa: E402

SUBJECT = "627549"
N_SEEDS = int(5e7)                 # about 1.3e8 streamlines (enough for the 1e8 panel)
LABEL = "1e8-synthseg"
MCAT_DIR = None                    # None: <out>/mcat; or the MCAT folder of a --synthseg-cst run (reuses its SynthSeg)
LINK_FROM = None                   # pyAFQ derivatives with these settings (SynthSeg, msmt CSD) to reuse
NOT_LINKED = ("tractography", "bundles", "profiles", "viz", "stats", ".html")   # recomputed here


def link_derivatives(src, dst):
    """dst mirrors src's pyAFQ derivatives as hard links (copies across file systems), leaving
    out tractography, bundles and profiles; files already in dst are kept."""
    if not op.isdir(src):
        return
    n = 0
    for root, dirs, files in os.walk(src):
        rel = op.relpath(root, src)
        dirs[:] = [d for d in dirs if not any(k in d for k in NOT_LINKED)]
        for f in files:
            if any(k in f for k in NOT_LINKED):
                continue
            out = op.join(dst, rel, f)
            if op.exists(out):
                continue
            os.makedirs(op.dirname(out), exist_ok=True)
            try:
                os.link(op.join(root, f), out)
            except OSError:
                shutil.copy2(op.join(root, f), out)
            n += 1
    if n:
        print(f"{n} derivatives linked from {src}")


def main():
    _, hcp_base_path = afd.fetch_hcp([SUBJECT], profile_name="hcp", study="HCP_1200",
                                     path=op.expanduser("~/AFQ_data/"))
    hcp_path = op.join(hcp_base_path, f"derivatives/dmriprep/sub-{SUBJECT}/ses-01/")
    out_path = op.join(hcp_base_path, f"derivatives/afq_seed_count/sub-{SUBJECT}/ses-01/nseeds-{LABEL}/")
    os.makedirs(out_path, exist_ok=True)
    if LINK_FROM:
        link_derivatives(LINK_FROM, out_path)

    inputs = dict(
        dwi_data_file=f"{hcp_path}/dwi/sub-{SUBJECT}_dwi.nii.gz",
        bval_file=f"{hcp_path}/dwi/sub-{SUBJECT}_dwi.bval",
        bvec_file=f"{hcp_path}/dwi/sub-{SUBJECT}_dwi.bvec",
        t1_file=f"{hcp_path}/anat/sub-{SUBJECT}_T1w.nii.gz",
        output_dir=out_path,             # brain mask and tissue: pyAFQ's SynthSeg defaults
        tracking_params=dict(n_seeds=N_SEEDS, directions="prob", odf_model="msmtcsd",
                             random_seeds=True, rng_seed=2026, trx=True))

    # the SynthSeg CST's ends, from MCAT (as MCAT's --synthseg-cst and --pyafq use them)
    from mcat.afq import synthseg_cst_for_pyafq

    bd = synthseg_cst_for_pyafq(ParticipantAFQ(**inputs), abd.BundleDict({}), out_dir=MCAT_DIR)
    left = dict(bd._dict["Left Corticospinal"], length={"min_len": 40}, mahal={"clean_rounds": 0})
    print("Left Corticospinal:", {k: (op.basename(v) if isinstance(v, str) else v) for k, v in left.items()})
    bundle_info = abd.BundleDict({"Left Corticospinal": left}, resample_subject_to=bd.resample_subject_to)
    myafq = ParticipantAFQ(**inputs, bundle_info=bundle_info)

    marker = op.join(out_path, "recog_synthseg_cst.txt")
    seg_file = op.join(out_path, f"sub-{SUBJECT}_desc-bundles_tractography.trx")
    if op.exists(marker) and op.exists(seg_file):
        print(f"recognition already done with this CST: {seg_file}")
        return
    if op.exists(seg_file):           # recognition from an earlier CST definition: redo it, keep tracking
        myafq.clobber(dependent_on="recog")
    path = myafq.export("bundles")
    with open(marker, "w") as fh:
        fh.write("recognition with the SynthSeg left CST (mcat.bundles.synthseg_cst_bd, exact ends)\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
