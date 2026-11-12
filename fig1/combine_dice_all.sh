#!/usr/bin/env bash
# Combine the per-subject test-retest CSVs of run_hcp_mat_full.py into all_desc-fuzzydice.csv
# (as its --combine does) and add, for every subject, bundle and technique:
#
#   fuzzy_dice  (already in the CSV)  sum_v min(p_v, q_v), maps scaled to sum 1 (Crum et al. 2006)
#   dice_vol    binary Dice on masks matched in volume (and vol_k): pyAFQ's mask in each session
#               is every voxel any streamline touched, and K is the mean of its two mask sizes;
#               MCAT/MPAT's mask in each session is its K densest voxels
#   wdsc_vol    weighted Dice of Kruper et al. 2021 / Cousineau et al. 2017 (pyAFQ's
#               AFQ.utils.volume.dice_coeff, weighted=True) on the same masks: each map is cut
#               to its own mask, then
#                   (sum_v a_v [v in mask_b] + sum_v b_v [v in mask_a]) / (sum a + sum b)
#               For pyAFQ the mask is every nonzero voxel, so this is pyAFQ's wDSC exactly.
#
#   bash combine_dice_all.sh                     # out-root from paths.py: <writeable>/hcp_default_mcat
#   bash combine_dice_all.sh /path/to/out-root
#
# The new measures need a pair: the bundle in both techniques (not the fornix, the hippocampal
# cingulum, or the single VOF, which pyAFQ splits).  Where pyAFQ found nothing in a session it
# scores 0, and MCAT/MPAT is matched to that bundle's median K over the subjects where pyAFQ
# found it in both sessions.  MCAT/MPAT should never miss a bundle: if it does, a warning and
# NaN.  Each map is read once.  Reads the density maps already written; reruns nothing.
# Works on the old layout (mcat/, "MCAT") and the renamed one (mpat/, "MPAT").
set -euo pipefail

OUT_ROOT="${1:-$(python -c 'from paths import writeable_path; print(writeable_path + "hcp_default_mcat")')}"
echo "out-root: $OUT_ROOT  (masks matched in volume: pyAFQ > 0 streamlines, MCAT/MPAT its K densest voxels)"

# -I: isolated, so nothing in the current folder, PYTHONPATH or ~/.local shadows the env's packages
python -I - "$OUT_ROOT" <<'EOF'
import glob, os.path as op, re, sys
import nibabel as nib
import numpy as np
import pandas as pd

root = sys.argv[1]
PYAFQ = "pyAFQ"
# MPAT bundles pyAFQ doesn't define (it splits the VOF into early visual, posterior and anterior VOF)
NOT_IN_PYAFQ = {"vertical occipital", "vof"}
COLS = ["dice_vol", "wdsc_vol"]


def desc_of(name):
    return re.sub(r"[^A-Za-z0-9]", "", name)


def load(mdir, sub, desc):
    """The density map, or None if missing or empty."""
    f = op.join(mdir, f"sub-{sub}_bundle-{desc}_desc-density_map.nii.gz")
    if not op.exists(f):
        return None
    d = np.clip(np.asarray(nib.load(f).dataobj, np.float64), 0, None)
    return d if d.sum() > 0 else None


def binary_dice(a, b):
    return float(2 * np.logical_and(a, b).sum() / max(a.sum() + b.sum(), 1))


def wdsc(a, b, sa, sb):
    """Weighted Dice (Cousineau et al. 2017; pyAFQ dice_coeff weighted=True) of the two maps each
    cut to its own support (sa, sb): the mass of each inside the other's support, over both maps'
    mass.  A support of every nonzero voxel (pyAFQ's) leaves the map as it is: pyAFQ's wDSC."""
    a, b = a * sa, b * sb
    return float(((a * sb).sum() + (b * sa).sum()) / (a.sum() + b.sum()))


def top_k(M, k):
    """The k highest-density voxels of M (no more than its nonzero ones)."""
    flat = M.ravel()
    k = int(min(k, np.count_nonzero(flat)))
    mask = np.zeros(flat.size, bool)
    if k > 0:
        mask[np.argpartition(flat, -k)[-k:]] = True
    return mask.reshape(M.shape)


files = sorted(glob.glob(op.join(root, "sub-*", "sub-*_desc-fuzzydice.csv")))
if not files:
    sys.exit(f"no sub-*/sub-*_desc-fuzzydice.csv under {root}")
df = pd.concat([pd.read_csv(f, dtype={"subject": str, "session_1": str, "session_2": str})
                for f in files], ignore_index=True)
techs = set(df["technique"])
markov = next((t for t in ("MPAT", "MCAT") if t in techs), None)
df["hemisphere"] = df["hemisphere"].fillna("")
for col in COLS + ["vol_k"]:
    df[col] = np.nan


def tech_dir(sub, ses, tech):
    base = op.join(root, f"sub-{sub}", f"ses-{ses}")
    if tech == PYAFQ:
        return op.join(base, "pyafq", "densities")
    return next((op.join(base, m) for m in ("mpat", "mcat") if op.isdir(op.join(base, m))), op.join(base, "mcat"))


def score_markov(sub, desc, s1, s2, K):
    """MCAT/MPAT's K densest voxels in each session (the same K); NaN, with a warning, if a map
    is missing (it should always find the bundle)."""
    M1, M2 = (load(tech_dir(sub, s, markov), sub, desc) for s in (s1, s2))
    if M1 is None or M2 is None or not K > 0:
        print(f"WARNING sub-{sub} {desc}: no {markov} map (or no K) in a session; left NaN", flush=True)
        return dict.fromkeys(COLS, np.nan)
    v1, v2 = top_k(M1, K), top_k(M2, K)
    return dict(dice_vol=binary_dice(v1, v2), wdsc_vol=wdsc(M1, M2, v1, v2))


def put(rows, tech, vals, K):
    i = rows.index[rows["technique"] == tech]
    vals = {**vals, "vol_k": K}
    df.loc[i, list(vals)] = list(vals.values())


groups = [(k, rows) for k, rows in df.groupby(["subject", "hemisphere", "bundle"])
          if markov is not None and {markov, PYAFQ} <= set(rows["technique"])        # pairs only
          and k[2].strip().lower() not in NOT_IN_PYAFQ]                              # (not the VOF)

# where pyAFQ found nothing in a session, it scores 0 and MPAT waits: it is matched to the
# bundle's median K over the pairs where pyAFQ found it
pooled, deferred = {}, []
for (sub, hemi, name), rows in groups:
    s1, s2 = rows.iloc[0]["session_1"], rows.iloc[0]["session_2"]
    desc = desc_of(f"{hemi} {name}".strip())
    P1, P2 = (load(tech_dir(sub, s, PYAFQ), sub, desc) for s in (s1, s2))
    if P1 is None or P2 is None:
        put(rows, PYAFQ, dict(dice_vol=0.0, wdsc_vol=0.0), np.nan)
        deferred.append(((sub, hemi, name), rows, desc, s1, s2))
        continue
    a1, a2 = P1 > 0, P2 > 0
    r_p = dict(dice_vol=binary_dice(a1, a2), wdsc_vol=wdsc(P1, P2, a1, a2))
    K = round((a1.sum() + a2.sum()) / 2)
    pooled.setdefault((hemi, name), []).append(K)
    r_m = score_markov(sub, desc, s1, s2, K)
    put(rows, PYAFQ, r_p, K)
    put(rows, markov, r_m, K)
    print(f"sub-{sub} {desc:<36} K={K:7d}  Dice pyAFQ {r_p['dice_vol']:.3f} {markov} {r_m['dice_vol']:.3f}"
          f"  weighted Dice pyAFQ {r_p['wdsc_vol']:.3f} {markov} {r_m['wdsc_vol']:.3f}", flush=True)

for (sub, hemi, name), rows, desc, s1, s2 in deferred:
    K = float(np.round(np.median(pooled.get((hemi, name), [np.nan]))))
    r_m = score_markov(sub, desc, s1, s2, K)
    put(rows, markov, r_m, K)
    print(f"sub-{sub} {desc:<36} pyAFQ found nothing in a session: pyAFQ 0; {markov} at the bundle's "
          f"median K={K:.0f}: Dice {r_m['dice_vol']:.3f}, weighted Dice {r_m['wdsc_vol']:.3f}", flush=True)

out = op.join(root, "all_desc-fuzzydice.csv")
df.to_csv(out, index=False)
print(f"wrote {out} ({df['subject'].nunique()} subjects, {len(df)} rows)")
if deferred:
    print(f"{len(deferred)} subject-bundles where pyAFQ found nothing in a session "
          f"(pyAFQ 0; {markov} at the bundle's median K):")
    print(pd.Series([d[2] for d in deferred]).value_counts().to_string())
k = ["subject", "hemisphere", "bundle"]
paired = df.groupby(k)["technique"].transform(lambda t: {markov, PYAFQ} <= set(t)).astype(bool)
paired &= ~df["bundle"].str.strip().str.lower().isin(NOT_IN_PYAFQ)
print("means over the pairs (as plot_fuzzydice.py; the fuzzy Dice of unpaired bundles excluded):")
print(df[paired].groupby("technique")[["fuzzy_dice"] + COLS].mean().round(3).to_string())
EOF
