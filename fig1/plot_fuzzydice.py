#!/usr/bin/env python
"""Bar plot of test-retest fuzzy Dice per tract, MPAT vs pyAFQ side by side (pyAFQ's colours),
with each subject as a jittered point.

    python plot_fuzzydice.py all_desc-fuzzydice.csv -o fuzzydice

Writes <o>.png and <o>.pdf.  Bars are the mean over subjects; tracts are in pyAFQ's order
(left / right partners side by side), with MPAT's hippocampal cingulum and fornix after the
cingulum cingulate (MPAT only: pyAFQ has no such bundles).  Within each tract the MPAT bar is
solid and the pyAFQ bar is hatched.

The VOF is left out for both techniques: pyAFQ splits it into Early Visual, Posterior and
Anterior VOF, so it isn't a like-for-like comparison with MPAT's single VOF.

Prints, for each technique, the mean, median, min and max across tracts of each tract's mean
fuzzy Dice, then MPAT vs pyAFQ on the tracts both have.

Panels are labelled c, d, e (--panel-labels).  The CSV's technique names are read through RENAME:
"MCAT" (the old name) as MPAT and "pyAFQ" as AFQ, the names printed and plotted.

Two more panels on pyAFQ's sampling (--no-sampling leaves them out): (d) one tract's pyAFQ
fuzzy Dice per subject against its streamline count N (--sampling-bundle, default the left
IFOF), fitted with alpha + beta / sqrt(N), and MPAT's mean dotted; (e) the MPAT - pyAFQ gap of
every subject and tract against N, with the pooled within-tract fit a_t + b / sqrt(N).

Tests (printed, and noted on the panels): (1) MPAT more reliable than pyAFQ: each subject's
fuzzy Dice averaged over the tracts both have, paired over subjects, one-sided Wilcoxon
signed-rank (paired t and Cohen's d_z as checks); (2) the gap left at N -> infinity (panel c's
limit) above 0: a bootstrap over subjects of the pooled fit, 95% interval and one-sided p.
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from AFQ.viz.utils import gen_color_dict  # noqa: E402

# the CSV's technique names -> the names printed and plotted (MPAT was MCAT; pyAFQ is AFQ)
RENAME = {"MCAT": "MPAT", "pyAFQ": "AFQ"}
TECHS = ["MPAT", "AFQ"]
HATCH = {"MPAT": None, "AFQ": "////"}
FIX = {"Frontooccipital": "Fronto-occipital", "Temporoparietal": "Temporo-parietal"}   # CSV -> pyAFQ names
# MPAT's bundles beyond pyAFQ's defaults, placed after this tract in pyAFQ's order
EXTRA_AFTER = {"Right Cingulum Cingulate": ["Left Cingulum Hippocampus", "Right Cingulum Hippocampus",
                                            "Left Fornix", "Right Fornix"]}
# VOF and pyAFQ's VOF sub-bundles (full names and abbreviations): left out of the plot
DROP = {"vertical occipital", "vof", "early visual", "ev", "posterior vertical occipital", "pvof",
        "anterior vertical occipital", "avof"}

FONTSIZE = 28
plt.rcParams.update({"font.size": FONTSIZE, "axes.labelsize": FONTSIZE, "axes.titlesize": FONTSIZE,
                     "xtick.labelsize": FONTSIZE - 2, "ytick.labelsize": FONTSIZE - 2,
                     "legend.fontsize": FONTSIZE - 2})


def pyafq_name(hemi, bundle):
    for a, b in FIX.items():
        bundle = bundle.replace(a, b)
    return f"{hemi} {bundle}" if isinstance(hemi, str) and hemi else bundle


def tract_order(names):
    """pyAFQ's order, MPAT's extra bundles (EXTRA_AFTER) inserted, then anything else sorted."""
    from AFQ.viz.utils import _COLOR_DICT as ORDER     # pyAFQ's own order

    order = []
    for n in ORDER:
        order += [n] + EXTRA_AFTER.get(n, [])
    known = [n for n in order if n in names]
    return known + sorted(set(names) - set(known))


def short(name):
    """Compact tick labels: L / R prefix, callosum as CC."""
    name = re.sub(r"^Left ", "L ", re.sub(r"^Right ", "R ", name))
    return name.replace("Callosum ", "CC ")


def summarise(label, s):
    print(f"{label}, each tract's mean over subjects, across {len(s)} tracts:")
    print(f"  mean   {s.mean():.3f}")
    print(f"  median {s.median():.3f}")
    print(f"  min    {s.min():.3f} ({s.idxmin()})")
    print(f"  max    {s.max():.3f} ({s.idxmax()})")


def sampling_fits(full):
    """pyAFQ vs MPAT, paired per subject and tract: N (pyAFQ streamlines, the two sessions'
    harmonic mean), both fuzzy Dice and their gap; per-tract fits pyAFQ FD = alpha + beta / sqrt(N),
    and the pooled within-tract fit gap = a_t + b / sqrt(N).  None without both techniques."""
    if "technique" not in full or not {"MPAT", "AFQ"} <= set(full["technique"]):
        return None
    f = full.copy()
    f["tract"] = [pyafq_name(h, b) for h, b in zip(f["hemisphere"], f["bundle"])]
    p = f[f.technique == "AFQ"].set_index(["subject", "tract"])
    m = f[f.technique == "MPAT"].set_index(["subject", "tract"])
    n1, n2 = p["expected_streamlines_session_1"], p["expected_streamlines_session_2"]
    x = pd.DataFrame({"fd_p": p["fuzzy_dice"], "fd_m": m["fuzzy_dice"],
                      "N": 2 / (1 / n1.where(n1 > 0) + 1 / n2.where(n2 > 0))}).dropna().reset_index()
    x["gap"], x["u"] = x.fd_m - x.fd_p, 1 / np.sqrt(x.N)
    per = {}
    for t, g in x.groupby("tract"):
        if len(g) >= 5:
            beta, alpha = np.polyfit(g.u, g.fd_p, 1)
            per[t] = dict(alpha=alpha, beta=beta, mcat=g.fd_m.mean(), r=np.corrcoef(g.u, g.gap)[0, 1],
                          decades=float(np.log10(g.N.max() / g.N.min())))
    uc = x.u - x.groupby("tract").u.transform("mean")
    gc = x.gap - x.groupby("tract").gap.transform("mean")
    b = float((uc * gc).sum() / (uc ** 2).sum())
    a_t = x.groupby("tract").gap.mean() - b * x.groupby("tract").u.mean()
    share = b * x.u.mean() / x.gap.mean()
    print(f"sampling: pooled gap = a_t + {b:.2f}/sqrt(N); a_t mean {a_t.mean():.3f}; "
          f"{100 * share:.0f}% of the mean gap ({x.gap.mean():.3f}) is the 1/sqrt(N) term at the observed N; "
          f"slope > 0 in {sum(v['r'] > 0 for v in per.values())} of {len(per)} tracts")
    return dict(x=x, per=per, b=b, a_t=a_t)


def sampling_panels(fig, gs, s, example, colors):
    """(b) the example tract: pyAFQ fuzzy Dice per subject vs N, its 1/sqrt(N) fit and asymptote,
    MPAT's mean dotted; (c) every pair's gap vs N, coloured by tract, with the pooled fit."""
    x, per, b = s["x"], s["per"], s["b"]
    if example not in per:      # the clearest trend over the widest range of N
        example = max(per, key=lambda t: per[t]["r"] * min(per[t]["decades"], 2.0))
    g, f = x[x.tract == example], per[example]
    nn = np.logspace(np.log10(g.N.min() / 1.5), np.log10(x.N.max() * 30), 200)    # from its data rightwards

    row = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.42], wspace=0.12)     # d, e, legend
    ax = fig.add_subplot(row[0])
    ax.scatter(g.N, g.fd_p, s=60, color=colors.get(example, "0.3"), edgecolors="black", linewidths=0.3,
               zorder=3)
    ax.plot(nn, f["alpha"] + f["beta"] / np.sqrt(nn), color="black", lw=1.4, label=r"$\alpha + \beta/\sqrt{N}$")
    ax.axhline(f["alpha"], color="0.5", lw=0.9, ls="--", label=r"limit, $N\to\infty$")
    ax.axhline(f["mcat"], color="black", lw=1.4, ls=":", label="MPAT (mean)")
    ax.set_xscale("log")
    ax.set_ylim(max(0.0, min(g.fd_p.min(), f["alpha"] + f["beta"] / np.sqrt(nn[0])) - 0.05), 1.0)
    ax.set_xlabel("AFQ streamlines (N)")
    ax.set_ylabel("fuzzy Dice")
    ax.set_title(example)
    ax.spines[["top", "right"]].set_visible(False)
    axd = ax

    ax = fig.add_subplot(row[1])          # x = 1 / sqrt(N): the pooled fit is a line, its limit at x = 0
    ax.scatter(x.u, x.gap, s=22, c=[colors.get(t, "0.5") for t in x.tract], alpha=0.6, linewidths=0, zorder=2)
    uu = np.linspace(0, x.u.max() * 1.02, 50)
    ax.plot(uu, s["a_t"].mean() + b * uu, color="black", lw=1.4,
            label=rf"$\bar a + {b:.1f}/\sqrt{{N}}$")
    dot, = ax.plot([0], [s["a_t"].mean()], "o", color="black", ms=11, zorder=4, clip_on=False)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(0, x.u.max() * 1.05)
    ax.set_xlabel(r"$1/\sqrt{N}$")
    ax.set_ylabel("MPAT $-$ AFQ")
    top = ax.secondary_xaxis("top")
    ticks = [n for n in (500, 1e3, 3e3, 1e4) if 1 / np.sqrt(n) <= x.u.max() * 1.05]
    top.set_xticks([0] + [1 / np.sqrt(n) for n in ticks])
    top.set_xticklabels([r"$\infty$"] + [f"{n:,.0f}" for n in ticks], fontsize=FONTSIZE - 2)
    top.set_xlabel("N")
    ax.spines[["right"]].set_visible(False)

    # one legend, in its own column right of e (so the row stays flush with c): d's entries, the
    # limit entry also showing e's dot
    from matplotlib.legend_handler import HandlerTuple

    handles, labels = axd.get_legend_handles_labels()
    i = labels.index(r"limit, $N\to\infty$")
    handles[i] = (handles[i], dot)
    lax = fig.add_subplot(row[2])
    lax.set_axis_off()
    lax.legend(handles, labels, frameon=False, loc="center left", borderaxespad=0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})

def test_overall(d):
    """Test 1: MPAT more reliable than pyAFQ overall.  The subject is the unit: each subject's
    fuzzy Dice averaged over the tracts both techniques have, MPAT vs pyAFQ paired over subjects
    (one-sided Wilcoxon signed-rank; paired t as a check)."""
    from scipy import stats

    both = d.groupby(["subject", "tract", "technique"])["fuzzy_dice"].mean().unstack().dropna()
    subj = both.groupby("subject").mean()
    diff = subj["MPAT"] - subj["AFQ"]
    w = stats.wilcoxon(subj["MPAT"], subj["AFQ"], alternative="greater")
    t = stats.ttest_rel(subj["MPAT"], subj["AFQ"], alternative="greater")
    dz = diff.mean() / diff.std(ddof=1)
    print(f"test 1 (MPAT > AFQ, {len(subj)} subjects, each averaged over {both.index.get_level_values('tract').nunique()} "
          f"tracts): mean difference {diff.mean():+.3f} (MPAT higher in {(diff > 0).sum()}/{len(diff)} subjects); "
          f"Wilcoxon W = {w.statistic:.0f}, p = {w.pvalue:.2g}; paired t({len(diff) - 1}) = {t.statistic:.1f}, "
          f"p = {t.pvalue:.2g}; Cohen's d_z = {dz:.2f}")
    return dict(diff=float(diff.mean()), p=float(w.pvalue), n=len(subj), higher=int((diff > 0).sum()))


def test_residual(x, n_boot=5000, seed=0):
    """Test 2: is the gap left at N -> infinity (the mean of the tracts' intercepts a_t in
    gap = a_t + b / sqrt(N), b shared) above 0?  Subjects resampled with replacement (all of a
    subject's tracts together, since they are not independent), the model refitted each time:
    a 95% percentile interval and a one-sided p (the share of resamples at or below 0)."""
    def fit(xx):
        uc = xx.u - xx.groupby("tract").u.transform("mean")
        gc = xx.gap - xx.groupby("tract").gap.transform("mean")
        b = (uc * gc).sum() / (uc ** 2).sum()
        return float((xx.groupby("tract").gap.mean() - b * xx.groupby("tract").u.mean()).mean()), float(b)

    a0, b0 = fit(x)
    subjects = x["subject"].unique()
    groups = {s_: g for s_, g in x.groupby("subject")}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(subjects, len(subjects), replace=True)
        boots[k] = fit(pd.concat([groups[s_] for s_ in pick], ignore_index=True))[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = (np.sum(boots <= 0) + 1) / (n_boot + 1)
    print(f"test 2 (gap left at N -> inf > 0): {a0:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] "
          f"(bootstrap over {len(subjects)} subjects, {n_boot} resamples), one-sided p = {p:.2g}")
    return dict(a=a0, lo=float(lo), hi=float(hi), p=float(p), b=b0)


def _p(p):
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("csv")
    p.add_argument("-o", "--out", default="fuzzydice")
    p.add_argument("--ymin", type=float, default=None, help="y-axis floor (default: just below the lowest point)")
    p.add_argument("--sampling-bundle", default="Left Anterior Thalamic",
                   help="the example tract of panel b (default: the left ATR)")
    p.add_argument("--n-boot", type=int, default=5000, help="bootstrap resamples (over subjects) for test 2")
    p.add_argument("--panel-labels", default="cde", help="labels of the three panels (default c, d, e)")
    p.add_argument("--no-sampling", action="store_true", help="leave out the sampling panels (b, c)")
    a = p.parse_args()

    d = pd.read_csv(a.csv, dtype={"subject": str})
    d["technique"] = d["technique"].replace(RENAME)
    d = d[d["technique"].isin(TECHS)]
    d = d[~d["bundle"].str.strip().str.lower().isin(DROP)]
    d["tract"] = [pyafq_name(h, b) for h, b in zip(d["hemisphere"], d["bundle"])]
    tracts = tract_order(set(d["tract"]))
    colors = gen_color_dict(tracts)

    means = d.groupby(["tract", "technique"])["fuzzy_dice"].mean().unstack().reindex(index=tracts, columns=TECHS)
    for t in TECHS:
        summarise(f"{t} fuzzy Dice", means[t].dropna())
        print()
    both = means.dropna()
    diff = both["MPAT"] - both["AFQ"]
    print(f"on the {len(both)} tracts both have: MPAT mean {both['MPAT'].mean():.3f}, "
          f"AFQ mean {both['AFQ'].mean():.3f}; MPAT higher on {(diff > 0).sum()} / {len(both)} "
          f"(mean difference {diff.mean():+.3f})")

    t1 = test_overall(d)
    samp = None if a.no_sampling else sampling_fits(d)
    t2 = test_residual(samp["x"], a.n_boot) if samp is not None else None
    rng = np.random.default_rng(0)
    width = max(10, 0.62 * len(tracts) + 2)
    if samp is None:
        fig, ax = plt.subplots(figsize=(width, 6.5), layout="constrained")
    else:
        fig = plt.figure(figsize=(width, 18), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.25, hspace=0.06, wspace=0.06)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05])
        ax = fig.add_subplot(gs[0, :])
    w = 0.38                                   # bar width; a tract's pair spans 2w
    for i, tract in enumerate(tracts):
        present = [t for t in TECHS if not np.isnan(means.at[tract, t])]
        offsets = (np.arange(len(present)) - (len(present) - 1) / 2) * w    # lone bars are centred
        for t, off in zip(present, offsets):
            ax.bar(i + off, means.at[tract, t], width=w, color=colors[tract], hatch=HATCH[t],
                   edgecolor="black", linewidth=0.5, zorder=2)
            v = d.loc[(d["tract"] == tract) & (d["technique"] == t), "fuzzy_dice"].to_numpy()
            ax.scatter(i + off + rng.uniform(-0.11, 0.11, len(v)), v, s=12, color="black", alpha=0.5,
                       linewidths=0, zorder=3)
    lo = a.ymin if a.ymin is not None else max(0.0, np.floor((d["fuzzy_dice"].min() - 0.03) * 20) / 20)
    ax.set_ylim(lo, 1.0)
    ax.set_xlim(-0.6, len(tracts) - 0.4)
    ax.set_xticks(np.arange(len(tracts)))
    ax.set_xticklabels([short(t) for t in tracts], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylabel("test-retest fuzzy Dice")
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor="0.8", edgecolor="black", lw=0.5, hatch=HATCH[t],
                             label=t)
                       for t in TECHS],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    ax.text(0.0, 1.02, f"MPAT > AFQ in {t1['higher']}/{t1['n']} subjects (mean {t1['diff']:+.3f}, "
                       f"Wilcoxon {_p(t1['p'])})", transform=ax.transAxes, ha="left", va="bottom")
    if samp is not None:
        sampling_panels(fig, gs, samp, a.sampling_bundle, colors)
        axc = fig.axes[2]
        # panel labels beside each plot (left of its y label), level with its top
        fig.canvas.draw()
        fig.set_layout_engine("none")            # freeze the layout, so the labels stay where placed
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for lab, axx in zip(a.panel_labels, (ax, fig.axes[1], axc)):
            left = inv.transform(axx.get_tightbbox(renderer).p0)[0]
            top = inv.transform(axx.transAxes.transform((0, 1)))[1]
            lift = 0.27 * 80 / 72 / fig.get_figheight()      # the font's space above the glyphs
            fig.text(left - 0.005, top + lift, lab, fontsize=60, fontweight="bold", ha="right", va="top")
        fig.savefig(f"{a.out}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
        fig.savefig(f"{a.out}.pdf", bbox_inches="tight")
        print(f"wrote {a.out}.png / .pdf")
        return
    fig.savefig(f"{a.out}.png", dpi=300)
    fig.savefig(f"{a.out}.pdf")
    print(f"wrote {a.out}.png / .pdf")


if __name__ == "__main__":
    main()