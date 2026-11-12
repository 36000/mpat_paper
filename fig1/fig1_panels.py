#!/usr/bin/env python
"""Figure 1: the three images stacked in one column, labels a and b in a gap to the left.

    python fig1_panels.py fig1b.png fig1c.png fuzzydice.png -o fig1.png

Rows: (a) the CST density projections (fig1b_mips.py), (b) the cortical endpoints
(fig1c_sphere.py), then the reliability figure (plot_fuzzydice.py), which carries its own
labels c, d, e.  Each image is trimmed as add_img trims it, each row is as tall as its image
at the column width (so the rows touch), and each label sits in the gap left of its image,
level with the image's top.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from AFQ.viz.utils import PanelFigure, trim  # noqa: E402

REDUCT = 2      # as add_img's reduct_count


def aspect(fname):
    """Height / width of the image as add_img shows it (trimmed the same way)."""
    im = Image.open(fname)
    for _ in range(REDUCT):
        im = trim(im, buffer=2)
    return im.size[1] / im.size[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("images", nargs="+", help="the row images, top to bottom")
    p.add_argument("-o", "--out", default="fig1.png")
    p.add_argument("--labels", default="ab", help="labels of the first rows (the last row labels itself)")
    p.add_argument("--width", type=float, default=12, help="figure width, inches")
    p.add_argument("--gap", type=float, default=0.06, help="label gap, as a fraction of the image width")
    p.add_argument("--pad", type=float, default=0.01, help="label to image distance, fraction of the figure width")
    a = p.parse_args()

    col_w = a.width / (1 + a.gap)
    heights = [col_w * aspect(f) for f in a.images]
    pf = PanelFigure(len(a.images), 1, a.width, sum(heights),
                     panel_label_kwargs=dict(color="black", fontfamily="sans-serif", fontsize=26,
                                             fontweight="bold"))
    pf.grid = plt.GridSpec(len(a.images), 2, width_ratios=[a.gap, 1], height_ratios=heights,
                           hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    axes = [pf.add_img(f, 1, row, reduct_count=REDUCT, add_panel_label=False) for row, f in enumerate(a.images)]

    pf.fig.canvas.draw()
    for lab, ax in zip(a.labels, axes):
        box = ax.get_position()
        pf.fig.text(box.x0 - a.pad, box.y1, lab, ha="right", va="top", **pf.panel_label_kwargs)
    pf.format_and_save_figure(a.out, tight_final=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
