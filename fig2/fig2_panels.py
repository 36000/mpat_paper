import matplotlib.pyplot as plt
from PIL import Image
from AFQ.viz.utils import PanelFigure, trim

plot1 = "../ixi_out/mpat/sub-IXI002Guys0828_desc-mpat_collage.png"
plot2 = "../baby_out/mpat/sub-01_ses-01_desc-mpat_collage.png"
plot3 = "../tumor_out/mpat/sub-0393_desc-ifoftumor_figure.png"
plot4 = "../rgvp_crossed_hbn.png"

ROWS = [(plot1, plot2), (plot3, plot4)]   # top row, bottom row
LABELS = "abcd"     # panel labels, in the order the images are added
GAP = 0.07          # width of each label gap, relative to an image column
PAD = 0.01          # space between a label and its image (fraction of the figure width)
WIDTH = 8           # figure width, inches (the height follows from the images)
REDUCT = 2          # as add_img's reduct_count


def aspect(fname):
    """Height / width of the image as add_img will show it (trimmed the same way)."""
    im = Image.open(fname)
    for _ in range(REDUCT):
        im = trim(im, buffer=2)
    return im.size[1] / im.size[0]


# each row as tall as its tallest image at the column width, so the rows touch
col_w = WIDTH / (2 + 2 * GAP)
heights = [col_w * max(aspect(f) for f in row) for row in ROWS]

pf = PanelFigure(2, 2, WIDTH, sum(heights))
pf.fig.patch.set_facecolor("black")
# gap | left images | gap | right images, rows sized to the images, no margins
pf.grid = plt.GridSpec(2, 4, width_ratios=[GAP, 1, GAP, 1], height_ratios=heights,
                       hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

axes = [pf.add_img(f, col, r, reduct_count=REDUCT, add_panel_label=False)
        for r, row in enumerate(ROWS) for col, f in zip((1, 3), row)]

# each label in the gap left of its image, level with the image's top edge
pf.fig.canvas.draw()
for lab, ax in zip(LABELS, axes):
    box = ax.get_position()
    pf.fig.text(box.x0 - PAD, box.y1, lab, ha="right", va="top", **pf.panel_label_kwargs)
pf.format_and_save_figure("fig2.png", tight_final=False)
