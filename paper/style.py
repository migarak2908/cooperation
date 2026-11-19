# Shared journal-style settings: Arial-metric sans-serif, figures drawn at printed size.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
SINGLE, DOUBLE = 8.5 * CM, 17.5 * CM
GREY = "#8c8c8c"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42,
})


def letter(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
