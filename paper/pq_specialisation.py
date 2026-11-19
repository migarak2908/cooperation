# Paper figure: specialisation of individuals (a) and of the population (b) over (p, q), for one
# (b1', b2') panel of a p_q_near_one results file, with q = p, q = (p + 1)/2 and the predicted
# division-of-labour wedge. Values are coop.schemes.individual / population, averaged over starts as in
# run_analysis.py. Saves pq_specialisation_<tag>.pdf/.png next to the results file.
#
# Usage:  python paper/pq_specialisation.py RESULTS.npz --panel b2=1.5
#         python paper/pq_specialisation.py RESULTS.npz --panel b2=1.5 --starts 0 3

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import DOUBLE, letter, plt
from coop import results, schemes
from coop.plots import NAN_COLOUR
from run_analysis import average_over_starts, describe, keep_starts, parse_values

mpl.rcParams["hatch.linewidth"] = 0.5
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f5ef", "#c5ddd4", "#7fb3a6", "#4a8f82"])
SEQ.set_bad(NAN_COLOUR)                                  # NaN: nobody invests
DOL = "#d9531e"
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
TICKS = [0.8, 0.9, 1.0, 1.1, 1.2]
PANELS = [("individual", "Specialisation of individuals"), ("population", "Specialisation of the population")]


def choose_panel(r, wanted):
    # (i, j) of the panel with the wanted values; a parameter with a single value may be left out
    names, values = r["panel_names"], r["panel_values"]
    for name, vs in zip(names, values):
        if name not in wanted and len(vs) == 1:
            wanted[name] = vs[0]
    options = "; ".join(f"{n} = {', '.join(f'{v:g}' for v in vs)}" for n, vs in zip(names, values))
    if set(wanted) != set(names):
        raise SystemExit(f"--panel: choose one panel, e.g. --panel {names[1]}={values[1][0]:g} ({options})")
    ij = results.panel_index(r, **wanted)
    if ij is None:
        raise SystemExit(f"--panel: no panel with {describe(wanted)} ({options})")
    return ij


def draw(ax, p, q, values, title, s):
    im = ax.pcolormesh(p, q, values, cmap=SEQ, vmin=0, vmax=1, shading="nearest", rasterized=True)
    x0, x1 = p[0] - (p[1] - p[0]) / 2, p[-1] + (p[1] - p[0]) / 2
    y0, y1 = q[0] - (q[1] - q[0]) / 2, q[-1] + (q[1] - q[0]) / 2
    lo, hi = np.linspace(x0, 1, 50), np.linspace(1, x1, 50)
    ax.plot(hi, hi, color="black", lw=0.8, ls=(0, (4, 2)))
    ax.plot(hi, (hi + 1) / 2, color="black", lw=0.8)
    ax.plot(lo, lo, color=DOL, lw=1.1, ls=(0, (4, 2)), path_effects=HALO, zorder=4)
    ax.plot(lo, (lo + 1) / 2, color=DOL, lw=1.1, path_effects=HALO, zorder=4)
    wedge = Polygon([(x0, x0), (1, 1), (x0, (x0 + 1) / 2)], closed=True, facecolor="none",
                    edgecolor=DOL, hatch="//////", lw=0, zorder=3)
    if hasattr(wedge, "set_hatch_linewidth"):
        wedge.set_hatch_linewidth(0.5)
    ax.add_patch(wedge)
    ax.annotate("Division of labour", xy=(0.835, 0.895), xytext=(0.81, 1.0), fontsize=7, color=DOL,
                ha="left", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6, color=DOL, shrinkA=2, shrinkB=0))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks(TICKS); ax.set_yticks(TICKS)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Benefit interaction, $p$")
    ax.set_title(title, loc="left", pad=4)
    letter(ax, s, x=-0.03, y=1.015)
    ax.text(1.115, 1.15, r"$q = p$", fontsize=7, ha="right", va="center")
    ax.text(1.19, 1.03, r"$q = (p+1)/2$", fontsize=7, ha="right", va="top")
    return im


def figure(maps, p, q, filename):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.78, DOUBLE * 0.78 / 2 * 0.92), sharey=True)
    for ax, (key, title), s in zip(axes, PANELS, "ab"):
        im = draw(ax, p, q, maps[key], title, s)
    axes[0].set_ylabel(r"Cost interaction, $q$")
    axes[1].tick_params(labelleft=False)

    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 0.9, 1))
    pos = axes[1].get_position()
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, pos.y0, 0.014, pos.height]))
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(direction="out", length=2, labelsize=7, right=True)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Specialisation", fontsize=7, labelpad=3)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{filename}.{ext}", bbox_inches="tight")
        print(f"saved {filename}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="results file from run_simulation.py (p_q_near_one sweep)")
    parser.add_argument("--panel", metavar="NAME=V[,NAME=V]", default="",
                        help="the (b1', b2') panel to plot, e.g. b2=1.5 (needed if the file has several)")
    parser.add_argument("--starts", nargs="+", type=int, metavar="I",
                        help="use only the runs from these starts (indices into the results' starts), e.g. 0 3")
    args = parser.parse_args()

    r = results.load(args.results)
    if r["grid_names"] != ["q", "p"]:
        raise SystemExit(f"needs a (p, q) grid; this file's grid is {r['grid_names']}")
    if args.starts:
        r = keep_starts(r, args.starts)
    i, j = choose_panel(r, parse_values(args.panel) if args.panel else {})
    maps = {key: average_over_starts(getattr(schemes, key)(r["xs"], r["fs"], r))[i, j] for key, _ in PANELS}

    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    print(f"panel: {r['labels'][a]} = {a_values[i]:g}, {r['labels'][b]} = {b_values[j]:g}")
    print(f"NaN cells (nobody invests): {int(np.isnan(maps['individual']).sum())} of {maps['individual'].size}")
    q, p = r["grid_axes"]
    figure(maps, p, q, os.path.join(os.path.dirname(os.path.abspath(args.results)), f"pq_specialisation_{r['tag']}"))
