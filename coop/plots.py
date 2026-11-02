# Figures from grid labels or values: one panel per (panel a, panel b) value pair, each an image over
# the grid, with the model's reference lines. Figures are saved to files and closed, never shown.
# Needs numpy and matplotlib only (no jax).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from coop import references
from coop.schemes import counted_groups, invested_threshold

OTHER_COLOUR = "#9e9e9e"
# differences from a grid point: teal negative, white 0, orange positive (red and blue mean the goods)
TEAL_ORANGE = LinearSegmentedColormap.from_list("teal_orange", ["#0F7B7B", "#FFFFFF", "#E07B00"])


def save(fig, filename):
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"saved {filename}")


def label_colours(labels, scheme):
    # the scheme's fixed legend, then any other labels present: tab20 colours if the scheme has no
    # fixed legend, grey "other" if it has one
    present = set(labels.ravel())
    if "colours" not in scheme:
        names = sorted(present)
        return names, [plt.cm.tab20(k % 20) for k in range(len(names))]
    names, colours = scheme["colours"](present)
    if not present <= set(names):
        names, colours = names + ["other"], colours + [OTHER_COLOUR]
    return names, colours


# ---------- panels ----------

def figure_size(r, columns=1):
    a, b = r["panel_values"]
    rows = len(a) if r["panel_rows"] == r["panel_names"][0] else len(b)
    return (4.5 * (len(a) * len(b) // rows) + 4) * columns, 4 * rows + 1.5


def panel_axes(container, r):
    # a grid of axes in container (a figure or subfigure), one per panel: r["panel_rows"] runs down,
    # the other panel parameter across; returns the axes and an iterator over (axis, i, j) with
    # i, j the panel a, panel b indices
    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    rows_are_a = r["panel_rows"] == a
    n_rows, n_cols = (len(a_values), len(b_values)) if rows_are_a else (len(b_values), len(a_values))
    axes = container.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    y_name, x_name = r["grid_names"]
    def each():
        for i, va in enumerate(a_values):
            for j, vb in enumerate(b_values):
                ax = axes[i, j] if rows_are_a else axes[j, i]
                ax.set_title(f"{r['labels'][a]} = {va:g}, {r['labels'][b]} = {vb:g}", fontsize=10)
                ax.set_xlabel(r["labels"][x_name]); ax.set_ylabel(r["labels"][y_name])
                yield ax, i, j
    return axes, each()


def extent(r):
    y_axis, x_axis = r["grid_axes"]
    return [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]


def draw_reference(ax, r):
    # the model's reference lines (and shading), clipped to the plotted range
    outside = references.shade(r)
    if outside is not None:
        overlay = np.zeros(outside.shape + (4,))
        overlay[outside] = (0.5, 0.5, 0.5, 0.4)
        ax.imshow(overlay, origin="lower", extent=extent(r), aspect="auto", interpolation="nearest")
    for line in references.lines(r):
        ax.plot(line["x"], line["y"], color=line.get("colour", "black"), linewidth=line.get("width", 1),
                linestyle=line["style"])
    x0, x1, y0, y1 = extent(r)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def reference_handles(r):
    return [Line2D([], [], color=line.get("colour", "black"), linewidth=line.get("width", 1),
                   linestyle=line["style"], label=line["label"])
            for line in references.lines(r) if line["label"]]


# ---------- category and scale plots ----------

def draw_categories(container, labels, r, names, colours, title):
    index = {l: k for k, l in enumerate(names)}
    codes = np.vectorize(lambda l: index.get(l, index.get("other")))(labels)
    axes, each = panel_axes(container, r)
    for ax, i, j in each:
        ax.imshow(codes[i, j], origin="lower", extent=extent(r), aspect="auto",
                  interpolation="nearest", cmap=ListedColormap(colours), vmin=-0.5, vmax=len(names) - 0.5)
        draw_reference(ax, r)
    container.suptitle(title, fontsize=11)


def plot_categories(labels, r, scheme, title, filename):
    names, colours = label_colours(labels, scheme)
    fig = plt.figure(figsize=figure_size(r))
    draw_categories(fig, labels, r, names, colours, title)
    handles = [Patch(color=c, label=l) for l, c in zip(names, colours)] + reference_handles(r)
    fig.legend(handles=handles, loc="center left", fontsize=8)
    fig.subplots_adjust(left=3.5 / fig.get_figwidth(), wspace=0.25, hspace=0.3, top=0.88)
    save(fig, filename)


def plot_side_by_side(left, right, r, scheme, titles, filename):
    # two sets of labels on the same grid, with one shared legend so the colours match
    names, colours = label_colours(np.concatenate([left.ravel(), right.ravel()]), scheme)
    fig = plt.figure(figsize=figure_size(r, columns=2))
    for k, (sub, labels, title) in enumerate(zip(fig.subfigures(1, 2), [left, right], titles)):
        draw_categories(sub, labels, r, names, colours, title)
        sub.subplots_adjust(left=0.25 if k == 0 else 0.08, wspace=0.25, hspace=0.3, top=0.88)   # room for the legend
    handles = [Patch(color=c, label=l) for l, c in zip(names, colours)] + reference_handles(r)
    fig.legend(handles=handles, loc="center left", fontsize=8)
    save(fig, filename)


def plot_scale(values, r, scheme, title, filename, relative_to=None, point=None):
    # relative_to: a description of the reference (e.g. "p = 1, q = 1") when values are differences
    # from it; then a diverging colour map centred on 0 with symmetric limits. point: the reference is
    # this grid point ({grid parameter: value}) in each panel: teal-orange, and a cross marks it
    label = scheme.get("label", "").format(scale=references.investment_scale(r)[1])
    cmap = None
    if relative_to:
        largest = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.0
        vmin, vmax = -(largest or 1.0), largest or 1.0
        cmap, label = TEAL_ORANGE if point else "RdBu_r", f"{label}, minus at {relative_to}"
    else:
        vmin, vmax = scheme["range"](r) if callable(scheme.get("range")) else scheme.get("range", (None, None))
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("lightgrey")                                           # NaN: no value (e.g. nobody invests)
    fig = plt.figure(figsize=figure_size(r))
    axes, each = panel_axes(fig, r)
    for ax, i, j in each:
        im = ax.imshow(values[i, j], origin="lower", extent=extent(r), aspect="auto",
                       interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        draw_reference(ax, r)
        if point:
            (y, x) = r["grid_names"]
            ax.plot(point[x], point[y], marker="x", color="black", markersize=6, markeredgewidth=1.5)
    fig.suptitle(title, fontsize=11)
    fig.colorbar(im, ax=axes, label=label)
    if reference_handles(r):
        fig.legend(handles=reference_handles(r), loc="lower left", fontsize=8)
    save(fig, filename)


# ---------- summary: one small grid per condition, one cell per panel ----------

def cell_edges(values):
    # edges of cells centred on the (sorted) values: midpoints, and half a step beyond each end
    v = np.asarray(values, float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = (v[1:] + v[:-1]) / 2
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def plot_summary(fractions, entries, r, filename):
    # fractions: one (panel a, panel b) array per entry; panel a across, panel b up
    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    fig, axes = plt.subplots(1, len(entries), figsize=(3.6 * len(entries) + 1, 4.2), squeeze=False)
    for ax, fraction, (title, scheme, relative_to, comparison, threshold) in zip(axes[0], fractions, entries):
        im = ax.pcolormesh(cell_edges(a_values), cell_edges(b_values), fraction.T, vmin=0, vmax=1, cmap="viridis")
        for i, va in enumerate(a_values):
            for j, vb in enumerate(b_values):
                ax.text(va, vb, f"{fraction[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if fraction[i, j] < 0.6 else "black")
        if {a, b} == {"p", "q"}:
            # q = p and q = (p + 1)/2, with p on whichever axis it is
            x0, x1 = ax.get_xlim()
            xs = np.linspace(x0, x1, 50)
            for style, line in [("--", lambda p: p), (":", lambda p: (p + 1) / 2)]:
                if a == "p":
                    ax.plot(xs, line(xs), color="black", linewidth=1, linestyle=style)
                else:                               # q across: q = line(p) -> p as a function of q
                    ax.plot(xs, xs if style == "--" else 2 * xs - 1, color="black", linewidth=1, linestyle=style)
            ax.set_xlim(x0, x1); ax.set_ylim(*cell_edges(b_values)[[0, -1]])
        condition = {"abs>": "|value| >", ">": ">", "<": "<"}[comparison] + f" {threshold:g}"
        relative = "" if relative_to is None else ", minus at " + ", ".join(f"{k} = {v:g}" for k, v in relative_to.items())
        ax.set_title(f"{title}\n{scheme} {condition}{relative}", fontsize=8)
        ax.set_xlabel(r["labels"][a]); ax.set_ylabel(r["labels"][b])
        ax.set_xticks(a_values); ax.set_yticks(b_values)
    fig.subplots_adjust(top=0.78, wspace=0.35)                # room for the titles
    fig.colorbar(im, ax=axes, label="fraction of the panel's grid points")
    fig.suptitle(f"summary - {r['tag']}", fontsize=10)
    save(fig, filename)


# ---------- cut at one grid column: each group's lean along the grid rows ----------

def plot_cut(r, x_value):
    # the grid column nearest x_value; one row per start, one column per panel; every counted group as
    # a dot (coloured by start) at (grid row value, lean), with lean = (x1 - x2) / (x1 + x2) from -1
    # (only good 2) to +1 (only good 1) and area ~ share; groups investing less than the "invested"
    # threshold in total sit at 0, hollow
    y_name, x_name = r["grid_names"]
    y_axis, x_axis = r["grid_axes"]
    k = int(np.argmin(np.abs(x_axis - x_value)))
    x = float(x_axis[k])
    xs, fs, counted = counted_groups(r["xs"][:, :, :, :, k], r["fs"][:, :, :, :, k], r)   # (start, a, b, row, slot, ...)
    d, t = xs[..., 0] - xs[..., 1], xs[..., 0] + xs[..., 1]
    invests = t >= invested_threshold(r)
    lean = np.where(invests, d / np.where(invests, t, 1.0), 0.0)
    Y = np.broadcast_to(y_axis[:, None], lean.shape[3:])                 # (row, slot)
    step = y_axis[1] - y_axis[0]

    starts, (a, b), (a_values, b_values) = r["starts"], r["panel_names"], r["panel_values"]
    columns = [(i, j, va, vb) for i, va in enumerate(a_values) for j, vb in enumerate(b_values)]
    fig, axes = plt.subplots(len(starts), len(columns), sharex=True, sharey=True,
                             figsize=(4 * len(columns) + 1.5, 2.6 * len(starts) + 1.5), squeeze=False)
    fig.subplots_adjust(left=0.12, bottom=0.12, top=0.9, wspace=0.15, hspace=0.25)

    # largest dot (share 1): 80% of the distance between neighbouring grid-row values, in points
    width = axes[0, 0].get_position().width * fig.get_figwidth() * 72
    size = (0.8 * width * step / (y_axis[-1] - y_axis[0] + 2 * step)) ** 2

    labels = r["labels"]
    lines = references.cut_lines(r, x)
    colours = plt.cm.tab10(np.arange(len(starts)))                     # one colour per start (row)
    for s, start in enumerate(starts):
        for c, (i, j, va, vb) in enumerate(columns):
            ax = axes[s, c]
            for filled in [True, False]:
                use = counted[s, i, j] & (invests[s, i, j] == filled)
                ax.scatter(Y[use], lean[s, i, j][use], s=size * fs[s, i, j][use],
                           facecolors=colours[s] if filled else "none", edgecolors=colours[s], linewidths=0.6)
            for position, style, _ in lines:
                ax.axvline(position, color="black", linewidth=1, linestyle=style)
            ax.set_xlim(y_axis[0] - step, y_axis[-1] + step); ax.set_ylim(-1.1, 1.1)
            if s == 0:
                ax.set_title(f"{labels[a]} = {va:g}, {labels[b]} = {vb:g}, {labels[x_name]} = {x:.3g}", fontsize=10)
            if s == len(starts) - 1:
                ax.set_xlabel(labels[y_name])
            if c == 0:
                ax.set_ylabel(f"start ({start[0]:g}, {start[1]:g})\nlean (x1 - x2) / (x1 + x2)", fontsize=9)
    handles = [Line2D([], [], color="black", linewidth=1, linestyle=style, label=label) for _, style, label in lines]
    handles += [Line2D([], [], marker="o", color="grey", linestyle="", label="group (area ~ share)"),
                Line2D([], [], marker="o", color="grey", markerfacecolor="none", linestyle="",
                       label=f"invests < {invested_threshold(r):g}")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(f"groups at {labels[x_name]} = {x:.3g} (nearest to {x_value:g}), one row per start - {r['tag']}",
                 fontsize=10)
    save(fig, f"cut_{x_name}{x_value:g}_{r['tag']}.png")
