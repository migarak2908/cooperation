# Figures from grid labels or values: one panel per (panel a, panel b) value pair, each an image over
# the grid, with the model's reference lines. Figures are saved to files and closed, never shown.
# Needs numpy and matplotlib only (no jax).

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle

from coop import references

OTHER_COLOUR = "#9e9e9e"
GOOD_COLOURS = ["#378ADD", "#E24B4A"]   # good 1 blue, good 2 red, in every figure
# differences from a grid point: teal negative, white 0, orange positive (red and blue mean the goods)
TEAL_ORANGE = LinearSegmentedColormap.from_list("teal_orange", ["#0F7B7B", "#FFFFFF", "#E07B00"])
# scale plots: one hue, light (low) to dark (high), away from the goods' red and blue; NaN in grey
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#F5F2EA", "#9FCBB9", "#3E8C7E", "#123F3A"])
NAN_COLOUR = "#CFCFCF"
HATCH_COLOUR = "#E8A33D"             # --hatch: amber lines and outline, clear on both ends of SEQUENTIAL
REFERENCE_GREY = "#7A7A7A"           # scale plots: reference lines, visible on both ends of SEQUENTIAL
# layers figure: total investment in white (with a thin grey outline, apart from cream) and greys, "in
# between" from cream (balanced) to mid teal (specialised), light enough that the black reference lines,
# branching outline and patterns stay visible on every region
LEVEL_COLOURS = {"defection": "#FFFFFF", "bistable": "#D3D1C7", "full": "#888780", "start-dependent": "#FFFFFF"}
SPECIALISATION = LinearSegmentedColormap.from_list("specialisation", ["#F1EFE8", "#1D9E75"])
DEFECTION_OUTLINE = "#9E9E9E"
KIND_HATCHES = {"between goods": "///", "in total": "..", "mixed": "///.."}
THREE_HATCH = "xx"
LAYER_LINE_COLOUR = "black"
LAYERS_CAPTION = ("Overlapping patterns: both kinds of branching. Specialists are fully specialised only where "
                  "x* < 0.25 (above the dashed line); below it they also invest in the second good, so the shade "
                  "is lighter. A few points in the division-of-labour panel have three groups.")


def save(fig, filename, **options):
    fig.savefig(filename, dpi=150, **options)
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

def figure_size(r):
    a, b = r["panel_values"]
    rows = len(a) if r["panel_rows"] == r["panel_names"][0] else len(b)
    return 4.5 * (len(a) * len(b) // rows) + 4, 4 * rows + 1.5


def panel_axes(fig, r):
    # a grid of axes in fig, one per panel: r["panel_rows"] runs down, the other panel parameter across;
    # returns the axes and an iterator over (axis, i, j) with i, j the panel a, panel b indices. Panel
    # titles ("p = 0.8, q = 1") only when there is more than one panel
    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    titled = len(a_values) * len(b_values) > 1
    rows_are_a = r["panel_rows"] == a
    n_rows, n_cols = (len(a_values), len(b_values)) if rows_are_a else (len(b_values), len(a_values))
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    y_name, x_name = r["grid_names"]
    def each():
        for i, va in enumerate(a_values):
            for j, vb in enumerate(b_values):
                ax = axes[i, j] if rows_are_a else axes[j, i]
                if titled:
                    ax.set_title(f"{r['labels'][a]} = {va:g}, {r['labels'][b]} = {vb:g}", fontsize=10)
                ax.set_xlabel(r["labels"][x_name]); ax.set_ylabel(r["labels"][y_name])
                yield ax, i, j
    return axes, each()


def extent(r):
    y_axis, x_axis = r["grid_axes"]
    return [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]


def draw_reference(ax, r, colour=None):
    # the model's reference lines, clipped to the plotted range; colour: all lines in it
    for line in references.lines(r):
        ax.plot(line["x"], line["y"], color=colour or line.get("colour", "black"), linewidth=line.get("width", 1),
                linestyle=line["style"])
    x0, x1, y0, y1 = extent(r)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def reference_handles(r, colour=None):
    return [Line2D([], [], color=colour or line.get("colour", "black"), linewidth=line.get("width", 1),
                   linestyle=line["style"], label=line["label"])
            for line in references.lines(r) if line["label"]]


# ---------- category and scale plots ----------

def plot_categories(labels, r, scheme, title, filename):
    names, colours = label_colours(labels, scheme)
    index = {l: k for k, l in enumerate(names)}
    codes = np.vectorize(lambda l: index.get(l, index.get("other")))(labels)
    fig = plt.figure(figsize=figure_size(r))
    axes, each = panel_axes(fig, r)
    for ax, i, j in each:
        ax.imshow(codes[i, j], origin="lower", extent=extent(r), aspect="auto",
                  interpolation="nearest", cmap=ListedColormap(colours), vmin=-0.5, vmax=len(names) - 0.5)
        draw_reference(ax, r)
    fig.suptitle(title, fontsize=11)
    handles = [Patch(color=c, label=l) for l, c in zip(names, colours)] + reference_handles(r)
    fig.legend(handles=handles, loc="center left", fontsize=8)
    fig.subplots_adjust(left=3.5 / fig.get_figwidth(), wspace=0.25, hspace=0.3, top=0.88)
    save(fig, filename)


def draw_hatch(ax, mask, r):
    # hatch the grid cells where mask is True and outline the hatched area along the cell edges; cells as
    # imshow places them over extent(r)
    x0, x1, y0, y1 = extent(r)
    rows, cols = mask.shape
    w, h = (x1 - x0) / cols, (y1 - y0) / rows
    padded = np.pad(mask, 1)
    edges = []
    for i, j in zip(*np.nonzero(mask)):
        x, y = x0 + j * w, y0 + i * h
        cell = Rectangle((x, y), w, h, fill=False, hatch="//", edgecolor=HATCH_COLOUR, linewidth=0)
        cell.set_hatch_linewidth(1.2)
        ax.add_patch(cell)
        for di, dj, edge in [(-1, 0, [(x, y), (x + w, y)]), (1, 0, [(x, y + h), (x + w, y + h)]),
                             (0, -1, [(x, y), (x, y + h)]), (0, 1, [(x + w, y), (x + w, y + h)])]:
            if not padded[i + 1 + di, j + 1 + dj]:                # a neighbour outside the area
                edges.append(edge)
    ax.add_collection(LineCollection(edges, colors=HATCH_COLOUR, linewidths=1.5))


def plot_scale(values, r, scheme, filename, relative_to=None, point=None, hatch=None):
    # relative_to: a description of the reference (e.g. "p = 1, q = 1") when values are differences
    # from it; then a diverging colour map centred on 0 with symmetric limits. point: the reference is
    # this grid point ({grid parameter: value}) in each panel: teal-orange, and a cross marks it.
    # hatch: (mask (panel a, panel b, row, column), legend label): hatch the grid points where mask is True.
    # The colour bar carries the scheme's short label, and its ends (scheme "ends") at vmin and vmax
    scale = references.investment_scale(r)[1]
    label = scheme.get("label", "").format(scale=scale)
    unit = f" ({scheme['unit'].format(scale=scale)})" if "unit" in scheme else ""
    ends = None
    if relative_to:
        largest = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.0
        vmin, vmax = -(largest or 1.0), largest or 1.0
        cmap, label = TEAL_ORANGE if point else "RdBu_r", f"change in {label} from {relative_to}"
    else:
        vmin, vmax = scheme["range"](r) if callable(scheme.get("range")) else scheme.get("range", (None, None))
        cmap = LinearSegmentedColormap.from_list(label, scheme["colours"]) if "colours" in scheme else SEQUENTIAL
        ends = scheme.get("ends")
    label += unit
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(NAN_COLOUR)                                            # NaN: no value (e.g. nobody invests)
    fig = plt.figure(figsize=figure_size(r))
    axes, each = panel_axes(fig, r)
    for ax, i, j in each:
        im = ax.imshow(values[i, j], origin="lower", extent=extent(r), aspect="auto",
                       interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        draw_reference(ax, r, REFERENCE_GREY)
        if point:
            (y, x) = r["grid_names"]
            ax.plot(point[x], point[y], marker="x", color="black", markersize=6, markeredgewidth=1.5)
        if hatch is not None:
            draw_hatch(ax, hatch[0][i, j], r)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    bar = fig.colorbar(im, ax=axes, shrink=0.8)
    bar.set_label(label, fontsize=10)
    bar.outline.set_visible(False)
    if ends:                            # words at the two ends instead of numbers there
        low, high = im.get_clim()
        ticks = [t for t in bar.get_ticks() if low < t < high]
        bar.set_ticks([low, *ticks, high], labels=[ends[0], *[f"{t:g}" for t in ticks], ends[1]])
    handles = reference_handles(r, REFERENCE_GREY)
    if hatch is not None:
        handles.append(Patch(facecolor="none", edgecolor=HATCH_COLOUR, hatch="//", linewidth=1.5, label=hatch[1]))
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 0.02), ncol=len(handles),
                   fontsize=9, frameon=False)
    save(fig, filename, bbox_inches="tight")


# ---------- individual-based simulation: the population across generations ----------

def plot_population(history, record_every, x_max, filename, bins=100):
    # history: (record, individual, trait), from generation 0. For each trait, the share of the
    # population at each investment against generation, good 1 blue and good 2 red, translucent so
    # they blend to purple where they overlap
    step = record_every
    extent = [-step / 2, (len(history) - 0.5) * step, 0, x_max]
    edges = np.linspace(0, x_max, bins + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for good, colour in enumerate(GOOD_COLOURS):
        counts = np.array([np.histogram(pop[:, good], edges)[0] for pop in history])
        image = np.zeros((bins, len(history), 4))
        image[..., :3] = to_rgb(colour)
        image[..., 3] = np.sqrt(counts.T / history.shape[1])       # square root: small groups still show
        ax.imshow(image, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    ax.set_xlim(extent[:2])
    ax.set_xlabel("generation")
    ax.set_ylabel("investment")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=c, label=f"good {g + 1}") for g, c in enumerate(GOOD_COLOURS)],
              frameon=False, fontsize=9)
    save(fig, filename, bbox_inches="tight")


# ---------- layers: total investment, specialisation and branching in one figure ----------

def draw_pattern(ax, X, Y, mask, hatch):
    # hatch the region where mask is True, without filling it
    if mask.any():
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors="none", hatches=[hatch])


def mark_unknown(ax, X, Y, mask):
    # a grey "?" at the centre of each connected region where mask is True
    from scipy import ndimage
    regions, n = ndimage.label(mask)
    for i, j in ndimage.center_of_mass(mask, regions, range(1, n + 1)):
        ax.text(np.interp(j, np.arange(X.shape[1]), X[0]), np.interp(i, np.arange(Y.shape[0]), Y[:, 0]), "?",
                color="grey", fontsize=8, ha="center", va="center")


def plot_layers(layer, r, lines, filename, panel_titles):
    # layer: coop.layers.layers(r); lines: reference lines (references.layer_lines); panel_titles: {(i, j): title}.
    # Drawn and saved inside rc_context, as the hatch colour is read when the figure is drawn
    with plt.rc_context({"hatch.color": LAYER_LINE_COLOUR}):
        fig = plt.figure(figsize=figure_size(r))
        axes, image = draw_layer_panels(fig, layer, r, lines, panel_titles)
        bar = fig.colorbar(image, ax=axes, shrink=0.6)
        bar.set_label("individual specialisation", fontsize=10)
        bar.set_ticks([0, 0.5, 1], labels=["balanced", "0.5", "specialised"])
        bar.outline.set_visible(False)
        for k, (title, handles) in enumerate(layer_legend(layer, lines).items()):
            fig.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(0.1 + 0.22 * k, 0.04),
                       frameon=False, fontsize=8, title_fontsize=9, alignment="left")
        fig.text(0.1, -0.04, textwrap.fill(LAYERS_CAPTION, 150), fontsize=9, va="top")
        save(fig, filename, bbox_inches="tight")


def draw_layer_panels(fig, layer, r, lines, panel_titles):
    y_axis, x_axis = r["grid_axes"]
    X, Y = np.meshgrid(x_axis, y_axis)
    axes, each = panel_axes(fig, r)
    for ax, i, j in each:
        level, kind = layer["level"][i, j], layer["kind"][i, j]
        greys = np.array([to_rgba(LEVEL_COLOURS.get(l, "none")) for l in level.ravel()]).reshape(level.shape + (4,))
        ax.imshow(greys, origin="lower", extent=extent(r), aspect="auto", interpolation="nearest")
        between = np.ma.masked_where(level != "in between", layer["specialisation"][i, j])
        image = ax.imshow(between, origin="lower", extent=extent(r), aspect="auto", interpolation="nearest",
                          cmap=SPECIALISATION, vmin=0, vmax=1)
        defection = level == "defection"
        if defection.any() and not defection.all():
            ax.contour(X, Y, defection.astype(float), levels=[0.5], colors=DEFECTION_OUTLINE, linewidths=0.5)
        mark_unknown(ax, X, Y, level == "start-dependent")
        branched = kind != "none"
        if branched.any() and not branched.all():
            ax.contour(X, Y, branched.astype(float), levels=[0.5], colors=LAYER_LINE_COLOUR, linewidths=0.8)
        for name, hatch in KIND_HATCHES.items():
            draw_pattern(ax, X, Y, kind == name, hatch)
        draw_pattern(ax, X, Y, layer["three"][i, j], THREE_HATCH)
        for line in lines:
            ax.plot(line["x"], line["y"], color=LAYER_LINE_COLOUR, linewidth=0.9, linestyle=line["style"])
        x0, x1, y0, y1 = extent(r)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(panel_titles[i, j], fontsize=10)
    return axes, image


def layer_legend(layer, lines):
    # three blocks, each with only the entries that occur
    levels, kinds = set(layer["level"].ravel()), set(layer["kind"].ravel())
    box = lambda colour, label: Patch(facecolor=colour, edgecolor="grey", linewidth=0.5, label=label)
    hatched = lambda hatch, label: Patch(facecolor="white", edgecolor=LAYER_LINE_COLOUR, hatch=hatch, linewidth=0.5,
                                         label=label)
    total = [box(LEVEL_COLOURS[l], l if l != "full" else "full investment")
             for l in ["defection", "bistable", "full"] if l in levels]
    branching = ([Line2D([], [], color=LAYER_LINE_COLOUR, linewidth=0.8, label="branched")] if kinds - {"none"} else [])
    branching += [hatched(KIND_HATCHES[k], k) for k in ["between goods", "in total"] if {k, "mixed"} & kinds]
    references = [Line2D([], [], color=LAYER_LINE_COLOUR, linewidth=0.9, linestyle=line["style"], label=line["label"])
                  for line in lines]
    return {title: handles for title, handles in [("Total investment", total), ("Branching", branching),
                                                  ("Reference lines", references)] if handles}
