# Labelling and plotting of results saved by two_good_simulate.py (quadratic payoff) or
# two_good_simulate_exp.py (exponential payoff), as a 3 x 3 grid of phase diagrams (columns p, rows q).
# Needs numpy, scipy and matplotlib only (no jax).
#
# A scheme turns the final state of each run into something to plot:
#   categories: a name per run ("CC and DD", ...); a grid point's label is the distinct names over
#               its starts, joined with " |or| "; rare labels are filtered out for a second plot
#   scale:      a number per run, averaged over starts and plotted with a colour bar
# To add a definition of cooperation: write a function like cc_dc_dd below and add it to SCHEMES.
#
# Usage:  python two_good_analysis.py                          (newest results file, scheme cc_dc_dd)
#         python two_good_analysis.py RESULTS.npz --scheme cc_dc_dd_division
#         python two_good_analysis.py --scheme mean_investment
#         python two_good_analysis.py two_good_results_exp_....npz --scheme cc_dc_dd mean_investment
#         python two_good_analysis.py RESULTS.npz --invested absolute     (override the "invested" rule)
#         in Colab: %run two_good_analysis.py --scheme cc_dc_dd

import argparse
import glob
import os
import numpy as np

LABEL_SHARE_MIN = 0.01               # a group counts if its (rounded) share is above this
RARE_FRAC = 0.003                    # labels on fewer than 0.3% of all grid points are rare
OTHER_COLOUR = "#9e9e9e"
SHADE_VALID_BAND = False             # True: grey out grid cells outside the band between the two lines
INVESTED_RULE = None                 # None: model default; "half_cap": above x_max / 2; "absolute": above INVESTED_MIN
INVESTED_MIN = 0.05


# ---------- loading ----------

def load(filename=None):
    # results of two_good_simulate.py or two_good_simulate_exp.py;
    # xs: (start, p, q, row, column, slot, good), fs: (..., slot)
    if filename is None:
        files = glob.glob("two_good_results_*.npz")
        if not files:
            raise SystemExit("no two_good_results_*.npz here: run two_good_simulate.py first")
        filename = max(files, key=os.path.getmtime)
    r = dict(np.load(filename))
    r["tag"] = str(r["tag"])
    r["x_max"], r["x_min"] = float(r["x_max"]), float(r["x_min"])
    r["p_values"], r["q_values"] = [float(v) for v in r["p_values"]], [float(v) for v in r["q_values"]]
    r["model"] = "exp" if "beta" in r else "quadratic"
    r.update(model_axes(r))
    print(f"loaded {filename} ({r['model']} model)")
    not_converged = int((r["status"] == 2).sum())
    if not_converged:
        print(f"  note: {not_converged} of {r['status'].size} runs stopped at {int(r['max_steps'])} steps "
              f"without converging")
    return r


def model(r):
    # results without a "model" field (e.g. built by two_good_toggles.py) are quadratic
    return r.get("model", "quadratic")


def model_axes(r):
    # column and row parameter values, and their names: quadratic b1', b2' (both r["b"]); exp b', beta'
    if model(r) == "exp":
        return dict(x_axis=r["b"], y_axis=r["beta"], x_name="b'", y_name="beta'")
    return dict(x_axis=r["b"], y_axis=r["b"], x_name="b1'", y_name="b2'")


def counted_groups(xs, fs, r):
    # the groups that count when labelling: traits rounded to 3 decimals, shares to 2; a group counts
    # if its share is above LABEL_SHARE_MIN and no earlier counted group is within merge_gap of it
    xs, fs = np.round(xs, 3), np.round(fs, 2)
    counted = fs > LABEL_SHARE_MIN
    K = xs.shape[-2]
    dist = np.linalg.norm(xs[..., :, None, :] - xs[..., None, :, :], axis=-1)
    earlier = np.tri(K, k=-1, dtype=bool)                                     # [j, i]: i < j
    counted &= ~np.any(counted[..., None, :] & earlier & (dist < r["merge_gap"]), axis=-1)
    return xs, fs, counted


# ---------- scheme: CC / DC / DD ----------
# A good is "invested" above invested_threshold(r): x_max / 2 for the quadratic model (half the cap),
# INVESTED_MIN for the exponential one (no natural cap; x_max / 2 would label almost everything DD).
# A group is CC if it invests in both goods, DC in one, DD in neither. A run's outcome is the distinct
# group types, alphabetical, joined with " and ".

def default_rule(r):
    return "absolute" if model(r) == "exp" else "half_cap"


def invested_rule(r):
    return INVESTED_RULE or default_rule(r)


def invested_threshold(r):
    return r["x_max"] / 2 if invested_rule(r) == "half_cap" else INVESTED_MIN


def cc_dc_dd(xs, fs, r, division=False):
    # xs: (..., K, 2), fs: (..., K) -> outcome name per run; with division, DC groups investing in
    # different goods make "DC and DC'"
    xs, fs, counted = counted_groups(xs, fs, r)
    invested = xs > invested_threshold(r)
    n_invested = invested.sum(axis=-1)
    has = lambda n: np.any(counted & (n_invested == n), axis=-1)
    dc_in = lambda good: np.any(counted & (n_invested == 1) & invested[..., good], axis=-1)
    division_of_labour = division & dc_in(0) & dc_in(1)

    # each run's combination as bits (1 = CC, 2 = DC, 4 = DD, 8 = division of labour), then its name
    bits = has(2) * 1 + has(1) * 2 + has(0) * 4 + division_of_labour * 8
    return np.array([outcome_name(k) for k in range(16)], dtype=object)[bits]


def outcome_name(bits):
    types = []
    if bits & 1:
        types.append("CC")
    if bits & 2:
        types.append("DC and DC'" if bits & 8 else "DC")
    if bits & 4:
        types.append("DD")
    return " and ".join(types) if types else "none"


def cc_dc_dd_division(xs, fs, r):
    return cc_dc_dd(xs, fs, r, division=True)


# fixed colours: hsv from 0 to 0.9 over these 11 labels in this order (as in the reference);
# "DC and DC'" teal, other labels containing it olive
CC_DC_DD_LABELS = ["CC", "CC and DC", "CC and DD", "CC |or| DC", "CC |or| DC |or| DD", "CC |or| DD",
                   "DC", "DC and DD", "DC and DD |or| DD", "DC |or| DD", "DD"]

def cc_dc_dd_colours(present):
    import matplotlib.pyplot as plt
    names = list(CC_DC_DD_LABELS)
    colours = [tuple(c) for c in plt.cm.hsv(np.linspace(0, 0.9, len(names)))]
    for l in sorted(l for l in present if "DC and DC'" in l):
        names.append(l)
        colours.append("#1b9e77" if l == "DC and DC'" else "#66a61e")
    return names, colours


def check_cc_dc_dd(r):
    # hand-made final states: full investment at x_max, and cases either side of the "invested"
    # threshold, so the checks hold for any x_max and rule
    m, t = r["x_max"], 1e-6
    half = invested_threshold(r)
    hi, lo = 1.2 * half, 0.8 * half                     # 0.6 x_max and 0.4 x_max for the half_cap rule
    rule = "x_max / 2" if invested_rule(r) == "half_cap" else f"{half:g}"
    def outcome(groups, shares, division=False):
        xs = np.full((4, 2), t); fs = np.zeros(4)
        xs[:len(groups)], fs[:len(shares)] = groups, shares
        return cc_dc_dd(xs, fs, r, division)
    cases = [
        (f"one group at ({m:g}, {m:g})",                     outcome([(m, m)], [1.0]),                          "CC"),
        (f"groups at ({m:g}, {m:g}) and (1e-6, 1e-6)",       outcome([(m, m), (t, t)], [0.5, 0.5]),             "CC and DD"),
        (f"groups at ({m:g}, 1e-6) and (1e-6, {m:g})",       outcome([(m, t), (t, m)], [0.5, 0.5]),             "DC"),
        ("same, division of labour on",                      outcome([(m, t), (t, m)], [0.5, 0.5], True),       "DC and DC'"),
        ("full group share 0.995 + (1e-6, 1e-6) share 0.005", outcome([(m, m), (t, t)], [0.995, 0.005]),        "CC"),
        ("three groups CC, DC, DD",                          outcome([(m, m), (m, t), (t, t)], [0.4, 0.3, 0.3]), "CC and DC and DD"),
        (f"one group at ({hi:g}, {hi:g}), above {rule}",     outcome([(hi, hi)], [1.0]),                        "CC"),
        (f"one group at ({lo:g}, {lo:g}), below {rule}",     outcome([(lo, lo)], [1.0]),                        "DD"),
    ]
    if round(half, 3) == half:       # rounding: just above the threshold rounds down onto it
        cases.append((f"one group at ({half + 0.0004:g}, {half + 0.0004:g}) (rounds to {half:g})",
                      outcome([(half + 0.0004, half + 0.0004)], [1.0]), "DD"))
    for name, got, expected in cases:
        print(f"  {'ok' if got == expected else 'FAILED':6s} {name:46s} -> {got!r}")
        assert got == expected
    got = grid_labels(np.array([["DD"], ["CC and DD"], ["DD"], ["CC"]], dtype=object))[0]
    print(f"  {'ok' if got == 'CC |or| CC and DD |or| DD' else 'FAILED':6s} {'grid label from DD, CC and DD, DD, CC':46s} -> {got!r}")
    assert got == "CC |or| CC and DD |or| DD"


def compare_analytic(labels, r):
    # p = q = 1 reduces to the one-good model: compare with its analytic diagram for this model
    if model(r) == "exp":
        compare_one_good_exp(labels, r)
    else:
        compare_one_good(labels, r)


def compare_one_good(labels, r):
    # p = q = 1 reduces to the one-good model: compare with its analytic diagram away from boundaries
    if 1.0 not in r["p_values"] or 1.0 not in r["q_values"]:
        print("p = q = 1 check skipped: p and q need to include 1")
        return
    b, x_max = r["b"], r["x_max"]
    B1, B2 = np.meshgrid(b, b)
    D0, Dmax = B1 - 1, (B1 - 1) + (2 - 4 * B2) * x_max
    with np.errstate(divide="ignore", invalid="ignore"):
        x_star = D0 / (4 * B2 - 2)
    analytic = np.select([(D0 > 0) & (Dmax > 0), (D0 < 0) & (Dmax < 0), (D0 < 0) & (Dmax > 0),
                          (D0 > 0) & (Dmax < 0) & (B2 < 1), (D0 > 0) & (Dmax < 0) & (B2 > 1)],
                         ["CC", "DD", "CC |or| DD", "CC and DD", np.where(x_star > x_max / 2, "CC", "DD")],
                         default="boundary")
    print_comparison(analytic, labels, r)


def compare_one_good_exp(labels, r):
    # exponential one-good model (one group per run, b = b', beta = beta'), with x* from the results:
    #   b > beta, beta < 1:      ESS at x*;  b > beta, 1 < beta < 2: branching;  b > beta, beta > 2: up to the wall
    #   b < beta, beta < 2:      no investment;  b < beta, beta > 2: bistable
    if 1.0 not in r["p_values"] or 1.0 not in r["q_values"]:
        print("p = q = 1 check skipped: p and q need to include 1")
        return
    B, BETA = np.meshgrid(r["b"], r["beta"])
    x_star = r["x_star"]
    ess = np.where(x_star > invested_threshold(r), "CC", "DD")
    analytic = np.select([(B > BETA) & (BETA < 1), (B > BETA) & (BETA > 1) & (BETA < 2), (B > BETA) & (BETA > 2),
                          (B < BETA) & (BETA < 2), (B < BETA) & (BETA > 2)],
                         [ess, "CC and DD", "CC", "DD", "CC |or| DD"], default="boundary")
    print_comparison(analytic, labels, r)


def print_comparison(analytic, labels, r):
    # simulated p = q = 1 labels vs an analytic diagram, leaving out points next to a region boundary
    pad = np.pad(analytic, 1, mode="edge")
    boundary = analytic == "boundary"
    rows, cols = analytic.shape
    for di, dj in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        boundary |= pad[di:di + rows, dj:dj + cols] != analytic
    sim = labels[r["p_values"].index(1.0), r["q_values"].index(1.0)]
    differ = (sim != analytic) & ~boundary
    print(f"p = q = 1 vs one-good analytic, away from boundaries: {differ.sum()} of {(~boundary).sum()} points differ")
    for a, s in sorted(set(zip(analytic[differ], sim[differ]))):
        print(f"  analytic {a!r:14s} simulated {s!r:28s} {int(((analytic == a) & (sim == s) & differ).sum()):4d} points")


# ---------- scheme: mean investment (a sliding scale) ----------

def investment_scale(r):
    # what mean investment is divided by: x_max for the quadratic model; the largest x* on the grid
    # for the exponential one, where x_max is just a wall
    if model(r) == "exp":
        return float(np.nanmax(r["x_star"])), "max x*"
    return r["x_max"], "x_max"


def mean_investment(xs, fs, r):
    # share-weighted average investment over groups and both goods, as a fraction of investment_scale
    return np.sum(fs[..., None] * xs, axis=(-2, -1)) / (2 * investment_scale(r)[0])


# ---------- schemes ----------
# kind: "categories" or "scale"; for categories, optionally colours (fixed legend), check (run
# before labelling) and compare (run on the unfiltered grid labels); a scale's label may use {scale}

SCHEMES = {
    "cc_dc_dd":          dict(kind="categories", outcome=cc_dc_dd, colours=cc_dc_dd_colours,
                              check=check_cc_dc_dd, compare=compare_analytic),
    "cc_dc_dd_division": dict(kind="categories", outcome=cc_dc_dd_division, colours=cc_dc_dd_colours,
                              check=check_cc_dc_dd, compare=compare_analytic),
    "mean_investment":   dict(kind="scale", outcome=mean_investment, label="mean investment / {scale}",
                              vmin=0, vmax=1),
}


# ---------- grid labels ----------

def grid_labels(outcomes):
    # outcomes: (start, ...) -> label per grid point: distinct outcomes, alphabetical, " |or| "
    labels = np.empty(outcomes.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        labels[idx] = " |or| ".join(sorted(set(outcomes[(slice(None),) + idx])))
    return labels


def replace_rare(labels, x_axis, y_axis):
    # labels on fewer than RARE_FRAC of all grid points (all panels together) are rare; each rare
    # point takes the label of the nearest non-rare point in the same panel
    from scipy.spatial import cKDTree
    values, counts = np.unique(labels, return_counts=True)
    rare = {v: int(c) for v, c in zip(values, counts) if c < RARE_FRAC * labels.size}
    X, Y = np.meshgrid(x_axis, y_axis)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    out = labels.copy()
    replaced = np.zeros(labels.shape[:2], int)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            flat = labels[i, j].ravel()
            is_rare = np.isin(flat, list(rare))
            if is_rare.any() and not is_rare.all():
                _, nearest = cKDTree(points[~is_rare]).query(points[is_rare])
                flat = flat.copy()
                flat[is_rare] = flat[~is_rare][nearest]
                out[i, j] = flat.reshape(labels.shape[2:])
                replaced[i, j] = is_rare.sum()
    return out, rare, replaced


# ---------- plots ----------

def label_colours(labels, scheme):
    # the scheme's fixed legend, then any other labels present: tab20 colours if the scheme has no
    # fixed legend, grey "other" if it has one
    import matplotlib.pyplot as plt
    present = set(labels.ravel())
    if "colours" not in scheme:
        names = sorted(present)
        return names, [plt.cm.tab20(k % 20) for k in range(len(names))]
    names, colours = scheme["colours"](present)
    if not present <= set(names):
        names, colours = names + ["other"], colours + [OTHER_COLOUR]
    return names, colours


def panels(r, title):
    # 3 x 3 figure: columns p left to right, rows q top to bottom; yields (axis, p index, q index)
    import matplotlib.pyplot as plt
    p_values, q_values = r["p_values"], r["q_values"]
    names = model_axes(r)
    fig, axes = plt.subplots(len(q_values), len(p_values), sharex=True, sharey=True,
                             figsize=(15, 11), squeeze=False)
    fig.suptitle(title)
    def each():
        for i, p in enumerate(p_values):
            for j, q in enumerate(q_values):
                ax = axes[j, i]
                ax.set_title(f"Phase diagram p = {p:g} q = {q:g}", fontsize=10)
                ax.set_xlabel(names["x_name"]); ax.set_ylabel(names["y_name"])
                yield ax, i, j
    return fig, axes, each()


def band_lines(b, x_max):
    # one-good reference lines, from B(z) = b1' z - b2' z^2 and x* = (b1' - 1) / (4 b2' - 2):
    #   upper: benefit still increasing at the largest pair total 2 x_max (valid below it)
    #   lower: x* inside [0, x_max] (valid above it)
    # with x_max = 0.5 both are b2' = b1'/2
    return b / (4 * x_max), (b - 1 + 2 * x_max) / (4 * x_max)


def line_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color="black", linewidth=1, linestyle="--", label="benefit increasing up to cap (below)"),
            Line2D([], [], color="black", linewidth=1, linestyle=":", label="x* inside [0, x_max] (above)")]


def draw_lines(ax, b, x_max):
    # the two reference lines, clipped to the plotted range; with SHADE_VALID_BAND, a grey overlay
    # on the cells outside the band (above the lower line and below the upper one)
    upper, lower = band_lines(b, x_max)
    extent = [b[0], b[-1], b[0], b[-1]]
    if SHADE_VALID_BAND:
        B1, B2 = np.meshgrid(b, b)                                   # cell centres, (b2', b1')
        upper_grid, lower_grid = band_lines(B1, x_max)
        outside = ~((B2 >= lower_grid) & (B2 <= upper_grid))
        overlay = np.zeros(B1.shape + (4,))
        overlay[outside] = (0.5, 0.5, 0.5, 0.4)
        ax.imshow(overlay, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    ax.plot(b, upper, color="black", linewidth=1, linestyle="--")
    ax.plot(b, lower, color="black", linewidth=1, linestyle=":")
    ax.set_xlim(b[0], b[-1]); ax.set_ylim(b[0], b[-1])


def exp_lines(r):
    # one-good reference lines for the exponential model, as (x points, y points, style, label),
    # keeping only those inside the plotted range
    b, beta = r["b"], r["beta"]
    lines = [(b, b, "--", "first unit of investment pays (below)")]
    for level, style, label in [(1, ":", "x* is a valley above, a peak below"), (2, "-.", "x* attracts below")]:
        if beta[0] <= level <= beta[-1]:
            lines.append(([b[0], b[-1]], [level, level], style, label))
    return lines


def draw_reference(ax, r):
    # the model's reference lines (SHADE_VALID_BAND is for the quadratic band only)
    if model(r) == "quadratic":
        draw_lines(ax, r["b"], r["x_max"])
        return
    for xs, ys, style, _ in exp_lines(r):
        ax.plot(xs, ys, color="black", linewidth=1, linestyle=style)
    ax.set_xlim(r["b"][0], r["b"][-1]); ax.set_ylim(r["beta"][0], r["beta"][-1])


def reference_handles(r):
    from matplotlib.lines import Line2D
    if model(r) == "quadratic":
        return line_handles()
    return [Line2D([], [], color="black", linewidth=1, linestyle=style, label=label) for _, _, style, label in exp_lines(r)]


def extent(r):
    x, y = model_axes(r)["x_axis"], model_axes(r)["y_axis"]
    return [x[0], x[-1], y[0], y[-1]]


def save(fig, filename):
    fig.savefig(filename, dpi=150)
    print(f"saved {filename}")


def plot_categories(labels, r, scheme, title, filename):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    names, colours = label_colours(labels, scheme)
    index = {l: k for k, l in enumerate(names)}
    codes = np.vectorize(lambda l: index.get(l, index.get("other")))(labels)
    fig, axes, each = panels(r, title)
    for ax, i, j in each:
        ax.imshow(codes[i, j], origin="lower", extent=extent(r), aspect="auto",
                  interpolation="nearest", cmap=ListedColormap(colours), vmin=-0.5, vmax=len(names) - 0.5)
        draw_reference(ax, r)
    handles = [Patch(color=c, label=l) for l, c in zip(names, colours)] + reference_handles(r)
    fig.legend(handles=handles, loc="center left", fontsize=8)
    fig.subplots_adjust(left=0.27, wspace=0.25, hspace=0.3)
    save(fig, filename)


def plot_scale(values, r, scheme, title, filename):
    fig, axes, each = panels(r, title)
    for ax, i, j in each:
        im = ax.imshow(values[i, j], origin="lower", extent=extent(r), aspect="auto",
                       interpolation="nearest", vmin=scheme.get("vmin"), vmax=scheme.get("vmax"))
        draw_reference(ax, r)
    fig.colorbar(im, ax=axes, label=scheme.get("label", "").format(scale=investment_scale(r)[1]))
    fig.legend(handles=reference_handles(r), loc="upper left", fontsize=8)
    save(fig, filename)


def print_counts(labels, r, scheme, title):
    names, _ = label_colours(labels, scheme)
    print(f"\n=== label counts: {title} ===")
    for i, p in enumerate(r["p_values"]):
        for j, q in enumerate(r["q_values"]):
            print(f"p = {p:g}, q = {q:g}:")
            values, counts = np.unique(labels[i, j], return_counts=True)
            for v, c in zip(values, counts):
                print(f"  {c:5d}  {v}" + ("" if v in names else "   <- other (grey)"))


# ---------- running a scheme ----------

def analyse(r, name):
    scheme = SCHEMES[name]
    values = scheme["outcome"](r["xs"], r["fs"], r)            # (start, p, q, row, column)
    base = f"{name}_{r['tag']}"

    if scheme["kind"] == "scale":
        plot_scale(values.mean(axis=0), r, scheme, f"{name}, averaged over starts - {r['tag']}", f"{base}.png")
        return

    if invested_rule(r) != default_rule(r):
        base += f"_invested-{invested_rule(r)}"              # not the model's default rule
    if "check" in scheme:
        print(f"{name} checks:"); scheme["check"](r)
    labels_raw = grid_labels(values)                           # (p, q, row, column)
    labels, rare, replaced = replace_rare(labels_raw, r["x_axis"], r["y_axis"])
    print(f"\nrare labels (fewer than {RARE_FRAC:.1%} of {labels_raw.size} points), replaced by their nearest neighbour:")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:5d}  {l}")
    print("points replaced per panel: " + ", ".join(
        f"p={p:g} q={q:g}: {replaced[i, j]}" for i, p in enumerate(r["p_values"]) for j, q in enumerate(r["q_values"])))
    if "compare" in scheme:
        print(); scheme["compare"](labels_raw, r)
    print_counts(labels_raw, r, scheme, f"{name}, unfiltered")
    print_counts(labels, r, scheme, f"{name}, filtered")
    plot_categories(labels, r, scheme, f"{name}, filtered (rare labels < {RARE_FRAC:.1%} replaced) - {r['tag']}",
                    f"{base}_filtered.png")
    plot_categories(labels_raw, r, scheme, f"{name}, unfiltered: what actually happened - {r['tag']}",
                    f"{base}_unfiltered.png")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help="results file (default: newest two_good_results_*.npz)")
    parser.add_argument("--scheme", nargs="+", default=["cc_dc_dd"], choices=list(SCHEMES),
                        help="one or more labelling schemes")
    parser.add_argument("--invested", choices=["half_cap", "absolute"],
                        help="rule for an invested good (default: the model's own)")
    args = parser.parse_args()
    if args.invested:
        INVESTED_RULE = args.invested

    r = load(args.results)
    for name in args.scheme:
        print(f"\n---------- {name} ----------")
        analyse(r, name)
    plt.show()
