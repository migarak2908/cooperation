# Labelling and plotting of results saved by two_good_pq_simulate.py: one phase diagram over (p, q)
# per (b, beta) panel (panel rows b, panel columns beta), with a check of the division-of-labour
# prediction for beta < 1, b > beta:
#   q < p                    -> single specialists (DC)
#   p < q < (p + 1) / 2      -> division of labour (DC and DC')
#   q > (p + 1) / 2          -> generalists (CC)
# Needs numpy, scipy and matplotlib only (no jax). Figures are saved, never shown.
#
# A scheme turns the final state of each run into something to plot:
#   categories: a name per run ("CC and DD", ...); a grid point's label is the distinct names over
#               its starts, joined with " |or| "; rare labels are filtered out for a second plot
#   scale:      a number per run, averaged over starts and plotted with a colour bar
# To add a definition of cooperation: write a function like cc_dc_dd below and add it to SCHEMES.
#
# Usage:  python two_good_pq_analysis.py                       (newest results file, scheme cc_dc_dd_division)
#         python two_good_pq_analysis.py RESULTS.npz --scheme cc_dc_dd_division mean_investment
#         in Colab: %run two_good_pq_analysis.py --scheme cc_dc_dd_division

import argparse
import glob
import os
from collections import Counter
import numpy as np

LABEL_SHARE_MIN = 0.01               # a group counts if its (rounded) share is above this
RARE_FRAC = 0.003                    # labels on fewer than 0.3% of all grid points are rare
OTHER_COLOUR = "#9e9e9e"
INVESTED_MIN = 0.05                  # a good counts as invested above this


# ---------- loading ----------

def load(filename=None):
    # results of two_good_pq_simulate.py; xs: (start, b, beta, q, p, slot, good), fs: (..., slot)
    if filename is None:
        files = glob.glob("two_good_pq_results_*.npz")
        if not files:
            raise SystemExit("no two_good_pq_results_*.npz here: run two_good_pq_simulate.py first")
        filename = max(files, key=os.path.getmtime)
    r = dict(np.load(filename))
    r["tag"] = str(r["tag"])
    r["x_max"], r["x_min"] = float(r["x_max"]), float(r["x_min"])
    r["b_values"], r["beta_values"] = [float(v) for v in r["b_values"]], [float(v) for v in r["beta_values"]]
    print(f"loaded {filename}")
    not_converged = int((r["status"] == 2).sum())
    if not_converged:
        print(f"  note: {not_converged} of {r['status'].size} runs stopped at {int(r['max_steps'])} steps "
              f"without converging")
    return r


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
# A good is "invested" above INVESTED_MIN: a group is CC if it invests in both goods, DC in one, DD in
# neither. A run's outcome is the distinct group types, alphabetical, joined with " and ".

def cc_dc_dd(xs, fs, r, division=False):
    # xs: (..., K, 2), fs: (..., K) -> outcome name per run; with division, DC groups investing in
    # different goods make "DC and DC'"
    xs, fs, counted = counted_groups(xs, fs, r)
    invested = xs > INVESTED_MIN
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
    # hand-made final states: full investment at x_max, and cases either side of INVESTED_MIN
    m, t = r["x_max"], 1e-6
    half = INVESTED_MIN
    hi, lo = 1.2 * half, 0.8 * half
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
        (f"one group at ({hi:g}, {hi:g}), above {half:g}",   outcome([(hi, hi)], [1.0]),                        "CC"),
        (f"one group at ({lo:g}, {lo:g}), below {half:g}",   outcome([(lo, lo)], [1.0]),                        "DD"),
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


def compare_prediction(labels, r, division=False):
    # the division-of-labour prediction for each panel, away from its two boundary lines:
    #   q < p -> DC;  p < q < (p + 1)/2 -> DC and DC' (DC without division);  q > (p + 1)/2 -> CC (DD if x* is tiny)
    p_axis, q_axis = r["p_axis"], r["q_axis"]
    P, Q = np.meshgrid(p_axis, q_axis)                                    # (q, p)
    cell = max(p_axis[1] - p_axis[0], q_axis[1] - q_axis[0])
    boundary = (np.abs(Q - P) < cell) | (np.abs(Q - (P + 1) / 2) < cell)
    print("prediction check (unfiltered labels, points within one grid cell of q = p or q = (p + 1)/2 left out):")
    for i, b in enumerate(r["b_values"]):
        for j, beta in enumerate(r["beta_values"]):
            if beta >= 1 or b <= beta:
                print(f"  b' = {b:g}, beta' = {beta:g}: skipped, the prediction needs beta < 1 and b > beta")
                continue
            generalists = "CC" if r["x_star"][i, j] > INVESTED_MIN else "DD"
            predicted = np.select([Q < P, (P < Q) & (Q < (P + 1) / 2), Q > (P + 1) / 2],
                                  ["DC", "DC and DC'" if division else "DC", generalists], default="boundary")
            sim = labels[i, j]
            used = ~boundary & (predicted != "boundary")
            differ = used & (sim != predicted)
            print(f"  b' = {b:g}, beta' = {beta:g}: {int((used & ~differ).sum())} agree, {int(differ.sum())} differ")
            for (a, s), c in Counter(zip(predicted[differ], sim[differ])).most_common(5):
                print(f"    predicted {str(a)!r:14s} simulated {s!r:28s} {c:4d} points")


def compare_prediction_division(labels, r):
    return compare_prediction(labels, r, division=True)


# ---------- scheme: mean investment (a sliding scale) ----------

def mean_investment(xs, fs, r):
    # share-weighted average investment over groups and both goods, as a fraction of the panel's x*
    x_star = r["x_star"][:, :, None, None]                                 # (b, beta) -> (b, beta, q, p)
    return np.sum(fs[..., None] * xs, axis=(-2, -1)) / (2 * x_star)


# ---------- schemes ----------
# kind: "categories" or "scale"; for categories, optionally colours (fixed legend), check (run
# before labelling) and compare (run on the unfiltered grid labels)

SCHEMES = {
    "cc_dc_dd":          dict(kind="categories", outcome=cc_dc_dd, colours=cc_dc_dd_colours,
                              check=check_cc_dc_dd, compare=compare_prediction),
    "cc_dc_dd_division": dict(kind="categories", outcome=cc_dc_dd_division, colours=cc_dc_dd_colours,
                              check=check_cc_dc_dd, compare=compare_prediction_division),
    "mean_investment":   dict(kind="scale", outcome=mean_investment, label="mean investment / x*",
                              vmin=0, vmax=2),
}


# ---------- grid labels ----------

def grid_labels(outcomes):
    # outcomes: (start, ...) -> label per grid point: distinct outcomes, alphabetical, " |or| "
    labels = np.empty(outcomes.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        labels[idx] = " |or| ".join(sorted(set(outcomes[(slice(None),) + idx])))
    return labels


def replace_rare(labels, p_axis, q_axis):
    # labels on fewer than RARE_FRAC of all grid points (all panels together) are rare; each rare
    # point takes the label of the nearest non-rare point in the same panel
    from scipy.spatial import cKDTree
    values, counts = np.unique(labels, return_counts=True)
    rare = {v: int(c) for v, c in zip(values, counts) if c < RARE_FRAC * labels.size}
    P, Q = np.meshgrid(p_axis, q_axis)
    points = np.stack([P.ravel(), Q.ravel()], axis=-1)
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
    # one panel per (b, beta): rows b, columns beta; yields (axis, b index, beta index)
    import matplotlib.pyplot as plt
    b_values, beta_values = r["b_values"], r["beta_values"]
    fig, axes = plt.subplots(len(b_values), len(beta_values), sharex=True, sharey=True,
                             figsize=(4.5 * len(beta_values) + 4, 4 * len(b_values) + 1.5), squeeze=False)
    fig.suptitle(title)
    def each():
        for i, b in enumerate(b_values):
            for j, beta in enumerate(beta_values):
                ax = axes[i, j]
                ax.set_title(f"b' = {b:g}, beta' = {beta:g}", fontsize=10)
                ax.set_xlabel("p"); ax.set_ylabel("q")
                yield ax, i, j
    return fig, axes, each()


def draw_reference(ax, r):
    # q = p (dashed), q = (p + 1)/2 (dotted) and p = 1 (thin grey), clipped to the plotted range
    p = r["p_axis"]
    ax.plot(p, p, color="black", linewidth=1, linestyle="--")
    ax.plot(p, (p + 1) / 2, color="black", linewidth=1, linestyle=":")
    ax.axvline(1.0, color="grey", linewidth=0.6)
    ax.set_xlim(p[0], p[-1]); ax.set_ylim(r["q_axis"][0], r["q_axis"][-1])


def reference_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color="black", linewidth=1, linestyle="--", label="q = p"),
            Line2D([], [], color="black", linewidth=1, linestyle=":", label="q = (p + 1)/2"),
            Line2D([], [], color="grey", linewidth=0.6, label="p = 1")]


def extent(r):
    return [r["p_axis"][0], r["p_axis"][-1], r["q_axis"][0], r["q_axis"][-1]]


def save(fig, filename):
    import matplotlib.pyplot as plt
    fig.savefig(filename, dpi=150)
    plt.close(fig)
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
    handles = [Patch(color=c, label=l) for l, c in zip(names, colours)] + reference_handles()
    fig.legend(handles=handles, loc="center left", fontsize=8)
    fig.subplots_adjust(left=3.5 / fig.get_figwidth(), wspace=0.25, hspace=0.3, top=0.85)
    save(fig, filename)


def plot_scale(values, r, scheme, title, filename):
    fig, axes, each = panels(r, title)
    for ax, i, j in each:
        im = ax.imshow(values[i, j], origin="lower", extent=extent(r), aspect="auto",
                       interpolation="nearest", vmin=scheme.get("vmin"), vmax=scheme.get("vmax"))
        draw_reference(ax, r)
    fig.colorbar(im, ax=axes, label=scheme.get("label", ""))
    fig.legend(handles=reference_handles(), loc="lower left", fontsize=8)
    save(fig, filename)


def print_counts(labels, r, scheme, title):
    names, _ = label_colours(labels, scheme)
    print(f"\n=== label counts: {title} ===")
    for i, b in enumerate(r["b_values"]):
        for j, beta in enumerate(r["beta_values"]):
            print(f"b' = {b:g}, beta' = {beta:g}:")
            values, counts = np.unique(labels[i, j], return_counts=True)
            for v, c in zip(values, counts):
                print(f"  {c:5d}  {v}" + ("" if v in names else "   <- other (grey)"))


# ---------- running a scheme ----------

def analyse(r, name):
    scheme = SCHEMES[name]
    values = scheme["outcome"](r["xs"], r["fs"], r)            # (start, b, beta, q, p)
    base = f"{name}_{r['tag']}"

    if scheme["kind"] == "scale":
        plot_scale(values.mean(axis=0), r, scheme, f"{name}, averaged over starts - {r['tag']}", f"{base}.png")
        return

    if "check" in scheme:
        print(f"{name} checks:"); scheme["check"](r)
    labels_raw = grid_labels(values)                           # (b, beta, q, p)
    labels, rare, replaced = replace_rare(labels_raw, r["p_axis"], r["q_axis"])
    print(f"\nrare labels (fewer than {RARE_FRAC:.1%} of {labels_raw.size} points), replaced by their nearest neighbour:")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:5d}  {l}")
    print("points replaced per panel: " + ", ".join(
        f"b'={b:g} beta'={beta:g}: {replaced[i, j]}" for i, b in enumerate(r["b_values"])
        for j, beta in enumerate(r["beta_values"])))
    if "compare" in scheme:
        print(); scheme["compare"](labels_raw, r)
    print_counts(labels_raw, r, scheme, f"{name}, unfiltered")
    print_counts(labels, r, scheme, f"{name}, filtered")
    plot_categories(labels, r, scheme, f"{name}, filtered (rare labels < {RARE_FRAC:.1%} replaced) - {r['tag']}",
                    f"{base}_filtered.png")
    plot_categories(labels_raw, r, scheme, f"{name}, unfiltered: what actually happened - {r['tag']}",
                    f"{base}_unfiltered.png")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")            # save figures only

    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help="results file (default: newest two_good_pq_results_*.npz)")
    parser.add_argument("--scheme", nargs="+", default=["cc_dc_dd_division"], choices=list(SCHEMES),
                        help="one or more labelling schemes")
    args = parser.parse_args()

    r = load(args.results)
    for name in args.scheme:
        print(f"\n---------- {name} ----------")
        analyse(r, name)
