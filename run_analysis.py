# Labels and plots a results file from run_simulation.py, with one or more labelling schemes
# (coop/schemes.py). Needs numpy, scipy and matplotlib only (no jax), so a results file from Colab
# can be analysed locally. Figures are saved to files, never shown. See USAGE.md for what to run.
#
# Usage:  python run_analysis.py                                  (newest results file, default scheme)
#         python run_analysis.py RESULTS.npz --scheme cc_dc_dd mean_investment
#         python run_analysis.py RESULTS.npz --invested absolute   (override the "invested" rule)
#         python run_analysis.py RESULTS.npz --cut 0.7             (each group's lean, at grid column 0.7)
#         python run_analysis.py RESULTS.npz --compare OTHER.npz   (side by side, and where labels differ)
#         python run_analysis.py RESULTS.npz --scheme mean_investment --relative-to p=1,q=1   (minus that panel)
#         python run_analysis.py RESULTS.npz --scheme mean_investment --relative-to-point p=1,q=1
#                                                                  (minus that grid point, in each panel)
#         python run_analysis.py RESULTS.npz --summary             (fraction of each panel meeting SUMMARIES)
#         python run_analysis.py RESULTS.npz --starts 0 3          (only the runs from starts 0 and 3)
#         in Colab: %run run_analysis.py RESULTS.npz

import argparse
from collections import Counter
import warnings
import numpy as np

from coop import plots, references, results, schemes
from coop.schemes import SCHEMES, SUMMARIES, RARE_FRAC


def default_schemes(r):
    # division of labour is the point of the (p, q) grid; elsewhere plain CC / DC / DD
    return ["cc_dc_dd_division"] if references.grid(r) == ("q", "p") else ["cc_dc_dd"]


def panel_names(r):
    # "p = 0.8, q = 1" for each panel (i, j)
    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    return {(i, j): f"{r['labels'][a]} = {va:g}, {r['labels'][b]} = {vb:g}"
            for i, va in enumerate(a_values) for j, vb in enumerate(b_values)}


def print_counts(labels, r, scheme, title):
    names, _ = plots.label_colours(labels, scheme)
    print(f"\n=== label counts: {title} ===")
    for (i, j), panel in panel_names(r).items():
        print(f"{panel}:")
        values, counts = np.unique(labels[i, j], return_counts=True)
        for v, c in zip(values, counts):
            print(f"  {c:5d}  {v}" + ("" if v in names else "   <- other (grey)"))


def parse_values(text):
    # "p=1,q=1" -> {"p": 1.0, "q": 1.0}
    return {name: float(value) for name, value in (item.split("=") for item in text.split(","))}


def describe(values):
    return ", ".join(f"{name} = {v:g}" for name, v in values.items())


def average_over_starts(values):
    # NaN only where every start is NaN (e.g. nobody invests from any start)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                 # "Mean of empty slice"
        return np.nanmean(values, axis=0)


def keep_starts(r, indices):
    # r with only the runs from these starts (indices into r["starts"]); every array's first axis is the
    # start, so all analysis then uses just them; adds "_starts-0-3" to the tag, so to every figure name
    n = len(r["starts"])
    if not indices or any(not 0 <= s < n for s in indices):
        raise SystemExit(f"--starts: indices 0 to {n - 1}, one per start: "
                         + ", ".join(f"{s} = ({a:g}, {b:g})" for s, (a, b) in enumerate(r["starts"])))
    for k, v in r.items():
        if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n and k != "grid_axes":
            r[k] = v[indices]
    r["starts"] = [r["starts"][s] for s in indices]
    r["tag"] += "_starts-" + "-".join(str(s) for s in indices)
    print("using starts " + ", ".join(f"{s} = ({a:g}, {b:g})" for s, (a, b) in zip(indices, r["starts"])))
    return r


def relative(values, r, reference):
    # values (panel a, panel b, row, column) minus, at the same grid point, the panel with the reference
    # values; a panel parameter not in reference keeps the panel's own value
    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    if not set(reference) <= {a, b}:
        raise SystemExit(f"--relative-to: the panel parameters are {a} and {b}, not {', '.join(reference)}")
    def index(name, own_values, k):
        if name not in reference:
            return k
        if reference[name] not in own_values:
            raise SystemExit(f"--relative-to: no panel has {name} = {reference[name]:g} "
                             f"(its values: {', '.join(f'{v:g}' for v in own_values)})")
        return own_values.index(reference[name])
    out = np.empty_like(values)
    for i in range(len(a_values)):
        for j in range(len(b_values)):
            out[i, j] = values[i, j] - values[index(a, a_values, i), index(b, b_values, j)]
    return out


def relative_point(values, r, point):
    # values (panel a, panel b, row, column) minus, in each panel, the value at the grid point with
    # point's grid-parameter values
    (y, x), (y_axis, x_axis) = r["grid_names"], r["grid_axes"]
    if set(point) != {y, x}:
        raise SystemExit(f"--relative-to-point: give both grid parameters, {y} and {x}, e.g. {x}=1,{y}=1")
    def index(name, axis):
        found = np.flatnonzero(axis == point[name])
        if not len(found):
            raise SystemExit(f"--relative-to-point: the grid has no {name} = {point[name]:g} "
                             f"(its values: {', '.join(f'{v:g}' for v in axis)})")
        return found[0]
    return values - values[:, :, index(y, y_axis), index(x, x_axis)][:, :, None, None]


def analyse(r, name, relative_to=None, point=None):
    scheme = SCHEMES[name]
    values = scheme["outcome"](r["xs"], r["fs"], r)            # (start, panel a, panel b, row, column)
    base = f"{name}_{r['tag']}"
    if schemes.invested_rule(r) != schemes.default_rule(r):
        base += f"_invested-{schemes.invested_rule(r)}"         # not the model's default rule

    # 1. checks of the scheme on hand-made cases
    if "check" in scheme:
        print(f"{name} checks:"); scheme["check"](r)

    # 2. a scale: averaged over starts, optionally minus the panel with relative_to's values, or the
    #    grid point with point's values
    if scheme["kind"] == "scale":
        averaged = average_over_starts(values)
        if point:
            suffix = "-".join(f"{k}{v:g}" for k, v in point.items())
            plots.plot_scale(relative_point(averaged, r, point), r, scheme,
                             f"{name}, averaged over starts, minus at {describe(point)} in each panel - {r['tag']}",
                             f"{base}_relpt-{suffix}.png", relative_to=describe(point), point=point)
        elif relative_to:
            suffix = "-".join(f"{k}{v:g}" for k, v in relative_to.items())
            plots.plot_scale(relative(averaged, r, relative_to), r, scheme,
                             f"{name}, averaged over starts, minus at {describe(relative_to)} - {r['tag']}",
                             f"{base}_rel-{suffix}.png", relative_to=describe(relative_to))
        else:
            plots.plot_scale(averaged, r, scheme, f"{name}, averaged over starts - {r['tag']}", f"{base}.png")
        return
    if relative_to or point:
        print(f"(--relative-to and --relative-to-point apply to scale schemes only; {name} is plotted as usual)")

    # 3. categories: grid labels, and a filtered version with rare labels replaced
    labels_raw = schemes.grid_labels(values)                  # (panel a, panel b, row, column)
    labels, rare, replaced = schemes.replace_rare(labels_raw, r)
    print(f"\nrare labels (fewer than {RARE_FRAC:.1%} of {labels_raw.size} points), replaced by their nearest neighbour:")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:5d}  {l}")
    print("points replaced per panel: " + ", ".join(f"{panel}: {replaced[ij]}" for ij, panel in panel_names(r).items()))

    # 4. the analytic prediction, counts, and the two figures
    if "compare" in scheme:
        print(); scheme["compare"](labels_raw, r)
    print_counts(labels_raw, r, scheme, f"{name}, unfiltered")
    print_counts(labels, r, scheme, f"{name}, filtered")
    plots.plot_categories(labels, r, scheme, f"{name}, filtered (rare labels < {RARE_FRAC:.1%} replaced) - {r['tag']}",
                          f"{base}_filtered.png")
    plots.plot_categories(labels_raw, r, scheme, f"{name}, unfiltered: what actually happened - {r['tag']}",
                          f"{base}_unfiltered.png")


def summarise(r):
    # for each SUMMARIES entry, the fraction of each panel's grid points meeting its condition;
    # printed as a table (rows panels, columns entries) and saved as one figure
    # entries relative to a panel this sweep doesn't have (e.g. p = q = 1 on a (p, q) grid) are skipped
    entries = [e for e in SUMMARIES if not e[2] or set(e[2]) <= set(r["panel_names"])]
    for title, _, relative_to, *_ in [e for e in SUMMARIES if e not in entries]:
        print(f"summary: {title!r} skipped: relative to {describe(relative_to)}, "
              f"but the panel parameters are {', '.join(r['panel_names'])}")
    fractions = []
    for title, name, relative_to, comparison, threshold in entries:
        values = average_over_starts(SCHEMES[name]["outcome"](r["xs"], r["fs"], r))   # (panel a, panel b, row, column)
        if relative_to:
            values = relative(values, r, {k: float(v) for k, v in relative_to.items()})
        with np.errstate(invalid="ignore"):                                     # NaN: not met
            met = {"abs>": np.abs(values) > threshold, ">": values > threshold, "<": values < threshold}[comparison]
        fractions.append(met.mean(axis=(-2, -1)))
    width = max(len(t) for t, *_ in entries)
    print("\nsummary: fraction of each panel's grid points meeting each condition")
    print(" " * 22 + "".join(f"{t:>{width + 2}s}" for t, *_ in entries))
    for (i, j), panel in panel_names(r).items():
        print(f"{panel:22s}" + "".join(f"{f[i, j]:>{width + 2}.2f}" for f in fractions))
    plots.plot_summary(fractions, entries, r, f"summary_{r['tag']}.png")


def compare_results(r, other, name):
    # two results files on the same grid (e.g. with and without an engine option): unfiltered labels
    # side by side, and where they differ: in total, per panel, and the 5 most common changes
    same_grid = (r["panel_names"] == other["panel_names"] and r["panel_values"] == other["panel_values"]
                 and r["grid_names"] == other["grid_names"]
                 and all(np.array_equal(a, b) for a, b in zip(r["grid_axes"], other["grid_axes"]))
                 and r["xs"].shape[0] == other["xs"].shape[0])
    if not same_grid:
        raise SystemExit("--compare needs two results files with the same sweep, grid size and starts")
    scheme = SCHEMES[name]
    if scheme["kind"] != "categories":
        raise SystemExit(f"--compare needs a categories scheme, not {name}")
    left, right = (schemes.grid_labels(scheme["outcome"](x["xs"], x["fs"], x)) for x in (r, other))
    plots.plot_side_by_side(left, right, r, scheme, [r["tag"], other["tag"]],
                            f"compare_{name}_{r['tag']}_vs_{other['tag']}.png")
    differ = left != right
    print(f"\ngrid points whose unfiltered {name} label differs (of {left.size}): {int(differ.sum())}")
    print("  per panel: " + ", ".join(f"{panel}: {int(differ[ij].sum())}" for ij, panel in panel_names(r).items()))
    for (a, b), c in Counter(zip(left[differ], right[differ])).most_common(5):
        print(f"  {a} -> {b} ({c})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help=f"results file (default: newest {results.PREFIX}*.npz)")
    parser.add_argument("--scheme", nargs="+", choices=list(SCHEMES),
                        help="one or more labelling schemes (default: cc_dc_dd_division on a (p, q) grid, else cc_dc_dd)")
    parser.add_argument("--invested", choices=["half_cap", "absolute"],
                        help="rule for an invested good (default: the model's own)")
    parser.add_argument("--cut", type=float, help="plot each group's lean along the grid rows, at the grid column nearest this")
    parser.add_argument("--compare", metavar="OTHER", help="another results file on the same grid to compare with")
    parser.add_argument("--relative-to", metavar="NAME=V[,NAME=V]",
                        help="scale schemes: plot each panel minus the panel with these values, e.g. p=1,q=1")
    parser.add_argument("--relative-to-point", metavar="NAME=V,NAME=V",
                        help="scale schemes: plot each grid point minus the grid point with these values, "
                             "in the same panel, e.g. p=1,q=1 on a (p, q) grid")
    parser.add_argument("--summary", action="store_true", help="fraction of each panel meeting each SUMMARIES condition")
    parser.add_argument("--starts", nargs="+", type=int, metavar="I",
                        help="analyse only the runs from these starts (indices into the results' starts), e.g. 0 3")
    args = parser.parse_args()

    r = results.load(args.results)
    r["invested_rule"] = args.invested
    if args.starts:
        r = keep_starts(r, args.starts)
    relative_to = parse_values(args.relative_to) if args.relative_to else None
    point = parse_values(args.relative_to_point) if args.relative_to_point else None
    if relative_to and point:
        raise SystemExit("give --relative-to or --relative-to-point, not both")
    names = args.scheme or ([] if args.cut is not None or args.summary else default_schemes(r))
    if args.cut is not None:
        plots.plot_cut(r, args.cut)
    if args.summary:
        summarise(r)
    if args.compare:
        other = results.load(args.compare)
        other["invested_rule"] = args.invested
        if args.starts:
            other = keep_starts(other, args.starts)
        compare_results(r, other, (names or default_schemes(r))[0])
    else:
        for name in names:
            print(f"\n---------- {name} ----------")
            analyse(r, name, relative_to, point)
