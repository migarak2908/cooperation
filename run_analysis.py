# Labels and plots a results file from run_simulation.py, with one or more labelling schemes
# (coop/schemes.py). Needs numpy, scipy and matplotlib only (no jax), so a results file from Colab
# can be analysed locally. Figures are saved to files, never shown. See USAGE.md for what to run.
#
# Usage:  python run_analysis.py                                  (newest results file, default scheme)
#         python run_analysis.py RESULTS.npz --scheme cc_dc_dd mean_investment
#         python run_analysis.py RESULTS.npz --invested absolute   (override the "invested" rule)
#         python run_analysis.py RESULTS.npz --cut 0.7             (each group's lean, at grid column 0.7)
#         python run_analysis.py RESULTS.npz --compare OTHER.npz   (side by side, and where labels differ)
#         in Colab: %run run_analysis.py RESULTS.npz

import argparse
from collections import Counter
import numpy as np

from coop import plots, references, results, schemes
from coop.schemes import SCHEMES, RARE_FRAC


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


def analyse(r, name):
    scheme = SCHEMES[name]
    values = scheme["outcome"](r["xs"], r["fs"], r)            # (start, panel a, panel b, row, column)
    base = f"{name}_{r['tag']}"
    if schemes.invested_rule(r) != schemes.default_rule(r):
        base += f"_invested-{schemes.invested_rule(r)}"         # not the model's default rule

    # 1. checks of the scheme on hand-made cases
    if "check" in scheme:
        print(f"{name} checks:"); scheme["check"](r)

    # 2. a scale: averaged over starts
    if scheme["kind"] == "scale":
        plots.plot_scale(values.mean(axis=0), r, scheme, f"{name}, averaged over starts - {r['tag']}", f"{base}.png")
        return

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
    args = parser.parse_args()

    r = results.load(args.results)
    r["invested_rule"] = args.invested
    names = args.scheme or ([] if args.cut is not None else default_schemes(r))
    if args.cut is not None:
        plots.plot_cut(r, args.cut)
    if args.compare:
        other = results.load(args.compare)
        other["invested_rule"] = args.invested
        compare_results(r, other, (names or default_schemes(r))[0])
    else:
        for name in names:
            print(f"\n---------- {name} ----------")
            analyse(r, name)
