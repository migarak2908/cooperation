# Labels and plots a results file from run_simulation.py, with one or more labelling schemes
# (coop/schemes.py). Needs numpy, scipy and matplotlib only (no jax), so a results file from Colab
# can be analysed locally. Figures are saved to files, never shown. See USAGE.md for what to run.
#
# Usage:  python run_analysis.py                                  (newest results file, default scheme)
#         python run_analysis.py RESULTS.npz --scheme cc_dc_dd mean_investment
#         python run_analysis.py RESULTS.npz --invested absolute   (override the "invested" rule)
#         python run_analysis.py RESULTS.npz --scheme mean_investment --relative-to p=1,q=1   (minus that panel)
#         python run_analysis.py RESULTS.npz --scheme mean_investment --relative-to-point p=1,q=1
#                                                                  (minus that grid point, in each panel)
#         python run_analysis.py RESULTS.npz --starts 0 3          (only the runs from starts 0 and 3)
#         python run_analysis.py RESULTS.npz --scheme individual --hatch division_of_labour_index:0.5
#                                                                  (hatch where that scheme is above 0.5)
#         python run_analysis.py RESULTS.npz --layers              (quadratic b1_b2: total investment,
#                                                                  specialisation and branching in one figure)
#         python run_analysis.py individual_sim.npz                (from run_individual_sim.py: the
#                                                                  population across generations)
#         python run_analysis.py individual_sim.npz --video        (and an .mp4 of every individual)
#         in Colab: %run run_analysis.py RESULTS.npz

import argparse
import warnings
import numpy as np

from coop import layers, plots, references, results, schemes
from coop.results import panel_index, panel_value
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


def hatch_mask(r, hatch):
    # hatch = (scheme, threshold): where that scheme, averaged over starts, is above threshold (NaN: not),
    # and its legend label
    name, threshold = hatch
    if SCHEMES[name]["kind"] != "scale":
        raise SystemExit(f"--hatch needs a scale scheme, not {name}")
    with np.errstate(invalid="ignore"):
        mask = average_over_starts(SCHEMES[name]["outcome"](r["xs"], r["fs"], r)) > threshold
    return mask, SCHEMES[name].get("short", f"{name} > {threshold:g}")


def plot_individual(name, video=False, fps=25):
    # a file from run_individual_sim.py: each trait's distribution across generations, and with video
    # every individual moving through the (x1, x2) plane
    with np.load(name) as f:
        history = np.concatenate([f["pop0"][None], f["history"]])          # generation 0 first
        record_every, x_max = int(f["record_every"]), float(f["x_max"])
    plots.plot_population(history, record_every, x_max, name.replace(".npz", ".png"))
    if video:
        plots.population_video(history, record_every, x_max, name.replace(".npz", ".mp4"), fps)


def analyse_layers(r):
    # the layers figure for the quadratic b1_b2 sweep, with per-panel counts, a robustness check and a
    # sanity check of the specialisation at p = q = 1
    if r["model"] != "quadratic" or references.grid(r) != ("b2", "b1"):
        raise SystemExit("--layers is for the quadratic model's (b1', b2') grid")
    print("layers checks:"); layers.check_layers(r)
    layer = layers.layers(r)
    print("\nlayers per panel (from starts 0 and 3):")
    for (i, j), panel in panel_names(r).items():
        level, kind = layer["level"][i, j], layer["kind"][i, j]
        print(f"{panel}:")
        print("  level:    " + ", ".join(f"{l} {int((level == l).sum())}" for l in layers.LEVELS))
        print("  branched: " + ", ".join(f"{k} {int((kind == k).sum())}" for k in layers.KINDS[1:])
              + f"; three or more groups {int(layer['three'][i, j].sum())}")

    print("\nrobustness: points whose level / branching kind changes with every tolerance halved, doubled")
    tolerances = (layers.EDGE, layers.SAME_TOTAL, layers.LEAN_TOL)
    scaled = {factor: layers.layers(r, *(factor * tol for tol in tolerances)) for factor in [0.5, 2]}
    for ij, panel in panel_names(r).items():
        changes = [f"x {factor:g}: {int((s['level'][ij] != layer['level'][ij]).sum())} / "
                   f"{int((s['kind'][ij] != layer['kind'][ij]).sum())}" for factor, s in scaled.items()]
        print(f"  {panel:22s}" + "   ".join(changes))

    ij = panel_index(r, p=1.0, q=1.0)
    between = layer["level"][ij] == "in between" if ij else None
    if ij and between.any():
        print(f"\nspecialisation at p = 1, q = 1 over {int(between.sum())} 'in between' points: "
              f"mean {layer['specialisation'][ij][between].mean():.3f} (about 0 expected from balanced starts)")
    titles = {(i, j): f"{panel}: {references.predicted_outcome(panel_value(r, 'p', i, j), panel_value(r, 'q', i, j))}"
              for (i, j), panel in panel_names(r).items()}
    plots.plot_layers(layer, r, references.layer_lines(r), f"layers_{r['tag']}.png", titles)


def analyse(r, name, relative_to=None, point=None, hatch=None):
    scheme = SCHEMES[name]
    values = scheme["outcome"](r["xs"], r["fs"], r)            # (start, panel a, panel b, row, column)
    base = f"{name}_{r['tag']}"
    if schemes.invested_rule(r) != schemes.default_rule(r):
        base += f"_invested-{schemes.invested_rule(r)}"         # not the model's default rule
    if hatch and scheme["kind"] == "scale":
        base += f"_hatch-{hatch[0]}"
        hatch = hatch_mask(r, hatch)

    # 1. checks of the scheme on hand-made cases
    if "check" in scheme:
        print(f"{name} checks:"); scheme["check"](r)

    # 2. a scale: averaged over starts, optionally minus the panel with relative_to's values, or the
    #    grid point with point's values
    if scheme["kind"] == "scale":
        averaged = average_over_starts(values)
        if point:
            suffix = "-".join(f"{k}{v:g}" for k, v in point.items())
            plots.plot_scale(relative_point(averaged, r, point), r, scheme, f"{base}_relpt-{suffix}.png", relative_to=describe(point), point=point, hatch=hatch)
        elif relative_to:
            suffix = "-".join(f"{k}{v:g}" for k, v in relative_to.items())
            plots.plot_scale(relative(averaged, r, relative_to), r, scheme, f"{base}_rel-{suffix}.png", relative_to=describe(relative_to), hatch=hatch)
        else:
            plots.plot_scale(averaged, r, scheme, f"{base}.png", hatch=hatch)
        return
    if relative_to or point or hatch:
        print(f"(--relative-to, --relative-to-point and --hatch apply to scale schemes only; {name} is plotted as usual)")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help=f"results file (default: newest {results.PREFIX}*.npz)")
    parser.add_argument("--scheme", nargs="+", choices=list(SCHEMES),
                        help="one or more labelling schemes (default: cc_dc_dd_division on a (p, q) grid, else cc_dc_dd)")
    parser.add_argument("--invested", choices=["half_cap", "absolute"],
                        help="rule for an invested good (default: the model's own)")
    parser.add_argument("--relative-to", metavar="NAME=V[,NAME=V]",
                        help="scale schemes: plot each panel minus the panel with these values, e.g. p=1,q=1")
    parser.add_argument("--relative-to-point", metavar="NAME=V,NAME=V",
                        help="scale schemes: plot each grid point minus the grid point with these values, "
                             "in the same panel, e.g. p=1,q=1 on a (p, q) grid")
    parser.add_argument("--hatch", metavar="SCHEME:THRESHOLD",
                        help="scale schemes: hatch the grid points where SCHEME (averaged over starts) is above "
                             "THRESHOLD, e.g. division_of_labour_index:0.5")
    parser.add_argument("--layers", action="store_true",
                        help="quadratic b1_b2: total investment, specialisation and branching in one figure, "
                             "from the balanced starts 0 and 3")
    parser.add_argument("--video", action="store_true",
                        help="individual_sim.npz: also a video of every individual in the (x1, x2) plane")
    parser.add_argument("--fps", type=int, default=25, help="--video: frames (records) per second")
    parser.add_argument("--starts", nargs="+", type=int, metavar="I",
                        help="analyse only the runs from these starts (indices into the results' starts), e.g. 0 3")
    args = parser.parse_args()
    if args.results:
        with np.load(args.results) as f:
            individual_based = "history" in f.files
        if individual_based:
            plot_individual(args.results, args.video, args.fps)
            raise SystemExit

    r = results.load(args.results)
    r["invested_rule"] = args.invested
    if args.starts:
        r = keep_starts(r, args.starts)
    relative_to = parse_values(args.relative_to) if args.relative_to else None
    point = parse_values(args.relative_to_point) if args.relative_to_point else None
    if relative_to and point:
        raise SystemExit("give --relative-to or --relative-to-point, not both")
    hatch = None
    if args.hatch:
        hatch_name, threshold = args.hatch.split(":")
        if hatch_name not in SCHEMES:
            raise SystemExit(f"--hatch: no scheme {hatch_name!r} (schemes: {', '.join(SCHEMES)})")
        hatch = (hatch_name, float(threshold))
    if args.layers:
        if args.starts:
            raise SystemExit("--layers uses the balanced starts 0 and 3 itself: leave out --starts")
        analyse_layers(r)
    for name in args.scheme or ([] if args.layers else default_schemes(r)):
        print(f"\n---------- {name} ----------")
        analyse(r, name, relative_to, point, hatch)
