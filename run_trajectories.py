# How the population evolves over time for a few (p, q) conditions, in the style of DHK's Fig. 1 time
# panels. Runs are recorded after every chunk and held at their final state up to a common end time
# (the longest run's finishing time by default). Two figures, each from its own runs:
#   trajectories_time_<tag>.png:   three runs per TIME_CONDITIONS entry, from TIME_STARTS (one line
#                                  style each); the split x1 - x2 against log time per condition, with
#                                  the totals as strips or shared
#   trajectories_plane_<tag>.png:  three runs per PLANE_CONDITIONS entry, from PLANE_STARTS; each
#                                  group's path in the (x1, x2) plane
# and the recorded data of each as trajectories_time_<tag>.npz and trajectories_plane_<tag>.npz.
# Every panel's third title line says which part of the division-of-labour rule its (p, q) satisfies.
# See USAGE.md.
#
# Usage:  python run_trajectories.py                       (totals as strips)
#         python run_trajectories.py --total shared
#         python run_trajectories.py --end-time 200          (default: the longest run's finishing time)
#         in Colab: %run run_trajectories.py

import argparse
import json
import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from coop import engine, models, plots, sweeps
from run_simulation import settings_for

# DHK's ESS point B, converted to the quadratic model; the benefit stays flat past its peak
B1, B2 = 7 / 4.6, 1.5
SETTINGS = dict(tol=1e-5, chunk=20)       # chunk: steps between records (20 steps = 0.2 time units)

# figure A: (p, q) and a title for each column, and the three starts drawn in every column (total 0.08),
# each with a line style: (on, off) dash lengths in points, None for solid. The balanced start is
# exactly on the diagonal: the engine's saddle push and split test take it off
TIME_CONDITIONS = [((1.0, 1.0),   "DHK"),
                   ((0.95, 1.0),  "benefit favours balance"),
                   ((0.95, 1.05), "both favour balance"),
                   ((1.0, 0.95),  "cost favours concentration"),
                   ((1.05, 0.95), "both favour concentration"),
                   ((0.9, 0.92),  "division of labour")]
TIME_STARTS = [((0.04, 0.04),   None,       "balanced"),
               ((0.05, 0.03),   (4, 2.5),   "slightly towards good 1"),
               ((0.075, 0.005), (1, 1.8),   "strongly towards good 1")]

# figure B: (p, q) and a title for each panel, and the three starts drawn on every panel (same total)
PLANE_CONDITIONS = [((1.0, 1.0),  "DHK"),
                    ((0.95, 1.0), "benefit favours balance"),
                    ((1.0, 0.95), "cost favours concentration"),
                    ((0.9, 0.92), "both: division of labour")]
PLANE_STARTS = [(0.07, 0.01), (0.04, 0.04), (0.01, 0.07)]     # towards good 1, balanced, towards good 2

SPLIT_LIMIT = 0.3                    # figure A: the split panels' x-axis, -0.3 to 0.3
PLANE_LIMIT = 0.3                    # figure B: x1 and x2 from 0 to 0.3
ARROW_TIMES = [0.5, 2, 8, 32, 128]   # figure B: arrowheads where each group is at these times
LEAN = LinearSegmentedColormap.from_list("lean", ["#E24B4A", "#7F77DD", "#378ADD"])   # lean -1, 0, +1
TIME_WIDTH = lambda f: 1 + f                         # figure A: line width, 1 to 2 pt by share
PLANE_WIDTH = lambda f: np.maximum(3 * f, 0.5)       # figure B: line width, up to 3 pt by share


def rule_part(p, q):
    # which part of the division-of-labour rule (p, q) satisfies
    if p == q == 1:
        return "neutral"
    if q < p:
        return "q < p"
    return "q > (p+1)/2" if q > (p + 1) / 2 else "p < q < (p+1)/2"


def title(pq, name):
    return f"{name}\np = {pq[0]:g}, q = {pq[1]:g}\n{rule_part(*pq)}"


# ---------- simulate and record ----------

def make_sweep(conditions):
    # p and q as panels (every combination of the conditions' values; only the conditions are used),
    # b1' and b2' fixed: a grid of one point
    ps = sorted({p for (p, _), _ in conditions})
    qs = sorted({q for (_, q), _ in conditions})
    return sweeps.Sweep("trajectories", panels=(("p", ps), ("q", qs)),
                        grid=(("b2", sweeps.Axis(B2, B2)), ("b1", sweeps.Axis(B1, B1))), panel_rows="q")


def simulate(conditions, starts):
    # one run per (condition, start); the engine's start slots (4) are set to starts. Returns, per
    # condition and start, the recorded times, xs and fs
    model = models.make_model("quadratic", flat_benefit=True)
    sweep = make_sweep(conditions)
    s = settings_for(model, sweep, **SETTINGS)
    if len(starts) > len(s.starts):
        raise SystemExit(f"at most {len(s.starts)} starts per condition")
    (_, ps), (_, qs) = sweep.panels
    shape = sweeps.layout(sweep, 1, len(s.starts))[0]                     # (start, p, q, 1, 1)
    ids = np.array([[np.ravel_multi_index((k, ps.index(p), qs.index(q), 0, 0), shape) for k in range(len(starts))]
                    for (p, q), _ in conditions])                           # (condition, start)
    start_of = np.unravel_index(np.arange(np.prod(shape)), shape)[0]       # each run's start slot
    frames = []

    def record(runs):
        if not frames:          # first call: the runs were just created; set the starts
            for k, start in enumerate(starts):
                runs.xs[start_of == k] = start
        frames.append((runs.stage_time[ids].copy(), runs.xs[ids].copy(), runs.fs[ids].copy()))

    engine.simulate(model, sweep, s, 1, record=record)
    times, xs, fs = (np.stack(a, axis=2) for a in zip(*frames))           # (condition, start, frame, ...)
    out = []
    for c in range(len(conditions)):
        # keep frame 0 and each frame where this run's time moved on (others advanced other runs)
        keep = [np.concatenate([[True], np.diff(times[c, k]) > 0]) for k in range(len(starts))]
        out.append([dict(times=times[c, k][kk], xs=xs[c, k][kk], fs=fs[c, k][kk]) for k, kk in enumerate(keep)])
    return out


def hold_until(d, t_end):
    # a finished run keeps its final state: add a last frame at t_end, so every line reaches the top
    if d["times"][-1] >= t_end:
        return d
    return dict(times=np.append(d["times"], t_end), xs=np.concatenate([d["xs"], d["xs"][-1:]]),
                fs=np.concatenate([d["fs"], d["fs"][-1:]]))


def held(data, end_time):
    # every run held up to end_time (default: the longest run's finishing time); and the finishing times
    finished = [[d["times"][-1] for d in runs] for runs in data]
    t_end = end_time or max(max(f) for f in finished)
    return [[hold_until(d, t_end) for d in runs] for runs in data], finished


def save_data(data, finished, meta, filename):
    arrays = {f"{k}_{c}_{s}": v for c, runs in enumerate(data) for s, d in enumerate(runs) for k, v in d.items()}
    np.savez(filename, **arrays, meta=json.dumps(dict(meta, finished=finished)))
    print(f"saved {filename}")


# ---------- plot ----------

def paths(fs):
    # every group's path as a list of (frame, slot): a slot's frames while it is alive, starting, for a
    # group that appeared in a split, from the nearest group alive the frame before (so it forks)
    out, current = [], {}
    for k in range(len(fs)):
        for i in range(fs.shape[1]):
            if fs[k, i] > 0 and i in current:
                current[i].append((k, i))
            elif fs[k, i] > 0:
                current[i] = [(k, i)]
            elif i in current:
                out.append(current.pop(i))
    return out + list(current.values())


def with_parents(d, groups):
    # prefix each path that starts after frame 0 with the nearest group alive the frame before
    xs, fs = d["xs"], d["fs"]
    out = []
    for path in groups:
        k, i = path[0]
        alive_before = np.flatnonzero(fs[k - 1] > 0) if k > 0 else []
        if len(alive_before):
            parent = alive_before[np.argmin(np.linalg.norm(xs[k - 1, alive_before] - xs[k, i], axis=-1))]
            path = [(k - 1, parent)] + path
        out.append(path)
    return out


def lines(ax, d, point, width, dashes=None, colour=None, first=1):
    # a LineCollection of every group's path through point(k, i), each edge coloured by lean (or colour)
    # and as wide as width(share); dashes: (on, off) in points, cut along each path's length on screen
    # (ax's limits and scales must be set first); first: the first frame drawn (1 leaves out time 0 on
    # a log axis)
    lean = lean_of(d["xs"])
    to_points = 72 / ax.figure.dpi
    segs, colours, widths = [], [], []
    for path in with_parents(d, paths(d["fs"])):
        path = [(k, i) for k, i in path if k >= first]
        length = 0.0                                    # along the path, in points
        for (k0, i0), (k1, i1) in zip(path, path[1:]):
            a, b = np.array(point(k0, i0), float), np.array(point(k1, i1), float)
            c = LEAN((lean[k1, i1] + 1) / 2) if colour is None else colour
            w = width(d["fs"][k1, i1])
            if dashes is None:
                segs.append([a, b]); colours.append(c); widths.append(w)
                continue
            screen = ax.transData.transform([a, b]) * to_points
            n = max(1, int(np.ceil(np.linalg.norm(screen[1] - screen[0]) / 0.5)))     # pieces of 0.5 pt at most
            pieces = ax.transData.inverted().transform(np.linspace(*screen, n + 1) / to_points)
            step = np.linalg.norm(screen[1] - screen[0]) / n
            for j in range(n):
                if (length + (j + 0.5) * step) % sum(dashes) < dashes[0]:
                    segs.append(pieces[j:j + 2]); colours.append(c); widths.append(w)
            length += n * step
    return LineCollection(segs, colors=colours, linewidths=widths, capstyle="butt" if dashes else "round")


def lean_of(xs):
    # (x1 - x2) / (x1 + x2): -1 all good 2, 0 equal, +1 all good 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((xs[..., 0] - xs[..., 1]) / (xs[..., 0] + xs[..., 1]))


def mean_total(d):
    # share-weighted total investment x1 + x2 per frame
    return np.sum(d["fs"] * d["xs"].sum(axis=-1), axis=-1)


def style_legend(fig):
    handles = [Line2D([], [], color="grey", linewidth=1.5, dashes=dashes or (None, None), label=name)
               for _, dashes, name in TIME_STARTS]
    fig.legend(handles=handles, loc="upper right", fontsize=7, frameon=False, ncol=len(handles))


def plot_time(data, total, filename):
    # A: one split panel per condition (x1 - x2 against log time, going up) with all its runs, one line
    # style per start; totals x1 + x2 as a narrow strip beside each (total "strips"), or one panel below
    # all the columns with each condition's balanced run overlaid ("shared")
    every = [d for runs in data for d in runs]
    t_min, t_max = min(d["times"][1] for d in every), max(d["times"][-1] for d in every)
    top_total = 1.1 * max(np.max(d["xs"].sum(axis=-1)) for d in every)
    n = len(data)
    strips = total == "strips"
    fig = plt.figure(figsize=(2.9 * n + 0.6, 6.5 if strips else 8.5))
    if strips:          # per condition: split, total (half as wide), gap
        grid = fig.add_gridspec(1, 3 * n - 1, width_ratios=([2, 1, 0.45] * n)[:-1], wspace=0.08,
                                left=0.05, right=0.98, top=0.8, bottom=0.14)
    else:
        grid = fig.add_gridspec(2, n, height_ratios=[3, 1.2], hspace=0.5, wspace=0.35,
                                left=0.06, right=0.98, top=0.85, bottom=0.1)
    for c, (runs, (pq, name)) in enumerate(zip(data, TIME_CONDITIONS)):
        ax = fig.add_subplot(grid[0, 3 * c] if strips else grid[0, c])
        ax.set_xlim(-SPLIT_LIMIT, SPLIT_LIMIT)
        ax.set_yscale("log")
        ax.set_ylim(t_min, t_max)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_xlabel("split x1 - x2", fontsize=8)
        if c == 0:
            ax.set_ylabel("time")
        ax.tick_params(labelsize=7)
        ax.set_title(title(pq, name), fontsize=9, loc="left" if strips else "center")
        if strips:
            strip = fig.add_subplot(grid[0, 3 * c + 1], sharey=ax)
            strip.set_xlim(0, top_total)
            strip.set_xlabel("total", fontsize=8)
            strip.tick_params(labelsize=7, labelleft=False)
        for d, (_, dashes, _) in zip(runs, TIME_STARTS):
            diff, totals = d["xs"][..., 0] - d["xs"][..., 1], d["xs"].sum(axis=-1)
            ax.add_collection(lines(ax, d, lambda k, i: (diff[k, i], d["times"][k]), TIME_WIDTH, dashes))
            if strips:
                strip.add_collection(lines(strip, d, lambda k, i: (totals[k, i], d["times"][k]), TIME_WIDTH, dashes))

    if not strips:
        total_ax = fig.add_subplot(grid[1, :])
        for runs, (pq, name), style in zip(data, TIME_CONDITIONS, ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 3))]):
            d = runs[0]
            total_ax.plot(d["times"][1:], mean_total(d)[1:], linestyle=style, color="black", linewidth=1,
                          label=f"{name} (p = {pq[0]:g}, q = {pq[1]:g})")
        total_ax.set_xscale("log")
        total_ax.set_xlim(t_min, t_max)
        total_ax.set_xlabel("time")
        total_ax.set_ylabel(f"total x1 + x2, share-weighted\n({TIME_STARTS[0][2]} start)", fontsize=8)
        total_ax.tick_params(labelsize=7)
        total_ax.legend(fontsize=7, frameon=False, ncol=2)
    style_legend(fig)
    fig.text(0.5, 0.01, "Three starts per condition: balanced (solid), slightly towards good 1 (dashed), strongly "
             "towards good 1 (dotted). Time in model units, log scale.", ha="center", fontsize=9)
    plots.save(fig, filename)


def plot_plane(data, filename):
    # B: one (x1, x2) panel per condition with all its runs: each group's path, arrowheads along it,
    # a hollow circle at each start and a dot sized by share where each group ends
    balanced = next(k for k, (x1, x2) in enumerate(PLANE_STARTS) if x1 == x2)
    reference = next((runs[balanced] for runs, (pq, _) in zip(data, PLANE_CONDITIONS) if pq == (1.0, 1.0)), None)
    same_total = mean_total(reference)[-1] if reference is not None else None
    fig, axes = plt.subplots(1, len(data), figsize=(3.3 * len(data) + 1, 4.6), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.78, bottom=0.16, wspace=0.3)
    for ax, runs, (pq, name) in zip(axes[0], data, PLANE_CONDITIONS):
        ax.set_xlim(0, PLANE_LIMIT)
        ax.set_ylim(0, PLANE_LIMIT)
        ax.set_aspect("equal")
        ax.plot([0, PLANE_LIMIT], [0, PLANE_LIMIT], color="grey", linewidth=0.5, alpha=0.5)
        if same_total is not None:
            ax.plot([0, same_total], [same_total, 0], color="grey", linestyle="--", linewidth=0.8,
                    label="same total as p = q = 1")
        for d in runs:
            xs, fs, times = d["xs"], d["fs"], d["times"]
            ax.add_collection(lines(ax, d, lambda k, i: tuple(xs[k, i]), PLANE_WIDTH, first=0))
            for t in ARROW_TIMES:           # arrowheads where each group is at these times
                k = np.searchsorted(times, t)
                if 0 < k < len(times):
                    for i in np.flatnonzero((fs[k] > 0) & (fs[k - 1] > 0)):
                        if np.linalg.norm(xs[k, i] - xs[k - 1, i]) > 1e-4:
                            ax.annotate("", xy=xs[k, i], xytext=xs[k - 1, i], zorder=4,
                                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.5,
                                                        mutation_scale=10, shrinkA=0, shrinkB=0))
            ax.scatter(*xs[0, 0], s=25, facecolors="none", edgecolors="black", linewidths=0.7, zorder=3)
            alive, lean = fs[-1] > 0, lean_of(xs)
            ax.scatter(xs[-1, alive, 0], xs[-1, alive, 1], s=60 * fs[-1, alive], c=LEAN((lean[-1, alive] + 1) / 2),
                       edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_xlabel("x1 (good 1)", fontsize=8)
        ax.set_ylabel("x2 (good 2)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title(pq, name), fontsize=9)
    if same_total is not None:
        axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=LEAN), cax=fig.add_axes([0.91, 0.25, 0.012, 0.45]))
    bar.set_ticks([-1, 0, 1], labels=["all good 2", "equal", "all good 1"])
    bar.set_label("lean (x1 - x2) / (x1 + x2)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.02, "Three starts per panel: towards good 1, balanced, towards good 2.", ha="center", fontsize=9)
    plots.save(fig, filename)


def print_final(data, finished, conditions, starts):
    for runs, times, (pq, name) in zip(data, finished, conditions):
        print(f"{name} (p = {pq[0]:g}, q = {pq[1]:g}, {rule_part(*pq)}):")
        for d, t, start in zip(runs, times, starts):
            alive = d["fs"][-1] > 0
            groups = ", ".join(f"({x1:.3f}, {x2:.3f}) share {f:.2f}" for (x1, x2), f in zip(d["xs"][-1][alive], d["fs"][-1][alive]))
            print(f"  from ({start[0]:g}, {start[1]:g}): finished at time {t:.0f}, final groups {groups}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", choices=["strips", "shared"], default="strips",
                        help="figure A: totals as a narrow strip beside each split panel, or in one panel below")
    parser.add_argument("--end-time", type=float, help="hold every run's final state up to this time "
                                                       "(default: the longest run's finishing time)")
    args = parser.parse_args()

    base = ["quadratic_flat", f"b1-{B1:.4g}", f"b2-{B2:g}", f"tol{SETTINGS['tol']:g}", f"chunk{SETTINGS['chunk']}"]
    end = [f"end{args.end_time:g}"] if args.end_time else []
    conditions = lambda cs: [f"p{p:g}-q{q:g}" for (p, q), _ in cs]
    time_tag = "_".join(base + conditions(TIME_CONDITIONS) + [f"total-{args.total}"] + end)
    plane_tag = "_".join(base + conditions(PLANE_CONDITIONS) + end)
    meta = dict(b1=B1, b2=B2, settings=SETTINGS)
    print("running on:", jax.devices())

    # figure A: three runs per condition, from TIME_STARTS
    starts = [start for start, _, _ in TIME_STARTS]
    data, finished = held(simulate(TIME_CONDITIONS, starts), args.end_time)
    save_data(data, finished, dict(meta, conditions=TIME_CONDITIONS, starts=starts), f"trajectories_time_{time_tag}.npz")
    plot_time(data, args.total, f"trajectories_time_{time_tag}.png")
    print_final(data, finished, TIME_CONDITIONS, starts)

    # figure B: three runs per condition, from PLANE_STARTS
    data, finished = held(simulate(PLANE_CONDITIONS, PLANE_STARTS), args.end_time)
    save_data(data, finished, dict(meta, conditions=PLANE_CONDITIONS, starts=PLANE_STARTS), f"trajectories_plane_{plane_tag}.npz")
    plot_plane(data, f"trajectories_plane_{plane_tag}.png")
    print_final(data, finished, PLANE_CONDITIONS, PLANE_STARTS)
