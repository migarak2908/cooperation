# How the population evolves over time for a few (p, q) conditions, in the style of DHK's Fig. 1 time
# panels. Runs are recorded after every chunk and held at their final state up to a common end time
# (the longest run's finishing time by default). Two figures, each from its own runs:
#   trajectories_time_<tag>.png:   one run per TIME_CONDITIONS entry, from --start; the split x1 - x2
#                                  against log time per condition, with the totals as strips or shared
#   trajectories_plane_<tag>.png:  three runs per PLANE_CONDITIONS entry, from PLANE_STARTS; each
#                                  group's path in the (x1, x2) plane
# and the recorded data of each as trajectories_time_<tag>.npz and trajectories_plane_<tag>.npz.
# See USAGE.md.
#
# Usage:  python run_trajectories.py                       (start (0.07, 0.01), totals as strips)
#         python run_trajectories.py --start 0.05,0.01 --total shared
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

from coop import engine, models, plots, sweeps
from run_simulation import settings_for

# DHK's ESS point B, converted to the quadratic model; the benefit stays flat past its peak
B1, B2 = 7 / 4.6, 1.5
SETTINGS = dict(tol=1e-5, chunk=20)       # chunk: steps between records (20 steps = 0.2 time units)

# figure A: (p, q) and a title for each column, and the start of every run
TIME_CONDITIONS = [((1.0, 1.0),   "DHK"),
                   ((0.95, 1.0),  "benefit favours balance"),
                   ((0.95, 1.05), "both favour balance"),
                   ((1.0, 0.95),  "cost favours concentration"),
                   ((1.05, 0.95), "both favour concentration"),
                   ((0.9, 0.92),  "division of labour")]
START = (0.07, 0.01)

# figure B: (p, q) and a title for each panel, and the three starts drawn on every panel (same total);
# the balanced start is exactly on the diagonal: the engine's saddle push and split test take it off
PLANE_CONDITIONS = [((1.0, 1.0),  "DHK"),
                    ((0.95, 1.0), "benefit favours balance"),
                    ((1.0, 0.95), "cost favours concentration"),
                    ((0.9, 0.92), "both: division of labour")]
PLANE_STARTS = [(0.07, 0.01), (0.04, 0.04), (0.01, 0.07)]     # towards good 1, balanced, towards good 2

SPLIT_LIMIT = 0.3                    # figure A: the split panels' x-axis, -0.3 to 0.3
PLANE_LIMIT = 0.3                    # figure B: x1 and x2 from 0 to 0.3
ARROW_TIMES = [0.5, 2, 8, 32, 128]   # figure B: arrowheads where each group is at these times
WIDTH = 3.0                          # line width of a group with share 1 (at least 0.5)
LEAN = LinearSegmentedColormap.from_list("lean", ["#E24B4A", "#7F77DD", "#378ADD"])   # lean -1, 0, +1


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

def steps(xs, fs):
    # every (frame k, group i, where it came from at frame k - 1) for each living group: itself, or for
    # a group that has just appeared (a split) the nearest group alive the frame before, so it forks
    out = []
    for k in range(1, len(fs)):
        alive_before = np.flatnonzero(fs[k - 1] > 0)
        for i in np.flatnonzero(fs[k] > 0):
            parent = i if fs[k - 1, i] > 0 else alive_before[np.argmin(np.linalg.norm(xs[k - 1, alive_before] - xs[k, i], axis=-1))]
            out.append((k, i, parent))
    return out


def lean_of(xs):
    # (x1 - x2) / (x1 + x2): -1 all good 2, 0 equal, +1 all good 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((xs[..., 0] - xs[..., 1]) / (xs[..., 0] + xs[..., 1]))


def lines(d, point, colour=None, first=1):
    # a LineCollection of every group's path, from point(k, i) to the next, width by share and coloured
    # by lean (or colour); first: the first frame drawn (1 leaves out time 0 on a log axis)
    lean = lean_of(d["xs"])
    used = [(k, i, parent) for k, i, parent in steps(d["xs"], d["fs"]) if k - 1 >= first]
    segs = [[point(k - 1, parent), point(k, i)] for k, i, parent in used]
    colours = colour or LEAN((np.array([lean[k, i] for k, i, _ in used]) + 1) / 2)
    widths = np.maximum([WIDTH * d["fs"][k, i] for k, i, _ in used], 0.5)
    return LineCollection(segs, colors=colours, linewidths=widths, capstyle="round")


def mean_total(d):
    # share-weighted total investment x1 + x2 per frame
    return np.sum(d["fs"] * d["xs"].sum(axis=-1), axis=-1)


def plot_time(data, total, filename):
    # A: one split panel per condition (x1 - x2 against log time, going up); totals x1 + x2 as a narrow
    # strip beside each (total "strips") or overlaid in one panel below all the columns ("shared")
    runs = [r[0] for r in data]
    t_min, t_max = runs[0]["times"][1], max(d["times"][-1] for d in runs)
    n = len(runs)
    strips = total == "strips"
    fig = plt.figure(figsize=(2.9 * n + 0.6, 6 if strips else 8))
    if strips:          # per condition: split, total (half as wide), gap
        grid = fig.add_gridspec(1, 3 * n - 1, width_ratios=([2, 1, 0.45] * n)[:-1], wspace=0.08,
                                left=0.06, right=0.98, top=0.86, bottom=0.14)
    else:
        grid = fig.add_gridspec(2, n, height_ratios=[3, 1.2], hspace=0.45, wspace=0.35,
                                left=0.07, right=0.98, top=0.9, bottom=0.1)
    for c, (d, ((p, q), title)) in enumerate(zip(runs, TIME_CONDITIONS)):
        ax = fig.add_subplot(grid[0, 3 * c] if strips else grid[0, c])
        diff = d["xs"][..., 0] - d["xs"][..., 1]
        ax.add_collection(lines(d, lambda k, i: (diff[k, i], d["times"][k])))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_xlim(-SPLIT_LIMIT, SPLIT_LIMIT)
        ax.set_yscale("log")
        ax.set_ylim(t_min, t_max)
        ax.set_xlabel("split x1 - x2", fontsize=8)
        if c == 0:
            ax.set_ylabel("time")
        ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\np = {p:g}, q = {q:g}", fontsize=9, loc="left" if strips else "center")
        if strips:
            strip = fig.add_subplot(grid[0, 3 * c + 1], sharey=ax)
            totals = d["xs"].sum(axis=-1)
            strip.add_collection(lines(d, lambda k, i: (totals[k, i], d["times"][k]), colour="grey"))
            strip.set_xlim(0, 1.1 * max(np.max(mean_total(r)) for r in runs))
            strip.set_xlabel("total", fontsize=8)
            strip.tick_params(labelsize=7, labelleft=False)

    if not strips:
        total_ax = fig.add_subplot(grid[1, :])
        for d, ((p, q), title), style in zip(runs, TIME_CONDITIONS, ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 3))]):
            total_ax.plot(d["times"][1:], mean_total(d)[1:], linestyle=style, color="black", linewidth=1,
                          label=f"{title} (p = {p:g}, q = {q:g})")
        total_ax.set_xscale("log")
        total_ax.set_xlim(t_min, t_max)
        total_ax.set_xlabel("time")
        total_ax.set_ylabel("total x1 + x2\n(share-weighted)", fontsize=8)
        total_ax.tick_params(labelsize=7)
        total_ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.text(0.5, 0.01, "All runs start towards good 1. Time in model units, log scale.", ha="center", fontsize=9)
    plots.save(fig, filename)


def plot_plane(data, filename):
    # B: one (x1, x2) panel per condition with all its runs: each group's path, arrowheads along it,
    # a hollow circle at each start and a dot sized by share where each group ends
    balanced = next(k for k, (x1, x2) in enumerate(PLANE_STARTS) if x1 == x2)
    reference = next((runs[balanced] for runs, (pq, _) in zip(data, PLANE_CONDITIONS) if pq == (1.0, 1.0)), None)
    same_total = mean_total(reference)[-1] if reference is not None else None
    fig, axes = plt.subplots(1, len(data), figsize=(3.3 * len(data) + 1, 4.3), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.82, bottom=0.17, wspace=0.3)
    for ax, runs, ((p, q), title) in zip(axes[0], data, PLANE_CONDITIONS):
        ax.plot([0, PLANE_LIMIT], [0, PLANE_LIMIT], color="grey", linewidth=0.5, alpha=0.5)
        if same_total is not None:
            ax.plot([0, same_total], [same_total, 0], color="grey", linestyle="--", linewidth=0.8,
                    label="same total as p = q = 1")
        for d in runs:
            xs, fs, times = d["xs"], d["fs"], d["times"]
            ax.add_collection(lines(d, lambda k, i: tuple(xs[k, i]), first=0))
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
        ax.set_xlim(0, PLANE_LIMIT)
        ax.set_ylim(0, PLANE_LIMIT)
        ax.set_aspect("equal")
        ax.set_xlabel("x1 (good 1)", fontsize=8)
        ax.set_ylabel("x2 (good 2)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\np = {p:g}, q = {q:g}", fontsize=9)
    if same_total is not None:
        axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=LEAN), cax=fig.add_axes([0.91, 0.25, 0.012, 0.5]))
    bar.set_ticks([-1, 0, 1], labels=["all good 2", "equal", "all good 1"])
    bar.set_label("lean (x1 - x2) / (x1 + x2)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.02, "Three starts per panel: towards good 1, balanced, towards good 2.", ha="center", fontsize=9)
    plots.save(fig, filename)


def print_final(data, finished, conditions, starts):
    for runs, times, ((p, q), title) in zip(data, finished, conditions):
        print(f"{title} (p = {p:g}, q = {q:g}):")
        for d, t, start in zip(runs, times, starts):
            alive = d["fs"][-1] > 0
            groups = ", ".join(f"({x1:.3f}, {x2:.3f}) share {f:.2f}" for (x1, x2), f in zip(d["xs"][-1][alive], d["fs"][-1][alive]))
            print(f"  from ({start[0]:g}, {start[1]:g}): finished at time {t:.0f}, final groups {groups}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=",".join(f"{v:g}" for v in START), metavar="X1,X2",
                        help=f"figure A: the start of every run (default {START[0]:g},{START[1]:g})")
    parser.add_argument("--total", choices=["strips", "shared"], default="strips",
                        help="figure A: totals as a narrow strip beside each split panel, or in one panel below")
    parser.add_argument("--end-time", type=float, help="hold every run's final state up to this time "
                                                       "(default: the longest run's finishing time)")
    args = parser.parse_args()
    start = tuple(float(v) for v in args.start.split(","))

    base = ["quadratic_flat", f"b1-{B1:.4g}", f"b2-{B2:g}", f"tol{SETTINGS['tol']:g}", f"chunk{SETTINGS['chunk']}"]
    end = [f"end{args.end_time:g}"] if args.end_time else []
    conditions = lambda cs: [f"p{p:g}-q{q:g}" for (p, q), _ in cs]
    time_tag = "_".join(base + conditions(TIME_CONDITIONS) + [f"start-{start[0]:g}-{start[1]:g}", f"total-{args.total}"] + end)
    plane_tag = "_".join(base + conditions(PLANE_CONDITIONS) + end)
    meta = dict(b1=B1, b2=B2, settings=SETTINGS)
    print("running on:", jax.devices())

    # figure A: one run per condition from start
    data, finished = held(simulate(TIME_CONDITIONS, [start]), args.end_time)
    save_data(data, finished, dict(meta, conditions=TIME_CONDITIONS, starts=[start]), f"trajectories_time_{time_tag}.npz")
    plot_time(data, args.total, f"trajectories_time_{time_tag}.png")
    print_final(data, finished, TIME_CONDITIONS, [start])

    # figure B: three runs per condition, from PLANE_STARTS
    data, finished = held(simulate(PLANE_CONDITIONS, PLANE_STARTS), args.end_time)
    save_data(data, finished, dict(meta, conditions=PLANE_CONDITIONS, starts=PLANE_STARTS), f"trajectories_plane_{plane_tag}.npz")
    plot_plane(data, f"trajectories_plane_{plane_tag}.png")
    print_final(data, finished, PLANE_CONDITIONS, PLANE_STARTS)
