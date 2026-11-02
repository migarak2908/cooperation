# How the population evolves over time for a few (p, q) conditions, in the style of DHK's Fig. 1 time
# panels: one run per condition, all from the same slightly unbalanced start, recorded after every
# chunk, and held at its final state up to a common end time. Saves two figures from the same runs,
#   trajectories_time_<tag>.png:   the split x1 - x2 against log time per condition, and the totals
#   trajectories_plane_<tag>.png:  each group's path in the (x1, x2) plane per condition
# and the recorded data as trajectories_<tag>.npz. See USAGE.md.
#
# Usage:  python run_trajectories.py                       (start (0.04, 0.01))
#         python run_trajectories.py --start 0.03,0.01
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
# (p, q) and a title for each column
CONDITIONS = [((1.0, 1.0),   "DHK"),
              ((0.95, 1.0),  "benefit favours balance"),
              ((1.0, 0.95),  "cost favours concentration"),
              ((0.9, 0.92),  "both: division of labour")]
START = (0.04, 0.01)

SPLIT_LIMIT = 0.3                    # figure A: the split panels' x-axis, -0.3 to 0.3
PLANE_LIMIT = 0.3                    # figure B: x1 and x2 from 0 to 0.3
ARROW_TIMES = [0.5, 2, 8, 32, 128]   # figure B: arrowheads where each group is at these times
WIDTH = 3.0                          # line width of a group with share 1
LEAN = LinearSegmentedColormap.from_list("lean", ["#E24B4A", "#7F77DD", "#378ADD"])   # lean -1, 0, +1


# ---------- simulate and record ----------

def make_sweep():
    # p and q as panels (every combination of the conditions' values; only the conditions are used),
    # b1' and b2' fixed: a grid of one point
    ps = sorted({p for (p, _), _ in CONDITIONS})
    qs = sorted({q for (_, q), _ in CONDITIONS})
    return sweeps.Sweep("trajectories", panels=(("p", ps), ("q", qs)),
                        grid=(("b2", sweeps.Axis(B2, B2)), ("b1", sweeps.Axis(B1, B1))), panel_rows="q")


def simulate(start):
    # one run per condition from start; returns, per condition, the recorded times, xs and fs
    model = models.make_model("quadratic", flat_benefit=True)
    sweep = make_sweep()
    s = settings_for(model, sweep, **SETTINGS)
    (_, ps), (_, qs) = sweep.panels
    shape = sweeps.layout(sweep, 1, len(s.starts))[0]                     # (start, p, q, 1, 1)
    ids = [np.ravel_multi_index((0, ps.index(p), qs.index(q), 0, 0), shape) for (p, q), _ in CONDITIONS]
    frames = []

    def record(runs):
        if not frames:          # first call: the runs were just created; every run starts at start
            runs.xs[:] = start
        frames.append((runs.stage_time[ids].copy(), runs.xs[ids].copy(), runs.fs[ids].copy()))

    engine.simulate(model, sweep, s, 1, record=record)
    times, xs, fs = (np.stack(a, axis=1) for a in zip(*frames))           # (condition, frame, ...)
    out = []
    for c in range(len(ids)):
        # keep frame 0 and each frame where this run's time moved on (others advanced other runs)
        keep = np.concatenate([[True], np.diff(times[c]) > 0])
        out.append(dict(times=times[c][keep], xs=xs[c][keep], fs=fs[c][keep]))
    return out


# ---------- plot ----------

def hold_until(d, t_end):
    # a finished run keeps its final state: add a last frame at t_end, so every line reaches the top
    if d["times"][-1] >= t_end:
        return d
    return dict(times=np.append(d["times"], t_end), xs=np.concatenate([d["xs"], d["xs"][-1:]]),
                fs=np.concatenate([d["fs"], d["fs"][-1:]]))


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


def lines(d, point, first=1):
    # a LineCollection of every group's path, from point(k, i) to the next, coloured by lean, width by share;
    # first: the first frame drawn (1 leaves out time 0 on a log axis)
    lean = lean_of(d["xs"])
    used = [(k, i, parent) for k, i, parent in steps(d["xs"], d["fs"]) if k - 1 >= first]
    segs = [[point(k - 1, parent), point(k, i)] for k, i, parent in used]
    leans = np.array([lean[k, i] for k, i, _ in used])
    widths = np.array([WIDTH * d["fs"][k, i] for k, i, _ in used])
    return LineCollection(segs, colors=LEAN((leans + 1) / 2), linewidths=np.maximum(widths, 0.5), capstyle="round")


def mean_total(d):
    # share-weighted total investment x1 + x2 per frame
    return np.sum(d["fs"] * d["xs"].sum(axis=-1), axis=-1)


def plot_time(data, filename):
    # A: one split panel per condition (x1 - x2 against log time, going up), and below one shared panel
    # of each condition's share-weighted total against log time
    t_min, t_max = data[0]["times"][1], max(d["times"][-1] for d in data)
    fig = plt.figure(figsize=(2.6 * len(data) + 0.6, 8))
    grid = fig.add_gridspec(2, len(data), height_ratios=[3, 1.2], hspace=0.45, wspace=0.35,
                            left=0.09, right=0.97, top=0.9, bottom=0.1)
    for c, (d, ((p, q), title)) in enumerate(zip(data, CONDITIONS)):
        ax = fig.add_subplot(grid[0, c])
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
        ax.set_title(f"{title}\np = {p:g}, q = {q:g}", fontsize=9)

    total_ax = fig.add_subplot(grid[1, :])
    for d, ((p, q), title), style in zip(data, CONDITIONS, ["-", "--", ":", "-."]):
        total_ax.plot(d["times"][1:], mean_total(d)[1:], linestyle=style, color="black", linewidth=1,
                      label=f"{title} (p = {p:g}, q = {q:g})")
    total_ax.set_xscale("log")
    total_ax.set_xlim(t_min, t_max)
    total_ax.set_xlabel("time")
    total_ax.set_ylabel("total x1 + x2\n(share-weighted)", fontsize=8)
    total_ax.tick_params(labelsize=7)
    total_ax.legend(fontsize=7, frameon=False)
    fig.text(0.5, 0.01, "All runs start slightly towards good 1. Time in model units.", ha="center", fontsize=9)
    plots.save(fig, filename)


def plot_plane(data, filename):
    # B: one (x1, x2) panel per condition: each group's path, arrowheads along it, a dot where it ends
    reference = next((d for d, (pq, _) in zip(data, CONDITIONS) if pq == (1.0, 1.0)), None)
    same_total = mean_total(reference)[-1] if reference is not None else None
    fig, axes = plt.subplots(1, len(data), figsize=(3.3 * len(data) + 1, 4.3), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.82, bottom=0.17, wspace=0.3)
    for ax, d, ((p, q), title) in zip(axes[0], data, CONDITIONS):
        xs, fs, times = d["xs"], d["fs"], d["times"]
        ax.plot([0, PLANE_LIMIT], [0, PLANE_LIMIT], color="grey", linewidth=0.5, alpha=0.5)
        if same_total is not None:
            ax.plot([0, same_total], [same_total, 0], color="grey", linestyle="--", linewidth=0.8,
                    label="same total as p = q = 1")
        ax.add_collection(lines(d, lambda k, i: tuple(xs[k, i]), first=0))
        for t in ARROW_TIMES:           # arrowheads where each group is at these times
            k = np.searchsorted(times, t)
            if 0 < k < len(times):
                for i in np.flatnonzero((fs[k] > 0) & (fs[k - 1] > 0)):
                    if np.linalg.norm(xs[k, i] - xs[k - 1, i]) > 1e-4:
                        ax.annotate("", xy=xs[k, i], xytext=xs[k - 1, i],
                                    arrowprops=dict(arrowstyle="-|>", color="#333333",
                                                    lw=0.5, mutation_scale=10, shrinkA=0, shrinkB=0), zorder=4)
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
    fig.text(0.5, 0.02, "All runs start slightly towards good 1.", ha="center", fontsize=9)
    plots.save(fig, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=",".join(f"{v:g}" for v in START), metavar="X1,X2",
                        help=f"the start of every run (default {START[0]:g},{START[1]:g})")
    parser.add_argument("--end-time", type=float, help="hold every run's final state up to this time "
                                                       "(default: the longest run's finishing time)")
    args = parser.parse_args()
    start = tuple(float(v) for v in args.start.split(","))

    tag = "_".join(["quadratic_flat", f"b1-{B1:.4g}", f"b2-{B2:g}", f"tol{SETTINGS['tol']:g}", f"chunk{SETTINGS['chunk']}"]
                   + [f"p{p:g}-q{q:g}" for (p, q), _ in CONDITIONS] + [f"start-{start[0]:g}-{start[1]:g}"]
                   + ([f"end{args.end_time:g}"] if args.end_time else []))
    print("running on:", jax.devices())
    data = simulate(start)
    finished = [d["times"][-1] for d in data]
    t_end = args.end_time or max(finished)
    data = [hold_until(d, t_end) for d in data]

    arrays = {f"{k}_{c}": v for c, d in enumerate(data) for k, v in d.items()}
    meta = dict(conditions=[list(pq) for pq, _ in CONDITIONS], titles=[t for _, t in CONDITIONS],
                start=list(start), b1=B1, b2=B2, settings=SETTINGS,
                finished=finished, end_time=t_end)       # the last frame of a run that finished early is held
    np.savez(f"trajectories_{tag}.npz", **arrays, meta=json.dumps(meta))
    print(f"saved trajectories_{tag}.npz")
    plot_time(data, f"trajectories_time_{tag}.png")
    plot_plane(data, f"trajectories_plane_{tag}.png")

    for d, t, ((p, q), title) in zip(data, finished, CONDITIONS):
        alive = d["fs"][-1] > 0
        groups = ", ".join(f"({x1:.3f}, {x2:.3f}) share {f:.2f}" for (x1, x2), f in zip(d["xs"][-1][alive], d["fs"][-1][alive]))
        print(f"{title} (p = {p:g}, q = {q:g}): finished at time {t:.0f}, final groups {groups}")
