# How the population evolves over time for a few (p, q) conditions, in the style of DHK's Fig. 1 time
# panels. Runs are recorded after every chunk and held at their final state up to a common end time
# (the longest run's finishing time by default). Three figures:
#   trajectories_time_<tag>.png:   one run per condition, from --start; each group's difference
#                                  x1 - x2 against time (linear to 1, log after), with the totals
#                                  x1 + x2 shared or as strips
#   trajectories_goods_<tag>.png:  the same runs: each group's x1 (blue) and x2 (red) against time
#   trajectories_plane_<tag>.png:  each condition's runs (its own plane starts), as paths in the
#                                  (x1, x2) plane
# and the recorded data of the time and plane runs as a .npz next to their figures. Lines are coloured
# by lean, as wide as the group's share, and a black diamond marks each split. See USAGE.md.
#
# Usage:  python run_trajectories.py                       (totals shared)
#         python run_trajectories.py --total strips --start 0.1,0.02
#         python run_trajectories.py --pq-diff 0.01          (p and q 0.01 from 1 instead of 0.05)
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D

from coop import engine, models, plots, sweeps
from run_simulation import settings_for

# DHK's ESS point B, converted to the quadratic model; the benefit stays flat past its peak
B1, B2 = 7 / 4.6, 1.5
SETTINGS = dict(tol=1e-5, chunk=20)       # chunk: steps between records (20 steps = 0.2 time units)

# figure A: every run starts at --start (default START), total 0.16
START = (0.155, 0.005)

# figure B's starts, all with total 0.16; the balanced one is exactly on the diagonal (the engine's
# saddle push and split test take it off)
GOOD_1, BALANCED, GOOD_2 = (0.11, 0.05), (0.08, 0.08), (0.05, 0.11)

# p and q move this far from 1 (--pq-diff); division of labour at (1 - 2D, 1 - 1.6D), inside
# p < q < (p+1)/2 for any D
PQ_DIFF = 0.05


def conditions(d):
    # all figures: (p, q), title and figure B's starts (all three with --plane-starts all), one column
    # (A, C) or panel (B) each
    at = lambda p, q: (round(p, 6), round(q, 6))
    return [(at(1, 1),             "arithmetic mean",            [GOOD_1, BALANCED, GOOD_2]),
            (at(1 - d, 1),         "benefit favours balance",    [GOOD_1, GOOD_2]),
            (at(1 - d, 1 + d),     "both favour balance",        [GOOD_1, GOOD_2]),
            (at(1, 1 - d),         "cost favours concentration", [GOOD_1, GOOD_2]),
            (at(1 + d, 1 - d),     "both favour concentration",  [GOOD_1, GOOD_2]),
            (at(1 - 2 * d, 1 - 1.6 * d), "division of labour",   [GOOD_1, BALANCED, GOOD_2])]


CONDITIONS = conditions(PQ_DIFF)

DIFF_LIMIT = 0.3                     # figure A: the difference x1 - x2 from -0.3 to 0.3
PLANE_LIMIT = 0.3                    # figure B: x1 and x2 from 0 to 0.3
ARROW_TIMES = [0.5, 2, 8, 32, 128]   # figure B: arrowheads where each group is at these times
TOTAL_STYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 3))]   # figure A, shared totals
LEAN = LinearSegmentedColormap.from_list("lean", ["#E24B4A", "#7F77DD", "#378ADD"])   # lean -1, 0, +1
SPLIT_MARK = dict(marker="D", color="black", markersize=6, linestyle="none", zorder=5)
GOOD_COLOURS, GOOD_ALPHA = ["#378ADD", "#E24B4A"], 0.55   # figure C: good 1 blue, good 2 red, translucent
TIME_AXIS = dict(value="symlog", linthresh=1)             # time from 0: linear up to 1, logarithmic after


def width(share):
    return 0.6 + 2.4 * share


def title(pq, name):
    # the division-of-labour panels also say which part of the rule they satisfy
    p, q = pq
    rule = "\np < q < (p+1)/2" if p < q < (p + 1) / 2 else ""
    return f"{name}\np = {p:g}, q = {q:g}{rule}"


# ---------- simulate and record ----------

def simulate(conditions, end_time):
    # conditions: [((p, q), starts)]; one run per (condition, start), recorded at time 0 and after
    # every chunk, and held at its final state up to end_time (default: the longest run's finishing time). Returns, per
    # condition, a list of runs dict(start, times, xs, fs, finished)
    model = models.make_model("quadratic", flat_benefit=True)
    ps = sorted({p for (p, _), _ in conditions})
    qs = sorted({q for (_, q), _ in conditions})
    # p and q as panels (every combination; only the conditions are used), b1' and b2' fixed
    sweep = sweeps.Sweep("trajectories", panels=(("p", ps), ("q", qs)),
                         grid=(("b2", sweeps.Axis(B2, B2)), ("b1", sweeps.Axis(B1, B1))), panel_rows="q")
    s = settings_for(model, sweep, **SETTINGS)
    starts = list(dict.fromkeys(start for _, ss in conditions for start in ss))   # the engine's start slots
    if len(starts) > len(s.starts):
        raise SystemExit(f"at most {len(s.starts)} different starts")
    shape = sweeps.layout(sweep, 1, len(s.starts))[0]                     # (start, p, q, 1, 1)
    ids = [[np.ravel_multi_index((starts.index(start), ps.index(p), qs.index(q), 0, 0), shape) for start in ss]
           for (p, q), ss in conditions]
    start_slot = np.unravel_index(np.arange(np.prod(shape)), shape)[0]
    frames = []

    def record(runs):
        if not frames:          # first call: the runs were just created; set the starts
            for k, start in enumerate(starts):
                runs.xs[start_slot == k] = start
        frames.append((runs.stage_time.copy(), runs.xs.copy(), runs.fs.copy()))

    engine.simulate(model, sweep, s, 1, record=record)
    times, xs, fs = (np.stack(a) for a in zip(*frames))                   # (frame, run, ...)
    out = [[dict(start=start, **one_run(times[:, i], xs[:, i], fs[:, i])) for start, i in zip(ss, row)]
           for (_, ss), row in zip(conditions, ids)]
    t_end = end_time or max(r["finished"] for runs in out for r in runs)
    return [[hold_until(r, t_end) for r in runs] for runs in out]


def one_run(times, xs, fs):
    # one run's frames: the first, and each where its time moved on (the others advanced other runs)
    keep = np.concatenate([[True], np.diff(times) > 0])
    return dict(times=times[keep], xs=xs[keep], fs=fs[keep], finished=times[-1])


def hold_until(r, t_end):
    # a finished run keeps its final state: add a last frame at t_end, so every line reaches the end
    if r["times"][-1] >= t_end:
        return r
    return dict(r, times=np.append(r["times"], t_end), xs=np.concatenate([r["xs"], r["xs"][-1:]]),
                fs=np.concatenate([r["fs"], r["fs"][-1:]]))


def save_data(data, conditions, filename):
    arrays = {f"{k}_{c}_{s}": r[k] for c, runs in enumerate(data) for s, r in enumerate(runs)
              for k in ["times", "xs", "fs"]}
    meta = dict(b1=B1, b2=B2, settings=SETTINGS, conditions=[list(pq) for pq, *_ in conditions],
                starts=[[r["start"] for r in runs] for runs in data],
                finished=[[r["finished"] for r in runs] for runs in data])
    np.savez(filename, **arrays, meta=json.dumps(meta))
    print(f"saved {filename}")


# ---------- groups, paths and splits ----------

def paths_and_splits(r):
    # every group's path as a list of (frame, slot), and every split as (frame, parent slot, new slot).
    # A slot only fills when a group splits (a saddle push moves a group, it fills no slot): its path
    # starts from the nearest group the frame before, so it forks from it
    xs, fs = r["xs"], r["fs"]
    done, current, splits = [], {}, []
    for k in range(len(fs)):
        for i in range(fs.shape[1]):
            if fs[k, i] > 0 and i in current:
                current[i].append((k, i))
            elif fs[k, i] > 0 and k > 0:
                before = np.flatnonzero(fs[k - 1] > 0)
                parent = before[np.argmin(np.linalg.norm(xs[k - 1, before] - xs[k, i], axis=-1))]
                current[i] = [(k - 1, parent), (k, i)]
                splits.append((k, parent, i))
            elif fs[k, i] > 0:
                current[i] = [(k, i)]
            elif i in current:
                done.append(current.pop(i))
    return done + list(current.values()), splits


def lean(x):
    # (x1 - x2) / (x1 + x2): -1 all good 2, 0 equal, +1 all good 1
    total = x[..., 0] + x[..., 1]
    return np.where(total > 0, (x[..., 0] - x[..., 1]) / np.where(total > 0, total, 1), 0.0)


def draw(ax, r, point, colour=None):
    # every group's path through point(x, k) (x = (x1, x2) at frame k), coloured by lean (or colour) and
    # as wide as its share, and a diamond at each split, where the parent was
    xs, fs = r["xs"], r["fs"]
    paths, splits = paths_and_splits(r)
    edges = [(a, b) for path in paths for a, b in zip(path, path[1:])]
    ax.add_collection(LineCollection(
        [[point(xs[a], a[0]), point(xs[b], b[0])] for a, b in edges],
        colors=colour or [LEAN((lean(xs[b]) + 1) / 2) for a, b in edges],
        linewidths=[width(fs[b]) for a, b in edges], capstyle="round"))
    for k, parent, child in splits:     # the split point: halfway between the two halves
        ax.plot(*point((xs[k, parent] + xs[k, child]) / 2, k), **SPLIT_MARK)


def mean_total(r):
    # share-weighted total investment x1 + x2 per frame
    return np.sum(r["fs"] * r["xs"].sum(axis=-1), axis=-1)


def split_legend(fig, extra=()):
    fig.legend(handles=[Line2D([], [], label="split", **SPLIT_MARK), *extra], loc="upper right",
               fontsize=8, frameon=False)


def lean_bar(fig, cax):
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=LEAN), cax=fig.add_axes(cax))
    bar.set_ticks([-1, 0, 1], labels=["all good 2", "equal", "all good 1"])
    bar.set_label("lean", fontsize=8)
    bar.ax.tick_params(labelsize=7)


# ---------- figure A: time panels ----------

def plot_time(data, total, filename):
    runs = [rs[0] for rs in data]
    t_max = max(r["times"][-1] for r in runs)
    n, strips = len(runs), total == "strips"
    fig = plt.figure(figsize=(2.9 * n + 0.6, 6.5 if strips else 8.5))
    if strips:          # per condition: split, total (half as wide), gap
        grid = fig.add_gridspec(1, 3 * n - 1, width_ratios=([2, 1, 0.45] * n)[:-1], wspace=0.08,
                                left=0.05, right=0.93, top=0.8, bottom=0.1)
    else:
        grid = fig.add_gridspec(2, n, height_ratios=[3, 1.2], hspace=0.5, wspace=0.35,
                                left=0.06, right=0.93, top=0.85, bottom=0.07)

    for c, (r, (pq, name, _)) in enumerate(zip(runs, CONDITIONS)):
        ax = fig.add_subplot(grid[0, 3 * c] if strips else grid[0, c])
        draw(ax, r, lambda x, k: (x[0] - x[1], r["times"][k]))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_yscale(**TIME_AXIS)
        ax.set(xlim=(-DIFF_LIMIT, DIFF_LIMIT), ylim=(0, t_max))
        ax.set_xlabel("difference", fontsize=8)
        ax.set_ylabel("time" if c == 0 else "")
        ax.tick_params(labelsize=7)
        ax.set_title(title(pq, name), fontsize=9, loc="left" if strips else "center")
        if strips:
            strip = fig.add_subplot(grid[0, 3 * c + 1], sharey=ax)
            draw(strip, r, lambda x, k: (x[0] + x[1], r["times"][k]), colour="grey")
            strip.set_xlim(0, 1.1 * max(np.max(r["xs"].sum(axis=-1)) for r in runs))
            strip.set_xlabel("total", fontsize=8)
            strip.tick_params(labelsize=7, labelleft=False)

    if not strips:
        ax = fig.add_subplot(grid[1, :])
        for r, (pq, name, _), style in zip(runs, CONDITIONS, TOTAL_STYLES):
            ax.plot(r["times"], mean_total(r), linestyle=style, color="black", linewidth=1,
                    label=f"{name} (p = {pq[0]:g}, q = {pq[1]:g})")
        ax.set_xscale(**TIME_AXIS)
        ax.set(xlim=(0, t_max), ylim=(min(mean_total(r)[0] for r in runs) - 0.01, None))
        ax.set_xlabel("time")
        ax.set_ylabel("total", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    split_legend(fig)
    lean_bar(fig, [0.95, 0.45, 0.008, 0.35])
    plots.save(fig, filename)


# ---------- figure C: each good's investment against time ----------

def plot_goods(data, filename):
    # each group's x1 (blue) and x2 (red) as one translucent line per path, so where they overlap they
    # blend to purple; a diamond on both lines at each split
    runs = [rs[0] for rs in data]
    t_max = max(r["times"][-1] for r in runs)
    fig, axes = plt.subplots(1, len(runs), sharey=True, figsize=(2.6 * len(runs) + 0.6, 6))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.74, bottom=0.17, wspace=0.25)
    for ax, r, (pq, name, _) in zip(axes, runs, CONDITIONS):
        xs, fs, times = r["xs"], r["fs"], r["times"]
        paths, splits = paths_and_splits(r)
        for good, colour in enumerate(GOOD_COLOURS):
            ax.add_collection(LineCollection([[(xs[k, i, good], times[k]) for k, i in path] for path in paths],
                                             colors=colour, alpha=GOOD_ALPHA, capstyle="butt",
                                             linewidths=[width(fs[path[-1]]) for path in paths]))
            for k, parent, child in splits:
                ax.plot((xs[k, parent, good] + xs[k, child, good]) / 2, times[k], **SPLIT_MARK)
        ax.set_yscale(**TIME_AXIS)
        ax.set(xlim=(0, 0.3), ylim=(0, t_max))
        ax.set_xlabel("investment", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title(pq, name), fontsize=9)
    axes[0].set_ylabel("time")

    blend = np.array([1.0, 1.0, 1.0])               # the legend's purple: red over blue over white
    for colour in GOOD_COLOURS:
        blend = (1 - GOOD_ALPHA) * blend + GOOD_ALPHA * np.array(to_rgb(colour))
    split_legend(fig, [Line2D([], [], color=c, alpha=GOOD_ALPHA, linewidth=2.5, label=l)
                       for c, l in zip(GOOD_COLOURS, ["good 1", "good 2"])]
                 + [Line2D([], [], color=blend, linewidth=2.5, label="overlap (purple)")])
    plots.save(fig, filename)


# ---------- figure B: investment plane ----------

def plot_plane(data, filename):
    # two rows of three panels
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.6))
    fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.08, wspace=0.3, hspace=0.4)

    for ax, runs, (pq, name, _) in zip(axes.flat, data, CONDITIONS):
        ax.set(xlim=(0, PLANE_LIMIT), ylim=(0, PLANE_LIMIT), aspect="equal")
        ax.plot([0, PLANE_LIMIT], [0, PLANE_LIMIT], color="grey", linewidth=0.5, alpha=0.5)
        for r in runs:
            xs, fs = r["xs"], r["fs"]
            draw(ax, r, lambda x, k: tuple(x))
            for t in ARROW_TIMES:       # arrowheads where each group is at these times
                k = np.searchsorted(r["times"], t)
                if 0 < k < len(fs):
                    for i in np.flatnonzero((fs[k] > 0) & (fs[k - 1] > 0)):
                        if np.linalg.norm(xs[k, i] - xs[k - 1, i]) > 1e-4:
                            ax.annotate("", xy=xs[k, i], xytext=xs[k - 1, i], zorder=4,
                                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.5,
                                                        mutation_scale=10, shrinkA=0, shrinkB=0))
            ax.scatter(*r["start"], s=25, facecolors="none", edgecolors="black", linewidths=0.7, zorder=3)
            alive = fs[-1] > 0
            ax.scatter(xs[-1, alive, 0], xs[-1, alive, 1], s=60 * fs[-1, alive], c=LEAN((lean(xs[-1, alive]) + 1) / 2),
                       edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_xlabel("x1 (good 1)", fontsize=8)
        ax.set_ylabel("x2 (good 2)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title(pq, name), fontsize=9)

    split_legend(fig, [Line2D([], [], marker="o", markersize=5, markerfacecolor="none", markeredgecolor="black",
                              linestyle="none", label="start"),
                       Line2D([], [], marker="o", markersize=7, markerfacecolor="#7F77DD", markeredgecolor="black",
                              linestyle="none", label="end (size = share)")])
    lean_bar(fig, [0.9, 0.3, 0.015, 0.4])
    plots.save(fig, filename)


def print_final(data, conditions):
    for runs, (pq, name, *_) in zip(data, conditions):
        print(f"{name} (p = {pq[0]:g}, q = {pq[1]:g}):")
        for r in runs:
            alive = r["fs"][-1] > 0
            groups = ", ".join(f"({x1:.3f}, {x2:.3f}) share {f:.2f}" for (x1, x2), f in zip(r["xs"][-1][alive], r["fs"][-1][alive]))
            print(f"  from ({r['start'][0]:g}, {r['start'][1]:g}): finished at time {r['finished']:.0f}, "
                  f"{len(paths_and_splits(r)[1])} splits, final groups {groups}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=",".join(f"{v:g}" for v in START), metavar="X1,X2",
                        help=f"figure A: the start of every run (default {START[0]:g},{START[1]:g})")
    parser.add_argument("--total", choices=["shared", "strips"], default="shared",
                        help="figure A: totals in one panel below all columns, or as a narrow strip beside each")
    parser.add_argument("--plane-starts", choices=["listed", "all"], default="listed",
                        help="figure B: each condition's own starts (balanced only in arithmetic mean and "
                             "division of labour), or all three starts in every panel")
    parser.add_argument("--pq-diff", type=float, default=PQ_DIFF, metavar="D",
                        help=f"all figures: how far p and q move from 1 (default {PQ_DIFF:g})")
    parser.add_argument("--end-time", type=float, help="hold every run's final state up to this time "
                                                       "(default: the longest run's finishing time)")
    args = parser.parse_args()
    start = tuple(float(v) for v in args.start.split(","))
    CONDITIONS = conditions(args.pq_diff)

    base = ["quadratic_flat", f"b1-{B1:.4g}", f"b2-{B2:g}", f"tol{SETTINGS['tol']:g}", f"chunk{SETTINGS['chunk']}"]
    end = [f"end{args.end_time:g}"] if args.end_time else []
    base += [f"p{p:g}-q{q:g}" for (p, q), *_ in CONDITIONS]
    goods_tag = "_".join(base + [f"start-{start[0]:g}-{start[1]:g}"] + end)
    time_tag = "_".join(base + [f"start-{start[0]:g}-{start[1]:g}", f"total-{args.total}"] + end)
    plane_tag = "_".join(base + (["starts-all"] if args.plane_starts == "all" else []) + end)
    print("running on:", jax.devices())

    data = simulate([(pq, [start]) for pq, _, _ in CONDITIONS], args.end_time)
    save_data(data, CONDITIONS, f"trajectories_time_{time_tag}.npz")
    plot_time(data, args.total, f"trajectories_time_{time_tag}.png")
    plot_goods(data, f"trajectories_goods_{goods_tag}.png")
    print_final(data, CONDITIONS)

    all_starts = args.plane_starts == "all"
    data = simulate([(pq, [GOOD_1, BALANCED, GOOD_2] if all_starts else starts) for pq, _, starts in CONDITIONS],
                    args.end_time)
    save_data(data, CONDITIONS, f"trajectories_plane_{plane_tag}.npz")
    plot_plane(data, f"trajectories_plane_{plane_tag}.png")
    print_final(data, CONDITIONS)
