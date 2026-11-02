# How the population evolves over time for a few (p, q) conditions, in the style of DHK's Fig. 1 time
# panels: one run per condition, all from the same slightly unbalanced start, recorded after every
# chunk. Saves trajectories_<tag>.png and the recorded data as trajectories_<tag>.npz. See USAGE.md.
#
# Usage:  python run_trajectories.py                       (start (0.04, 0.01))
#         python run_trajectories.py --start 0.03,0.01
#         in Colab: %run run_trajectories.py

import argparse
import json
import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

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

SPLIT_LIMIT = 0.3                    # the split panel's x-axis: -0.3 to 0.3
WIDTH = 4.0                          # line width of a group with share 1
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

def segments(times, values, fs, lean):
    # one segment per living group per frame step, from (value, time) to the next; a group that
    # appears (a split) starts from the nearest group alive the frame before, so it forks from it.
    # Returns the segments, their share and their lean
    segs, shares, leans = [], [], []
    for k in range(1, len(times)):
        alive_before, alive = fs[k - 1] > 0, fs[k] > 0
        for i in np.flatnonzero(alive):
            if alive_before[i]:
                parent = i
            else:
                candidates = np.flatnonzero(alive_before)
                parent = candidates[np.argmin(np.abs(values[k - 1, candidates] - values[k, i]))]
            segs.append([(values[k - 1, parent], times[k - 1]), (values[k, i], times[k])])
            shares.append(fs[k, i])
            leans.append(lean[k, i])
    return segs, np.array(shares), np.array(leans)


def plot(data, start, filename):
    t_max = max(d["times"][-1] for d in data)
    fig = plt.figure(figsize=(3.2 * len(data), 5.5))
    columns = fig.subfigures(1, len(data))
    for column, d, ((p, q), title) in zip(columns, data, CONDITIONS):
        x1, x2 = d["xs"][..., 0], d["xs"][..., 1]
        total, diff = x1 + x2, x1 - x2
        with np.errstate(divide="ignore", invalid="ignore"):
            lean = np.nan_to_num(diff / total)
        total_ax, split_ax = column.subplots(1, 2, sharey=True)
        column.subplots_adjust(left=0.2, bottom=0.12, top=0.84)

        segs, shares, _ = segments(d["times"], total, d["fs"], lean)
        total_ax.add_collection(LineCollection(segs, colors="grey", linewidths=WIDTH * shares, capstyle="round"))
        total_ax.set_xlim(0, 1.1 * np.max(np.where(d["fs"] > 0, total, 0)))
        total_ax.set_title("total", fontsize=9)
        total_ax.set_ylabel("time")

        segs, shares, leans = segments(d["times"], diff, d["fs"], lean)
        split_ax.add_collection(LineCollection(segs, colors=LEAN((leans + 1) / 2), linewidths=WIDTH * shares,
                                           capstyle="round"))
        split_ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        split_ax.set_xlim(-SPLIT_LIMIT, SPLIT_LIMIT)
        split_ax.set_title("split", fontsize=9)

        total_ax.set_ylim(0, t_max)
        for ax in (total_ax, split_ax):
            ax.tick_params(labelsize=7)
        column.suptitle(f"{title}\np = {p:g}, q = {q:g}", fontsize=10)
    fig.text(0.5, 0.01, "All runs start slightly towards good 1. Time in model units.", ha="center", fontsize=9)
    plots.save(fig, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=",".join(f"{v:g}" for v in START), metavar="X1,X2",
                        help=f"the start of every run (default {START[0]:g},{START[1]:g})")
    args = parser.parse_args()
    start = tuple(float(v) for v in args.start.split(","))

    tag = "_".join(["quadratic_flat", f"b1-{B1:.4g}", f"b2-{B2:g}", f"tol{SETTINGS['tol']:g}", f"chunk{SETTINGS['chunk']}"]
                   + [f"p{p:g}-q{q:g}" for (p, q), _ in CONDITIONS] + [f"start-{start[0]:g}-{start[1]:g}"])
    print("running on:", jax.devices())
    data = simulate(start)

    arrays = {f"{k}_{c}": v for c, d in enumerate(data) for k, v in d.items()}
    meta = dict(conditions=[list(pq) for pq, _ in CONDITIONS], titles=[t for _, t in CONDITIONS],
                start=list(start), b1=B1, b2=B2, settings=SETTINGS)
    np.savez(f"trajectories_{tag}.npz", **arrays, meta=json.dumps(meta))
    print(f"saved trajectories_{tag}.npz")
    plot(data, start, f"trajectories_{tag}.png")

    for d, ((p, q), title) in zip(data, CONDITIONS):
        alive = d["fs"][-1] > 0
        groups = ", ".join(f"({x1:.3f}, {x2:.3f}) share {f:.2f}" for (x1, x2), f in zip(d["xs"][-1][alive], d["fs"][-1][alive]))
        print(f"{title} (p = {p:g}, q = {q:g}): time {d['times'][-1]:.0f}, final groups {groups}")
