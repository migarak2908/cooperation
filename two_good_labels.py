# Second labelling scheme for the two-good simulation: label each grid point by the
# group types (CC / DC / DD) its runs end with. Uses the final states from
# multiple_pop_run in two_good_integration_phase.py; the simulation itself is unchanged.

import os
import numpy as np

x_max = 2.0
x_min = 1e-6
min_share = 1e-4      # same as multiple_pop_run: smaller groups don't count
merge_gap = 1e-3      # same as multiple_pop_run: closer groups count as one

TYPES = ["CC", "DC", "DD"]           # type index 0, 1, 2

# the 11 expected labels, in order, with their colours
LABELS = ["CC", "CC and DC", "CC and DD", "CC |or| DC", "CC |or| DC |or| DD", "CC |or| DD",
          "DC", "DC and DD", "DC and DD |or| DD", "DC |or| DD", "DD"]
COLOURS = ["#d7301f", "#e6803a", "#e8d84a", "#a6e05a", "#5ee05e", "#7ff0a8",
           "#6ccbe0", "#3a63d8", "#4b1fd8", "#b43ad8", "#d6338f"]
OTHER_COLOUR = "#9e9e9e"


# ---------- step 1: type of each group ----------

def group_types(xs, fs, x_max=x_max, min_share=min_share, merge_gap=merge_gap):
    # xs: (..., 2 groups, 2 goods), fs: (..., 2 groups)
    # returns types (..., 2 groups) with CC = 0, DC = 1, DD = 2, and alive (..., 2 groups)

    # groups closer than merge_gap count as one: put both at their share-weighted average
    close = np.linalg.norm(xs[..., 0, :] - xs[..., 1, :], axis=-1) < merge_gap
    average = np.sum(fs[..., None] * xs, axis=-2) / np.sum(fs, axis=-1)[..., None]
    xs = np.where(close[..., None, None], average[..., None, :], xs)

    # a good is "invested" if the group puts more than x_max / 2 into it
    invested = xs > x_max / 2                     # (..., group, good)
    n_invested = invested.sum(axis=-1)            # 2 -> CC, 1 -> DC, 0 -> DD
    types = 2 - n_invested

    alive = fs > min_share
    return types, alive


# ---------- step 2: outcome of each run = set of group types ----------
# A set of types is stored as a number with one bit per type:
# bit 0 = CC, bit 1 = DC, bit 2 = DD; e.g. CC and DD -> 1 + 4 = 5.

def run_outcomes(xs, fs, **kwargs):
    types, alive = group_types(xs, fs, **kwargs)
    bits = np.where(alive, 1 << types, 0)         # (..., group)
    return np.bitwise_or(bits[..., 0], bits[..., 1])


def outcome_types(outcome):
    # e.g. 5 -> (0, 2)  (CC and DD)
    return tuple(t for t in range(3) if outcome & (1 << t))


def outcome_name(outcome):
    return " and ".join(TYPES[t] for t in outcome_types(outcome))


# ---------- step 3: label of each grid point = set of run outcomes over starts ----------

def point_labels(outcomes, min_count=1):
    # outcomes: (start, ...) -> array of label strings with shape (...)
    # an outcome enters a point's set only if at least min_count starts produce it
    all_outcomes = range(1, 8)
    counts = {o: (outcomes == o).sum(axis=0) for o in all_outcomes}
    order = sorted(all_outcomes, key=outcome_types)      # canonical order: CC < DC < DD, then second type
    labels = np.empty(outcomes.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        present = [o for o in order if counts[o][idx] >= min_count]
        labels[idx] = " |or| ".join(outcome_name(o) for o in present)
    return labels


# ---------- checks ----------

def check_classifier():
    tiny = 0.000001
    cases = [
        ("one group at (2, 2)",                  [[2, 2], [2, 2]],          [0.5, 0.5], "CC"),
        ("one group at (2, 1e-6)",               [[2, tiny], [2, tiny]],    [0.5, 0.5], "DC"),
        ("one group at (1e-6, 2)",               [[tiny, 2], [tiny, 2]],    [0.5, 0.5], "DC"),
        ("groups at (2, 2) and (1e-6, 1e-6)",    [[2, 2], [tiny, tiny]],    [0.5, 0.5], "CC and DD"),
        ("one group at (1.2, 1.2)",              [[1.2, 1.2], [1.2, 1.2]],  [0.5, 0.5], "CC"),
        ("one group at (0.8, 0.8)",              [[0.8, 0.8], [0.8, 0.8]],  [0.5, 0.5], "DD"),
    ]
    for name, xs, fs, expected in cases:
        got = outcome_name(int(run_outcomes(np.array(xs, float), np.array(fs, float))))
        status = "ok" if got == expected else "FAILED"
        print(f"  {status:6s} {name:38s} -> {got!r} (expected {expected!r})")
        assert got == expected


def one_good_analytic_labels(b1, b2, x_max=x_max):
    # the one-good analytic diagram, in the new labels: with p = q = 1 the payoff only depends
    # on the average investment, so a single group at level z sits at (z, z)
    B1, B2 = np.meshgrid(b1, b2)
    D0 = B1 - 1
    Dmax = D0 + (2 - 4 * B2) * x_max
    with np.errstate(divide="ignore", invalid="ignore"):
        x_star = D0 / (4 * B2 - 2)
    ess_type = np.where(x_star > x_max / 2, "CC", "DD")
    return np.select(
        [(D0 > 0) & (Dmax > 0), (D0 < 0) & (Dmax < 0), (D0 < 0) & (Dmax > 0),
         (D0 > 0) & (Dmax < 0) & (B2 < 1), (D0 > 0) & (Dmax < 0) & (B2 > 1)],
        ["CC", "DD", "CC |or| DD", "CC and DD", ess_type], default="boundary")


# ---------- simulation (run once, results cached) ----------

def simulate(p, q, b1, b2, starts, time_step=0.01, steps=10000):
    import jax
    import jax.numpy as jnp
    from two_good_integration_phase import multiple_pop_run

    I, Pg, Qg, Y, X = jnp.meshgrid(jnp.arange(len(starts)), p, q, b2, b1, indexing="ij")
    run_one = lambda p, q, b1, b2, x0: multiple_pop_run(p, q, b1, b2, x0, time_step, steps,
                                                        x_max=x_max, x_min=x_min)
    run_all = jax.jit(jax.vmap(run_one))
    xs, fs = run_all(Pg.ravel(), Qg.ravel(), X.ravel(), Y.ravel(), jnp.asarray(starts)[I.ravel()])
    return np.asarray(xs).reshape(*I.shape, 2, 2), np.asarray(fs).reshape(*I.shape, 2)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    min_count = 1                        # raise to drop outcomes that only a few starts reach
    cache = "two_good_results.npz"       # delete this file to rerun the simulation

    b1 = np.linspace(0, 4, 30)           # b1' values (across)
    b2 = np.linspace(0, 4, 30)           # b2' values (up)
    p = np.array([0.8, 1, 1.25])
    q = np.array([0.8, 1, 1.25])
    s = np.linspace(x_min, x_max, 5)
    S1, S2 = np.meshgrid(s, s)
    starts = np.stack([S1.ravel(), S2.ravel()], axis=-1)     # (25, 2)

    if os.path.exists(cache):
        data = np.load(cache)
        xs, fs = data["xs"], data["fs"]
        print(f"loaded final states from {cache}")
    else:
        print(f"simulating {len(starts) * len(p) * len(q) * len(b2) * len(b1)} runs ...")
        t = time.time()
        xs, fs = simulate(p, q, b1, b2, starts)
        print(f"simulation took {time.time() - t:.0f}s")
        np.savez(cache, xs=xs, fs=fs, p=p, q=q, b1=b1, b2=b2, starts=starts)
    # xs: (start, p, q, b2', b1', group, good), fs: (start, p, q, b2', b1', group)

    print("classifier checks:")
    check_classifier()

    outcomes = run_outcomes(xs, fs)                      # (start, p, q, b2', b1')
    labels = point_labels(outcomes, min_count)           # (p, q, b2', b1')

    # check: p = q = 1 against the one-good analytic diagram
    i1, j1 = int(np.argmin(np.abs(p - 1))), int(np.argmin(np.abs(q - 1)))
    analytic = one_good_analytic_labels(b1, b2)
    sim = labels[i1, j1]
    differ = analytic != sim
    print(f"p = q = 1 vs one-good analytic: {differ.sum()} of {sim.size} grid points differ")
    for a, s_ in sorted(set(zip(analytic[differ], sim[differ]))):
        n = int(((analytic == a) & (sim == s_)).sum())
        print(f"  analytic {a!r:16s} simulated {s_!r:28s} {n:3d} points")
    with_dc = np.array(["DC" in l for l in sim.ravel()]).reshape(sim.shape)
    print(f"  points with DC in the p = q = 1 panel: {with_dc.sum()}")

    # per panel: how many grid points got each label, and any labels outside the 11
    index = {l: k for k, l in enumerate(LABELS)}
    for i in range(len(p)):
        for j in range(len(q)):
            panel = labels[i, j]
            values, counts = np.unique(panel, return_counts=True)
            print(f"p = {p[i]:g}, q = {q[j]:g}:")
            for v, c in sorted(zip(values, counts), key=lambda vc: index.get(vc[0], len(LABELS))):
                print(f"  {c:4d}  {v}" + ("" if v in index else "   <- not one of the 11 (grey)"))

    # plot
    codes = np.vectorize(lambda l: index.get(l, len(LABELS)))(labels)     # other labels -> grey
    cmap = ListedColormap(COLOURS + [OTHER_COLOUR])
    extent = [b1[0], b1[-1], b2[0], b2[-1]]

    fig, axes = plt.subplots(len(p), len(q), sharex=True, sharey=True, figsize=(13, 10))
    for i in range(len(p)):
        for j in range(len(q)):
            ax = axes[i, j]
            ax.imshow(codes[i, j], origin="lower", extent=extent, aspect="auto",
                      cmap=cmap, vmin=-0.5, vmax=len(LABELS) + 0.5, interpolation="nearest")
            ax.set_title(f"p = {p[i]:g}, q = {q[j]:g}", fontsize=10)
    handles = [Patch(color=c, label=l) for l, c in zip(LABELS, COLOURS)]
    if (codes == len(LABELS)).any():
        handles.append(Patch(color=OTHER_COLOUR, label="other"))
    fig.legend(handles=handles, loc="center right", fontsize=9)
    fig.supxlabel("b1'"); fig.supylabel("b2'")
    fig.suptitle(f"group types at the end of the runs (min_count = {min_count})")
    fig.subplots_adjust(right=0.78)
    fig.savefig(f"two_good_labels_min{min_count}.png", dpi=150)
    print(f"saved two_good_labels_min{min_count}.png")
    plt.show()
