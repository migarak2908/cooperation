# Second labelling scheme for the two-good simulation: label each grid point by the
# group types (CC / DC / DD) its runs end with. Uses the final states from
# multiple_pop_run in two_good_integration_phase.py; the simulation itself is unchanged.

import os
import numpy as np

x_max = 0.5
x_min = 1e-6
min_share = 1e-4      # same as multiple_pop_run: smaller groups don't count
merge_gap = 1e-3      # same as multiple_pop_run: closer groups count as one

TYPES = ["CC", "DC", "DD"]           # type index 0, 1, 2

# A run's outcome is stored as a number with one bit per group type:
# bit 0 (1) = CC, bit 1 (2) = DC, bit 2 (4) = DD, e.g. CC and DD -> 1 + 4 = 5.
# Division of labour (two DC groups investing in different goods) gets its own bit 3 (8).
# Canonical order of outcomes when building "|or|" labels:
OUTCOMES = [(1, "CC"), (3, "CC and DC"), (7, "CC and DC and DD"), (5, "CC and DD"),
            (2, "DC"), (8, "DC and DC'"), (6, "DC and DD"), (4, "DD")]
OUTCOME_NAME = dict(OUTCOMES)

# the known labels, in order, with their colours
LABELS = ["CC", "CC and DC", "CC and DD", "CC |or| DC", "CC |or| DC |or| DD", "CC |or| DD",
          "DC", "DC and DC'", "DC and DD", "DC and DD |or| DD", "DC |or| DD", "DD"]
COLOURS = ["#d7301f", "#e6803a", "#e8d84a", "#a6e05a", "#5ee05e", "#7ff0a8",
           "#6ccbe0", "#1b9e77", "#3a63d8", "#4b1fd8", "#b43ad8", "#d6338f"]
OTHER_COLOUR = "#9e9e9e"


# ---------- step 1: type of each group ----------

def group_types(xs, fs, x_max=x_max, share_min=min_share, merge_gap=merge_gap):
    # xs: (..., 2 groups, 2 goods), fs: (..., 2 groups)
    # returns, per group: type (CC = 0, DC = 1, DD = 2), alive, and for DC groups the
    # good it invests in (0 or 1); plus close (...): the two groups count as one

    # groups closer than merge_gap count as one: put both at their share-weighted average
    close = np.linalg.norm(xs[..., 0, :] - xs[..., 1, :], axis=-1) < merge_gap
    average = np.sum(fs[..., None] * xs, axis=-2) / np.sum(fs, axis=-1)[..., None]
    xs = np.where(close[..., None, None], average[..., None, :], xs)

    # a good is "invested" if the group puts more than x_max / 2 into it
    invested = xs > x_max / 2                     # (..., group, good)
    n_invested = invested.sum(axis=-1)            # 2 -> CC, 1 -> DC, 0 -> DD
    types = 2 - n_invested
    good = np.argmax(invested, axis=-1)           # for DC groups: which good they invest in

    alive = fs > share_min
    return types, alive, good, close


# ---------- step 2: outcome of each run ----------

def run_outcomes(xs, fs, comparison=False, comparison_share_min=0.01, **kwargs):
    # comparison mode: round investments to 3 decimals and shares to 2 decimals, and
    # ignore groups with a share at or below comparison_share_min
    if comparison:
        xs, fs = np.round(xs, 3), np.round(fs, 2)
        kwargs["share_min"] = comparison_share_min
    types, alive, good, close = group_types(xs, fs, **kwargs)

    bits = np.where(alive, 1 << types, 0)         # (..., group)
    outcome = np.bitwise_or(bits[..., 0], bits[..., 1])

    # division of labour: two alive, separate DC groups investing in different goods
    division = (alive.all(axis=-1) & ~close & (types == 1).all(axis=-1)
                & (good[..., 0] != good[..., 1]))
    return np.where(division, 8, outcome)


# ---------- step 3: label of each grid point = set of run outcomes over starts ----------

def point_labels(outcomes, min_count=1):
    # outcomes: (start, ...) -> array of label strings with shape (...)
    # an outcome enters a point's set only if at least min_count starts produce it
    counts = {o: (outcomes == o).sum(axis=0) for o, _ in OUTCOMES}
    labels = np.empty(outcomes.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        present = [name for o, name in OUTCOMES if counts[o][idx] >= min_count]
        labels[idx] = " |or| ".join(present)
    return labels


# ---------- comparison mode: rare-label filter ----------

def replace_rare(labels, b1, b2, rare_frac=0.003):
    # labels: (p, q, b2', b1'). A label covering fewer than rare_frac of all grid points
    # (all panels together) is rare; each rare point takes the label of the nearest
    # non-rare point in the same panel.
    from scipy.spatial import cKDTree

    values, counts = np.unique(labels, return_counts=True)
    rare = {v: int(c) for v, c in zip(values, counts) if c < rare_frac * labels.size}

    B1, B2 = np.meshgrid(b1, b2)
    coords = np.stack([B1.ravel(), B2.ravel()], axis=-1)     # (b2' * b1', 2), same order as ravel
    out = labels.copy()
    replaced = np.zeros(labels.shape[:2], dtype=int)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            flat = labels[i, j].ravel()
            is_rare = np.array([l in rare for l in flat])
            if is_rare.any() and (~is_rare).any():
                _, nearest = cKDTree(coords[~is_rare]).query(coords[is_rare])
                new = flat.copy()
                new[is_rare] = flat[~is_rare][nearest]
                out[i, j] = new.reshape(labels.shape[2:])
                replaced[i, j] = is_rare.sum()
    return out, rare, replaced


# ---------- checks ----------

def check_classifier(x_max=x_max):
    # hand-made final states, written relative to x_max so the checks hold for any x_max
    # (the threshold for "invested" is x_max / 2)
    tiny = 0.000001
    hi, lo = 0.6 * x_max, 0.4 * x_max            # just above and just below x_max / 2
    m, almost = x_max, 0.98 * x_max
    cases = [
        (f"one group at ({m:g}, {m:g})",                     [[m, m], [m, m]],               "CC"),
        (f"one group at ({m:g}, 1e-6)",                      [[m, tiny], [m, tiny]],         "DC"),
        (f"one group at (1e-6, {m:g})",                      [[tiny, m], [tiny, m]],         "DC"),
        (f"groups at ({m:g}, {m:g}) and (1e-6, 1e-6)",       [[m, m], [tiny, tiny]],         "CC and DD"),
        (f"one group at ({hi:g}, {hi:g})",                   [[hi, hi], [hi, hi]],           "CC"),
        (f"one group at ({lo:g}, {lo:g})",                   [[lo, lo], [lo, lo]],           "DD"),
        (f"groups at ({m:g}, 1e-6) and (1e-6, {m:g})",       [[m, tiny], [tiny, m]],         "DC and DC'"),
        (f"groups at ({m:g}, 1e-6) and ({m:g}, 1e-6)",       [[m, tiny], [m, tiny]],         "DC"),
        (f"groups at ({m:g}, 1e-6) and ({almost:g}, 1e-6)",  [[m, tiny], [almost, tiny]],    "DC"),
        (f"groups at (1e-6, {m:g}) and (1e-6, 1e-6)",        [[tiny, m], [tiny, tiny]],      "DC and DD"),
    ]
    for name, xs, expected in cases:
        got = OUTCOME_NAME[int(run_outcomes(np.array(xs, float), np.array([0.5, 0.5]), x_max=x_max))]
        status = "ok" if got == expected else "FAILED"
        print(f"  {status:6s} {name:46s} -> {got!r} (expected {expected!r})")
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


def load_cache(path, settings):
    # returns (xs, fs) if the file exists and was made with exactly these settings, else None
    if not os.path.exists(path):
        return None
    data = np.load(path)
    for key, value in settings.items():
        if key not in data or np.shape(data[key]) != np.shape(value) or not np.allclose(data[key], value):
            print(f"{path} was made with different settings ({key}), ignoring it")
            return None
    return data["xs"], data["fs"]


# ---------- output helpers ----------

def print_counts(labels, p, q, title):
    index = {l: k for k, l in enumerate(LABELS)}
    print(f"\n=== label counts: {title} ===")
    for i in range(len(p)):
        for j in range(len(q)):
            values, counts = np.unique(labels[i, j], return_counts=True)
            print(f"p = {p[i]:g}, q = {q[j]:g}:")
            for v, c in sorted(zip(values, counts), key=lambda vc: index.get(vc[0], len(LABELS))):
                print(f"  {c:4d}  {v}" + ("" if v in index else "   <- not a known label (grey)"))


def report_division(labels, p, q, title):
    found = False
    for i in range(len(p)):
        for j in range(len(q)):
            n = int(np.array(["DC and DC'" in l for l in labels[i, j].ravel()]).sum())
            exact = int((labels[i, j] == "DC and DC'").sum())
            if n:
                found = True
                print(f"  {title}: p = {p[i]:g}, q = {q[j]:g}: {n} points contain DC and DC' "
                      f"({exact} labelled exactly \"DC and DC'\")")
    if not found:
        print(f"  {title}: DC and DC' appears in no panel")


def plot_labels(labels, p, q, b1, b2, title, filename):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    index = {l: k for k, l in enumerate(LABELS)}
    codes = np.vectorize(lambda l: index.get(l, len(LABELS)))(labels)     # other labels -> grey
    cmap = ListedColormap(COLOURS + [OTHER_COLOUR])
    extent = [b1[0], b1[-1], b2[0], b2[-1]]

    # p increases across the columns, q increases down the rows
    fig, axes = plt.subplots(len(q), len(p), sharex=True, sharey=True, figsize=(13, 10), squeeze=False)
    for i in range(len(p)):
        for j in range(len(q)):
            ax = axes[j, i]
            ax.imshow(codes[i, j], origin="lower", extent=extent, aspect="auto",
                      cmap=cmap, vmin=-0.5, vmax=len(LABELS) + 0.5, interpolation="nearest")
            ax.set_title(f"p = {p[i]:g}, q = {q[j]:g}", fontsize=10)
    handles = [Patch(color=c, label=l) for l, c in zip(LABELS, COLOURS)]
    if (codes == len(LABELS)).any():
        handles.append(Patch(color=OTHER_COLOUR, label="other"))
    fig.legend(handles=handles, loc="center right", fontsize=9)
    fig.supxlabel("b1'"); fig.supylabel("b2'")
    fig.suptitle(title)
    fig.subplots_adjust(right=0.78)
    fig.savefig(filename, dpi=150)
    print(f"saved {filename}")


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    min_count = 1                        # raise to drop outcomes that only a few starts reach
    comparison_share_min = 0.01          # comparison mode: ignore groups with share <= this
    rare_frac = 0.003                    # comparison mode: labels on < 0.3% of all points are rare
    cache = "two_good_results_v2.npz"    # results with the new starts
    old_cache = "two_good_results.npz"   # previous run (starts on the walls and diagonal), if present
    time_step, steps = 0.01, 10000

    b1 = np.linspace(0, 4, 50)           # b1' values (across)
    b2 = np.linspace(0, 4, 50)           # b2' values (up)
    p = np.array([0.5, 1, 1.5])
    q = np.array([0.5, 1, 1.5])

    # starts: 5 x 5 grid from 5% to 95% of x_max in each good (off the walls), each nudged by
    # up to 0.002 per good (off the exact diagonal), with a fixed seed so runs are reproducible
    s = np.linspace(0.05 * x_max, 0.95 * x_max, 5)
    S1, S2 = np.meshgrid(s, s)
    starts = np.stack([S1.ravel(), S2.ravel()], axis=-1)     # (25, 2)
    rng = np.random.default_rng(0)
    starts = np.clip(starts + rng.uniform(-0.002, 0.002, starts.shape), x_min, x_max)

    settings = dict(x_max=x_max, x_min=x_min, p=p, q=q, b1=b1, b2=b2, starts=starts,
                    time_step=time_step, steps=steps)
    cached = load_cache(cache, settings)
    if cached is not None:
        xs, fs = cached
        print(f"loaded final states from {cache}")
    else:
        print(f"NEW SIMULATION with the new starting points: "
              f"{len(starts) * len(p) * len(q) * len(b2) * len(b1)} runs ...")
        t = time.time()
        xs, fs = simulate(p, q, b1, b2, starts, time_step, steps)
        print(f"simulation took {time.time() - t:.0f}s")
        np.savez(cache, xs=xs, fs=fs, **settings)
    # xs: (start, p, q, b2', b1', group, good), fs: (start, p, q, b2', b1', group)

    print("classifier checks:")
    check_classifier()

    # --- comparison mode off: what actually happened ---
    labels = point_labels(run_outcomes(xs, fs), min_count)            # (p, q, b2', b1')
    print_counts(labels, p, q, "comparison mode OFF")

    # check: p = q = 1 against the one-good analytic diagram (only valid at p = q = 1,
    # where the model reduces to one good, so skip it if that panel isn't in the grid)
    if np.any(np.isclose(p, 1)) and np.any(np.isclose(q, 1)):
        i1, j1 = int(np.argmax(np.isclose(p, 1))), int(np.argmax(np.isclose(q, 1)))
        analytic = one_good_analytic_labels(b1, b2)
        sim = labels[i1, j1]
        differ = analytic != sim
        print(f"\np = q = 1 vs one-good analytic: {differ.sum()} of {sim.size} grid points differ")
        for a, s_ in sorted(set(zip(analytic[differ], sim[differ]))):
            n = int(((analytic == a) & (sim == s_)).sum())
            print(f"  analytic {a!r:16s} simulated {s_!r:28s} {n:3d} points")
    else:
        print("\np = q = 1 check skipped: p and q need to include 1")

    # --- comparison mode on: thresholds, rounding, rare-label filter ---
    labels_cmp_raw = point_labels(run_outcomes(xs, fs, comparison=True,
                                               comparison_share_min=comparison_share_min), min_count)
    labels_cmp, rare, replaced = replace_rare(labels_cmp_raw, b1, b2, rare_frac)
    print(f"\nrare labels (fewer than {rare_frac:.1%} of {labels_cmp_raw.size} points = "
          f"{rare_frac * labels_cmp_raw.size:g} points, all panels together):")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:4d}  {l}")
    if not rare:
        print("  none")
    for i in range(len(p)):
        for j in range(len(q)):
            print(f"  p = {p[i]:g}, q = {q[j]:g}: {replaced[i, j]} points replaced")
    print_counts(labels_cmp, p, q, "comparison mode ON")

    # --- division of labour ---
    print("\nDC and DC' (division of labour):")
    report_division(labels, p, q, "mode OFF")
    report_division(labels_cmp_raw, p, q, "mode ON, before rare filter")
    report_division(labels_cmp, p, q, "mode ON, after rare filter")
    division_label = "DC and DC'"
    if division_label in rare:
        print(f"  note: {division_label!r} was classed as rare ({rare[division_label]} points) "
              f"and replaced by the rare-label filter")

    # --- previous run (old starts) vs this run, both with comparison mode off ---
    if os.path.exists(old_cache):
        old = np.load(old_cache)
        same_grid = all(np.shape(old[k]) == np.shape(v) and np.allclose(old[k], v)
                        for k, v in dict(p=p, q=q, b1=b1, b2=b2).items())
        if same_grid:
            labels_old = point_labels(run_outcomes(old["xs"], old["fs"]), min_count)
            known = set(LABELS)
            print(f"\nprevious run ({old_cache}) vs new starts, comparison mode off:")
            for i in range(len(p)):
                for j in range(len(q)):
                    grey_old = sum(l not in known for l in labels_old[i, j].ravel())
                    grey_new = sum(l not in known for l in labels[i, j].ravel())
                    changed = labels_old[i, j] != labels[i, j]
                    print(f"  p = {p[i]:g}, q = {q[j]:g}: grey {grey_old} -> {grey_new}, "
                          f"{changed.sum()} points changed label")
                    moves = {}
                    for a, b_ in zip(labels_old[i, j][changed], labels[i, j][changed]):
                        moves[(a, b_)] = moves.get((a, b_), 0) + 1
                    for (a, b_), n in sorted(moves.items(), key=lambda kv: -kv[1])[:5]:
                        print(f"      {n:4d}  {a!r} -> {b_!r}")
        else:
            print(f"\n{old_cache} uses a different grid, skipping the comparison with the previous run")
    else:
        print(f"\nno previous run ({old_cache}) found, skipping the comparison")

    plot_labels(labels, p, q, b1, b2,
                f"Comparison mode OFF: what actually happened (min_count = {min_count})",
                "two_good_labels_actual.png")
    plot_labels(labels_cmp, p, q, b1, b2,
                f"Comparison mode ON: share > {comparison_share_min:g}, rounded, "
                f"rare labels (< {rare_frac:.1%}) replaced by nearest",
                "two_good_labels_comparison.png")
    plt.show()
