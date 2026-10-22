# Two-good adaptive dynamics with repeated splitting, run until convergence, labelled with the
# reference scheme (CC / DC / DD) and plotted as a 3 x 3 grid of phase diagrams.
#
# Each run (one (p, q, b1', b2') and one start) is advanced CHUNK steps at a time. After each
# chunk it is checked once: groups that have stopped at a branching point split, settled runs
# finish, runs that cycle get a smaller time step. Runs start with 2 group slots and move up to
# 4, then 8, only if a split needs room.
#
# Usage:  python two_good_chunked.py              (100 x 100 grid)
#         python two_good_chunked.py --n 10       (quick test)
#         python two_good_chunked.py --division   (label division of labour as "DC and DC'")
#         in Colab: %run two_good_chunked.py --n 10

import argparse
import os
import time
from collections import deque
import numpy as np
import jax

USE_X64 = False                      # 32-bit gave identical labels and is much faster on GPU
jax.config.update("jax_enable_x64", USE_X64)
import jax.numpy as jnp

# ---------- settings ----------
x_max, x_min = 2.0, 1e-6
P_VALUES = [0.8, 1.0, 1.25]
Q_VALUES = [0.8, 1.0, 1.25]
GRID_N = 100                         # b1' and b2' from 0 to 4 on a GRID_N x GRID_N grid
STARTS = [(0.01, 0.01), (0.01, 1.99), (1.99, 0.01), (1.99, 1.99)]

TIME_STEP = 0.01                     # starting time step (halved for runs that cycle)
MIN_DT = 1e-5
CHUNK = 1000                         # steps between checks
MAX_STEPS = 200_000                  # cap per run
TOL = 1e-4                           # settled: gradients and payoff gaps below this
K_LEVELS = [2, 4, 8]                 # group slots per run
BATCH = {2: 16384, 4: 2048, 8: 512}  # runs advanced together at each slot count
SMALL_BATCH = 256                    # batch size once few runs are left
SPLIT_GAP = 0.01
MERGE_GAP = 1e-3
MIN_SHARE = 1e-4                     # smaller groups are removed
LAM_MIN = 1e-6                       # split only if the curvature is above this
CASCADE_DIST = 0.05                  # a group splits again only after moving this far from its last split
CYCLE_RATIO = 0.1                    # cycling: net movement < this x (CHUNK * dt * gradient)

LABEL_SHARE_MIN = 0.01               # labelling: a group counts if its (rounded) share is above this
RARE_FRAC = 0.003                    # labels on fewer than 0.3% of all grid points are rare
DIVISION_OF_LABOUR = False           # True: DC groups in different goods -> "DC and DC'"

# fixed colours: hsv from 0 to 0.9 over these 11 labels in this order (as in the reference)
STANDARD_LABELS = ["CC", "CC and DC", "CC and DD", "CC |or| DC", "CC |or| DC |or| DD", "CC |or| DD",
                   "DC", "DC and DD", "DC and DD |or| DD", "DC |or| DD", "DD"]
DIVISION_COLOUR = "#1b9e77"                                  # "DC and DC'"
DIVISION_OTHER_COLOURS = ["#66a61e", "#a6761d", "#7570b3"]   # other labels containing it
OTHER_COLOUR = "#9e9e9e"


# ---------- model ----------

M = lambda pow, a, b: ((a ** pow + b ** pow) / 2) ** (1 / pow)

def P(y, x, b1, b2, p, q):
    # y: own traits (goods,), x: others' traits (..., goods) -- [..., k] picks good k
    Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
    Mq = M(q, y[..., 0], y[..., 1])
    return b1 * Mp - b2 * Mp ** 2 - Mq + Mq ** 2


def top_eigen(H):
    # largest eigenvalue and its eigenvector of a 2x2 symmetric matrix [[a, b], [b, d]]
    a, b, d = H[0, 0], H[0, 1], H[1, 1]
    lam = (a + d) / 2 + jnp.sqrt(((a - d) / 2)**2 + b**2)
    v = jnp.array([b, lam - a])
    v = jnp.where(b != 0, v, jnp.where(a >= d, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))
    return lam, v / jnp.linalg.norm(v)


# ---------- one population: CHUNK steps, and the check after them ----------
# xs: (K slots, 2 goods), fs: (K,); a slot with share 0 is unused.

def run_chunk(p, q, b1, b2, xs, fs, dt):

    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    group_payoffs = jax.vmap(payoff, in_axes=(0, None, None))
    group_grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))
    K = xs.shape[0]
    slots = jnp.arange(K)

    def step(state, _):
        xs, fs = state
        alive = fs > 0

        # 1. compute group grads
        grads = group_grads(xs, xs, fs)

        # 2. update investments (used slots only)
        new_xs = jnp.where(alive[:, None], jnp.clip(xs + dt * grads, x_min, x_max), xs)

        # 3. calculate per group payoffs and average payoff
        payoffs = group_payoffs(xs, xs, fs)
        avg_payoff = jnp.sum(fs * payoffs)

        # 4. update population shares
        new_fs = fs + dt * fs * (payoffs - avg_payoff)
        xs, fs = new_xs, new_fs / jnp.sum(new_fs)

        # 5. min check: remove groups below MIN_SHARE
        fs = jnp.where(fs < MIN_SHARE, 0.0, fs)
        fs = fs / jnp.sum(fs)
        alive = fs > 0

        # 6. merge check: the closest pair of groups, if closer than MERGE_GAP
        dist = jnp.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
        pair = alive[:, None] & alive[None, :] & (slots[:, None] < slots[None, :])
        dist = jnp.where(pair, dist, jnp.inf)
        i, j = jnp.argmin(dist) // K, jnp.argmin(dist) % K
        merge = dist[i, j] < MERGE_GAP
        merged_x = (fs[i] * xs[i] + fs[j] * xs[j]) / (fs[i] + fs[j])
        xs = jnp.where(merge, xs.at[i].set(merged_x), xs)
        fs = jnp.where(merge, fs.at[i].add(fs[j]).at[j].set(0.0), fs)

        return (xs, fs), None

    (xs, fs), _ = jax.lax.scan(step, (xs, fs), None, length=CHUNK)
    return xs, fs


def check(p, q, b1, b2, xs, fs):
    # g:   per group, size of the gradient (a good pinned at a wall and pushed outwards counts as 0)
    # gap: largest difference between a group's payoff and the average (shares still changing)
    # lam, v: per group, largest curvature and its direction, goods at a wall left out
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    alive = fs > 0

    grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)
    pushing_out = ((xs <= x_min) & (grads <= 0)) | ((xs >= x_max) & (grads >= 0))
    g = jnp.max(jnp.where(pushing_out, 0.0, jnp.abs(grads)), axis=-1)

    payoffs = jax.vmap(payoff, in_axes=(0, None, None))(xs, xs, fs)
    gap = jnp.max(jnp.where(alive, jnp.abs(payoffs - jnp.sum(fs * payoffs)), 0.0))

    free = (x_min < xs) & (xs < x_max)
    H = jax.vmap(jax.hessian(payoff), in_axes=(0, None, None))(xs, xs, fs)
    lam, v = jax.vmap(top_eigen)(H * free[:, :, None] * free[:, None, :])
    return g, gap, lam, v


run_chunk_all = jax.jit(jax.vmap(run_chunk))     # vmap: many populations at once
check_all = jax.jit(jax.vmap(check))


# ---------- running every population until it has settled ----------

class Runs:
    # state of every run; slots up to the largest K, unused ones with share 0
    def __init__(self, n):
        dtype = np.float64 if USE_X64 else np.float32
        b = np.linspace(0, 4, n)
        I, Pg, Qg, Y, X = np.meshgrid(np.arange(len(STARTS)), P_VALUES, Q_VALUES, b, b, indexing="ij")
        self.shape = I.shape                              # (start, p, q, b2', b1')
        N, k = I.size, K_LEVELS[-1]
        self.params = np.stack([Pg.ravel(), Qg.ravel(), X.ravel(), Y.ravel()], axis=1).astype(dtype)
        self.xs = np.repeat(np.array(STARTS)[I.ravel()][:, None, :], k, axis=1).astype(dtype)
        self.fs = np.zeros((N, k), dtype)
        self.fs[:, 0] = 1.0
        self.origin = np.full((N, k, 2), np.nan)          # where each group last split
        self.dt = np.full(N, TIME_STEP, dtype)
        self.steps = np.zeros(N, int)
        self.status = np.zeros(N, int)                    # 0 running, 1 converged, 2 hit MAX_STEPS
        self.max_k = np.full(N, K_LEVELS[0])              # most slots the run needed
        self.overflow = np.zeros(N, bool)                 # wanted more than K_LEVELS[-1] slots
        self.blocked = np.zeros(N, bool)                  # ended with a split blocked by CASCADE_DIST


def advance(runs, ids, K, size):
    # one chunk for the runs in ids (all with K slots); returns the runs that stay at this K
    ids = np.array(ids)
    padded = np.concatenate([ids, np.repeat(ids[:1], size - len(ids))])     # fixed batch size
    p, q, b1, b2 = runs.params[padded].T
    x_start = runs.xs[ids, :K]
    xs, fs = run_chunk_all(p, q, b1, b2, runs.xs[padded, :K], runs.fs[padded, :K], runs.dt[padded])
    g, gap, lam, v = (np.asarray(a)[:len(ids)] for a in check_all(p, q, b1, b2, xs, fs))
    xs, fs = np.asarray(xs)[:len(ids)], np.asarray(fs)[:len(ids)]
    runs.xs[ids, :K], runs.fs[ids, :K] = xs, fs
    runs.steps[ids] += CHUNK

    # which groups want to split, and which runs have settled
    alive = fs > 0
    stopped = alive & (g < TOL)
    far = ~(np.linalg.norm(xs - runs.origin[ids, :K], axis=-1) < CASCADE_DIST)   # never split -> far
    want = stopped & (lam > LAM_MIN) & far
    blocked = stopped & (lam > LAM_MIN) & ~far
    g_max = np.where(alive, g, 0.0).max(axis=1)
    converged = (g_max < TOL) & (gap < TOL) & ~want.any(axis=1)

    # cycling (large gradient, hardly any net movement): halve dt; otherwise let it grow back
    net = np.where(alive, np.linalg.norm(xs - x_start, axis=-1), 0.0).max(axis=1)
    cycling = ~converged & ~want.any(axis=1) & (net < CYCLE_RATIO * CHUNK * runs.dt[ids] * g_max)
    runs.dt[ids] = np.where(cycling, np.maximum(runs.dt[ids] / 2, MIN_DT),
                            np.minimum(runs.dt[ids] * 2, TIME_STEP))

    # splits; a run that needs more slots moves up a level
    moved_up = np.zeros(len(ids), bool)
    for r in np.flatnonzero(want.any(axis=1)):
        new_K = split(runs, ids[r], np.flatnonzero(want[r]), v[r], K)
        moved_up[r] = new_K != K

    done = converged | (runs.steps[ids] >= MAX_STEPS)
    runs.status[ids] = np.where(converged, 1, np.where(runs.steps[ids] >= MAX_STEPS, 2, 0))
    runs.blocked[ids] = np.where(done, blocked.any(axis=1), runs.blocked[ids])
    return ids[~done & ~moved_up], ids[moved_up & ~done]


def split(runs, run, groups, v, K):
    # split each group in groups along its direction v into a free slot; returns the slot count used
    levels = K_LEVELS[K_LEVELS.index(K):]
    K = next((k for k in levels if (runs.fs[run, :k] == 0).sum() >= len(groups)), levels[-1])
    free = list(np.flatnonzero(runs.fs[run, :K] == 0))
    if len(free) < len(groups):
        runs.overflow[run] = True
    for w in groups[:len(free)]:
        f = free.pop(0)
        x = runs.xs[run, w].copy()
        runs.xs[run, w] = np.clip(x - SPLIT_GAP * v[w], x_min, x_max)
        runs.xs[run, f] = np.clip(x + SPLIT_GAP * v[w], x_min, x_max)
        runs.fs[run, w] = runs.fs[run, f] = runs.fs[run, w] / 2
        runs.origin[run, w] = runs.origin[run, f] = x
    runs.max_k[run] = max(runs.max_k[run], K)
    return K


def simulate(n):
    runs = Runs(n)
    N = runs.status.size
    waiting = {K: deque() for K in K_LEVELS}           # runs waiting for a place in a batch
    running = {K: [] for K in K_LEVELS}                # runs in the batch at each slot count
    waiting[K_LEVELS[0]].extend(range(N))
    t0, chunks = time.time(), 0

    while any(waiting[K] or running[K] for K in K_LEVELS):
        for K in K_LEVELS:
            if not (waiting[K] or running[K]):
                continue
            # fixed batch size (smaller once few runs are left), topped up from the waiting runs
            big = min(BATCH[K], N)
            size = big if len(waiting[K]) + len(running[K]) > SMALL_BATCH else min(SMALL_BATCH, big)
            while len(running[K]) < size and waiting[K]:
                running[K].append(waiting[K].popleft())

            stay, moved_up = advance(runs, running[K], K, size)
            running[K] = list(stay)
            for run in moved_up:
                waiting[runs.max_k[run]].append(run)
            chunks += 1

        if chunks % 20 == 0:
            print(f"  {time.time() - t0:6.0f}s  finished {int((runs.status > 0).sum())}/{N}  "
                  + ", ".join(f"K={K}: {len(waiting[K]) + len(running[K])} left" for K in K_LEVELS), flush=True)
    return runs


# ---------- labelling ----------

def outcome_name(bits):
    # bits: 1 = CC present, 2 = DC present, 4 = DD present, 8 = DC groups in both goods
    # -> distinct group types, alphabetical, joined with " and "
    types = []
    if bits & 1:
        types.append("CC")
    if bits & 2:
        types.append("DC and DC'" if bits & 8 else "DC")
    if bits & 4:
        types.append("DD")
    return " and ".join(types) if types else "none"


def run_outcomes(xs, fs, division=DIVISION_OF_LABOUR):
    # xs: (..., K, 2), fs: (..., K) -> outcome name per run
    xs, fs = np.round(xs, 3), np.round(fs, 2)
    counted = fs > LABEL_SHARE_MIN

    # groups closer than MERGE_GAP count as one: drop a group if an earlier counted one is close
    K = xs.shape[-2]
    dist = np.linalg.norm(xs[..., :, None, :] - xs[..., None, :, :], axis=-1)
    earlier = np.tri(K, k=-1, dtype=bool)                                     # [j, i]: i < j
    counted &= ~np.any(counted[..., None, :] & earlier & (dist < MERGE_GAP), axis=-1)

    # a good is "invested" above x_max / 2: CC = both goods, DC = one, DD = neither
    invested = xs > x_max / 2
    n_invested = invested.sum(axis=-1)
    has = lambda n: np.any(counted & (n_invested == n), axis=-1)
    dc_in = lambda good: np.any(counted & (n_invested == 1) & invested[..., good], axis=-1)
    division_of_labour = division & dc_in(0) & dc_in(1)

    bits = has(2) * 1 + has(1) * 2 + has(0) * 4 + division_of_labour * 8
    names = np.array([outcome_name(k) for k in range(16)], dtype=object)
    return names[bits]


def grid_labels(outcomes):
    # outcomes: (start, ...) -> label per grid point: distinct outcomes, alphabetical, " |or| "
    labels = np.empty(outcomes.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        labels[idx] = " |or| ".join(sorted(set(outcomes[(slice(None),) + idx])))
    return labels


def replace_rare(labels, b):
    # labels on fewer than RARE_FRAC of all grid points (all panels together) are rare; each rare
    # point takes the label of the nearest non-rare point in the same panel
    from scipy.spatial import cKDTree
    values, counts = np.unique(labels, return_counts=True)
    rare = {v: int(c) for v, c in zip(values, counts) if c < RARE_FRAC * labels.size}
    B1, B2 = np.meshgrid(b, b)
    points = np.stack([B1.ravel(), B2.ravel()], axis=-1)
    out = labels.copy()
    replaced = np.zeros(labels.shape[:2], int)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            flat = labels[i, j].ravel()
            is_rare = np.isin(flat, list(rare))
            if is_rare.any() and not is_rare.all():
                _, nearest = cKDTree(points[~is_rare]).query(points[is_rare])
                flat = flat.copy()
                flat[is_rare] = flat[~is_rare][nearest]
                out[i, j] = flat.reshape(labels.shape[2:])
                replaced[i, j] = is_rare.sum()
    return out, rare, replaced


# ---------- checks ----------

def check_payoff():
    for y, x, b1, b2, p, q, expected in [((0.5, 0.5), (0.5, 0.5), 2.0, 0.75, 1.0, 1.0, 1.0),
                                         ((1.0, 0.0), (1.0, 0.0), 2.0, 0.75, 2.0, 0.5, 1.141)]:
        got = float(P(jnp.array(y), jnp.array(x), b1, b2, p, q))
        print(f"  {'ok' if abs(got - expected) < 1e-3 else 'FAILED':6s} P(y={y}, x={x}, b1'={b1}, "
              f"b2'={b2}, p={p}, q={q}) = {got:.4f} (expected {expected})")
        assert abs(got - expected) < 1e-3


def check_labeller():
    t = 1e-6
    def outcome(groups, shares, division=False):
        xs = np.full((4, 2), t); fs = np.zeros(4)
        xs[:len(groups)], fs[:len(shares)] = groups, shares
        return run_outcomes(xs, fs, division)
    cases = [
        ("one group at (2, 2)",                          outcome([(2, 2)], [1.0]),                          "CC"),
        ("groups at (2, 2) and (1e-6, 1e-6)",            outcome([(2, 2), (t, t)], [0.5, 0.5]),             "CC and DD"),
        ("groups at (2, 1e-6) and (1e-6, 2)",            outcome([(2, t), (t, 2)], [0.5, 0.5]),             "DC"),
        ("same, division of labour on",                  outcome([(2, t), (t, 2)], [0.5, 0.5], True),       "DC and DC'"),
        ("(2, 2) share 0.995 + (1e-6, 1e-6) share 0.005", outcome([(2, 2), (t, t)], [0.995, 0.005]),        "CC"),
        ("one group at (1.0004, 1.0004) (rounds to 1)",  outcome([(1.0004, 1.0004)], [1.0]),                "DD"),
        ("three groups CC, DC, DD",                      outcome([(2, 2), (2, t), (t, t)], [0.4, 0.3, 0.3]), "CC and DC and DD"),
    ]
    for name, got, expected in cases:
        print(f"  {'ok' if got == expected else 'FAILED':6s} {name:46s} -> {got!r}")
        assert got == expected
    got = grid_labels(np.array([["DD"], ["CC and DD"], ["DD"], ["CC"]], dtype=object))[0]
    print(f"  {'ok' if got == 'CC |or| CC and DD |or| DD' else 'FAILED':6s} {'grid label from DD, CC and DD, DD, CC':46s} -> {got!r}")
    assert got == "CC |or| CC and DD |or| DD"


def check_one_good(labels, b):
    # p = q = 1 reduces to the one-good model: compare with its analytic diagram away from boundaries
    if 1.0 not in P_VALUES or 1.0 not in Q_VALUES:
        print("p = q = 1 check skipped: p and q need to include 1")
        return
    B1, B2 = np.meshgrid(b, b)
    D0, Dmax = B1 - 1, (B1 - 1) + (2 - 4 * B2) * x_max
    with np.errstate(divide="ignore", invalid="ignore"):
        x_star = D0 / (4 * B2 - 2)
    analytic = np.select([(D0 > 0) & (Dmax > 0), (D0 < 0) & (Dmax < 0), (D0 < 0) & (Dmax > 0),
                          (D0 > 0) & (Dmax < 0) & (B2 < 1), (D0 > 0) & (Dmax < 0) & (B2 > 1)],
                         ["CC", "DD", "CC |or| DD", "CC and DD", np.where(x_star > x_max / 2, "CC", "DD")],
                         default="boundary")
    pad = np.pad(analytic, 1, mode="edge")
    boundary = analytic == "boundary"
    for di, dj in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        boundary |= pad[di:di + len(b), dj:dj + len(b)] != analytic
    sim = labels[P_VALUES.index(1.0), Q_VALUES.index(1.0)]
    differ = (sim != analytic) & ~boundary
    print(f"p = q = 1 vs one-good analytic, away from boundaries: {differ.sum()} of {(~boundary).sum()} points differ")
    for a, s in sorted(set(zip(analytic[differ], sim[differ]))):
        print(f"  analytic {a!r:14s} simulated {s!r:28s} {int(((analytic == a) & (sim == s) & differ).sum()):4d} points")


# ---------- output ----------

def label_colours(labels):
    # legend labels and colours: the 11 standard labels, then labels with "DC and DC'", then "other"
    import matplotlib.pyplot as plt
    present = set(labels.ravel())
    names = list(STANDARD_LABELS)
    colours = [tuple(c) for c in plt.cm.hsv(np.linspace(0, 0.9, len(STANDARD_LABELS)))]
    division = sorted(l for l in present if "DC and DC'" in l)
    if DIVISION_OF_LABOUR and "DC and DC'" not in division:
        division = ["DC and DC'"] + division
    extra = [l for l in division if l != "DC and DC'"]
    for l in division:
        names.append(l)
        if l == "DC and DC'":
            colours.append(DIVISION_COLOUR)
        else:
            few = len(extra) <= len(DIVISION_OTHER_COLOURS)
            colours.append(DIVISION_OTHER_COLOURS[extra.index(l)] if few else DIVISION_OTHER_COLOURS[0])
    if not present <= set(names):
        names.append("other")
        colours.append(OTHER_COLOUR)
    return names, colours


def plot(labels, b, title, filename):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    names, colours = label_colours(labels)
    index = {l: k for k, l in enumerate(names)}
    codes = np.vectorize(lambda l: index.get(l, index.get("other")))(labels)
    extent = [b[0], b[-1], b[0], b[-1]]

    # columns: p left to right, rows: q top to bottom
    fig, axes = plt.subplots(len(Q_VALUES), len(P_VALUES), sharex=True, sharey=True,
                             figsize=(15, 11), squeeze=False)
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            ax = axes[j, i]
            ax.imshow(codes[i, j], origin="lower", extent=extent, aspect="auto", interpolation="nearest",
                      cmap=ListedColormap(colours), vmin=-0.5, vmax=len(names) - 0.5)
            ax.set_title(f"Phase diagram p = {p:g} q = {q:g}", fontsize=10)
            ax.set_xlabel("b1'"); ax.set_ylabel("b2'")
    fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(names, colours)], loc="center left", fontsize=8)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.27, wspace=0.25, hspace=0.3)
    fig.savefig(filename, dpi=150)
    print(f"saved {filename}")


def print_counts(labels, title):
    names, _ = label_colours(labels)
    print(f"\n=== label counts: {title} ===")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            print(f"p = {p:g}, q = {q:g}:")
            values, counts = np.unique(labels[i, j], return_counts=True)
            for v, c in zip(values, counts):
                print(f"  {c:5d}  {v}" + ("" if v in names else "   <- other (grey)"))


def print_convergence(info):
    status, steps = info["status"], info["steps"]
    print(f"\nruns converged: {int((status == 1).sum())} of {status.size}; "
          f"stopped at MAX_STEPS = {MAX_STEPS} without converging: {int((status == 2).sum())}")
    print(f"steps used: median {int(np.median(steps))}, 90% {int(np.percentile(steps, 90))}, "
          f"99% {int(np.percentile(steps, 99))}, max {int(steps.max())}")
    print(f"runs that ended with a reduced time step (still cycling): {int((info['dt'] < TIME_STEP).sum())}")
    print("runs that needed more slots: " + ", ".join(f"K={K}: {int((info['max_k'] >= K).sum())}" for K in K_LEVELS))
    print(f"runs wanting more than {K_LEVELS[-1]} slots: {int(info['overflow'].sum())}; "
          f"runs that ended with a split blocked by CASCADE_DIST: {int(info['blocked'].sum())}")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            capped = status[:, i, j] == 2
            print(f"  p = {p:g}, q = {q:g}: {int(capped.sum())} runs not converged, "
                  f"{int(capped.any(axis=0).sum())} grid points with at least one")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    parser.add_argument("--division", action="store_true", help="label division of labour as \"DC and DC'\"")
    args = parser.parse_args()
    DIVISION_OF_LABOUR = DIVISION_OF_LABOUR or args.division
    b = np.linspace(0, 4, args.n)

    tag = (f"n{args.n}_chunk{CHUNK}_max{MAX_STEPS}_dt{TIME_STEP:g}_tol{TOL:g}"
           f"_K{'-'.join(map(str, K_LEVELS))}_{'f64' if USE_X64 else 'f32'}")
    dol_tag = f"division_of_labour={'on' if DIVISION_OF_LABOUR else 'off'}"
    cache = f"chunked_results_{tag}.npz"

    print("running on:", jax.devices())
    print("payoff checks:"); check_payoff()
    print("labeller checks:"); check_labeller()

    if os.path.exists(cache):
        info = dict(np.load(cache))
        print(f"loaded final states from {cache}")
    else:
        n_runs = len(STARTS) * len(P_VALUES) * len(Q_VALUES) * args.n**2
        print(f"simulating {n_runs} runs in chunks ({tag}) ...")
        t = time.time()
        runs = simulate(args.n)
        print(f"simulation took {time.time() - t:.0f}s")
        s = runs.shape
        info = dict(xs=runs.xs.reshape(*s, -1, 2), fs=runs.fs.reshape(*s, -1), steps=runs.steps.reshape(s),
                    status=runs.status.reshape(s), dt=runs.dt.reshape(s), max_k=runs.max_k.reshape(s),
                    overflow=runs.overflow.reshape(s), blocked=runs.blocked.reshape(s))
        np.savez(cache, **info)
    print_convergence(info)

    labels_raw = grid_labels(run_outcomes(info["xs"], info["fs"], DIVISION_OF_LABOUR))   # (p, q, b2', b1')
    labels, rare, replaced = replace_rare(labels_raw, b)
    print(f"\nrare labels (fewer than {RARE_FRAC:.1%} of {labels_raw.size} points = {RARE_FRAC * labels_raw.size:g}):")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:5d}  {l}")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            print(f"  p = {p:g}, q = {q:g}: {replaced[i, j]} points replaced")

    print()
    check_one_good(labels_raw, b)
    print_counts(labels_raw, f"unfiltered, {dol_tag}")
    print_counts(labels, f"filtered, {dol_tag}")
    plot(labels, b, f"Filtered (rare labels < {RARE_FRAC:.1%} replaced), {dol_tag} - {tag}",
         f"chunked_filtered_{tag}_{dol_tag}.png")
    plot(labels_raw, b, f"Unfiltered: what actually happened, {dol_tag} - {tag}",
         f"chunked_unfiltered_{tag}_{dol_tag}.png")
    plt.show()
