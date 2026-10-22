# Two-good simulation with repeated splitting (K group slots per run), labelled with
# settings chosen to match a reference set of phase diagrams.
# Reuses the payoff from two_good_integration_phase.py and the rare-label filter from
# two_good_labels.py; neither file is changed.
#
# Usage:  python two_good_reference.py            (full 100 x 100 grid)
#         python two_good_reference.py --n 10     (quick 10 x 10 test)
#         in Colab: %run two_good_reference.py --n 10

import argparse
import os
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)          # 64-bit floats, before any arrays are made
import jax.numpy as jnp

from two_good_integration_phase import P
from two_good_labels import replace_rare, one_good_analytic_labels

# ---------- settings ----------
x_max, x_min = 2.0, 1e-6
P_VALUES = [0.8, 1.0, 1.25]
Q_VALUES = [0.8, 1.0, 1.25]
GRID_N = 100                         # b1' and b2' each from 0 to 4 on a GRID_N x GRID_N grid
STARTS = [(0.01, 0.01), (0.01, 1.99), (1.99, 0.01), (1.99, 1.99)]

# dynamics (the reference values are unknown; these are the current defaults)
K = 4                                # group slots per run
TIME_STEP = 0.01
STEPS = 10000
STILL = 1e-6                         # "stopped" if speed < STILL / TIME_STEP
SPLIT_GAP = 0.01
MERGE_GAP = 1e-3
MIN_SHARE = 1e-4                     # groups below this share are removed
LAM_MIN = 1e-6                       # split only if the largest curvature eigenvalue is above this
SPLIT_COOLDOWN = 100                 # a group can't split again within this many steps of a split
ANNEAL_FRAC, ANNEAL_TO = 0.2, 1e-3   # last 20% of steps: dt shrinks to dt * 1e-3

# labelling (reference scheme)
LABEL_SHARE_MIN = 0.01               # a group counts only if its (rounded) share is above this
RARE_FRAC = 0.003                    # labels on fewer than 0.3% of all grid points are rare
DIVISION_OF_LABOUR = False           # True: two DC groups in different goods -> "DC and DC'"


# ---------- dynamics ----------

def top_eigen(H):
    # largest eigenvalue and its eigenvector of a 2x2 symmetric matrix [[a, b], [b, d]]
    a, b, d = H[0, 0], H[0, 1], H[1, 1]
    lam = (a + d) / 2 + jnp.sqrt(((a - d) / 2)**2 + b**2)
    v = jnp.array([b, lam - a])
    v = jnp.where(b != 0, v, jnp.where(a >= d, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))
    return lam, v / jnp.linalg.norm(v)


def k_group_run(p, q, b1, b2, x0, time_step=TIME_STEP, steps=STEPS, K=K, still=STILL,
                split_gap=SPLIT_GAP, merge_gap=MERGE_GAP, min_share=MIN_SHARE, lam_min=LAM_MIN,
                split_cooldown=SPLIT_COOLDOWN, anneal_frac=ANNEAL_FRAC, anneal_to=ANNEAL_TO):
    # One run for one (p, q, b1', b2') and one start x0 = (good 1, good 2).
    # xs: (K slots, 2 goods), fs: (K,); a slot with share 0 is unused.
    # Returns final xs, fs and overflow = a split was wanted but no slot was free.

    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    group_payoffs = jax.vmap(payoff, in_axes=(0, None, None))                        # (K,)
    group_grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))                # (K, 2)
    group_curvs = jax.vmap(jax.hessian(payoff), in_axes=(0, None, None))             # (K, 2, 2)
    slots = jnp.arange(K)

    anneal_start = steps - int(steps * anneal_frac)
    anneal_len = max(steps - anneal_start, 1)

    def step(state, t):
        xs, fs, age, overflow = state
        dt = time_step * anneal_to ** jnp.clip((t - anneal_start) / anneal_len, 0.0, 1.0)
        alive = fs > 0

        # 1. gradients, 2. investment update (only for used slots)
        grads = group_grads(xs, xs, fs)
        new_xs = jnp.where(alive[:, None], jnp.clip(xs + dt * grads, x_min, x_max), xs)

        # 3. payoffs, 4. share update
        payoffs = group_payoffs(xs, xs, fs)
        avg_payoff = jnp.sum(fs * payoffs)
        new_fs = fs + dt * fs * (payoffs - avg_payoff)
        new_fs = new_fs / jnp.sum(new_fs)

        moved = jnp.linalg.norm(new_xs - xs, axis=-1)
        xs, fs, age = new_xs, new_fs, age + 1

        # 5. remove groups below min_share (slot freed)
        fs = jnp.where(fs < min_share, 0.0, fs)
        fs = fs / jnp.sum(fs)
        alive = fs > 0

        # 6. merge the closest pair of used slots, if closer than merge_gap (one merge per step)
        dist = jnp.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
        pair = alive[:, None] & alive[None, :] & (slots[:, None] < slots[None, :])
        dist = jnp.where(pair, dist, jnp.inf)
        k = jnp.argmin(dist)
        i, j = k // K, k % K
        merge = dist[i, j] < merge_gap
        merged_x = (fs[i] * xs[i] + fs[j] * xs[j]) / (fs[i] + fs[j])
        xs = jnp.where(merge, xs.at[i].set(merged_x), xs)
        fs = jnp.where(merge, fs.at[i].add(fs[j]).at[j].set(0.0), fs)
        moved = jnp.where(merge, moved.at[i].set(0.0), moved)
        alive = fs > 0

        # 7. split one group that has stopped and whose payoff curves upwards (largest eigenvalue
        #    of the curvature matrix, goods at a wall removed), into the first free slot
        free_goods = (x_min < xs) & (xs < x_max)                                     # (K, 2)
        H = group_curvs(xs, xs, fs) * free_goods[:, :, None] * free_goods[:, None, :]
        lam, v = jax.vmap(top_eigen)(H)
        stopped = moved / dt < still / time_step
        wants = alive & stopped & (lam > lam_min) & (age >= split_cooldown)
        has_free = jnp.any(~alive)
        i = jnp.argmax(wants)                     # first group that wants to split
        j = jnp.argmax(~alive)                    # first free slot
        split = wants[i] & has_free
        overflow = overflow | (jnp.any(wants) & ~has_free)
        new_i = jnp.clip(xs[i] - split_gap * v[i], x_min, x_max)
        new_j = jnp.clip(xs[i] + split_gap * v[i], x_min, x_max)
        xs = jnp.where(split, xs.at[i].set(new_i).at[j].set(new_j), xs)
        fs = jnp.where(split, fs.at[j].set(fs[i] / 2).at[i].set(fs[i] / 2), fs)
        age = jnp.where(split, age.at[i].set(0).at[j].set(0), age)

        return (xs, fs, age, overflow), None

    xs = jnp.tile(jnp.asarray(x0, dtype=float), (K, 1))
    fs = jnp.zeros(K).at[0].set(1.0)
    age = jnp.full(K, split_cooldown)
    (xs, fs, age, overflow), _ = jax.lax.scan(step, (xs, fs, age, False), jnp.arange(steps))
    return xs, fs, overflow


def simulate(n, settings):
    b = np.linspace(0, 4, n)
    starts = np.array(STARTS)
    I, Pg, Qg, Y, X = np.meshgrid(np.arange(len(starts)), P_VALUES, Q_VALUES, b, b, indexing="ij")
    run_one = lambda p, q, b1, b2, x0: k_group_run(p, q, b1, b2, x0, **settings)
    run_all = jax.jit(jax.vmap(run_one))
    xs, fs, overflow = run_all(Pg.ravel(), Qg.ravel(), X.ravel(), Y.ravel(), starts[I.ravel()])
    shape = I.shape                                       # (start, p, q, b2', b1')
    return (np.asarray(xs).reshape(*shape, -1, 2), np.asarray(fs).reshape(*shape, -1),
            np.asarray(overflow).reshape(shape))


# ---------- labelling (reference scheme) ----------

def outcome_bits(xs, fs, share_min=LABEL_SHARE_MIN, merge_gap=MERGE_GAP, division=DIVISION_OF_LABOUR):
    # xs: (..., K, 2), fs: (..., K) -> one number per run: bit 1 = CC, 2 = DC, 4 = DD,
    # 8 = DC groups in both goods (only used if division is on)
    xs, fs = np.round(xs, 3), np.round(fs, 2)
    counted = fs > share_min

    # groups closer than merge_gap count as one: drop a group if an earlier counted group is close
    k = xs.shape[-2]
    dist = np.linalg.norm(xs[..., :, None, :] - xs[..., None, :, :], axis=-1)    # (..., K, K)
    earlier = np.tri(k, k=-1, dtype=bool)                                         # [j, i] = i < j
    duplicate = np.any(counted[..., None, :] & earlier & (dist < merge_gap), axis=-1)
    counted = counted & ~duplicate

    invested = xs > x_max / 2                       # (..., K, 2)
    types = 2 - invested.sum(axis=-1)               # CC = 0, DC = 1, DD = 2
    good = np.argmax(invested, axis=-1)             # for DC groups: which good
    has = lambda t: np.any(counted & (types == t), axis=-1)
    dc_both = (np.any(counted & (types == 1) & (good == 0), axis=-1)
               & np.any(counted & (types == 1) & (good == 1), axis=-1))
    return has(0) * 1 + has(1) * 2 + has(2) * 4 + (division & dc_both) * 8


def outcome_name(bits):
    # distinct types, alphabetical ("CC" < "DC" < "DC and DC'" < "DD"), joined with " and "
    tokens = []
    if bits & 1: tokens.append("CC")
    if bits & 2: tokens.append("DC and DC'" if bits & 8 else "DC")
    if bits & 4: tokens.append("DD")
    return " and ".join(tokens) if tokens else "none"


def grid_labels(bits):
    # bits: (start, ...) -> labels (...): distinct run outcomes, alphabetical, " |or| "
    names = np.vectorize(outcome_name, otypes=[object])(bits)
    labels = np.empty(bits.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        labels[idx] = " |or| ".join(sorted(set(names[(slice(None),) + idx])))
    return labels


# ---------- checks ----------

def check_payoff():
    cases = [((0.5, 0.5), (0.5, 0.5), 2.0, 0.75, 1.0, 1.0, 1.0),
             ((1.0, 0.0), (1.0, 0.0), 2.0, 0.75, 2.0, 0.5, 1.141)]
    for y, x, b1, b2, p, q, expected in cases:
        got = float(P(jnp.array(y), jnp.array(x), b1, b2, p, q))
        ok = abs(got - expected) < 1e-3
        print(f"  {'ok' if ok else 'FAILED':6s} P(y={y}, x={x}, b1'={b1}, b2'={b2}, p={p}, q={q}) "
              f"= {got:.4f} (expected {expected})")
        assert ok


def check_labeller():
    t = 1e-6
    def run(groups, shares, division=False):
        xs = np.full((4, 2), t); fs = np.zeros(4)
        xs[:len(groups)] = groups; fs[:len(shares)] = shares
        return outcome_name(int(outcome_bits(xs, fs, division=division)))
    cases = [
        ("one group at (2, 2)",                         run([(2, 2)], [1.0]),                         "CC"),
        ("groups at (2, 2) and (1e-6, 1e-6)",           run([(2, 2), (t, t)], [0.5, 0.5]),            "CC and DD"),
        ("groups at (2, 1e-6) and (1e-6, 2)",           run([(2, t), (t, 2)], [0.5, 0.5]),            "DC"),
        ("same, division of labour on",                 run([(2, t), (t, 2)], [0.5, 0.5], True),      "DC and DC'"),
        ("(2, 2) share 0.995 + (1e-6, 1e-6) share 0.005", run([(2, 2), (t, t)], [0.995, 0.005]),      "CC"),
        ("one group at (1.0004, 1.0004) (rounds to 1)", run([(1.0004, 1.0004)], [1.0]),               "DD"),
        ("three groups CC, DC, DD",                     run([(2, 2), (2, t), (t, t)], [0.4, 0.3, 0.3]), "CC and DC and DD"),
    ]
    for name, got, expected in cases:
        print(f"  {'ok' if got == expected else 'FAILED':6s} {name:46s} -> {got!r} (expected {expected!r})")
        assert got == expected
    bits = np.array([[4], [5], [4], [1]])      # DD, CC and DD, DD, CC over four starts
    got = grid_labels(bits)[0]
    expected = "CC |or| CC and DD |or| DD"
    print(f"  {'ok' if got == expected else 'FAILED':6s} {'grid label from DD, CC and DD, DD, CC':46s} -> {got!r}")
    assert got == expected


def check_one_good(labels, b):
    # p = q = 1 panel vs the one-good analytic diagram, away from boundaries
    i, j = P_VALUES.index(1.0), Q_VALUES.index(1.0)
    analytic = one_good_analytic_labels(b, b, x_max=x_max)
    sim = labels[i, j]
    # boundary: analytic label differs from a neighbour (or is undefined)
    pad = np.pad(analytic, 1, mode="edge")
    boundary = (analytic == "boundary")
    for di, dj in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        boundary |= pad[di:di + len(b), dj:dj + len(b)] != analytic
    differ = (sim != analytic) & ~boundary
    print(f"p = q = 1 vs one-good analytic, away from boundaries: {differ.sum()} of "
          f"{(~boundary).sum()} points differ")
    for a, s in sorted(set(zip(analytic[differ], sim[differ]))):
        n = int(((analytic == a) & (sim == s) & differ).sum())
        print(f"  analytic {a!r:14s} simulated {s!r:28s} {n:4d} points")


# ---------- plotting ----------

def plot_labels(labels, b, title, filename):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    present = sorted(set(labels.ravel()))                         # alphabetical
    colours = plt.cm.hsv(np.linspace(0, 0.9, len(present)))
    index = {l: k for k, l in enumerate(present)}
    codes = np.vectorize(index.get)(labels)
    extent = [b[0], b[-1], b[0], b[-1]]

    # columns: p left to right, rows: q top to bottom
    fig, axes = plt.subplots(len(Q_VALUES), len(P_VALUES), sharex=True, sharey=True,
                             figsize=(15, 11), squeeze=False)
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            ax = axes[j, i]
            ax.imshow(codes[i, j], origin="lower", extent=extent, aspect="auto",
                      cmap=ListedColormap(colours), vmin=-0.5, vmax=len(present) - 0.5,
                      interpolation="nearest")
            ax.set_title(f"Phase diagram p = {p:g} q = {q:g}", fontsize=10)
            ax.set_xlabel("b1'"); ax.set_ylabel("b2'")
    fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(present, colours)],
               loc="center left", fontsize=8)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.27, wspace=0.25, hspace=0.3)
    fig.savefig(filename, dpi=150)
    print(f"saved {filename}")


def print_counts(labels, title):
    print(f"\n=== label counts: {title} ===")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            values, counts = np.unique(labels[i, j], return_counts=True)
            print(f"p = {p:g}, q = {q:g}:")
            for v, c in sorted(zip(values, counts)):
                print(f"  {c:5d}  {v}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--dt", type=float, default=TIME_STEP)
    parser.add_argument("--still", type=float, default=STILL)
    parser.add_argument("--K", type=int, default=K)
    args = parser.parse_args()

    settings = dict(time_step=args.dt, steps=args.steps, K=args.K, still=args.still,
                    split_gap=SPLIT_GAP, merge_gap=MERGE_GAP, min_share=MIN_SHARE, lam_min=LAM_MIN,
                    split_cooldown=SPLIT_COOLDOWN, anneal_frac=ANNEAL_FRAC, anneal_to=ANNEAL_TO)
    tag = f"n{args.n}_steps{args.steps}_dt{args.dt:g}_still{args.still:g}_K{args.K}"
    cache = f"reference_results_{tag}.npz"
    b = np.linspace(0, 4, args.n)

    print("running on:", jax.devices())
    print("payoff checks:"); check_payoff()
    print("labeller checks:"); check_labeller()

    if os.path.exists(cache):
        data = np.load(cache)
        xs, fs, overflow = data["xs"], data["fs"], data["overflow"]
        print(f"loaded final states from {cache}")
    else:
        n_runs = len(STARTS) * len(P_VALUES) * len(Q_VALUES) * args.n**2
        print(f"simulating {n_runs} runs ({tag}) ...")
        t = time.time()
        xs, fs, overflow = simulate(args.n, settings)
        print(f"simulation took {time.time() - t:.0f}s")
        np.savez(cache, xs=xs, fs=fs, overflow=overflow)
    print(f"runs where a split was wanted but no slot was free (K = {args.K} too small): "
          f"{int(overflow.sum())} of {overflow.size}")
    # xs: (start, p, q, b2', b1', K, 2), fs: (start, p, q, b2', b1', K)

    labels_raw = grid_labels(outcome_bits(xs, fs))                    # (p, q, b2', b1')
    labels, rare, replaced = replace_rare(labels_raw, b, b, RARE_FRAC)
    print(f"\nrare labels (fewer than {RARE_FRAC:.1%} of {labels_raw.size} points = "
          f"{RARE_FRAC * labels_raw.size:g} points):")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:5d}  {l}")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            print(f"  p = {p:g}, q = {q:g}: {replaced[i, j]} points replaced")

    print()
    check_one_good(labels_raw, b)
    print_counts(labels_raw, "unfiltered")
    print_counts(labels, "filtered")

    plot_labels(labels, b, f"Filtered (rare labels < {RARE_FRAC:.1%} replaced) - {tag}",
                f"reference_filtered_{tag}.png")
    plot_labels(labels_raw, b, f"Unfiltered: what actually happened - {tag}",
                f"reference_unfiltered_{tag}.png")
    plt.show()
