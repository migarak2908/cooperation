# Two-good simulation run in chunks until convergence (instead of a fixed number of steps),
# labelled with the reference scheme from two_good_reference.py.
#
# Each run advances CHUNK steps at a time (dynamics, removal of small groups and merging every
# step, but no splitting). After each chunk, every run is checked once:
#   - groups that have stopped, curve upwards, and have moved at least CASCADE_DIST since their
#     last split are split (several per run if needed)
#   - a run whose traits and shares have settled, with no split wanted, is finished
#   - a run that is moving back and forth without getting anywhere has its time step halved
#   - a run that reaches MAX_STEPS is stopped and flagged as not converged
# Runs start with 2 group slots and move up to 4, then 8, if they need more for their splits.
# Each slot count has a fixed-size batch; finished runs are replaced by waiting ones, so the
# compiled step function is reused.
#
# Usage:  python two_good_chunked.py            (full grid)
#         python two_good_chunked.py --n 10     (quick 10 x 10 test)
#         in Colab: %run two_good_chunked.py --n 10

import argparse
import os
import time
from collections import deque
import numpy as np

USE_X64 = False                      # 32-bit gave identical labels on the 10 x 10 test and is much faster on GPU

import jax
import two_good_reference as ref     # labelling, checks and plots (this import turns 64-bit on)
jax.config.update("jax_enable_x64", USE_X64)
import jax.numpy as jnp
from two_good_integration_phase import P

x_max, x_min = ref.x_max, ref.x_min
P_VALUES, Q_VALUES, STARTS = ref.P_VALUES, ref.Q_VALUES, ref.STARTS

# ---------- settings ----------
GRID_N = 100
TIME_STEP = 0.01                     # starting time step (halved for runs that cycle)
MIN_DT = 1e-5                        # smallest time step
CHUNK = 1000                         # steps between checks
MAX_STEPS = 200_000                  # cap per run
CONVERGENCE_TOL = 1e-4               # settled: gradients and payoff gaps below this
K_LEVELS = [2, 4, 8]                 # group slots: start with 2, move up when splits need room
BATCH = {2: 16384, 4: 2048, 8: 512}  # runs per batch at each slot count
SMALL_BATCH = 256                    # batch size once few runs are left
SPLIT_GAP = ref.SPLIT_GAP
MERGE_GAP = ref.MERGE_GAP
MIN_SHARE = ref.MIN_SHARE
LAM_MIN = ref.LAM_MIN
CASCADE_DIST = 0.05                  # a group splits again only after moving this far from its last split
CYCLE_RATIO = 0.1                    # cycling: net movement over a chunk < this x (CHUNK * dt * gradient)


# ---------- compiled pieces (shape-generic: K and batch size come from the arrays) ----------

def _advance_one(p, q, b1, b2, xs, fs, dt):
    # CHUNK steps of the dynamics for one run, without splitting
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    group_payoffs = jax.vmap(payoff, in_axes=(0, None, None))
    group_grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))
    K = xs.shape[0]
    slots = jnp.arange(K)

    def step(state, _):
        xs, fs = state
        alive = fs > 0

        # gradients, investment update (used slots only)
        grads = group_grads(xs, xs, fs)
        new_xs = jnp.where(alive[:, None], jnp.clip(xs + dt * grads, x_min, x_max), xs)

        # payoffs, share update
        payoffs = group_payoffs(xs, xs, fs)
        new_fs = fs + dt * fs * (payoffs - jnp.sum(fs * payoffs))
        xs, fs = new_xs, new_fs / jnp.sum(new_fs)

        # remove groups below MIN_SHARE
        fs = jnp.where(fs < MIN_SHARE, 0.0, fs)
        fs = fs / jnp.sum(fs)
        alive = fs > 0

        # merge the closest pair of used slots if closer than MERGE_GAP
        dist = jnp.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
        pair = alive[:, None] & alive[None, :] & (slots[:, None] < slots[None, :])
        dist = jnp.where(pair, dist, jnp.inf)
        k = jnp.argmin(dist)
        i, j = k // K, k % K
        merge = dist[i, j] < MERGE_GAP
        merged_x = (fs[i] * xs[i] + fs[j] * xs[j]) / (fs[i] + fs[j])
        xs = jnp.where(merge, xs.at[i].set(merged_x), xs)
        fs = jnp.where(merge, fs.at[i].add(fs[j]).at[j].set(0.0), fs)
        return (xs, fs), None

    (xs, fs), _ = jax.lax.scan(step, (xs, fs), None, length=CHUNK)
    return xs, fs


def _assess_one(p, q, b1, b2, xs, fs):
    # per group: gradient size (goods pinned at a wall and pushed outwards ignored), largest
    # curvature eigenvalue and its direction (goods at a wall removed); per run: payoff gap
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    alive = fs > 0
    grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)
    pushing_out = ((xs <= x_min) & (grads <= 0)) | ((xs >= x_max) & (grads >= 0))
    g = jnp.max(jnp.where(pushing_out, 0.0, jnp.abs(grads)), axis=-1)               # (K,)

    payoffs = jax.vmap(payoff, in_axes=(0, None, None))(xs, xs, fs)
    gap = jnp.max(jnp.where(alive, jnp.abs(payoffs - jnp.sum(fs * payoffs)), 0.0))

    free_goods = (x_min < xs) & (xs < x_max)
    H = jax.vmap(jax.hessian(payoff), in_axes=(0, None, None))(xs, xs, fs)
    H = H * free_goods[:, :, None] * free_goods[:, None, :]
    lam, v = jax.vmap(ref.top_eigen)(H)
    return g, gap, lam, v


advance = jax.jit(jax.vmap(_advance_one))
assess = jax.jit(jax.vmap(_assess_one))


# ---------- the chunked driver ----------

def simulate_chunked(n, verbose=True):
    b = np.linspace(0, 4, n)
    starts = np.array(STARTS, dtype=float)
    I, Pg, Qg, Y, X = np.meshgrid(np.arange(len(starts)), P_VALUES, Q_VALUES, b, b, indexing="ij")
    shape = I.shape
    N = I.size
    k_max = K_LEVELS[-1]
    dtype = np.float64 if USE_X64 else np.float32
    params = np.stack([Pg.ravel(), Qg.ravel(), X.ravel(), Y.ravel()], axis=1).astype(dtype)

    # state of every run, with room for the largest slot count
    XS = np.repeat(starts[I.ravel()][:, None, :], k_max, axis=1).astype(dtype)  # (N, k_max, 2)
    FS = np.zeros((N, k_max), dtype); FS[:, 0] = 1.0
    ORIGIN = np.full((N, k_max, 2), np.nan)          # where each group last split
    DT = np.full(N, TIME_STEP, dtype)
    STEPS = np.zeros(N, int)
    LEVEL = np.zeros(N, int)                         # index into K_LEVELS
    STATUS = np.zeros(N, int)                        # 0 running, 1 converged, 2 hit MAX_STEPS
    OVERFLOW = np.zeros(N, bool)                     # wanted more slots than k_max
    BLOCKED = np.zeros(N, bool)                      # ended with a split blocked by CASCADE_DIST
    REACHED = np.zeros((N, len(K_LEVELS)), bool); REACHED[:, 0] = True

    queues = [deque() for _ in K_LEVELS]
    queues[0].extend(range(N))
    batches = [np.full(0, -1) for _ in K_LEVELS]
    t0, chunks = time.time(), 0

    def batch_size(level, waiting, running):
        big = min(BATCH[K_LEVELS[level]], N)
        return big if waiting + running > SMALL_BATCH else min(SMALL_BATCH, big)

    while any(queues) or any((bt >= 0).any() for bt in batches):
        for level, K in enumerate(K_LEVELS):
            batch, queue = batches[level], queues[level]
            running = batch[batch >= 0]
            if len(running) == 0 and not queue:
                continue
            # refill the batch (resize it when few runs are left)
            size = batch_size(level, len(queue), len(running))
            new = list(running)
            while len(new) < size and queue:
                new.append(queue.popleft())
            batch = np.array(new + [-1] * (size - len(new)))
            batches[level] = batch
            ids = batch[batch >= 0]
            padded = np.where(batch >= 0, batch, ids[0])           # empty places repeat a real run

            # one chunk of dynamics, then the end-of-chunk check
            pr = params[padded]
            x_start = XS[padded, :K]
            xs, fs = advance(pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3], x_start, FS[padded, :K], DT[padded])
            g, gap, lam, v = assess(pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3], xs, fs)
            xs, fs, g, gap, lam, v = (np.asarray(a)[batch >= 0] for a in (xs, fs, g, gap, lam, v))
            x_start = x_start[batch >= 0]
            chunks += 1

            XS[ids, :K], FS[ids, :K] = xs, fs
            STEPS[ids] += CHUNK
            alive = fs > 0
            stopped = alive & (g < CONVERGENCE_TOL)
            far = ~(np.linalg.norm(xs - ORIGIN[ids, :K], axis=-1) < CASCADE_DIST)    # NaN origin -> True
            want = stopped & (lam > LAM_MIN) & far
            blocked = stopped & (lam > LAM_MIN) & ~far
            settled = (np.where(alive, g, 0.0).max(axis=1) < CONVERGENCE_TOL) & (gap < CONVERGENCE_TOL)
            converged = settled & ~want.any(axis=1)

            # cycling: large gradient but hardly any net movement over the chunk -> halve dt;
            # otherwise let dt grow back towards TIME_STEP (a small dt would make shares crawl)
            net = np.where(alive, np.linalg.norm(xs - x_start, axis=-1), 0.0).max(axis=1)
            gmax = np.where(alive, g, 0.0).max(axis=1)
            cycling = ~converged & ~want.any(axis=1) & (net < CYCLE_RATIO * CHUNK * DT[ids] * gmax)
            DT[ids] = np.where(cycling, np.maximum(DT[ids] / 2, MIN_DT), np.minimum(DT[ids] * 2, TIME_STEP))

            keep = np.ones(len(ids), bool)
            for r in np.flatnonzero(want.any(axis=1)):
                run = ids[r]
                groups = np.flatnonzero(want[r])
                lvl = level
                while (FS[run, :K_LEVELS[lvl]] == 0).sum() < len(groups) and lvl < len(K_LEVELS) - 1:
                    lvl += 1                                       # move up to more slots
                free = list(np.flatnonzero(FS[run, :K_LEVELS[lvl]] == 0))
                if len(free) < len(groups):
                    OVERFLOW[run] = True
                for w in groups[:len(free)]:
                    f = free.pop(0)
                    x = XS[run, w].copy()
                    XS[run, w] = np.clip(x - SPLIT_GAP * v[r, w], x_min, x_max)
                    XS[run, f] = np.clip(x + SPLIT_GAP * v[r, w], x_min, x_max)
                    FS[run, w] = FS[run, f] = FS[run, w] / 2
                    ORIGIN[run, w] = ORIGIN[run, f] = x
                if lvl != level:
                    LEVEL[run] = lvl
                    REACHED[run, lvl] = True
                    queues[lvl].append(run)
                    keep[r] = False

            done = converged | (STEPS[ids] >= MAX_STEPS)
            STATUS[ids] = np.where(converged, 1, np.where(STEPS[ids] >= MAX_STEPS, 2, 0))
            BLOCKED[ids] = np.where(done, blocked.any(axis=1), BLOCKED[ids])
            keep &= ~done
            batch[np.flatnonzero(batch >= 0)[~keep]] = -1

        if verbose and chunks % 20 == 0:
            print(f"  {time.time() - t0:6.0f}s  finished {int((STATUS > 0).sum())}/{N}  "
                  f"waiting/running per slot count: "
                  + ", ".join(f"K={K}: {len(queues[l]) + int((batches[l] >= 0).sum())}"
                              for l, K in enumerate(K_LEVELS)), flush=True)

    info = dict(steps=STEPS.reshape(shape), status=STATUS.reshape(shape), dt=DT.reshape(shape),
                overflow=OVERFLOW.reshape(shape), blocked=BLOCKED.reshape(shape),
                reached=REACHED.reshape(*shape, len(K_LEVELS)))
    return XS.reshape(*shape, k_max, 2), FS.reshape(*shape, k_max), info


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    args = parser.parse_args()
    b = np.linspace(0, 4, args.n)

    tag = (f"n{args.n}_chunk{CHUNK}_max{MAX_STEPS}_dt{TIME_STEP:g}_tol{CONVERGENCE_TOL:g}"
           f"_K{'-'.join(map(str, K_LEVELS))}_{'f64' if USE_X64 else 'f32'}")
    cache = f"chunked_results_{tag}.npz"

    print("running on:", jax.devices())
    print("payoff checks:"); ref.check_payoff()
    print("labeller checks:"); ref.check_labeller()

    if os.path.exists(cache):
        data = np.load(cache)
        xs, fs = data["xs"], data["fs"]
        info = {k: data[k] for k in ("steps", "status", "dt", "overflow", "blocked", "reached")}
        print(f"loaded final states from {cache}")
    else:
        print(f"simulating {len(STARTS) * len(P_VALUES) * len(Q_VALUES) * args.n**2} runs in chunks ({tag}) ...")
        t = time.time()
        xs, fs, info = simulate_chunked(args.n)
        print(f"simulation took {time.time() - t:.0f}s")
        np.savez(cache, xs=xs, fs=fs, **info)

    steps, status = info["steps"], info["status"]
    print(f"\nruns converged: {int((status == 1).sum())} of {status.size}; "
          f"stopped at MAX_STEPS = {MAX_STEPS} without converging: {int((status == 2).sum())}")
    print(f"steps used: median {int(np.median(steps))}, 90% {int(np.percentile(steps, 90))}, "
          f"99% {int(np.percentile(steps, 99))}, max {int(steps.max())}")
    print(f"runs that ended with a reduced time step (still cycling): {int((info['dt'] < TIME_STEP).sum())}")
    print("runs that needed more slots: " + ", ".join(
        f"K={K}: {int(info['reached'][..., l].sum())}" for l, K in enumerate(K_LEVELS)))
    print(f"runs wanting more than {K_LEVELS[-1]} slots: {int(info['overflow'].sum())}; "
          f"runs that ended with a split blocked by CASCADE_DIST: {int(info['blocked'].sum())}")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            capped = status[:, i, j] == 2
            print(f"  p = {p:g}, q = {q:g}: {int(capped.sum())} runs not converged, "
                  f"{int(capped.any(axis=0).sum())} grid points with at least one")

    labels_raw = ref.grid_labels(ref.outcome_bits(xs, fs))
    labels, rare, replaced = ref.replace_rare(labels_raw, b, b, ref.RARE_FRAC)
    print(f"\nrare labels (fewer than {ref.RARE_FRAC:.1%} of {labels_raw.size} points = "
          f"{ref.RARE_FRAC * labels_raw.size:g} points):")
    for l, c in sorted(rare.items(), key=lambda lc: -lc[1]):
        print(f"  {c:5d}  {l}")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            print(f"  p = {p:g}, q = {q:g}: {replaced[i, j]} points replaced")

    print()
    ref.check_one_good(labels_raw, b)
    ref.print_counts(labels_raw, "unfiltered")
    ref.print_counts(labels, "filtered")
    ref.plot_labels(labels, b, f"Filtered (rare labels < {ref.RARE_FRAC:.1%} replaced) - chunked, {tag}",
                    f"chunked_filtered_{tag}.png")
    ref.plot_labels(labels_raw, b, f"Unfiltered: what actually happened - chunked, {tag}",
                    f"chunked_unfiltered_{tag}.png")
    plt.show()
