# Two-good adaptive dynamics with repeated branching, run until convergence. Saves the final state of
# every run, with everything needed to interpret it, to two_good_results_<tag>.npz; labelling and
# plotting are done from that file by two_good_analysis.py.
#
# Each run (one (p, q, b1', b2') and one start) is advanced CHUNK steps at a time. After each chunk:
#   - a stopped group on a saddle is pushed off it along the unstable direction
#   - a stopped group at an attracting point whose payoff curves upwards splits in two
#   - a run whose traits and shares have settled is finished
#   - a run that is cycling gets a smaller time step
# Runs start with 2 group slots and move up to 4, then 8, only if a split needs room.
#
# Usage:  python two_good_simulate.py             (100 x 100 grid)
#         python two_good_simulate.py --n 10      (quick test)
#         python two_good_simulate.py --flat-benefit   (benefit stays at its maximum past its peak)
#         in Colab: %run two_good_simulate.py --n 10

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
x_max, x_min = 0.5, 1e-6
P_VALUES = [0.8, 1.0, 1.25]
Q_VALUES = [0.8, 1.0, 1.25]
GRID_N = 100                         # b1' and b2' from 0 to B_MAX on a GRID_N x GRID_N grid
B_MAX = 4.0
LO, HI = 0.005 * x_max, 0.995 * x_max                  # 0.0025 and 0.4975 when x_max = 0.5
STARTS = [(LO, LO), (LO, HI), (HI, LO), (HI, HI)]
FLAT_BENEFIT = False                 # True: the benefit stays at its maximum past its peak instead of falling

TIME_STEP = 0.01                     # starting time step (halved while a run is cycling)
MIN_DT = 1e-5
CHUNK = 1000                         # steps between checks
MAX_STEPS = 200_000                  # cap per run
TOL = 1e-4                           # stopped / settled: gradients and payoff gaps below this
K_LEVELS = [2, 4, 8]                 # group slots per run
BATCH = {2: 16384, 4: 2048, 8: 512}  # runs advanced together at each slot count
SMALL_BATCH = 256                    # batch size once few runs are left
SPLIT_GAP = 0.01                     # distance of a split, and of a push off a saddle
MERGE_GAP = 1e-3
MIN_SHARE = 1e-4                     # smaller groups are removed
LAM_MIN = 1e-6                       # split only if the curvature is above this
SADDLE_TOL = 1e-4                    # Jacobian eigenvalues above this make a saddle (0 = neutral, as when p = q)
CASCADE_DIST = 0.05                  # a group splits again only after moving this far from its last split
CYCLE_RATIO = 0.1                    # cycling: net movement < this x (CHUNK * dt * gradient)


# ---------- model ----------

M = lambda pow, a, b: ((a ** pow + b ** pow) / 2) ** (1 / pow)

def P(y, x, b1, b2, p, q):
    # y: own traits (goods,), x: others' traits (..., goods) -- [..., k] picks good k
    Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
    Mq = M(q, y[..., 0], y[..., 1])
    # the benefit peaks at Mp = b1 / (2 b2); b2 can be 0, so a safe divisor keeps NaNs out of the gradients
    safe_b2 = jnp.where(b2 > 0, b2, 1.0)
    peak = jnp.where(b2 > 0, b1 / (2 * safe_b2), jnp.inf)
    m = jnp.minimum(Mp, peak) if FLAT_BENEFIT else Mp
    return b1 * m - b2 * m ** 2 - Mq + Mq ** 2


def largest_eigen(A):
    # largest real part of the eigenvalues of a 2x2 matrix [[a, b], [c, d]] and a matching direction
    # (its eigenvector when the eigenvalues are real); works for symmetric and non-symmetric A
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    half_trace, det = (a + d) / 2, a * d - b * c
    lam = half_trace + jnp.sqrt(jnp.maximum(half_trace**2 - det, 0.0))
    v = jnp.where(b != 0, jnp.array([b, lam - a]),
                  jnp.where(c != 0, jnp.array([lam - d, c]),
                            jnp.where(a >= d, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))))
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
        dist = jnp.where(alive[:, None] & alive[None, :] & (slots[:, None] < slots[None, :]), dist, jnp.inf)
        i, j = jnp.argmin(dist) // K, jnp.argmin(dist) % K
        merge = dist[i, j] < MERGE_GAP
        merged_x = (fs[i] * xs[i] + fs[j] * xs[j]) / (fs[i] + fs[j])
        xs = jnp.where(merge, xs.at[i].set(merged_x), xs)
        fs = jnp.where(merge, fs.at[i].add(fs[j]).at[j].set(0.0), fs)

        return (xs, fs), None

    (xs, fs), _ = jax.lax.scan(step, (xs, fs), None, length=CHUNK)
    return xs, fs


def check(p, q, b1, b2, xs, fs):
    # Per group, with goods at a wall left out:
    #   g:      size of the selection gradient (a good pushed against its wall counts as 0)
    #   lam, v: largest curvature of a mutant's payoff and its direction (branching if lam > 0)
    #   mu, u:  largest eigenvalue of the selection-gradient Jacobian, i.e. as the whole group moves,
    #           and its direction (a saddle if mu > 0)
    # Per run:
    #   gap:    largest difference between a group's payoff and the average (shares still changing)
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    alive = fs > 0
    free = (x_min < xs) & (xs < x_max)
    free2 = free[:, :, None] & free[:, None, :]

    grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)
    pushed_against_wall = ((xs <= x_min) & (grads <= 0)) | ((xs >= x_max) & (grads >= 0))
    g = jnp.max(jnp.where(pushed_against_wall, 0.0, jnp.abs(grads)), axis=-1)

    payoffs = jax.vmap(payoff, in_axes=(0, None, None))(xs, xs, fs)
    gap = jnp.max(jnp.where(alive, jnp.abs(payoffs - jnp.sum(fs * payoffs)), 0.0))

    H = jax.vmap(jax.hessian(payoff), in_axes=(0, None, None))(xs, xs, fs)
    lam, v = jax.vmap(largest_eigen)(jnp.where(free2, H, 0.0))

    def jacobian(i):        # selection gradient of group i as its trait moves (other groups fixed)
        return jax.jacfwd(lambda x: jax.grad(payoff)(x, xs.at[i].set(x), fs))(xs[i])
    J = jax.vmap(jacobian)(jnp.arange(xs.shape[0]))
    J = jnp.where(free2, J, 0.0) - jnp.eye(2) * ~free[:, None, :]           # goods at a wall: stable
    mu, u = jax.vmap(largest_eigen)(J)
    return g, gap, lam, v, mu, u


def selection_gradients(p, q, b1, b2, xs, fs):
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    return jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)


run_chunk_all = jax.jit(jax.vmap(run_chunk))     # vmap: many populations at once
check_all = jax.jit(jax.vmap(check))
selection_gradients_all = jax.jit(jax.vmap(selection_gradients))


# ---------- running every population until it has settled ----------

class Runs:
    # state of every run; slots up to the largest K, unused ones with share 0
    def __init__(self, n):
        dtype = np.float64 if USE_X64 else np.float32
        b = np.linspace(0, B_MAX, n)
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
        self.pushes = np.zeros(N, int)                    # times a group was pushed off a saddle

    def results(self):
        s = self.shape
        return dict(xs=self.xs.reshape(*s, -1, 2), fs=self.fs.reshape(*s, -1), steps=self.steps.reshape(s),
                    status=self.status.reshape(s), dt=self.dt.reshape(s), max_k=self.max_k.reshape(s),
                    overflow=self.overflow.reshape(s), pushes=self.pushes.reshape(s))


def advance(runs, ids, K, size):
    # one chunk for the runs in ids (all with K slots), then the check;
    # returns the runs that carry on at this K, and those that moved up to more slots
    ids = np.array(ids)
    padded = np.concatenate([ids, np.repeat(ids[:1], size - len(ids))])     # fixed batch size
    p, q, b1, b2 = runs.params[padded].T
    x_start = runs.xs[ids, :K]
    xs, fs = run_chunk_all(p, q, b1, b2, runs.xs[padded, :K], runs.fs[padded, :K], runs.dt[padded])
    g, gap, lam, v, mu, u = (np.asarray(a)[:len(ids)] for a in check_all(p, q, b1, b2, xs, fs))
    xs, fs = np.asarray(xs)[:len(ids)], np.asarray(fs)[:len(ids)]
    runs.steps[ids] += CHUNK

    alive = fs > 0
    stopped = alive & (g < TOL)

    # a stopped group on a saddle is pushed off it along the unstable direction
    saddle = stopped & (mu > SADDLE_TOL)
    xs = np.where(saddle[..., None], np.clip(xs + SPLIT_GAP * u, x_min, x_max), xs)
    runs.xs[ids, :K], runs.fs[ids, :K] = xs, fs
    runs.pushes[ids] += saddle.sum(axis=1)

    # a stopped group at an attracting point with upward curvature splits, unless it split nearby
    moved_away = ~(np.linalg.norm(xs - runs.origin[ids, :K], axis=-1) < CASCADE_DIST)   # never split: True
    split_wanted = stopped & ~saddle & (lam > LAM_MIN) & moved_away
    busy = split_wanted.any(axis=1) | saddle.any(axis=1)

    # settled: no movement, no share changes, nothing to split or push
    g_max = np.where(alive, g, 0.0).max(axis=1)
    converged = (g_max < TOL) & (gap < TOL) & ~busy

    # cycling (large gradient, hardly any net movement): halve dt; otherwise let it grow back
    net = np.where(alive, np.linalg.norm(xs - x_start, axis=-1), 0.0).max(axis=1)
    cycling = ~converged & ~busy & (net < CYCLE_RATIO * CHUNK * runs.dt[ids] * g_max)
    runs.dt[ids] = np.where(cycling, np.maximum(runs.dt[ids] / 2, MIN_DT), np.minimum(runs.dt[ids] * 2, TIME_STEP))

    moved_up = np.zeros(len(ids), bool)
    for r in np.flatnonzero(split_wanted.any(axis=1)):
        moved_up[r] = split(runs, ids[r], np.flatnonzero(split_wanted[r]), v[r], K) != K

    capped = runs.steps[ids] >= MAX_STEPS
    done = converged | capped
    runs.status[ids] = np.where(converged, 1, np.where(capped, 2, 0))
    return ids[~done & ~moved_up], ids[moved_up & ~done]


def split(runs, run, groups, v, K):
    # split each group in groups along its direction v into a free slot, moving the run up to more
    # slots if needed; returns the slot count used
    levels = K_LEVELS[K_LEVELS.index(K):]
    K = next((k for k in levels if (runs.fs[run, :k] == 0).sum() >= len(groups)), levels[-1])
    free = list(np.flatnonzero(runs.fs[run, :K] == 0))
    runs.overflow[run] |= len(free) < len(groups)
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


def load_or_simulate(n):
    # the results file holds the final state of every run plus the settings needed to read it,
    # so two_good_analysis.py needs nothing from this script
    tag = f"xmax{x_max:g}{'_flat' if FLAT_BENEFIT else ''}_n{n}_dt{TIME_STEP:g}_chunk{CHUNK}_max{MAX_STEPS}_tol{TOL:g}_{'f64' if USE_X64 else 'f32'}"
    filename = f"two_good_results_{tag}.npz"
    if os.path.exists(filename):
        print(f"{filename} already exists: loading it instead of simulating (delete it to re-run)")
        return dict(np.load(filename)), filename
    print(f"simulating {len(STARTS) * len(P_VALUES) * len(Q_VALUES) * n**2} runs ({tag}) ...")
    t = time.time()
    results = simulate(n).results()
    print(f"simulation took {time.time() - t:.0f}s")
    results.update(tag=tag, x_max=x_max, x_min=x_min, flat_benefit=FLAT_BENEFIT, p_values=P_VALUES, q_values=Q_VALUES,
                   b=np.linspace(0, B_MAX, n), starts=STARTS, merge_gap=MERGE_GAP, max_steps=MAX_STEPS)
    np.savez(filename, **results)
    print(f"saved {filename}")
    return results, filename


def check_payoff():
    # expected values with the benefit falling past its peak, and flat there (FLAT_BENEFIT);
    # the second case is just past the peak (Mp = 1.414, peak 1.333), the third well past it (Mp = 1, peak 0.8)
    for y, x, b1, b2, p, q, falling, flat in [((0.5, 0.5), (0.5, 0.5), 2.0, 0.75, 1.0, 1.0, 1.0, 1.0),
                                              ((1.0, 0.0), (1.0, 0.0), 2.0, 0.75, 2.0, 0.5, 1.141, 1.146),
                                              ((0.5, 0.5), (0.5, 0.5), 1.2, 0.75, 1.0, 1.0, 0.20, 0.23)]:
        expected = flat if FLAT_BENEFIT else falling
        got = float(P(jnp.array(y), jnp.array(x), b1, b2, p, q))
        print(f"  {'ok' if abs(got - expected) < 1e-3 else 'FAILED':6s} P(y={y}, x={x}, b1'={b1}, "
              f"b2'={b2}, p={p}, q={q}) = {got:.4f} (expected {expected})")
        assert abs(got - expected) < 1e-3


# ---------- output ----------

def print_summary(results):
    status, steps = results["status"], results["steps"]
    print(f"\nruns converged: {int((status == 1).sum())} of {status.size}; "
          f"stopped at MAX_STEPS = {MAX_STEPS}: {int((status == 2).sum())}")
    print(f"steps used: median {int(np.median(steps))}, 90% {int(np.percentile(steps, 90))}, "
          f"99% {int(np.percentile(steps, 99))}, max {int(steps.max())}")
    print("runs that needed more slots: " + ", ".join(f"K={K}: {int((results['max_k'] >= K).sum())}" for K in K_LEVELS)
          + f"; wanted more than {K_LEVELS[-1]}: {int(results['overflow'].sum())}")
    print(f"runs pushed off a saddle at least once: {int((results['pushes'] > 0).sum())}")
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            capped, pushed = status[:, i, j] == 2, results["pushes"][:, i, j] > 0
            if capped.any() or pushed.any():
                print(f"  p = {p:g}, q = {q:g}: {int(capped.sum())} runs not converged, "
                      f"{int(pushed.sum())} pushed off a saddle")
    if FLAT_BENEFIT:
        print(f"runs with a group at x_max and a selection gradient of exactly 0 there "
              f"(extra benefit and extra cost both 0): {zero_gradient_at_x_max(results)}")


def zero_gradient_at_x_max(results):
    # number of runs with a group at x_max in a good whose selection gradient in that good is exactly 0
    xs, fs = results["xs"], results["fs"]
    K = fs.shape[-1]
    b = results["b"]
    _, Pg, Qg, Y, X = np.meshgrid(np.arange(len(STARTS)), P_VALUES, Q_VALUES, b, b, indexing="ij")
    grads = np.asarray(selection_gradients_all(*(a.ravel().astype(xs.dtype) for a in (Pg, Qg, X, Y)),
                                               xs.reshape(-1, K, 2), fs.reshape(-1, K)))
    alive = fs.reshape(-1, K, 1) > 0
    corner = alive & (xs.reshape(-1, K, 2) >= x_max) & (grads == 0)
    return int(corner.any(axis=(1, 2)).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    parser.add_argument("--flat-benefit", action="store_true", help="benefit stays at its maximum past its peak")
    args = parser.parse_args()
    FLAT_BENEFIT = args.flat_benefit

    print("running on:", jax.devices())
    print("payoff checks:"); check_payoff()
    results, filename = load_or_simulate(args.n)
    print_summary(results)
    print(f"\nnext: python two_good_analysis.py {filename} --scheme cc_dc_dd")
