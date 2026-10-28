# Two-good adaptive dynamics with saturating exponential benefit and cost, and repeated branching,
# run until convergence. Saves the final state of every run, with everything needed to interpret it,
# to two_good_results_exp_<tag>.npz, for labelling and plotting by an analysis script.
# Payoff: b (1 - exp(-M_p(x + y) / beta)) - (1 - exp(-M_q(y))), with b and beta in place of b1', b2'.
#
# Each run (one (p, q, b, beta) and one start) is advanced CHUNK steps at a time. After each chunk:
#   - a stopped group on a saddle is pushed off it along the unstable direction
#   - a stopped group at an attracting point whose payoff curves upwards splits in two
#   - a run whose traits and shares have settled is finished
#   - a run that is cycling gets a smaller time step, one that isn't a larger one (up to MAX_DT)
# Runs start with 2 group slots and move up to 4, then 8, only if a split needs room.
#
# Usage:  python two_good_simulate_exp.py             (100 x 100 grid)
#         python two_good_simulate_exp.py --n 10      (quick test)
#         in Colab: %run two_good_simulate_exp.py --n 10

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
x_max, x_min = 10.0, 1e-6       # above the singular point everywhere on the grid (largest about 7)
P_VALUES = [0.8, 1.0, 1.25]
Q_VALUES = [0.8, 1.0, 1.25]
GRID_N = 100
B_MAX = 6.0        # b from 0 to B_MAX
BETA_MAX = 1.7     # beta from BETA_MAX / GRID_N to BETA_MAX (beta = 0 would divide by zero);
                   # stops below 2 because the singular point grows very large as beta -> 2
LO, HI = 0.01, 4.0              # set directly, not as fractions of x_max: at large investment both
                                # curves are flat and the gradient falls below TOL, so a start near
                                # x_max would count as converged without moving
STARTS = [(LO, LO), (LO, HI), (HI, LO), (HI, HI)]

TIME_STEP = 0.01                     # starting time step (halved while a run is cycling)
MIN_DT = 1e-5
MAX_DT = 1.0                         # dt doubles each chunk a run isn't cycling, up to this: the flat
                                     # exponential curves give small gradients, so fixed steps crawl
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

def P(y, x, b, beta, p, q):
    # y: own traits (goods,), x: others' traits (..., goods) -- [..., k] picks good k
    Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
    Mq = M(q, y[..., 0], y[..., 1])
    return b * (1 - jnp.exp(-Mp / beta)) - (1 - jnp.exp(-Mq))


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

def run_chunk(p, q, b, beta, xs, fs, dt):
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b, beta, p, q))
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


def check(p, q, b, beta, xs, fs):
    # Per group, with goods at a wall left out:
    #   g:      size of the selection gradient (a good pushed against its wall counts as 0)
    #   lam, v: largest curvature of a mutant's payoff and its direction (branching if lam > 0)
    #   mu, u:  largest eigenvalue of the selection-gradient Jacobian, i.e. as the whole group moves,
    #           and its direction (a saddle if mu > 0)
    # Per run:
    #   gap:    largest difference between a group's payoff and the average (shares still changing)
    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b, beta, p, q))
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


run_chunk_all = jax.jit(jax.vmap(run_chunk))     # vmap: many populations at once
check_all = jax.jit(jax.vmap(check))


# ---------- running every population until it has settled ----------

class Runs:
    # state of every run; slots up to the largest K, unused ones with share 0
    def __init__(self, n):
        dtype = np.float64 if USE_X64 else np.float32
        b_axis = np.linspace(0, B_MAX, n)
        beta_axis = np.linspace(BETA_MAX / n, BETA_MAX, n)
        I, Pg, Qg, Y, X = np.meshgrid(np.arange(len(STARTS)), P_VALUES, Q_VALUES, beta_axis, b_axis, indexing="ij")
        self.shape = I.shape                              # (start, p, q, beta, b)
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
    p, q, b, beta = runs.params[padded].T
    x_start = runs.xs[ids, :K]
    xs, fs = run_chunk_all(p, q, b, beta, runs.xs[padded, :K], runs.fs[padded, :K], runs.dt[padded])
    g, gap, lam, v, mu, u = (np.asarray(a)[:len(ids)] for a in check_all(p, q, b, beta, xs, fs))
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

    # cycling (large gradient, hardly any net movement): halve dt; otherwise let it grow, up to MAX_DT
    net = np.where(alive, np.linalg.norm(xs - x_start, axis=-1), 0.0).max(axis=1)
    cycling = ~converged & ~busy & (net < CYCLE_RATIO * CHUNK * runs.dt[ids] * g_max)
    runs.dt[ids] = np.where(cycling, np.maximum(runs.dt[ids] / 2, MIN_DT), np.minimum(runs.dt[ids] * 2, MAX_DT))

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
    tag = f"exp_bmax{B_MAX:g}_betamax{BETA_MAX:g}_xmax{x_max:g}_n{n}_dt{TIME_STEP:g}-{MAX_DT:g}_chunk{CHUNK}_max{MAX_STEPS}_tol{TOL:g}_{'f64' if USE_X64 else 'f32'}"
    filename = f"two_good_results_{tag}.npz"
    if os.path.exists(filename):
        print(f"{filename} already exists: loading it instead of simulating (delete it to re-run)")
        return dict(np.load(filename)), filename
    print(f"simulating {len(STARTS) * len(P_VALUES) * len(Q_VALUES) * n**2} runs ({tag}) ...")
    t = time.time()
    results = simulate(n).results()
    print(f"simulation took {time.time() - t:.0f}s")
    b_axis = np.linspace(0, B_MAX, n)
    beta_axis = np.linspace(BETA_MAX / n, BETA_MAX, n)
    # one-good singular point x* = ln(b / beta) / (2 / beta - 1) for each (beta, b); NaN where it
    # doesn't exist (b <= beta, or beta >= 2)
    B, BETA = np.meshgrid(b_axis, beta_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_star = np.log(B / BETA) / (2 / BETA - 1)
    x_star = np.where((B > BETA) & (BETA < 2), x_star, np.nan)
    results.update(tag=tag, x_max=x_max, x_min=x_min, p_values=P_VALUES, q_values=Q_VALUES,
                   b=b_axis, beta=beta_axis, x_star=x_star, starts=STARTS, merge_gap=MERGE_GAP,
                   max_steps=MAX_STEPS)
    np.savez(filename, **results)
    print(f"saved {filename}")
    return results, filename


def check_payoff():
    for y, x, b, beta, p, q, expected in [((0.5, 0.5), (0.5, 0.5), 2.0, 1.5, 1.0, 1.0, 0.5797),
                                          ((1.0, 1e-12), (1.0, 1e-12), 2.0, 1.5, 2.0, 0.5, 0.9997)]:
        got = float(P(jnp.array(y), jnp.array(x), b, beta, p, q))
        print(f"  {'ok' if abs(got - expected) < 1e-3 else 'FAILED':6s} P(y={y}, x={x}, b={b}, "
              f"beta={beta}, p={p}, q={q}) = {got:.4f} (expected {expected})")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    args = parser.parse_args()

    print("running on:", jax.devices())
    print("payoff checks:"); check_payoff()
    results, filename = load_or_simulate(args.n)
    print_summary(results)
    print(f"\nnext: two_good_analysis.py, but it still needs updating for this model: the axes are b and "
          f"beta, and the 'invested above x_max / 2' rule must be replaced (results: {filename})")
