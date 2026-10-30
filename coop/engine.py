# Two-good adaptive dynamics with repeated branching, for any payoff model (coop/models.py) over any
# sweep (coop/sweeps.py). Knows nothing about a particular model: gradients, Hessians and Jacobians
# all come from jax applied to the model's payoff.
#
# Each run (one parameter point and one start) is advanced `chunk` steps at a time. After each chunk:
#   - a stopped group on a saddle is pushed off it along the unstable direction
#   - a stopped group at an attracting point whose payoff curves upwards splits in two
#   - a run whose traits and shares have settled is finished
#   - a run that is cycling gets a smaller time step, one that isn't a larger one (up to max_dt)
# Runs start with 2 group slots and move up to 4, then 8, only if a split needs room.
#
# Engine options (Settings below; defaults first) for comparing with a reference implementation:
#   hessian       exact: jax.hessian  |  silas: chain rule with a different mixed derivative of M
#                 (quadratic model only)
#   split_mode    test_repeat: split stopped groups whose payoff curves upwards, repeatedly
#                 forced_once: after the first stage split every group once along the top
#                 eigenvector of the full Hessian, run one more stage, then stop
#   stop          converged: a stage ends when the run has settled
#                 fixed_time: a stage ends after exactly fixed_time time units
#   start_jitter  False | True: one random offset per (start, panel, good), normal(0, jitter_sd)

import time
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp

from coop import sweeps


@dataclass
class Settings:
    # traits and starts
    x_max: float = 0.5
    x_min: float = 1e-6
    lo: float = None                 # starts: the four corners of (lo, hi) in both goods;
    hi: float = None                 # None: 0.5% and 99.5% of x_max
    # time stepping and stopping
    time_step: float = 0.01          # starting time step (halved while a run is cycling)
    min_dt: float = 1e-5
    max_dt: float = 0.01             # dt doubles each chunk a run isn't cycling, up to this
    chunk: int = 1000                # steps between checks
    max_steps: int = 200_000         # cap per run
    tol: float = 1e-4                # stopped / settled: gradients and payoff gaps below this
    # groups
    k_levels: tuple = (2, 4, 8)      # group slots per run
    batch: dict = field(default_factory=lambda: {2: 16384, 4: 2048, 8: 512})   # runs advanced together per slot count
    small_batch: int = 256           # batch size once few runs are left
    split_gap: float = 0.01          # distance of a split, and of a push off a saddle
    merge_gap: float = 1e-3
    min_share: float = 1e-4          # smaller groups are removed
    lam_min: float = 1e-6            # split only if the curvature is above this
    saddle_tol: float = 1e-4         # Jacobian eigenvalues above this make a saddle (0 = neutral, as when p = q)
    cascade_dist: float = 0.05       # a group splits again only after moving this far from its last split
    cycle_ratio: float = 0.1         # cycling: net movement < this x (chunk * dt * gradient)
    # engine options (see the top of this file)
    hessian: str = "exact"
    split_mode: str = "test_repeat"
    stop: str = "converged"
    fixed_time: float = 100.0
    start_jitter: bool = False
    jitter_seed: int = 0
    jitter_sd: float = 0.01
    x64: bool = False                # 32-bit gave identical labels and is much faster on GPU

    @property
    def starts(self):
        lo = 0.005 * self.x_max if self.lo is None else self.lo
        hi = 0.995 * self.x_max if self.hi is None else self.hi
        return [(lo, lo), (lo, hi), (hi, lo), (hi, hi)]

    @property
    def dtype(self):
        return np.float64 if self.x64 else np.float32


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


# ---------- one population: `chunk` steps, and the check after them ----------
# xs: (K slots, 2 goods), fs: (K,); a slot with share 0 is unused. params: the run's parameter values
# in model.params order. Built per (model, settings): the settings are fixed while a function is
# traced, so the Python ifs on them choose which code is compiled.

def make_engine(model, s):
    fixed = s.stop == "fixed_time"
    prm = lambda params: dict(zip(model.params, params))
    payoff_of = lambda params: lambda y, xs, fs: jnp.sum(fs * model.payoff(y, xs, prm(params)))

    def run_chunk(params, xs, fs, dt, time_left):
        payoff = payoff_of(params)
        group_payoffs = jax.vmap(payoff, in_axes=(0, None, None))
        group_grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))
        K = xs.shape[0]
        slots = jnp.arange(K)

        def step(state, _):
            xs, fs, elapsed = state
            old = xs, fs
            # fixed_time: the last step is shortened to end exactly at time_left, later ones do nothing
            step_dt = jnp.clip(time_left - elapsed, 0.0, dt) if fixed else dt
            alive = fs > 0

            # 1. compute group grads
            grads = group_grads(xs, xs, fs)

            # 2. update investments (used slots only)
            new_xs = jnp.where(alive[:, None], jnp.clip(xs + step_dt * grads, s.x_min, s.x_max), xs)

            # 3. calculate per group payoffs and average payoff
            payoffs = group_payoffs(xs, xs, fs)
            avg_payoff = jnp.sum(fs * payoffs)

            # 4. update population shares
            new_fs = fs + step_dt * fs * (payoffs - avg_payoff)
            xs, fs = new_xs, new_fs / jnp.sum(new_fs)

            # 5. min check: remove groups below min_share
            fs = jnp.where(fs < s.min_share, 0.0, fs)
            fs = fs / jnp.sum(fs)
            alive = fs > 0

            # 6. merge check: the closest pair of groups, if closer than merge_gap
            dist = jnp.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
            dist = jnp.where(alive[:, None] & alive[None, :] & (slots[:, None] < slots[None, :]), dist, jnp.inf)
            i, j = jnp.argmin(dist) // K, jnp.argmin(dist) % K
            merge = dist[i, j] < s.merge_gap
            merged_x = (fs[i] * xs[i] + fs[j] * xs[j]) / (fs[i] + fs[j])
            xs = jnp.where(merge, xs.at[i].set(merged_x), xs)
            fs = jnp.where(merge, fs.at[i].add(fs[j]).at[j].set(0.0), fs)

            if fixed:
                xs, fs = jnp.where(step_dt > 0, xs, old[0]), jnp.where(step_dt > 0, fs, old[1])
            return (xs, fs, elapsed + step_dt), None

        (xs, fs, elapsed), _ = jax.lax.scan(step, (xs, fs, jnp.zeros_like(dt)), None, length=s.chunk)
        return xs, fs, elapsed

    def check(params, xs, fs):
        # Per group, with goods at a wall left out:
        #   g:      size of the selection gradient (a good pushed against its wall counts as 0)
        #   lam, v: largest curvature of a mutant's payoff and its direction (branching if lam > 0)
        #   mu, u:  largest eigenvalue of the selection-gradient Jacobian, i.e. as the whole group moves,
        #           and its direction (a saddle if mu > 0)
        #   lam_full, v_full: largest curvature of the full Hessian, walls included (for forced_once)
        # Per run:
        #   gap:    largest difference between a group's payoff and the average (shares still changing)
        payoff = payoff_of(params)
        alive = fs > 0
        free = (s.x_min < xs) & (xs < s.x_max)
        free2 = free[:, :, None] & free[:, None, :]

        grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)
        pushed_against_wall = ((xs <= s.x_min) & (grads <= 0)) | ((xs >= s.x_max) & (grads >= 0))
        g = jnp.max(jnp.where(pushed_against_wall, 0.0, jnp.abs(grads)), axis=-1)

        payoffs = jax.vmap(payoff, in_axes=(0, None, None))(xs, xs, fs)
        gap = jnp.max(jnp.where(alive, jnp.abs(payoffs - jnp.sum(fs * payoffs)), 0.0))

        if s.hessian == "exact":
            H = jax.vmap(jax.hessian(payoff), in_axes=(0, None, None))(xs, xs, fs)
        else:
            H = jax.vmap(model.chain_hessian, in_axes=(0, None, None, None))(xs, xs, fs, prm(params))
        lam, v = jax.vmap(largest_eigen)(jnp.where(free2, H, 0.0))
        lam_full, v_full = jax.vmap(largest_eigen)(H)

        def jacobian(i):        # selection gradient of group i as its trait moves (other groups fixed)
            return jax.jacfwd(lambda x: jax.grad(payoff)(x, xs.at[i].set(x), fs))(xs[i])
        J = jax.vmap(jacobian)(jnp.arange(xs.shape[0]))
        J = jnp.where(free2, J, 0.0) - jnp.eye(2) * ~free[:, None, :]           # goods at a wall: stable
        mu, u = jax.vmap(largest_eigen)(J)
        return g, gap, lam, v, mu, u, lam_full, v_full

    return jax.jit(jax.vmap(run_chunk)), jax.jit(jax.vmap(check))     # vmap: many populations at once


def selection_gradients(model, params, xs, fs):
    # every group's selection gradient, for many runs at once: params (N, parameters), xs (N, K, 2), fs (N, K)
    def one(params, xs, fs):
        payoff = lambda y, xs, fs: jnp.sum(fs * model.payoff(y, xs, dict(zip(model.params, params))))
        return jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)
    return np.asarray(jax.jit(jax.vmap(one))(params, xs, fs))


# ---------- running every population until it has finished ----------

class Runs:
    # state of every run; slots up to the largest K, unused ones with share 0
    def __init__(self, model, sweep, s, n):
        starts = s.starts
        self.shape, (I, Ia, Ib), values = sweeps.layout(sweep, n, len(starts))   # (start, panel a, panel b, row, column)
        N, k = I.size, s.k_levels[-1]
        self.params = np.stack([values[name] for name in model.params], axis=1).astype(s.dtype)

        x0 = np.array(starts)[I]
        if s.start_jitter:      # one offset per (start, panel, good), shared by the whole panel
            offsets = np.random.default_rng(s.jitter_seed).normal(0, s.jitter_sd, self.shape[:3] + (2,))
            x0 = np.clip(x0 + offsets[I, Ia, Ib], s.x_min, s.x_max)
        self.xs = np.repeat(x0[:, None, :], k, axis=1).astype(s.dtype)
        self.fs = np.zeros((N, k), s.dtype)
        self.fs[:, 0] = 1.0
        self.origin = np.full((N, k, 2), np.nan)          # where each group last split
        self.dt = np.full(N, s.time_step, s.dtype)
        self.steps = np.zeros(N, int)
        self.status = np.zeros(N, int)                    # 0 running, 1 finished, 2 hit max_steps
        self.max_k = np.full(N, s.k_levels[0])            # most slots the run needed
        self.overflow = np.zeros(N, bool)                 # wanted more than k_levels[-1] slots
        self.pushes = np.zeros(N, int)                    # times a group was pushed off a saddle
        self.stage = np.zeros(N, int)                     # splits made so far (forced_once and fixed_time)
        self.stage_time = np.zeros(N)                     # time since the stage started

    def results(self):
        s = self.shape
        return dict(xs=self.xs.reshape(*s, -1, 2), fs=self.fs.reshape(*s, -1), steps=self.steps.reshape(s),
                    status=self.status.reshape(s), dt=self.dt.reshape(s), max_k=self.max_k.reshape(s),
                    overflow=self.overflow.reshape(s), pushes=self.pushes.reshape(s), stage=self.stage.reshape(s))


def advance(engine, s, runs, ids, K, size):
    # one chunk for the runs in ids (all with K slots), then the check and the decisions;
    # returns the runs that carry on at this K, and those that moved up to more slots
    run_chunk_all, check_all = engine
    fixed, forced = s.stop == "fixed_time", s.split_mode == "forced_once"
    ids = np.array(ids)
    padded = np.concatenate([ids, np.repeat(ids[:1], size - len(ids))])     # fixed batch size
    params = runs.params[padded]
    x_start = runs.xs[ids, :K]
    time_left = (s.fixed_time - runs.stage_time[padded]).astype(s.dtype)
    xs, fs, elapsed = run_chunk_all(params, runs.xs[padded, :K], runs.fs[padded, :K], runs.dt[padded], time_left)
    g, gap, lam, v, mu, u, lam_full, v_full = (np.asarray(a)[:len(ids)] for a in check_all(params, xs, fs))
    xs, fs = np.asarray(xs)[:len(ids)], np.asarray(fs)[:len(ids)]
    runs.steps[ids] += s.chunk
    runs.stage_time[ids] += np.asarray(elapsed)[:len(ids)]

    alive = fs > 0
    stopped = alive & (g < s.tol)

    # a stopped group on a saddle is pushed off it along the unstable direction
    saddle = stopped & (mu > s.saddle_tol)
    xs = np.where(saddle[..., None], np.clip(xs + s.split_gap * u, s.x_min, s.x_max), xs)
    runs.xs[ids, :K], runs.fs[ids, :K] = xs, fs
    runs.pushes[ids] += saddle.sum(axis=1)

    # a group at an attracting point with upward curvature wants to split, unless it split nearby
    moved_away = ~(np.linalg.norm(xs - runs.origin[ids, :K], axis=-1) < s.cascade_dist)   # never split: True
    upward = ~saddle & (lam > s.lam_min) & moved_away
    g_max = np.where(alive, g, 0.0).max(axis=1)
    settled = (g_max < s.tol) & (gap < s.tol)        # no movement, no share changes

    # when does a stage end, which groups split, and when is the run finished
    if fixed:
        stage_end = runs.stage_time[ids] >= s.fixed_time * (1 - 1e-6)
    else:
        stage_end = settled & ~saddle.any(axis=1)
    if forced:          # stage 0 ends: every group splits along the full Hessian; stage 1 ends: done
        split_wanted = (stage_end & (runs.stage[ids] == 0))[:, None] & alive
        directions = v_full
        finished = stage_end & (runs.stage[ids] >= 1)
    elif fixed:         # each stage end: test, split the groups that want to; done if none do
        split_wanted = stage_end[:, None] & alive & upward
        directions = v
        finished = stage_end & ~split_wanted.any(axis=1)
    else:               # default: split stopped groups whenever they want to; done once settled
        split_wanted = stopped & upward
        directions = v
        finished = settled & ~(split_wanted.any(axis=1) | saddle.any(axis=1))
    busy = split_wanted.any(axis=1) | saddle.any(axis=1)

    # cycling (large gradient, hardly any net movement): halve dt; otherwise let it grow, up to max_dt
    net = np.where(alive, np.linalg.norm(xs - x_start, axis=-1), 0.0).max(axis=1)
    cycling = ~finished & ~busy & (net < s.cycle_ratio * s.chunk * runs.dt[ids] * g_max)
    runs.dt[ids] = np.where(cycling, np.maximum(runs.dt[ids] / 2, s.min_dt), np.minimum(runs.dt[ids] * 2, s.max_dt))

    moved_up = np.zeros(len(ids), bool)
    for r in np.flatnonzero(split_wanted.any(axis=1)):
        moved_up[r] = split(s, runs, ids[r], np.flatnonzero(split_wanted[r]), directions[r], K) != K
        if fixed or forced:                               # a new stage starts
            runs.stage[ids[r]] += 1
            runs.stage_time[ids[r]] = 0.0

    capped = runs.steps[ids] >= s.max_steps
    done = finished | capped
    runs.status[ids] = np.where(finished, 1, np.where(capped, 2, 0))
    return ids[~done & ~moved_up], ids[moved_up & ~done]


def split(s, runs, run, groups, v, K):
    # split each group in groups along its direction v into a free slot, moving the run up to more
    # slots if needed; returns the slot count used
    levels = s.k_levels[s.k_levels.index(K):]
    K = next((k for k in levels if (runs.fs[run, :k] == 0).sum() >= len(groups)), levels[-1])
    free = list(np.flatnonzero(runs.fs[run, :K] == 0))
    runs.overflow[run] |= len(free) < len(groups)
    for w in groups[:len(free)]:
        f = free.pop(0)
        x = runs.xs[run, w].copy()
        runs.xs[run, w] = np.clip(x - s.split_gap * v[w], s.x_min, s.x_max)
        runs.xs[run, f] = np.clip(x + s.split_gap * v[w], s.x_min, s.x_max)
        runs.fs[run, w] = runs.fs[run, f] = runs.fs[run, w] / 2
        runs.origin[run, w] = runs.origin[run, f] = x
    runs.max_k[run] = max(runs.max_k[run], K)
    return K


def simulate(model, sweep, s, n):
    # every run of the sweep until it has finished; returns the final state (Runs.results)
    engine = make_engine(model, s)
    runs = Runs(model, sweep, s, n)
    N = runs.status.size
    waiting = {K: deque() for K in s.k_levels}         # runs waiting for a place in a batch
    running = {K: [] for K in s.k_levels}              # runs in the batch at each slot count
    waiting[s.k_levels[0]].extend(range(N))
    t0, chunks = time.time(), 0

    while any(waiting[K] or running[K] for K in s.k_levels):
        for K in s.k_levels:
            if not (waiting[K] or running[K]):
                continue
            # fixed batch size (smaller once few runs are left), topped up from the waiting runs
            big = min(s.batch[K], N)
            size = big if len(waiting[K]) + len(running[K]) > s.small_batch else min(s.small_batch, big)
            while len(running[K]) < size and waiting[K]:
                running[K].append(waiting[K].popleft())

            stay, moved_up = advance(engine, s, runs, running[K], K, size)
            running[K] = list(stay)
            for run in moved_up:
                waiting[runs.max_k[run]].append(run)
            chunks += 1

        if chunks % 20 == 0:
            print(f"  {time.time() - t0:6.0f}s  finished {int((runs.status > 0).sum())}/{N}  "
                  + ", ".join(f"K={K}: {len(waiting[K]) + len(running[K])} left" for K in s.k_levels), flush=True)
    return runs.results()
