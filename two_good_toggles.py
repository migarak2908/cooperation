# Two-good adaptive dynamics with toggles for the ways a reference implementation differs from
# two_good_simulate.py. With every toggle at its default the results are exactly those of
# two_good_simulate.py.
#
# Toggles (SETTINGS below, or command-line flags; default first):
#   BENEFIT       pair_totals: m_p = M_p(x + y)  |  individual_sum: m_p = M_p(y) + M_p(x)
#   HESSIAN       exact: jax.hessian  |  silas: chain rule with a different mixed derivative of M
#   SPLIT_MODE    test_repeat: split stopped groups whose payoff curves upwards, repeatedly
#                 forced_once: after the first stage split every group once along the top
#                 eigenvector of the full Hessian, run one more stage, then stop
#   STOP          converged: a stage ends when the run has settled
#                 fixed_time: a stage ends after exactly FIXED_TIME time units
#   START_JITTER  off | on: one random offset per (start, p, q, good), normal(0, 0.01)
#
# Usage:  python two_good_toggles.py --n 10                       (one config, from the flags)
#         python two_good_toggles.py --n 10 --benefit individual_sum --stop fixed_time
#         python two_good_toggles.py --n 10 --ablation            (baseline, benefit_only, reference_all)
# Figures are saved to files, never shown.

import argparse
import csv
import os
import time
from collections import Counter, deque
import numpy as np
import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_good_analysis as analysis

# ---------- toggles ----------
SETTINGS = dict(
    BENEFIT="pair_totals",
    HESSIAN="exact",
    SPLIT_MODE="test_repeat",
    STOP="converged",
    START_JITTER="off",
    FIXED_TIME=100.0,
    JITTER_SEED=0,
)
CHOICES = dict(BENEFIT=["pair_totals", "individual_sum"], HESSIAN=["exact", "silas"],
               SPLIT_MODE=["test_repeat", "forced_once"], STOP=["converged", "fixed_time"],
               START_JITTER=["off", "on"])

# ---------- settings (as in two_good_simulate.py) ----------
x_max, x_min = 2.0, 1e-6
P_VALUES = [0.8, 1.0, 1.25]
Q_VALUES = [0.8, 1.0, 1.25]
GRID_N = 100
B_MAX = 4.0
LO, HI = 0.005 * x_max, 0.995 * x_max
STARTS = [(LO, LO), (LO, HI), (HI, LO), (HI, HI)]

TIME_STEP = 0.01
MIN_DT = 1e-5
CHUNK = 1000
MAX_STEPS = 200_000
TOL = 1e-4
K_LEVELS = [2, 4, 8]
BATCH = {2: 16384, 4: 2048, 8: 512}
SMALL_BATCH = 256
SPLIT_GAP = 0.01
MERGE_GAP = 1e-3
MIN_SHARE = 1e-4
LAM_MIN = 1e-6
SADDLE_TOL = 1e-4
CASCADE_DIST = 0.05
CYCLE_RATIO = 0.1
JITTER_SD = 0.01


# ---------- model ----------

M = lambda pow, a, b: ((a ** pow + b ** pow) / 2) ** (1 / pow)

def P(y, x, b1, b2, p, q, benefit="pair_totals"):
    # y: own traits (goods,), x: others' traits (..., goods) -- [..., k] picks good k
    if benefit == "pair_totals":
        Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
    else:
        Mp = M(p, y[..., 0], y[..., 1]) + M(p, x[..., 0], x[..., 1])
    Mq = M(q, y[..., 0], y[..., 1])
    return b1 * Mp - b2 * Mp ** 2 - Mq + Mq ** 2


def M_parts(r, a, mixed):
    # gradient and Hessian of M_r at a = (a1, a2); mixed = "silas" replaces the mixed derivative
    f = lambda a: M(r, a[0], a[1])
    g, H = jax.grad(f)(a), jax.hessian(f)(a)
    if mixed == "silas":
        o = -2 ** (-1 / r) * a[0] ** (r - 1) * a[1] ** (r - 1) * (a[0] ** r + a[1] ** r) ** (1 / r - 2)
        H = H.at[0, 1].set(o).at[1, 0].set(o)
    return g, H


def chain_hessian(y, xs, fs, b1, b2, p, q, benefit, mixed="silas"):
    # mutant Hessian by the chain rule, averaged over partner groups with their shares:
    # B'(m_p) Hp + B'' gp gp' - C'(m_q) Hq - C'' gq gq', with B(m) = b1' m - b2' m^2, C(m) = m - m^2
    gq, Hq = M_parts(q, y, mixed)
    mq = M(q, y[0], y[1])
    cost = -(1 - 2 * mq) * Hq + 2 * jnp.outer(gq, gq)
    def with_partner(x):
        a = x + y if benefit == "pair_totals" else y
        gp, Hp = M_parts(p, a, mixed)
        mp = M(p, a[0], a[1]) if benefit == "pair_totals" else M(p, y[0], y[1]) + M(p, x[0], x[1])
        return (b1 - 2 * b2 * mp) * Hp - 2 * b2 * jnp.outer(gp, gp) + cost
    return jnp.sum(fs[:, None, None] * jax.vmap(with_partner)(xs), axis=0)


def largest_eigen(A):
    # largest real part of the eigenvalues of a 2x2 matrix [[a, b], [c, d]] and a matching direction
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    half_trace, det = (a + d) / 2, a * d - b * c
    lam = half_trace + jnp.sqrt(jnp.maximum(half_trace**2 - det, 0.0))
    v = jnp.where(b != 0, jnp.array([b, lam - a]),
                  jnp.where(c != 0, jnp.array([lam - d, c]),
                            jnp.where(a >= d, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))))
    return lam, v / jnp.linalg.norm(v)


# ---------- one population: CHUNK steps, and the check after them ----------
# Built per config: the toggles are fixed while a function is traced, so the Python ifs on them
# below choose which code is compiled; nothing inside depends on the data through a Python if.

def make_engine(cfg):
    benefit, fixed = cfg["BENEFIT"], cfg["STOP"] == "fixed_time"

    def run_chunk(p, q, b1, b2, xs, fs, dt, time_left):
        payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q, benefit))
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
            new_xs = jnp.where(alive[:, None], jnp.clip(xs + step_dt * grads, x_min, x_max), xs)

            # 3. calculate per group payoffs and average payoff
            payoffs = group_payoffs(xs, xs, fs)
            avg_payoff = jnp.sum(fs * payoffs)

            # 4. update population shares
            new_fs = fs + step_dt * fs * (payoffs - avg_payoff)
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

            if fixed:
                xs, fs = jnp.where(step_dt > 0, xs, old[0]), jnp.where(step_dt > 0, fs, old[1])
            return (xs, fs, elapsed + step_dt), None

        (xs, fs, elapsed), _ = jax.lax.scan(step, (xs, fs, jnp.zeros_like(dt)), None, length=CHUNK)
        return xs, fs, elapsed

    def check(p, q, b1, b2, xs, fs):
        # per group: g (gradient size, goods pushed against a wall count as 0), lam, v (largest
        # curvature with walls masked, and its direction), mu, u (Jacobian: saddle if mu > 0),
        # lam_full, v_full (largest curvature of the full Hessian); per run: gap (payoff spread)
        payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q, benefit))
        alive = fs > 0
        free = (x_min < xs) & (xs < x_max)
        free2 = free[:, :, None] & free[:, None, :]

        grads = jax.vmap(jax.grad(payoff), in_axes=(0, None, None))(xs, xs, fs)
        pushed_against_wall = ((xs <= x_min) & (grads <= 0)) | ((xs >= x_max) & (grads >= 0))
        g = jnp.max(jnp.where(pushed_against_wall, 0.0, jnp.abs(grads)), axis=-1)

        payoffs = jax.vmap(payoff, in_axes=(0, None, None))(xs, xs, fs)
        gap = jnp.max(jnp.where(alive, jnp.abs(payoffs - jnp.sum(fs * payoffs)), 0.0))

        if cfg["HESSIAN"] == "exact":
            H = jax.vmap(jax.hessian(payoff), in_axes=(0, None, None))(xs, xs, fs)
        else:
            H = jax.vmap(chain_hessian, in_axes=(0, None, None) + (None,) * 5)(xs, xs, fs, b1, b2, p, q, benefit)
        lam, v = jax.vmap(largest_eigen)(jnp.where(free2, H, 0.0))
        lam_full, v_full = jax.vmap(largest_eigen)(H)

        def jacobian(i):        # selection gradient of group i as its trait moves (other groups fixed)
            return jax.jacfwd(lambda x: jax.grad(payoff)(x, xs.at[i].set(x), fs))(xs[i])
        J = jax.vmap(jacobian)(jnp.arange(xs.shape[0]))
        J = jnp.where(free2, J, 0.0) - jnp.eye(2) * ~free[:, None, :]           # goods at a wall: stable
        mu, u = jax.vmap(largest_eigen)(J)
        return g, gap, lam, v, mu, u, lam_full, v_full

    return jax.jit(jax.vmap(run_chunk)), jax.jit(jax.vmap(check))


# ---------- running every population until it has finished ----------

class Runs:
    # state of every run; slots up to the largest K, unused ones with share 0
    def __init__(self, cfg, n):
        dtype = np.float32
        b = np.linspace(0, B_MAX, n)
        I, Ip, Iq, Y, X = np.meshgrid(np.arange(len(STARTS)), np.arange(len(P_VALUES)), np.arange(len(Q_VALUES)),
                                      b, b, indexing="ij")
        self.shape = I.shape                              # (start, p, q, b2', b1')
        N, k = I.size, K_LEVELS[-1]
        Pg, Qg = np.array(P_VALUES)[Ip], np.array(Q_VALUES)[Iq]
        self.params = np.stack([Pg.ravel(), Qg.ravel(), X.ravel(), Y.ravel()], axis=1).astype(dtype)

        starts = np.array(STARTS)[I.ravel()]
        if cfg["START_JITTER"] == "on":   # one offset per (start, p, q, good), shared by the whole panel
            offsets = np.random.default_rng(cfg["JITTER_SEED"]).normal(
                0, JITTER_SD, (len(STARTS), len(P_VALUES), len(Q_VALUES), 2))
            starts = np.clip(starts + offsets[I.ravel(), Ip.ravel(), Iq.ravel()], x_min, x_max)
        self.xs = np.repeat(starts[:, None, :], k, axis=1).astype(dtype)
        self.fs = np.zeros((N, k), dtype)
        self.fs[:, 0] = 1.0
        self.origin = np.full((N, k, 2), np.nan)          # where each group last split
        self.dt = np.full(N, TIME_STEP, dtype)
        self.steps = np.zeros(N, int)
        self.status = np.zeros(N, int)                    # 0 running, 1 finished, 2 hit MAX_STEPS
        self.max_k = np.full(N, K_LEVELS[0])
        self.overflow = np.zeros(N, bool)
        self.pushes = np.zeros(N, int)
        self.stage = np.zeros(N, int)                     # splits made so far (forced_once and fixed_time)
        self.stage_time = np.zeros(N)                     # time since the stage started

    def results(self):
        s = self.shape
        return dict(xs=self.xs.reshape(*s, -1, 2), fs=self.fs.reshape(*s, -1), steps=self.steps.reshape(s),
                    status=self.status.reshape(s), dt=self.dt.reshape(s), max_k=self.max_k.reshape(s),
                    overflow=self.overflow.reshape(s), pushes=self.pushes.reshape(s), stage=self.stage.reshape(s))


def advance(engine, cfg, runs, ids, K, size):
    # one chunk for the runs in ids (all with K slots), then the check and the decisions;
    # returns the runs that carry on at this K, and those that moved up to more slots
    run_chunk_all, check_all = engine
    fixed, forced = cfg["STOP"] == "fixed_time", cfg["SPLIT_MODE"] == "forced_once"
    ids = np.array(ids)
    padded = np.concatenate([ids, np.repeat(ids[:1], size - len(ids))])     # fixed batch size
    p, q, b1, b2 = runs.params[padded].T
    x_start = runs.xs[ids, :K]
    time_left = (cfg["FIXED_TIME"] - runs.stage_time[padded]).astype(np.float32)
    xs, fs, elapsed = run_chunk_all(p, q, b1, b2, runs.xs[padded, :K], runs.fs[padded, :K], runs.dt[padded], time_left)
    g, gap, lam, v, mu, u, lam_full, v_full = (np.asarray(a)[:len(ids)] for a in check_all(p, q, b1, b2, xs, fs))
    xs, fs = np.asarray(xs)[:len(ids)], np.asarray(fs)[:len(ids)]
    runs.steps[ids] += CHUNK
    runs.stage_time[ids] += np.asarray(elapsed)[:len(ids)]

    alive = fs > 0
    stopped = alive & (g < TOL)

    # a stopped group on a saddle is pushed off it along the unstable direction
    saddle = stopped & (mu > SADDLE_TOL)
    xs = np.where(saddle[..., None], np.clip(xs + SPLIT_GAP * u, x_min, x_max), xs)
    runs.xs[ids, :K], runs.fs[ids, :K] = xs, fs
    runs.pushes[ids] += saddle.sum(axis=1)

    moved_away = ~(np.linalg.norm(xs - runs.origin[ids, :K], axis=-1) < CASCADE_DIST)   # never split: True
    upward = ~saddle & (lam > LAM_MIN) & moved_away
    g_max = np.where(alive, g, 0.0).max(axis=1)
    settled = (g_max < TOL) & (gap < TOL)

    # when does a stage end, which groups split, and when is the run finished
    if fixed:
        stage_end = runs.stage_time[ids] >= cfg["FIXED_TIME"] * (1 - 1e-6)
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
    else:               # current: split stopped groups whenever they want to; done once settled
        split_wanted = stopped & upward
        directions = v
        finished = settled & ~(split_wanted.any(axis=1) | saddle.any(axis=1))
    busy = split_wanted.any(axis=1) | saddle.any(axis=1)

    # cycling (large gradient, hardly any net movement): halve dt; otherwise let it grow back
    net = np.where(alive, np.linalg.norm(xs - x_start, axis=-1), 0.0).max(axis=1)
    cycling = ~finished & ~busy & (net < CYCLE_RATIO * CHUNK * runs.dt[ids] * g_max)
    runs.dt[ids] = np.where(cycling, np.maximum(runs.dt[ids] / 2, MIN_DT), np.minimum(runs.dt[ids] * 2, TIME_STEP))

    moved_up = np.zeros(len(ids), bool)
    for r in np.flatnonzero(split_wanted.any(axis=1)):
        moved_up[r] = split(runs, ids[r], np.flatnonzero(split_wanted[r]), directions[r], K) != K
        if fixed or forced:                               # a new stage starts
            runs.stage[ids[r]] += 1
            runs.stage_time[ids[r]] = 0.0

    capped = runs.steps[ids] >= MAX_STEPS
    done = finished | capped
    runs.status[ids] = np.where(finished, 1, np.where(capped, 2, 0))
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


def simulate(cfg, n):
    engine = make_engine(cfg)
    runs = Runs(cfg, n)
    N = runs.status.size
    waiting = {K: deque() for K in K_LEVELS}
    running = {K: [] for K in K_LEVELS}
    waiting[K_LEVELS[0]].extend(range(N))
    t0, chunks = time.time(), 0

    while any(waiting[K] or running[K] for K in K_LEVELS):
        for K in K_LEVELS:
            if not (waiting[K] or running[K]):
                continue
            big = min(BATCH[K], N)
            size = big if len(waiting[K]) + len(running[K]) > SMALL_BATCH else min(SMALL_BATCH, big)
            while len(running[K]) < size and waiting[K]:
                running[K].append(waiting[K].popleft())

            stay, moved_up = advance(engine, cfg, runs, running[K], K, size)
            running[K] = list(stay)
            for run in moved_up:
                waiting[runs.max_k[run]].append(run)
            chunks += 1

        if chunks % 20 == 0:
            print(f"  {time.time() - t0:6.0f}s  finished {int((runs.status > 0).sum())}/{N}  "
                  + ", ".join(f"K={K}: {len(waiting[K]) + len(running[K])} left" for K in K_LEVELS), flush=True)
    return runs


def config_tag(cfg, n):
    # every toggle value, plus the grid and run settings
    stop = f"fixed_time{cfg['FIXED_TIME']:g}" if cfg["STOP"] == "fixed_time" else "converged"
    jitter = f"on{cfg['JITTER_SEED']}" if cfg["START_JITTER"] == "on" else "off"
    return (f"benefit-{cfg['BENEFIT']}_hessian-{cfg['HESSIAN']}_split-{cfg['SPLIT_MODE']}_stop-{stop}"
            f"_jitter-{jitter}_xmax{x_max:g}_n{n}")


def load_or_simulate(cfg, n):
    tag = config_tag(cfg, n)
    filename = f"toggles_results_{tag}.npz"
    if os.path.exists(filename):
        print(f"loaded {filename}")
        return dict(np.load(filename)), tag
    print(f"simulating {len(STARTS) * len(P_VALUES) * len(Q_VALUES) * n**2} runs ({tag}) ...")
    t = time.time()
    results = simulate(cfg, n).results()
    results["seconds"] = time.time() - t
    print(f"simulation took {results['seconds']:.0f}s")
    np.savez(filename, **results)
    return results, tag


# ---------- tests ----------

def run_tests():
    # run in 64-bit so the comparison to 1e-6 is about the formulas, not rounding
    jax.config.update("jax_enable_x64", True)
    try:
        print("(a) chain-rule Hessian with the correct mixed derivative vs jax.hessian:")
        rng = np.random.default_rng(1)
        worst = 0.0
        for benefit in CHOICES["BENEFIT"]:
            for _ in range(5):
                p, q = rng.choice([0.8, 1.0, 1.25, 2.0], 2)
                b1, b2 = rng.uniform(0, 4, 2)
                xs = jnp.array(rng.uniform(0.05, 2.0, (3, 2)))
                fs = jnp.array(rng.dirichlet(np.ones(3)))
                y = jnp.array(rng.uniform(0.05, 2.0, 2))
                exact = jax.hessian(lambda y: jnp.sum(fs * P(y, xs, b1, b2, p, q, benefit)))(y)
                chain = chain_hessian(y, xs, fs, b1, b2, p, q, benefit, mixed="exact")
                err = float(jnp.max(jnp.abs(exact - chain)))
                worst = max(worst, err)
                print(f"  {'ok' if err < 1e-6 else 'FAILED':6s} {benefit:14s} p={p:g} q={q:g}: max difference {err:.1e}")
        print("(b) silas Hessian, individual_sum, b1'=1.5, b2'=0.9, y = x = (0.3, 0.2), one group:")
        y = jnp.array([0.3, 0.2])
        ok_b = True
        for p, q, expected in [(1.25, 1.25, [[0.07068, -0.04690], [-0.04690, 0.08071]]),
                               (0.8, 1.25, [[-0.03412, -0.08057], [-0.08057, -0.41142]])]:
            H = chain_hessian(y, y[None], jnp.array([1.0]), 1.5, 0.9, p, q, "individual_sum")
            err = float(np.max(np.abs(np.asarray(H) - expected)))
            ok_b &= err < 1e-4
            print(f"  {'ok' if err < 1e-4 else 'FAILED':6s} p={p:g} q={q:g}: H = {np.round(np.asarray(H), 5).tolist()} "
                  f"(max difference {err:.1e})")
    finally:
        jax.config.update("jax_enable_x64", False)
    assert worst < 1e-6 and ok_b, "Hessian tests failed"


# ---------- labels, figures and the comparison table ----------

LABEL_INFO = dict(x_max=x_max, merge_gap=MERGE_GAP, p_values=P_VALUES, q_values=Q_VALUES)
SCHEME = analysis.SCHEMES["cc_dc_dd"]


def unfiltered_labels(results):
    return analysis.grid_labels(analysis.cc_dc_dd(results["xs"], results["fs"], LABEL_INFO))   # (p, q, b2', b1')


def save_figure(labels, b, title, filename):
    analysis.plot_categories(labels, dict(LABEL_INFO, b=b), SCHEME, title, filename)
    plt.close("all")


def save_side_by_side(left, right, b, left_title, right_title, filename):
    # two 3 x 3 panels (columns p, rows q) with one shared legend, so the colours match
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    names, colours = analysis.label_colours(np.concatenate([left.ravel(), right.ravel()]), SCHEME)
    index = {l: k for k, l in enumerate(names)}
    fig = plt.figure(figsize=(28, 11))
    for k, (sub, labels, title) in enumerate(zip(fig.subfigures(1, 2), [left, right], [left_title, right_title])):
        codes = np.vectorize(lambda l: index.get(l, index.get("other")))(labels)
        axes = sub.subplots(len(Q_VALUES), len(P_VALUES), sharex=True, sharey=True, squeeze=False)
        for i, p in enumerate(P_VALUES):
            for j, q in enumerate(Q_VALUES):
                ax = axes[j, i]
                ax.imshow(codes[i, j], origin="lower", extent=[b[0], b[-1], b[0], b[-1]], aspect="auto",
                          interpolation="nearest", cmap=ListedColormap(colours), vmin=-0.5, vmax=len(names) - 0.5)
                analysis.draw_line(ax, b)
                ax.set_title(f"Phase diagram p = {p:g} q = {q:g}", fontsize=10)
                ax.set_xlabel("b1'"); ax.set_ylabel("b2'")
        sub.suptitle(title, fontsize=11)
        sub.subplots_adjust(left=0.25 if k == 0 else 0.08, wspace=0.25, hspace=0.3)     # room for the legend
    handles = [Patch(color=c, label=l) for l, c in zip(names, colours)]
    handles.append(Line2D([], [], color="black", linewidth=1, linestyle="--", label="b2' = b1'/2"))
    fig.legend(handles=handles, loc="center left", fontsize=8)
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"saved {filename}")


def compare(labels, reference):
    # differences from the reference labels: total, per (p, q) panel, and the 5 most common changes
    differ = labels != reference
    panels = {f"p={p:g} q={q:g}": int(differ[i, j].sum()) for i, p in enumerate(P_VALUES) for j, q in enumerate(Q_VALUES)}
    changes = Counter(zip(reference[differ], labels[differ])).most_common(5)
    return int(differ.sum()), panels, changes


# ---------- run plan ----------

def config(**changes):
    return dict(SETTINGS, **changes)

CONFIGS = {
    "baseline": config(),
    "benefit_only": config(BENEFIT="individual_sum"),
    "reference_all": config(BENEFIT="individual_sum", HESSIAN="silas", SPLIT_MODE="forced_once",
                            STOP="fixed_time", START_JITTER="on"),
}


def run_ablation(n=GRID_N, fixed_time=None):
    b = np.linspace(0, B_MAX, n)
    labels, info, saved = {}, {}, []
    for name, cfg in CONFIGS.items():
        if fixed_time is not None:
            cfg = dict(cfg, FIXED_TIME=fixed_time)
        print(f"\n---------- {name} ----------")
        results, tag = load_or_simulate(cfg, n)
        labels[name] = unfiltered_labels(results)
        info[name] = dict(tag=tag, seconds=float(results.get("seconds", np.nan)),
                          capped=int((results["status"] == 2).sum()) if cfg["STOP"] == "converged" else None)
        filename = f"toggles_unfiltered_{name}_{tag}.png"
        save_figure(labels[name], b, f"{name}, unfiltered - {tag}", filename)
        saved.append(filename)

    rows = []
    for name in CONFIGS:
        if name == "baseline":
            continue
        filename = f"toggles_side_by_side_baseline_vs_{name}_{info[name]['tag']}.png"
        save_side_by_side(labels["baseline"], labels[name], b, f"baseline - {info['baseline']['tag']}",
                          f"{name} - {info[name]['tag']}", filename)
        saved.append(filename)
        total, panels, changes = compare(labels[name], labels["baseline"])
        rows.append(dict(config=name, differ_from_baseline=total, **panels,
                         **{f"change_{k + 1}": f"{a} -> {c} ({m})" for k, ((a, c), m) in enumerate(changes)}))

    print(f"\nrun time and runs that hit MAX_STEPS = {MAX_STEPS} (converged mode only):")
    for name, d in info.items():
        capped = "n/a (fixed_time)" if d["capped"] is None else d["capped"]
        print(f"  {name:14s} {d['seconds']:8.0f}s   capped: {capped}")

    print(f"\ngrid points whose unfiltered label differs from baseline (of {labels['baseline'].size}):")
    for row in rows:
        print(f"\n{row['config']}: {row['differ_from_baseline']}")
        print("  per panel: " + ", ".join(f"{k}: {row[k]}" for k in row if k.startswith("p=")))
        for k in range(1, 6):
            if f"change_{k}" in row:
                print(f"  {row[f'change_{k}']}")

    with open("ablation_summary.csv", "w", newline="") as f:
        fields = ["config", "differ_from_baseline"] + [k for k in rows[0] if k.startswith("p=")] + \
                 [f"change_{k}" for k in range(1, 6)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    saved.append("ablation_summary.csv")
    print("\nsaved files:\n  " + "\n  ".join(saved))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    parser.add_argument("--ablation", action="store_true", help="run the three configs of the run plan")
    for key, choices in CHOICES.items():
        parser.add_argument("--" + key.lower().replace("_", "-"), choices=choices, default=SETTINGS[key])
    parser.add_argument("--fixed-time", type=float, default=SETTINGS["FIXED_TIME"])
    args = parser.parse_args()

    print("running on:", jax.devices())
    print("tests:"); run_tests()
    if args.ablation:
        run_ablation(args.n, args.fixed_time)
    else:
        cfg = config(**{key: getattr(args, key.lower()) for key in CHOICES}, FIXED_TIME=args.fixed_time)
        results, tag = load_or_simulate(cfg, args.n)
        save_figure(unfiltered_labels(results), np.linspace(0, B_MAX, args.n), f"unfiltered - {tag}",
                    f"toggles_unfiltered_{tag}.png")
