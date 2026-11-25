import argparse
import numpy as np
import jax
import jax.numpy as jnp
from coop.models import quadratic


POP_SIZE = 10_000
TRAITS = 2
GENERATIONS = 75_000
RECORD_EVERY = 100                   # save the population every this many generations
UPDATE_FRACTION = 0.01               # share of the population updated at once (1 / POP_SIZE: one at a time, 1: all)
MUTATION_PROB = 0.01                 # per individual: both goods mutate in one event
MUTATION_STD = 0.005
X_MAX = 0.5
START = (0.1, 0.4)                   # every individual's (trait 1, trait 2) at generation 0
HIST_BINS = 50                       # --record summary: bins of the u and v histograms
FULL_REFUSE_GB = 4                   # --record full: refuse a history bigger than this, warn above FULL_WARN_GB
FULL_WARN_GB = 1


prm = {"b1": 7 / 4.6, "b2": 1.5, "p": 0.98, "q": 0.984}
P = quadratic(flat_benefit=True).payoff          # P(own, partner, prm), the benefit flat past its peak


"""
Choose random focal individual x (in batches: each generation every individual is focal once, a
batch of them at a time, all reading the population as it was before the batch)
x replaced by offspring as follows
Payoff_x is determined by single interaction with random member z of population (z != x)
Payoff_x is compared to Payoff_y of another randomly chosen member y (y != x, y's partner != y)
w = (P_y - P_x) / alpha - clip to [0, 1]
alpha = largest possible payoff difference

offspring replacing x has y as parent with w prob, else has x as parent with 1-w prob
offspring inherits parental strategy, except with mu prob both traits mutate in one event:
steps sigma*z1 and sigma*(rho*z1 + sqrt(1-rho^2)*z2), z1, z2 standard normal
rho > 0: more variation in total investment, rho < 0: more in allocation between the goods

Summaries use u = (x1 + x2)/sqrt(2) (total) and v = (x1 - x2)/sqrt(2) (allocation).
"""


def payoff_range(prm, x_max, n=21):
    # largest payoff minus smallest, over every own and partner strategy in [0, x_max]^2 (n x n grid)
    v = jnp.linspace(0, x_max, n)
    traits = jnp.stack(jnp.meshgrid(v, v), axis=-1).reshape(-1, TRAITS)
    payoffs = jax.vmap(lambda own: P(own, traits, prm))(traits)
    return float(payoffs.max() - payoffs.min())


def other(key, N, not_this):
    # for each index in not_this, a random index from 0 to N - 1 that isn't it
    k = jax.random.randint(key, not_this.shape, 0, N - 1)
    return k + (k >= not_this)


def correlated_steps(key, shape, sd, rho):
    # normal steps (shape + (2,)), sd in each good, correlation rho between the goods
    z = jax.random.normal(key, shape + (TRAITS,))
    return sd * jnp.stack([z[..., 0], rho * z[..., 0] + jnp.sqrt(1 - rho**2) * z[..., 1]], axis=-1)


def hist_edges(x_max, bins=HIST_BINS):
    # fixed bin edges covering every possible u and v for traits in [0, x_max]
    r = x_max / np.sqrt(2)
    return np.linspace(0, 2 * r, bins + 1), np.linspace(-r, r, bins + 1)


def summarise(pop, x_max, u_edges, v_edges):
    # pop (N, 2) -> dict of summaries (see the module docstring for u and v)
    u = (pop[:, 0] + pop[:, 1]) / np.sqrt(2)
    v = (pop[:, 0] - pop[:, 1]) / np.sqrt(2)
    du, dv = u - u.mean(), v - v.mean()

    def hist(a, edges):
        lo, hi, bins = edges[0], edges[-1], len(edges) - 1
        idx = jnp.clip(jnp.floor((a - lo) / (hi - lo) * bins).astype(jnp.int32), 0, bins - 1)
        return jnp.bincount(idx, length=bins).astype(jnp.uint16)

    return {"mean_u": u.mean(), "mean_v": v.mean(),
            "var_u": (du**2).mean(), "var_v": (dv**2).mean(), "cov_uv": (du * dv).mean(),
            "n_zero": (pop == 0).sum(0).astype(jnp.int32), "n_max": (pop == x_max).sum(0).astype(jnp.int32),
            "hist_u": hist(u, u_edges), "hist_v": hist(v, v_edges)}


def main_sim(key, alpha, sigma, mu, pop0, prm, x_max, n_records, record_every, update_fraction, rho=0.0,
             record="full"):
    # record "full": (history (n_records, N, TRAITS), final population)
    # record "summary": (dict of summaries, each with a leading n_records axis, final population)
    N = pop0.shape[0]
    batch = max(1, round(update_fraction * N))
    if N % batch:
        raise SystemExit(f"update fraction {update_fraction:g} gives batches of {batch}, which don't divide {N}")
    u_edges, v_edges = hist_edges(x_max)

    def one_batch(carry, i):
        # i: the focal individuals of this batch, (batch,)
        pop, key = carry
        k1, k2, k3, k4, k5, k6, key = jax.random.split(key, 7)
        j = other(k1, N, i)                  # each i's partner
        pi = other(k2, N, i)                 # the individual each i compares with
        pj = other(k3, N, pi)                # pi's partner
        Px = P(pop[i], pop[j], prm)
        Py = P(pop[pi], pop[pj], prm)
        w = jnp.clip((Py - Px) / alpha, 0.0, 1.0)
        copy = jax.random.uniform(k4, (batch,)) < w
        child = jnp.where(copy[:, None], pop[pi], pop[i])
        # with prob mu both goods mutate in one event, sd sigma each, correlation rho
        mutate = jax.random.uniform(k5, (batch,)) < mu
        child = jnp.clip(child + mutate[:, None] * correlated_steps(k6, (batch,), sigma, rho), 0, x_max)
        pop = pop.at[i].set(child)

        return (pop, key), None

    def one_generation(carry, _):
        # every individual is focal once, in random order, a batch at a time
        pop, key = carry
        key, k = jax.random.split(key)
        order = jax.random.permutation(k, N).reshape(-1, batch)
        carry, _ = jax.lax.scan(one_batch, (pop, key), order)
        return carry, None

    def one_record(carry, _):
        carry, _ = jax.lax.scan(one_generation, carry, None, length=record_every)
        out = carry[0] if record == "full" else summarise(carry[0], x_max, u_edges, v_edges)
        return carry, out

    (pop, _), out = jax.lax.scan(one_record, (pop0, key), None, length=n_records)
    return out, pop


def start_population(key, start, sd, rho, pop_size, x_max):
    # every individual at start, plus correlated normal deviations if sd > 0, clipped to [0, x_max]
    pop0 = jnp.tile(jnp.array(start, jnp.float32), (pop_size, 1))
    if sd > 0:
        pop0 = jnp.clip(pop0 + correlated_steps(key, (pop_size,), sd, rho), 0, x_max)
    return pop0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="individual-based simulation; defaults are the settings above")
    for name in ("b1", "b2", "p", "q"):
        ap.add_argument(f"--{name}", type=float, default=prm[name])
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--start", type=float, nargs=2, default=START, metavar=("X1", "X2"))
    ap.add_argument("--start-sd", type=float, default=0.0,
                    help="sd of each individual's starting deviation from --start in each good (0: all at --start)")
    ap.add_argument("--start-rho", type=float, default=0.0, help="correlation of the starting deviations")
    ap.add_argument("--mutation-prob", type=float, default=MUTATION_PROB)
    ap.add_argument("--mutation-std", type=float, default=MUTATION_STD)
    ap.add_argument("--rho", type=float, default=0.0, help="correlation of the two goods' mutation steps")
    ap.add_argument("--pop-size", type=int, default=POP_SIZE)
    ap.add_argument("--record-every", type=int, default=RECORD_EVERY)
    ap.add_argument("--update-fraction", type=float, default=UPDATE_FRACTION)
    ap.add_argument("--record", choices=("full", "summary"), default="full",
                    help="full: the whole population every record; summary: moments, edge counts, u and v histograms")
    ap.add_argument("--replicates", type=int, default=1, help="independent runs in parallel (leading axis if > 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="individual_sim.npz")
    args = ap.parse_args()
    for flag in ("rho", "start_rho"):
        if not -1 <= getattr(args, flag) <= 1:
            raise SystemExit(f"--{flag.replace('_', '-')} must be between -1 and 1, got {getattr(args, flag):g}")
    if args.start_sd < 0 or args.replicates < 1 or args.pop_size < 2 or args.record_every < 1:
        raise SystemExit("need --start-sd >= 0, --replicates >= 1, --pop-size >= 2, --record-every >= 1")
    if args.record == "summary" and args.pop_size > np.iinfo(np.uint16).max:
        raise SystemExit(f"--record summary stores histogram counts as 16-bit, so --pop-size must be at most "
                         f"{np.iinfo(np.uint16).max}")
    prm = {name: getattr(args, name) for name in ("b1", "b2", "p", "q")}
    R, n_records = args.replicates, args.generations // args.record_every

    if args.record == "full":
        gb = R * n_records * args.pop_size * TRAITS * 4 / 1e9
        if gb > FULL_REFUSE_GB:
            raise SystemExit(f"--record full would keep a {gb:.1f} GB history (limit {FULL_REFUSE_GB} GB); "
                             f"use --record summary, fewer --replicates or a larger --record-every")
        if gb > FULL_WARN_GB:
            print(f"warning: --record full keeps a {gb:.1f} GB history in memory and on disk")

    # replicate 0 runs on the seed's key itself, so one replicate matches the single-run script;
    # the starting cloud has its own key folded from each replicate's, leaving the run's draws unchanged
    key = jax.random.key(args.seed)
    keys = jnp.stack([key] + [jax.random.fold_in(key, r) for r in range(1, R)])
    start_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2**31 - 1))(keys)
    pop0 = jax.vmap(lambda k: start_population(k, args.start, args.start_sd, args.start_rho,
                                               args.pop_size, X_MAX))(start_keys)
    alpha = payoff_range(prm, X_MAX)
    print("running on:", jax.devices())
    print("params:", prm, "rho:", args.rho)
    print(f"alpha = {alpha:.4f}")

    sim = lambda k, p0: main_sim(k, alpha, args.mutation_std, args.mutation_prob, p0, prm, X_MAX, n_records,
                                 args.record_every, args.update_fraction, args.rho, args.record)
    out, final = jax.jit(jax.vmap(sim))(keys, pop0)

    squeeze = (lambda a: np.asarray(a)[0]) if R == 1 else np.asarray     # one replicate: no replicate axis
    data = {"history": squeeze(out)} if args.record == "full" else {k: squeeze(a) for k, a in out.items()}
    u_edges, v_edges = hist_edges(X_MAX)
    np.savez(args.out, **data, pop0=squeeze(pop0), final=squeeze(final), alpha=alpha,
             record_every=args.record_every, update_fraction=args.update_fraction, x_max=X_MAX, rho=args.rho,
             **prm, generations=args.generations, start=np.array(args.start), start_sd=args.start_sd,
             start_rho=args.start_rho, mutation_prob=args.mutation_prob, mutation_std=args.mutation_std,
             pop_size=args.pop_size, replicates=R, record=args.record, seed=args.seed,
             u_edges=u_edges, v_edges=v_edges)
    print(f"saved {args.out}")
    if args.record == "full" and R == 1:
        print(f"next: python run_analysis.py {args.out}")
