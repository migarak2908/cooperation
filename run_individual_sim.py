import argparse
import numpy as np
import jax
import jax.numpy as jnp
from coop.models import quadratic


POP_SIZE = 10_000
TRAITS = 2
GENERATIONS = 125_000
RECORD_EVERY = 100                   # save the population every this many generations
UPDATE_FRACTION = 0.01               # share of the population updated at once (1 / POP_SIZE: one at a time, 1: all)
MUTATION_PROB = 0.01                 # per trait
MUTATION_STD = 0.005
X_MAX = 0.5
START = (0.1, 0.4)                   # every individual's (trait 1, trait 2) at generation 0


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


def main_sim(key, alpha, sigma, mu, pop0, prm, x_max, n_records, record_every, update_fraction, rho=0.0):
    # the population every record_every generations: (n_records, N, TRAITS)
    N = pop0.shape[0]
    batch = max(1, round(update_fraction * N))
    if N % batch:
        raise SystemExit(f"update fraction {update_fraction:g} gives batches of {batch}, which don't divide {N}")

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
        z = jax.random.normal(k6, (batch, TRAITS))
        dx = sigma * jnp.stack([z[:, 0], rho * z[:, 0] + jnp.sqrt(1 - rho**2) * z[:, 1]], axis=1)
        child = jnp.clip(child + mutate[:, None] * dx, 0, x_max)
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
        return carry, carry[0]

    run = jax.jit(lambda carry: jax.lax.scan(one_record, carry, None, length=n_records)[1])
    return run((pop0, key))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="individual-based simulation; defaults are the settings above")
    for name in ("b1", "b2", "p", "q"):
        ap.add_argument(f"--{name}", type=float, default=prm[name])
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--start", type=float, nargs=2, default=START, metavar=("X1", "X2"))
    ap.add_argument("--mutation-prob", type=float, default=MUTATION_PROB)
    ap.add_argument("--mutation-std", type=float, default=MUTATION_STD)
    ap.add_argument("--rho", type=float, default=0.0, help="correlation of the two goods' mutation steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="individual_sim.npz")
    args = ap.parse_args()
    if not -1 <= args.rho <= 1:
        raise SystemExit(f"--rho must be between -1 and 1, got {args.rho:g}")
    prm = {name: getattr(args, name) for name in ("b1", "b2", "p", "q")}

    key = jax.random.key(args.seed)
    pop0 = jnp.tile(jnp.array(args.start, jnp.float32), (POP_SIZE, 1))
    alpha = payoff_range(prm, X_MAX)
    print("running on:", jax.devices())
    print("params:", prm, "rho:", args.rho)
    print(f"alpha = {alpha:.4f}")

    history = main_sim(key, alpha, args.mutation_std, args.mutation_prob, pop0, prm, X_MAX,
                       args.generations // RECORD_EVERY, RECORD_EVERY, UPDATE_FRACTION, args.rho)
    np.savez(args.out, history=np.asarray(history), pop0=np.asarray(pop0), alpha=alpha,
             record_every=RECORD_EVERY, update_fraction=UPDATE_FRACTION, x_max=X_MAX, rho=args.rho, **prm)
    print(f"saved {args.out}\nnext: python run_analysis.py {args.out}")
