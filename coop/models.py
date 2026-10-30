# Payoff models. A model is the payoff P(y, x, prm) of an individual investing y = (y1, y2) whose
# partner invests x, with prm a dict of parameter values, plus what the rest of the code needs to
# know about it: its parameter names, default settings, payoff checks and default sweep.
# Gradients and Hessians come from jax, so a new model only needs its payoff.
#
# To add a model: write a function like quadratic below that returns a Model, and add it to MODELS.
# Options (e.g. flat_benefit) are keyword arguments of that function; to set one from the command
# line, add it to OPTIONS.

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Model:
    name: str
    params: tuple                  # parameter names, in the order they are stored per run
    labels: dict                   # parameter name -> axis label
    payoff: Callable               # payoff(y, x, prm): y own traits (goods,), x partners' (..., goods)
    checks: list                   # (y, x, prm, expected payoff)
    default_sweep: str
    options: dict = field(default_factory=dict)       # the options this model was built with
    settings: dict = field(default_factory=dict)      # overrides of the engine Settings defaults
    chain_hessian: Optional[Callable] = None          # needed for the engine's hessian="silas"

    @property
    def tag(self):
        # name plus the options that differ from their default, e.g. quadratic_flat
        short = dict(flat_benefit="flat", flat_cost="flatcost")
        on = [short.get(k, f"{k}-{v}") if v is True else f"{k}-{v}" for k, v in self.options.items()
              if v != OPTIONS[k][0][0]]
        return "_".join([self.name] + on)


# power (generalised) mean of a and b
M = lambda pow, a, b: ((a ** pow + b ** pow) / 2) ** (1 / pow)


# ---------- quadratic: b1' Mp - b2' Mp^2 - Mq + Mq^2 ----------

def quadratic(flat_benefit=False, flat_cost=False, benefit="pair_totals"):
    # benefit "pair_totals": Mp = M_p(x + y) of the pair's totals per good;
    #         "individual_sum": Mp = M_p(y) + M_p(x)
    def P(y, x, prm):
        b1, b2, p, q = prm["b1"], prm["b2"], prm["p"], prm["q"]
        if benefit == "pair_totals":
            Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
        else:
            Mp = M(p, y[..., 0], y[..., 1]) + M(p, x[..., 0], x[..., 1])
        Mq = M(q, y[..., 0], y[..., 1])
        # the benefit peaks at Mp = b1 / (2 b2); b2 can be 0, so a safe divisor keeps NaNs out of the gradients
        safe_b2 = jnp.where(b2 > 0, b2, 1.0)
        peak = jnp.where(b2 > 0, b1 / (2 * safe_b2), jnp.inf)
        m = jnp.minimum(Mp, peak) if flat_benefit else Mp
        # the cost Mq - Mq^2 peaks at Mq = 1/2 (never passed while x_max <= 1/2, as Mq <= x_max)
        c = jnp.minimum(Mq, 0.5) if flat_cost else Mq
        return b1 * m - b2 * m ** 2 - c + c ** 2

    # expected values with: both falling past their peaks, flat benefit, flat cost, both flat;
    # the second case is just past the benefit's peak (Mp = 1.414, peak 1.333), the third well past it
    # (Mp = 1, peak 0.8), the fourth past both (Mp = 2, peak 1.333; Mq = 1, peak 0.5).
    # y = x in every case, so the individual_sum benefit gives the same values.
    pick = lambda values: values[flat_benefit + 2 * flat_cost]
    prm = lambda b1, b2, p, q: dict(b1=b1, b2=b2, p=p, q=q)
    checks = [((0.5, 0.5), (0.5, 0.5), prm(2.0, 0.75, 1.0, 1.0), pick((1.0, 1.0, 1.0, 1.0))),
              ((1.0, 0.0), (1.0, 0.0), prm(2.0, 0.75, 2.0, 0.5), pick((1.141, 1.146, 1.141, 1.146))),
              ((0.5, 0.5), (0.5, 0.5), prm(1.2, 0.75, 1.0, 1.0), pick((0.20, 0.23, 0.20, 0.23))),
              ((1.0, 1.0), (1.0, 1.0), prm(2.0, 0.75, 1.0, 1.0), pick((1.0, 1.333, 0.75, 1.083)))]

    def chain_hessian(y, xs, fs, prm, mixed="silas"):
        # mutant Hessian by the chain rule, averaged over partner groups with their shares:
        # B'(m_p) Hp + B'' gp gp' - C'(m_q) Hq - C'' gq gq', with B(m) = b1' m - b2' m^2, C(m) = m - m^2;
        # mixed = "silas" uses a different mixed derivative of M (the reference implementation's)
        b1, b2, p, q = prm["b1"], prm["b2"], prm["p"], prm["q"]
        gq, Hq = M_parts(q, y, mixed)
        mq = M(q, y[0], y[1])
        cost = -(1 - 2 * mq) * Hq + 2 * jnp.outer(gq, gq)
        def with_partner(x):
            a = x + y if benefit == "pair_totals" else y
            gp, Hp = M_parts(p, a, mixed)
            mp = M(p, a[0], a[1]) if benefit == "pair_totals" else M(p, y[0], y[1]) + M(p, x[0], x[1])
            return (b1 - 2 * b2 * mp) * Hp - 2 * b2 * jnp.outer(gp, gp) + cost
        return jnp.sum(fs[:, None, None] * jax.vmap(with_partner)(xs), axis=0)

    return Model(name="quadratic", params=("p", "q", "b1", "b2"),
                 labels=dict(p="p", q="q", b1="b1'", b2="b2'"), payoff=P, checks=checks,
                 default_sweep="b1_b2", options=dict(flat_benefit=flat_benefit, flat_cost=flat_cost, benefit=benefit),
                 settings=dict(x_max=0.5),
                 chain_hessian=None if (flat_benefit or flat_cost) else chain_hessian)


def M_parts(r, a, mixed):
    # gradient and Hessian of M_r at a = (a1, a2); mixed = "silas" replaces the mixed derivative
    f = lambda a: M(r, a[0], a[1])
    g, H = jax.grad(f)(a), jax.hessian(f)(a)
    if mixed == "silas":
        o = -2 ** (-1 / r) * a[0] ** (r - 1) * a[1] ** (r - 1) * (a[0] ** r + a[1] ** r) ** (1 / r - 2)
        H = H.at[0, 1].set(o).at[1, 0].set(o)
    return g, H


# ---------- exponential: b (1 - exp(-Mp / beta)) - (1 - exp(-Mq)) ----------

def exponential():
    def P(y, x, prm):
        b, beta, p, q = prm["b"], prm["beta"], prm["p"], prm["q"]
        Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
        Mq = M(q, y[..., 0], y[..., 1])
        return b * (1 - jnp.exp(-Mp / beta)) - (1 - jnp.exp(-Mq))

    prm = lambda b, beta, p, q: dict(b=b, beta=beta, p=p, q=q)
    checks = [((0.5, 0.5), (0.5, 0.5), prm(2.0, 1.5, 1.0, 1.0), 0.5797),
              ((1.0, 1e-12), (1.0, 1e-12), prm(2.0, 1.5, 2.0, 0.5), 0.9997)]

    # x_max = 10 is above the singular point x* everywhere on the default grid (largest about 7).
    # Starts are set directly, not as fractions of x_max: at large investment both curves are flat and
    # the gradient falls below tol, so a start near x_max would count as converged without moving.
    # max_dt = 1: the flat exponential curves give small gradients, so fixed steps of 0.01 crawl.
    return Model(name="exp", params=("p", "q", "b", "beta"),
                 labels=dict(p="p", q="q", b="b'", beta="beta'"), payoff=P, checks=checks,
                 default_sweep="b_beta", settings=dict(x_max=10.0, lo=0.01, hi=4.0, max_dt=1.0))


MODELS = {"quadratic": quadratic, "exp": exponential}

# options that can be set from the command line: name -> (choices, the first being the default; help)
OPTIONS = {
    "flat_benefit": ([False, True], "quadratic: the benefit stays at its maximum past its peak instead of falling"),
    "flat_cost":    ([False, True], "quadratic: the cost stays at its maximum past its peak (Mq = 1/2) instead of falling"),
    "benefit":      (["pair_totals", "individual_sum"],
                     "quadratic: Mp of the pair's totals per good, or the sum of each partner's own Mp"),
}


def make_model(name, **options):
    # the model with the given options; an option the model doesn't have is an error
    try:
        return MODELS[name](**options)
    except TypeError:
        raise SystemExit(f"model {name!r} doesn't take the option(s) {', '.join(options)}")


def check_payoff(model):
    for y, x, prm, expected in model.checks:
        got = float(model.payoff(jnp.array(y), jnp.array(x), prm))
        values = ", ".join(f"{k}={v}" for k, v in prm.items())
        print(f"  {'ok' if abs(got - expected) < 1e-3 else 'FAILED':6s} P(y={y}, x={x}, {values}) = {got:.4f} "
              f"(expected {expected})")
        assert abs(got - expected) < 1e-3


def check_chain_hessian(model):
    # the chain-rule Hessian (with the exact mixed derivative) against jax.hessian, and the reference
    # implementation's ("silas") values for two hand-checked cases; run in 64-bit so the comparison to
    # 1e-6 is about the formulas, not rounding
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(1)
        worst = 0.0
        for _ in range(5):
            p, q = rng.choice([0.8, 1.0, 1.25, 2.0], 2)
            b1, b2 = rng.uniform(0, 4, 2)
            prm = dict(b1=b1, b2=b2, p=p, q=q)
            xs = jnp.array(rng.uniform(0.05, 2.0, (3, 2)))
            fs = jnp.array(rng.dirichlet(np.ones(3)))
            y = jnp.array(rng.uniform(0.05, 2.0, 2))
            exact = jax.hessian(lambda y: jnp.sum(fs * model.payoff(y, xs, prm)))(y)
            worst = max(worst, float(jnp.max(jnp.abs(exact - model.chain_hessian(y, xs, fs, prm, mixed="exact")))))
        print(f"  {'ok' if worst < 1e-6 else 'FAILED':6s} chain-rule Hessian vs jax.hessian: max difference {worst:.1e}")
        ok = worst < 1e-6
        if model.options["benefit"] == "individual_sum":
            y = jnp.array([0.3, 0.2])
            for p, q, expected in [(1.25, 1.25, [[0.07068, -0.04690], [-0.04690, 0.08071]]),
                                   (0.8, 1.25, [[-0.03412, -0.08057], [-0.08057, -0.41142]])]:
                H = model.chain_hessian(y, y[None], jnp.array([1.0]), dict(b1=1.5, b2=0.9, p=p, q=q))
                err = float(np.max(np.abs(np.asarray(H) - expected)))
                ok &= err < 1e-4
                print(f"  {'ok' if err < 1e-4 else 'FAILED':6s} silas Hessian, b1'=1.5, b2'=0.9, y = x = (0.3, 0.2), "
                      f"p={p:g} q={q:g}: max difference {err:.1e}")
    finally:
        jax.config.update("jax_enable_x64", x64)
    assert ok, "Hessian checks failed"
