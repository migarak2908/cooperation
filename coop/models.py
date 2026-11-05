# Payoff models. A model is the payoff P(y, x, prm) of an individual investing y = (y1, y2) whose
# partner invests x, with prm a dict of parameter values, plus its parameter names, default settings,
# payoff checks and default sweep. Gradients and Hessians come from jax, so a model only needs its payoff.
#
# To add a model: write a function like quadratic below that returns a Model, and add it to MODELS.
# Options (e.g. flat_benefit) are its keyword arguments; add them to OPTIONS for a command-line flag.

from dataclasses import dataclass, field
from typing import Callable
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

    @property
    def tag(self):
        # name plus the options that are on, e.g. quadratic_flat
        short = dict(flat_benefit="flat", flat_cost="flatcost")
        return "_".join([self.name] + [short[k] for k, v in self.options.items() if v])


# power (generalised) mean of a and b
M = lambda pow, a, b: ((a ** pow + b ** pow) / 2) ** (1 / pow)


# ---------- quadratic: b1' Mp - b2' Mp^2 - Mq + Mq^2 ----------

def quadratic(flat_benefit=False, flat_cost=False):
    def P(y, x, prm):
        b1, b2, p, q = prm["b1"], prm["b2"], prm["p"], prm["q"]
        Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])     # the pair's totals per good
        Mq = M(q, y[..., 0], y[..., 1])                              # own investment only
        # the benefit peaks at Mp = b1 / (2 b2); b2 can be 0, so a safe divisor keeps NaNs out of the gradients
        peak = jnp.where(b2 > 0, b1 / (2 * jnp.where(b2 > 0, b2, 1.0)), jnp.inf)
        m = jnp.minimum(Mp, peak) if flat_benefit else Mp
        # the cost Mq - Mq^2 peaks at Mq = 1/2 (never passed while x_max <= 1/2, as Mq <= x_max)
        c = jnp.minimum(Mq, 0.5) if flat_cost else Mq
        return b1 * m - b2 * m ** 2 - c + c ** 2

    # expected values without and with each flat option (none, flat benefit, flat cost, both):
    # the second case is just past the benefit's peak (Mp = 1.414, peak 1.333), the third well past it
    # (Mp = 1, peak 0.8), the fourth past both (Mp = 2, peak 1.333; Mq = 1, peak 0.5)
    pick = lambda values: values[flat_benefit + 2 * flat_cost]
    prm = lambda b1, b2, p, q: dict(b1=b1, b2=b2, p=p, q=q)
    checks = [((0.5, 0.5), (0.5, 0.5), prm(2.0, 0.75, 1.0, 1.0), pick((1.0, 1.0, 1.0, 1.0))),
              ((1.0, 0.0), (1.0, 0.0), prm(2.0, 0.75, 2.0, 0.5), pick((1.141, 1.146, 1.141, 1.146))),
              ((0.5, 0.5), (0.5, 0.5), prm(1.2, 0.75, 1.0, 1.0), pick((0.20, 0.23, 0.20, 0.23))),
              ((1.0, 1.0), (1.0, 1.0), prm(2.0, 0.75, 1.0, 1.0), pick((1.0, 1.333, 0.75, 1.083)))]

    return Model(name="quadratic", params=("p", "q", "b1", "b2"),
                 labels=dict(p="p", q="q", b1="b1'", b2="b2'"), payoff=P, checks=checks,
                 default_sweep="b1_b2", options=dict(flat_benefit=flat_benefit, flat_cost=flat_cost),
                 settings=dict(x_max=0.5))


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

# options that can be set from the command line (all on/off): name -> help
OPTIONS = {
    "flat_benefit": "quadratic: the benefit stays at its maximum past its peak instead of falling",
    "flat_cost":    "quadratic: the cost stays at its maximum past its peak (Mq = 1/2) instead of falling",
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
