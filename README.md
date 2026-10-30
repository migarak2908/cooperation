# Cooperation

Adaptive dynamics of cooperative investment in public goods. A population invests
an amount `x` in a shared good; selection can drive it to full cooperation, full
defection, an intermediate level, or make it branch into coexisting cooperators and
defectors. This repo maps out which happens where in parameter space, analytically
and by simulation, first for one good and then for two.

## Model

### One good

An individual investing `y` among others investing `x` gets

```
P(y, x) = b1'(x + y) - b2'(x + y)^2 - y + y^2
```

(benefit from the total investment minus the cost of its own), with traits in
`[0, x_max]`. Everything follows from the selection gradient `D(x) = dP/dy` at `y = x`:

| Quantity | Formula |
|---|---|
| gradient at 0 | `D0 = b1' - 1` |
| gradient at `x_max` | `Dmax = (b1' - 1) + (2 - 4 b2') x_max` |
| singular point | `x* = (b1' - 1) / (4 b2' - 2)` |
| curvature (branching if > 0) | `2 - 2 b2'` |

| Region | Condition | Outcome |
|---|---|---|
| C | `D0 > 0`, `Dmax > 0` | everyone invests `x_max` |
| D | `D0 < 0`, `Dmax < 0` | everyone invests 0 |
| C or D | `D0 < 0`, `Dmax > 0` | bistable: `x*` is a tipping point, the start decides |
| ESS | `D0 > 0`, `Dmax < 0`, `b2' > 1` | everyone settles at `x*` |
| C and D | `D0 > 0`, `Dmax < 0`, `b2' < 1` | branching into defectors (0) and cooperators (`x_max`) |

In **C and D** the cooperators' share settles where both groups earn the same payoff:
`f = (b1' - 1 + m - b2' m) / (2 b2' m)` with `m = x_max`. In **C or D**, with starts
spread evenly over `[0, x_max]`, the probability of ending up cooperating is
`1 - x*/x_max`.

### Two goods

Each individual invests in two goods: `y = (y1, y2)` for the focal individual, `x = (x1, x2)` for its partner. Each good is pooled across the pair, and the two pooled amounts are combined with a power (generalised) mean, `M_p(a, b) = ((a^p + b^p) / 2)^(1/p)`:

```
P(y, x) = b1' M_p(x1 + y1, x2 + y2) - b2' M_p(x1 + y1, x2 + y2)^2 - M_q(y1, y2) + M_q(y1, y2)^2
```

`p` sets how the goods combine in the benefit and `q` how the investments combine in
the cost: `p = 1` is a plain average, `p < 1` makes the goods complements, `p > 1`
substitutes. With `p = q = 1` the model reduces exactly to the one-good model in the
average investment.

The exponential model replaces the benefit and cost with saturating curves,
`b (1 - exp(-M_p / beta)) - (1 - exp(-M_q))`; for `beta < 1` and `b > beta` theory predicts single
specialists for `q < p`, division of labour for `p < q < (p + 1)/2`, and generalists for `q > (p + 1)/2`.

## Running

**See [USAGE.md](USAGE.md) for what to run for each model, sweep and option, and what each gives.**
In short:

```
python run_simulation.py --model quadratic --n 10       # saves results_<tag>.npz
python run_analysis.py results_<tag>.npz                # labels it and saves the figures
```

## Layout

| Path | What it holds |
|---|---|
| `run_simulation.py` | runs a model over a sweep and saves the final state of every run |
| `run_analysis.py` | labels and plots a results file (no JAX needed) |
| `coop/models.py` | payoff models (quadratic, exponential) and their options |
| `coop/sweeps.py` | parameter sweeps: which parameters are panels and which form the grid |
| `coop/engine.py` | the adaptive dynamics, for any model and sweep |
| `coop/results.py` | saving and loading results files |
| `coop/schemes.py` | labelling schemes: what counts as cooperation |
| `coop/references.py` | theory: `x*`, reference lines, analytic predictions |
| `coop/plots.py` | figures |
| `single_good/` | small self-contained one-good scripts (analytic phase diagram and simulations) |
| `archive/two_good_integration_phase.py` | an early two-good version with exactly two groups per run (kept for reference) |

### How the two-good simulation works

Each run (one parameter point and one start) has a number of group slots (2, moving up to 4 and then
8 only when a split needs room) and is advanced 1000 steps at a time; every step updates the traits
along the selection gradient and the shares by replicator dynamics, drops groups below a minimum
share and merges groups that come within `1e-3`. After each chunk:

- a group that has stopped on a **saddle** (the selection-gradient Jacobian has a positive
  eigenvalue) is pushed off it along the unstable direction, so symmetric starts don't branch
  where they shouldn't
- a group that has stopped at an attracting point where the mutant payoff curves upwards
  **splits** along that direction, unless it split nearby just before
- a run whose gradients and payoff differences are all below `tol` has **converged** and stops
- a run that is **cycling** (large gradient, little net movement) has its time step halved; it
  grows back (up to `max_dt`) once the run moves normally

Gradients, Hessians and Jacobians come from JAX applied to the model's payoff, so a new model only
needs its payoff function. Thousands of runs are advanced together with `jax.vmap` inside `jax.jit`.
Traits are kept in `[x_min, x_max]` with `x_min > 0` because the power mean's gradient is infinite
at 0 when `p < 1`.

## Checks and caveats

- The one-good simulation matches the analytic regions at all 10,000 points of a
  100 × 100 grid, and the cooperator share matches `f` (e.g. 0.194 at `b1' = 1.5`,
  `b2' = 0.9`).
- The two-good simulation with `p = q = 1` reproduces the one-good results.
- Close to the edges of the C and D region (`x*` near 0 or `x_max`) and near
  `b2' = 0.5` the dynamics are very slow; the region labels are still right there, but
  the average trait may need far more steps (50,000+) to converge.
