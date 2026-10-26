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

## Files

| File | What it does |
|---|---|
| `single_good_single_pop.py` | one monomorphic population following the selection gradient |
| `single_good_multiple_pop.py` | loop version: a population that can split into two groups, with share dynamics, merging and dropping of small groups |
| `single_good_analytic_phase.py` | analytic phase diagram and average trait for one good |
| `single_good_integration_phase.py` | vectorised simulation of the two-group dynamics over a `(b1', b2')` grid and several starting values |
| `two_good_integration_phase.py` | the same for two goods, over `(p, q, b1', b2')` |
| `two_good_simulate.py` | two goods with repeated branching, each run continued until it converges; saves every final state to `two_good_results_<tag>.npz` |
| `two_good_analysis.py` | labels and plots a results file; labelling schemes (`cc_dc_dd`, `cc_dc_dd_division`, `mean_investment`) are listed in `SCHEMES` |

### How the simulations work

`multiple_pop_run` simulates one population for one parameter point and one start,
following the same steps as `multiple_pop`: gradients, trait update, payoffs, share
update, then drop / merge / split checks. To make it fast:

- there are always two groups; a single group is both groups at the same trait,
  which behaves identically
- the `if` checks are written as `jnp.where(condition, if_true, if_false)`
- the time loop is `jax.lax.scan`, and `jax.vmap` runs every
  (start, parameter) combination at once inside one `jax.jit`

Gradients and curvature come from `jax.grad` / `jax.hessian`, so changing the model
only means editing `P`. In the two-good version a group splits when the curvature
matrix (restricted to goods that are not at a wall) has a positive eigenvalue, along
the corresponding eigenvector; traits are kept in `[x_min, x_max]` with
`x_min = 1e-6` because the power mean's gradient is infinite at 0 when `p < 1`.
With `p < 1` the push on a good near 0 doesn't fade, so fixed time steps can cycle
around a corner instead of reaching it; the time step therefore shrinks smoothly over
the last 20% of a run (`anneal_frac`, `anneal_to`), and outcomes are classified with
small tolerances (`wall_tol`, `same_tol`) rather than exact equality.

Each run is classified by its final state (C, D, ESS, or C and D — per good in the
two-good version); each grid point takes the most common outcome over starts, and is
labelled **C or D** when some starts end C and others D. Starts include both walls
(and all four corners for two goods) so bistable points are always detected.

### Two goods with repeated branching: simulate, then analyse

`two_good_simulate.py` lets a population split more than once. Each run has a number of group
slots (2, moving up to 4 and then 8 only when a split needs room) and is advanced 1000 steps at a
time. After each chunk:

- a group that has stopped on a **saddle** (the selection-gradient Jacobian has a positive
  eigenvalue) is pushed off it along the unstable direction, so symmetric starts don't branch
  where they shouldn't
- a group that has stopped at an attracting point where the mutant payoff curves upwards
  **splits** along that direction, unless it split nearby just before
- a run whose gradients and payoff differences are all below `TOL` has **converged** and stops
- a run that is **cycling** (large gradient, little net movement) has its time step halved; it
  grows back once the run moves normally

Runs stop at `MAX_STEPS` if they never converge. The final traits and shares of every run are
saved, with the settings needed to read them (`x_max`, p and q values, b grid, starts), in
`two_good_results_<tag>.npz`; if that file exists the simulation is skipped.

`two_good_analysis.py` turns a results file into plots. A **scheme** defines what cooperation means:

| Scheme | Kind | What it gives |
|---|---|---|
| `cc_dc_dd` | categories | a good counts as invested above `x_max / 2`; each group is CC (both goods), DC (one) or DD (neither); a run's outcome is its distinct group types, e.g. `CC and DD`, and a grid point's label is the distinct outcomes over its starts, e.g. `CC \|or\| DD` |
| `cc_dc_dd_division` | categories | the same, but DC groups investing in different goods give `DC and DC'` (division of labour) |
| `mean_investment` | scale | share-weighted mean investment / `x_max`, averaged over starts |

Category schemes produce two figures: the labels as they came out, and a filtered version where
labels on fewer than 0.3% of all grid points are replaced by their nearest neighbour. Every
figure has a panel per `(p, q)` (p across, q down) and the line `b2' = b1'/2`. To add a scheme,
write a function `(xs, fs, r)` returning a name or number per run and add it to `SCHEMES`.

## Running

Requires `jax`, `numpy`, `scipy` and `matplotlib` (`two_good_analysis.py` doesn't need `jax`).

```
python single_good_analytic_phase.py
python single_good_integration_phase.py
python two_good_integration_phase.py
python two_good_simulate.py --n 30
python two_good_analysis.py --scheme cc_dc_dd mean_investment
```

`two_good_analysis.py` uses the newest results file unless one is given, and needs no JAX,
so a results file from Colab can be analysed locally.

Grid sizes, starts, step size and number of steps are set at the top of each
`__main__` block; for `two_good_simulate.py` they are in the settings at the top of the file.

### On Google Colab (GPU)

1. Runtime → Change runtime type → T4 GPU
2. Upload the script with the Files panel
3. In a cell: `%run two_good_integration_phase.py` (`%run`, not `!python`, so the figures show)
4. The first line printed should list a `CudaDevice`; if it shows `CpuDevice`, run
   `!pip install -U "jax[cuda12]"`, restart the runtime and run again

The first run includes about a minute of compilation.

## Checks and caveats

- The one-good simulation matches the analytic regions at all 10,000 points of a
  100 × 100 grid, and the cooperator share matches `f` (e.g. 0.194 at `b1' = 1.5`,
  `b2' = 0.9`).
- The two-good simulation with `p = q = 1` reproduces the one-good results.
- Close to the edges of the C and D region (`x*` near 0 or `x_max`) and near
  `b2' = 0.5` the dynamics are very slow; the region labels are still right there, but
  the average trait may need far more steps (50,000+) to converge.
