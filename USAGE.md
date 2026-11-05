# What to run

Every two-good experiment is two steps:

1. **`run_simulation.py`** runs a model over a parameter sweep and saves `results_<tag>.npz`
2. **`run_analysis.py`** labels and plots that file (no JAX needed, so it can run locally on a file from Colab)

The simulation ends by printing the exact analysis command to run next.

## Setup

Locally: `pip install jax numpy scipy matplotlib`, then run from the repo folder.

On Colab (GPU):

```
!git clone https://github.com/migarak2908/cooperation.git
%cd cooperation
%run run_simulation.py --model quadratic --n 10
```

Runtime → Change runtime type → T4 GPU first. The first line printed should list a `CudaDevice`;
if it shows `CpuDevice`, run `!pip install -U "jax[cuda12]"`, restart the runtime and run again.
Figures are saved as `.png` files in the folder, not shown in the notebook.

## Quick reference

| I want… | Simulation | Analysis | Main output |
|---|---|---|---|
| Quadratic phase diagrams: (b1', b2') grid, a panel per (p, q) | `--model quadratic` | `run_analysis.py RESULTS.npz` | `cc_dc_dd_..._filtered.png`, `..._unfiltered.png` |
| The same with the benefit flat past its peak | `--model quadratic --flat-benefit` | same | same, with the branching permissible region `1/2 < b2' < 1` drawn instead of the band |
| The same with the cost flat past its peak | `--model quadratic --flat-cost` | same | identical to the plain run while `x_max <= 0.5` (Mq never passes 1/2) |
| Exponential phase diagrams: (b', beta') grid, a panel per (p, q) | `--model exp` | same | same, with the exponential reference lines |
| Test the division-of-labour rule: (p, q) grid, a panel per (b', beta') | `--model exp --sweep p_q` | same (uses `cc_dc_dd_division`) | same, with `q = p`, `q = (p+1)/2`, `p = 1`, plus a prediction check printed |
| The same (p, q) grid for the quadratic model, a panel per (b1', b2') | `--model quadratic --sweep p_q_quadratic` | same | same, with `q = p`, `q = (p+1)/2`, `p = 1`, plus a prediction check printed |
| Other panel values, without editing code | add e.g. `--values beta=0.3,0.5` or `--values b2=0.7,0.95` | same | the changed values are added to the file name |
| How investment is spread (a colour scale) | any | `--scheme mean_investment` | `mean_investment_<tag>.png` |
| How cleanly groups divide the goods | any (made for `p_q`) | `--scheme division_of_labour_index` | `division_of_labour_index_<tag>.png` |
| How total investment and the split between goods change as p and q move away from 1 | `--model quadratic --values p=0.9,1,1.1 --values q=0.9,1,1.1` | `--scheme mean_investment individual --relative-to p=1,q=1` | `<scheme>_<tag>_rel-p1-q1.png` |

Add `--n 10` (or `--n 20` for `p_q`) to any simulation for a quick test; the default is a 100 × 100 grid.

## run_simulation.py

```
python run_simulation.py --model MODEL [--sweep SWEEP] [--n N] [model options] [settings]
```

### Models and sweeps

| Model | Payoff | Parameters | Default sweep | Model defaults |
|---|---|---|---|---|
| `quadratic` | `b1' Mp - b2' Mp^2 - Mq + Mq^2` | p, q, b1, b2 | `b1_b2` | `x_max = 0.5`, starts at 0.5% and 99.5% of `x_max` |
| `exp` | `b (1 - exp(-Mp / beta)) - (1 - exp(-Mq))` | p, q, b, beta | `b_beta` | `x_max = 10`, starts at 0.01 and 4, `max_dt = 1` |

`Mp = M_p(x1 + y1, x2 + y2)` (the pair's totals per good), `Mq = M_q(y1, y2)` (own investments),
with the power mean `M_r(a, b) = ((a^r + b^r) / 2)^(1/r)`.

| Sweep | For model | Panels | Grid | Sweep defaults |
|---|---|---|---|---|
| `b1_b2` | quadratic | p, q ∈ {0.8, 1, 1.25} | b1' and b2' from 0 to 4 | |
| `b_beta` | exp | p, q ∈ {0.8, 1, 1.25} | b' from 0 to 6, beta' from 0.017 to 1.7 | |
| `p_q` | exp | b' = 2, beta' ∈ {0.5, 0.75, 0.9} | p and q from 0.3 to 1.5 | `x_min = 1e-3` |
| `p_q_quadratic` | quadratic | b1' = 2, b2' ∈ {1.2, 1.5, 2} (inside the rule's region: b1' > 1, b2' above both 1 and b1'/2) | p and q from 0.3 to 1.5 | `x_min = 1e-3` |
| `p_q_near_one` | quadratic | as `p_q_quadratic` | p and q from 0.8 to 1.2 (zoomed in on p = q = 1) | `x_min = 1e-3` |

`--values NAME=V1,V2,...` replaces a sweep's panel values (repeat it for the other panel parameter),
e.g. `--sweep p_q_quadratic --values b1=1.2,1.6 --values b2=0.7`. The new values go into the file
name (`..._p_q_quadratic_b1-1.2-1.6_b2-0.7_n100_...`), so runs with different values never overwrite
each other. Grid ranges are changed in `coop/sweeps.py`.

Each run starts from one of four corners, `(lo, lo)`, `(lo, hi)`, `(hi, lo)`, `(hi, hi)`, so every
grid point is run four times. `--start X1,X2` (repeatable) replaces the four corners with the given start(s) at every grid point,
e.g. `--start 0.11,0.05`. The starts are added to the file name (`_start-0.11-0.05`) and recorded in
the results file, so `run_analysis.py --starts` indexes them (0, 1, ... in the order given).

### Model options (quadratic only)

| Flag | Effect |
|---|---|
| `--flat-benefit` | the benefit rises as before, then stays at its maximum past its peak `Mp = b1'/(2 b2')` instead of falling |
| `--flat-cost` | the cost stays at its maximum past `Mq = 1/2`; no effect while `x_max <= 0.5` |

### Settings

`--x-max`, `--x-min`, `--max-dt`, `--max-steps`, `--tol`, `--start` and `--x64` override the settings for one
run. Everything else (time step, chunk size, batch sizes, split and merge distances, …) is in
`Settings` in `coop/engine.py`. Priority, lowest first: `Settings` defaults → the model's defaults →
the sweep's → the command line.

### What it prints and saves

- payoff checks for the model
- a summary: runs finished and capped at `max_steps`, steps used, runs that needed more group slots or
  were pushed off a saddle (per panel)
- `results_<tag>.npz`: the final traits and shares of every run plus everything needed to read them.
  The tag is the model (and its options), the sweep, `n`, `x_max`, and any setting changed on the
  command line, e.g. `results_quadratic_flat_b1_b2_n100_xmax0.5.npz`. If the file already exists it
  is loaded instead of re-run (delete it to re-run); if it was made with different settings (e.g.
  after editing the sweep values) the run stops and asks you to delete or rename it.

## run_analysis.py

```
python run_analysis.py [RESULTS.npz] [--scheme S ...] [--invested RULE] [--relative-to ...] [--hatch ...] [--starts ...]
```

With no file it uses the newest `results_*.npz` in the folder.

### Schemes

| Scheme | Kind | What it gives |
|---|---|---|
| `cc_dc_dd` | categories | each group is CC (invests in both goods), DC (one) or DD (neither); a run's outcome is its distinct group types (e.g. `CC and DD`), a grid point's label the distinct outcomes over its four starts (e.g. `CC \|or\| DD`) |
| `cc_dc_dd_division` | categories | the same, but DC groups investing in different goods give `DC and DC'` (division of labour) |
| `mean_investment` | scale | share-weighted mean investment over groups and goods, divided by `x_max` (quadratic), the panel's `x*` (exp with b and beta per panel, scale up to 2) or the largest `x*` on the grid (exp otherwise). The colour scale runs from the lowest to the highest value in the figure |
| `specialisation` | scale | −1 to 1: 2 × population − individual lopsidedness; +1 = all in the same good, 0 = balanced, −1 = division of labour. Green (−1), white (0), blue (+1); NaN where nobody invests |
| `division_of_labour_index` | scale | 0 to 1: how lopsided individuals are minus how lopsided the population is; 1 = groups specialise in opposite goods |
| `individual` | scale | 0 to 1: how lopsided individuals are, `sum f_i \|x1_i - x2_i\| / sum f_i t_i`; 0 = everyone invests equally in both goods, 1 = everyone in one good only. Unsigned: it says how specialised individuals are, not in which good. Colour bar "individual specialisation", from "balanced" to "specialised" |
| `population` | scale | 0 to 1: how lopsided the population is as a whole, `\|sum f_i (x1_i - x2_i)\| / sum f_i t_i`; 0 = the goods are balanced overall. Division of labour (half in each good) gives 0. Colour bar "population bias", from "balanced" to "specialised" |
| `not_converged` | scale | 0 to 1: the fraction of starts that stopped at `max_steps` without converging |
| `dhk_type` | categories | by structure, not by the half-cap threshold. With each group's total `t_i = x1_i + x2_i` and top = `2 x_max`: `branching` if two groups' totals differ by more than 0.05 top; else, with the share-weighted mean total m, `defection` (m < 0.01 top), `full cooperation` (m > 0.99 top) or `ESS`. `defection \|or\| full cooperation` is the repeller (bistable). Compared with the one-good diagram at p = q = 1 |

`individual`, `population` and `division_of_labour_index` are NaN (light grey) where nobody invests: a share-weighted total `sum f_i t_i` below 1% of twice the scale (this also covers runs sitting near `x_min`). Averages over starts ignore NaN, so a grid point is NaN only if every start is.

Default: `cc_dc_dd_division` on a (p, q) grid (`p_q`, `p_q_quadratic`, `p_q_near_one`), `cc_dc_dd` otherwise. Several at once: `--scheme cc_dc_dd mean_investment`.

A category scheme gives two figures, `<scheme>_<tag>_unfiltered.png` (what happened) and
`..._filtered.png` (labels on fewer than 0.3% of grid points replaced by their nearest neighbour),
and prints label counts per panel and a check against theory:

| Model and sweep | Check printed |
|---|---|
| quadratic, `b1_b2` | the p = q = 1 panel against the one-good analytic diagram |
| exp, `b_beta` | the p = q = 1 panel against the one-good exponential diagram |
| exp, `p_q` | each panel against the division-of-labour rule, where the one-good `x*` is an ESS (beta' < 1, b' > beta') |
| quadratic, `p_q_quadratic` | the same, for b1' > 1 and b2' above both 1 and b1'/2 |

A scale scheme gives one figure, `<scheme>_<tag>.png`, averaged over starts. It has no overall title, and the colour bar carries a short label. The colours run from cream (low) to dark teal (high), keeping clear of the goods' red and blue; NaN is grey. Reference lines are mid-grey and listed in a legend below the panels.

### Other options

| Flag | Effect |
|---|---|
| `--invested half_cap` / `absolute` | when a good counts as invested: above `x_max / 2` (quadratic default) or above 0.05 (exp default); adds `_invested-<rule>` to the figure names when not the default |
| `--relative-to NAME=V[,NAME=V]` | scale schemes: each panel minus the panel with these values at the same grid point (after averaging over starts), on a blue-red scale centred on 0; e.g. `p=1,q=1`. A parameter left out keeps each panel's own value. Adds `_rel-p1-q1` to the figure name |
| `--relative-to-point NAME=V,NAME=V` | scale schemes: each grid point minus the grid point with these grid-parameter values in the same panel (after averaging over starts), e.g. `p=1,q=1` on a (p, q) grid. Both grid parameters are needed, and the grid must have exactly those values (else it stops and lists them). Diverging teal (negative), white (0), orange (positive) scale with symmetric limits; red and blue are kept for the goods. A black cross marks the reference point. Adds `_relpt-p1-q1` to the figure name. Not with `--relative-to` |
| `--hatch SCHEME:THRESHOLD` | scale schemes: hatch the grid cells where another scale scheme, averaged over starts (NaN ignored), is above THRESHOLD, e.g. `--hatch division_of_labour_index:0.5`: groups specialised in opposite goods. The cells get amber `//` lines with an amber outline along the cell edges, and the legend says "division of labour" (or `<scheme> > <threshold>` for a scheme without a short name). Works with `--relative-to` and `--relative-to-point`. Adds `_hatch-<scheme>` to the figure name |
| `--starts I [I ...]` | use only the runs from these starts (indices into the results' starts: 0 = (lo, lo), 1 = (lo, hi), 2 = (hi, lo), 3 = (hi, hi)) for everything: scales, labels, `--relative-to`, `--hatch`. E.g. `--starts 0 3` keeps the two balanced starts, so a lean comes from selection, not from the start. Adds `_starts-0-3` to the figure names; the simulation is unchanged |

## run_trajectories.py

How the population evolves over time for a few (p, q) conditions, in the style of DHK's Fig. 1 time panels. Makes three figures.

```
python run_trajectories.py [--start X1,X2] [--total shared|strips] [--plane-starts listed|all] [--pq-diff D] [--end-time T]
```

- Quadratic model with the flat benefit, `tol = 1e-5`, DHK's ESS point B converted: b1' = 7/4.6, b2' = 1.5.
- Runs are recorded every 20 steps (0.2 time units) through `engine.simulate`'s `record` argument.
- Each run is held at its final state up to a common end time, `--end-time` (default: the longest run's finishing time), so all lines reach the end.
- Lean is (x1 − x2)/(x1 + x2), from −1 (all good 2) to +1 (all good 1). Lines are coloured by it: red, purple at 0, blue. Figures A and B have a colour bar for it. The figures have no overall title.
- Line width is 0.6 + 2.4 × share pt, so the branches after a split are thinner. Each branch forks from its parent.
- A black diamond marks every split, where the parent was. A split is a group appearing in an empty slot; a saddle push is not marked.
- The division-of-labour panels have a third title line, `p < q < (p+1)/2`, the part of the rule they satisfy.

**Conditions (all figures):** (1, 1) arithmetic mean, (1 − D, 1) benefit favours balance, (1 − D, 1 + D) both favour balance, (1, 1 − D) cost favours concentration, (1 + D, 1 − D) both favour concentration, (1 − 2D, 1 − 1.6D) division of labour (inside p < q < (p+1)/2 for any D). D is `--pq-diff`, default 0.05: (0.95, 1), (0.95, 1.05), (1, 0.95), (1.05, 0.95), (0.9, 0.92). The p and q values are in the file names.

**Figure A, `trajectories_time_<tag>.png`:** one run per condition, all from `--start` (default 0.155, 0.005: x1 − x2 = 0.15, total 0.16).
- Every line starts at time 0 at its initial values. The time axis is linear from 0 to 1 and logarithmic after (symlog).
- Per column, each group's difference x1 − x2, from −0.3 to 0.3, with a dashed line at 0, time going up. The lines are coloured by lean. At p = q = 1 the difference stays fixed while the total grows, so the lean (difference / total) shrinks; that's why the axis shows the difference.
- `--total shared` (default) adds one panel below all the columns, with every condition's total x1 + x2 (share-weighted) overlaid in its own line style. Its y-axis starts just below the initial total.
- `--total strips` instead puts a narrow grey total panel beside each difference panel.

**Figure C, `trajectories_goods_<tag>.png`:** the same runs as figure A, one panel per condition.
- Each group's good-1 investment x1 (blue) and good-2 investment x2 (red), from 0 to 0.3, against the same time axis as figure A (time going up, from 0).
- Both lines are translucent (alpha 0.55), so where they overlap they blend to purple. In division of labour, the purple lines are one group's good 1 overlapping the other group's good 2.
- Diamonds mark the splits; the legend shows good 1, good 2, overlap and split.

**Figure B, `trajectories_plane_<tag>.png`:** the same six conditions in two rows of three, each with its own starts, all with total 0.16.
- Starts, all with total 0.16: towards good 1 (0.11, 0.05), balanced (0.08, 0.08), towards good 2 (0.05, 0.11). The balanced start is exactly on the diagonal; the engine's saddle push and split test take it off.
- `--plane-starts listed` (default): arithmetic mean and division of labour use all three starts, the generalist and specialist panels only the two offset ones. `--plane-starts all`: all three starts in every panel (adds `_starts-all` to the file names).
- Each panel shows x1 across and x2 up, from 0 to 0.3, with a faint diagonal x1 = x2.
- Paths have arrowheads. Each start is a hollow circle, and each group's end is a dot sized by share; both are in the legend with the split diamond. The colour bar for lean is on the right.

`conditions` and the starts are at the top of the script. The recorded data are saved next to each figure as `trajectories_time_<tag>.npz` and `trajectories_plane_<tag>.npz`. These hold `times_<c>_<s>`, `xs_<c>_<s>` and `fs_<c>_<s>` per condition c and start s, with the held last frame. The meta includes the starts and finishing times.

## One-good scripts

The scripts in `single_good/` are small and self-contained; run them directly:

| Script | What it does |
|---|---|
| `single_good_analytic_phase.py` | analytic phase diagram and average trait (`FLAT_BENEFIT` and `x_max` at the top; flat benefit needs `x_max <= 0.5`) |
| `single_good_integration_phase.py` | vectorised simulation of the two-group dynamics over a (b1', b2') grid |
| `single_good_multiple_pop.py` | loop version: a population that can split into two groups |
| `single_good_single_pop.py` | one monomorphic population following the selection gradient |

They show their figures with `plt.show()`; on Colab put `%matplotlib inline` before `%run`.

## Adding things

| To add | Where | How |
|---|---|---|
| a model | `coop/models.py` | write a function returning a `Model` (payoff, parameter names, checks, default sweep and settings) and add it to `MODELS`; options are its keyword arguments, listed in `OPTIONS` to get a command-line flag |
| a sweep | `coop/sweeps.py` | add an entry to `SWEEPS`: two panel parameters with values, two grid parameters with an `Axis`, and which panel parameter runs down the figure |
| a labelling scheme | `coop/schemes.py` | write a function `(xs, fs, r)` returning a name or number per run and add it to `SCHEMES` |
| reference lines or a prediction | `coop/references.py` | add a branch for the model and grid to `lines` (and `compare`) |

## Layout

```
run_simulation.py       simulation entry point
run_analysis.py         analysis entry point
run_trajectories.py     a few runs over time (DHK Fig. 1 style)
coop/
  models.py             payoff models and their checks
  sweeps.py             parameter sweeps
  engine.py             the adaptive dynamics (Settings, simulate)
  results.py            saving and loading results files
  schemes.py            labelling schemes
  references.py         theory: x*, reference lines, predictions
  plots.py              figures
single_good/            the one-good scripts
archive/                old scripts kept for reference
```
