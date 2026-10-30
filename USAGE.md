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
| Each group's lean along one column of the grid | any (made for `p_q`) | `--cut 0.7` | `cut_p0.7_<tag>.png`: one row per start |
| Compare with the reference implementation (ablation) | the three runs under [Comparing runs](#comparing-runs) | `--compare` | `compare_<scheme>_<tagA>_vs_<tagB>.png` and the differences printed |

Add `--n 10` (or `--n 20` for `p_q`) to any simulation for a quick test; the default is a 100 × 100 grid.

## run_simulation.py

```
python run_simulation.py --model MODEL [--sweep SWEEP] [--n N] [model options] [engine options]
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

`--values NAME=V1,V2,...` replaces a sweep's panel values (repeat it for the other panel parameter),
e.g. `--sweep p_q_quadratic --values b1=1.2,1.6 --values b2=0.7`. The new values go into the file
name (`..._p_q_quadratic_b1-1.2-1.6_b2-0.7_n100_...`), so runs with different values never overwrite
each other. Grid ranges are changed in `coop/sweeps.py`.

Each run starts from one of four corners, `(lo, lo)`, `(lo, hi)`, `(hi, lo)`, `(hi, hi)`, so every
grid point is run four times.

### Model options (quadratic only)

| Flag | Effect |
|---|---|
| `--flat-benefit` | the benefit rises as before, then stays at its maximum past its peak `Mp = b1'/(2 b2')` instead of falling |
| `--flat-cost` | the cost stays at its maximum past `Mq = 1/2`; no effect while `x_max <= 0.5` |
| `--benefit individual_sum` | `Mp = M_p(y) + M_p(x)` instead of the pair's totals (as in the reference implementation) |

### Engine options

These change how the dynamics are run; the defaults are the standard method. The non-default
choices reproduce a reference implementation, for comparison.

| Flag | Choices (default first) | Effect |
|---|---|---|
| `--hessian` | `exact`, `silas` | branching test with `jax.hessian`, or the reference's chain-rule Hessian (quadratic without flat options only) |
| `--split-mode` | `test_repeat`, `forced_once` | split groups whose payoff curves upwards, repeatedly; or split every group once after the first stage and stop after the second |
| `--stop` | `converged`, `fixed_time` | a stage ends when the run has settled, or after exactly `--fixed-time` time units (default 100) |
| `--start-jitter` | off, on | one random offset per (start, panel, good), sd 0.01; `--jitter-seed` sets the seed (default 0) |

### Settings

`--x-max`, `--x-min`, `--max-dt`, `--max-steps`, `--tol` and `--x64` override the settings for one
run. Everything else (time step, chunk size, batch sizes, split and merge distances, …) is in
`Settings` in `coop/engine.py`. Priority, lowest first: `Settings` defaults → the model's defaults →
the sweep's → the command line.

### What it prints and saves

- payoff checks for the model (and Hessian checks with `--hessian silas`)
- a summary: runs finished and capped at `max_steps`, steps used, runs that needed more group slots or
  were pushed off a saddle (per panel), and groups stuck at `x_max` with an exactly zero gradient
- `results_<tag>.npz`: the final traits and shares of every run plus everything needed to read them.
  The tag is the model (and its options), the sweep, `n`, `x_max`, and any setting changed on the
  command line, e.g. `results_quadratic_flat_b1_b2_n100_xmax0.5.npz`. If the file already exists it
  is loaded instead of re-run (delete it to re-run); if it was made with different settings (e.g.
  after editing the sweep values) the run stops and asks you to delete or rename it.

## run_analysis.py

```
python run_analysis.py [RESULTS.npz] [--scheme S ...] [--invested RULE] [--cut VALUE] [--compare OTHER.npz]
```

With no file it uses the newest `results_*.npz` in the folder.

### Schemes

| Scheme | Kind | What it gives |
|---|---|---|
| `cc_dc_dd` | categories | each group is CC (invests in both goods), DC (one) or DD (neither); a run's outcome is its distinct group types (e.g. `CC and DD`), a grid point's label the distinct outcomes over its four starts (e.g. `CC \|or\| DD`) |
| `cc_dc_dd_division` | categories | the same, but DC groups investing in different goods give `DC and DC'` (division of labour) |
| `mean_investment` | scale | share-weighted mean investment over groups and goods, divided by `x_max` (quadratic), the panel's `x*` (exp with b and beta per panel, scale up to 2) or the largest `x*` on the grid (exp otherwise) |
| `division_of_labour_index` | scale | 0 to 1: how lopsided individuals are minus how lopsided the population is; 1 = groups specialise in opposite goods |

Default: `cc_dc_dd_division` on a (p, q) grid (`p_q`, `p_q_quadratic`), `cc_dc_dd` otherwise. Several at once: `--scheme cc_dc_dd mean_investment`.

A category scheme gives two figures, `<scheme>_<tag>_unfiltered.png` (what happened) and
`..._filtered.png` (labels on fewer than 0.3% of grid points replaced by their nearest neighbour),
and prints label counts per panel and a check against theory:

| Model and sweep | Check printed |
|---|---|
| quadratic, `b1_b2` | the p = q = 1 panel against the one-good analytic diagram |
| exp, `b_beta` | the p = q = 1 panel against the one-good exponential diagram |
| exp, `p_q` | each panel against the division-of-labour rule, where the one-good `x*` is an ESS (beta' < 1, b' > beta') |
| quadratic, `p_q_quadratic` | the same, for b1' > 1 and b2' above both 1 and b1'/2 |

A scale scheme gives one figure, `<scheme>_<tag>.png`, averaged over starts.

### Other options

| Flag | Effect |
|---|---|
| `--invested half_cap` / `absolute` | when a good counts as invested: above `x_max / 2` (quadratic default) or above 0.05 (exp default); adds `_invested-<rule>` to the figure names when not the default |
| `--cut VALUE` | for the grid column nearest VALUE (e.g. p = 0.7 on the `p_q` sweep): every group as a dot at (grid row value, lean), lean = (x1 − x2)/(x1 + x2), area ~ share, one row of panels per start. Runs no scheme unless `--scheme` is given |
| `--compare OTHER.npz` | the two files' unfiltered labels side by side with one legend, and where they differ: in total, per panel, and the 5 most common changes. Both files need the same sweep, `n` and starts |

### Comparing runs

The ablation against the reference implementation (run at `x_max = 2`, as in the reference):

```
python run_simulation.py --model quadratic --x-max 2
python run_simulation.py --model quadratic --x-max 2 --benefit individual_sum
python run_simulation.py --model quadratic --x-max 2 --benefit individual_sum --hessian silas \
                         --split-mode forced_once --stop fixed_time --start-jitter

python run_analysis.py results_quadratic_b1_b2_n100_xmax2.npz \
       --compare results_quadratic_benefit-individual_sum_b1_b2_n100_xmax2.npz
python run_analysis.py results_quadratic_b1_b2_n100_xmax2.npz \
       --compare results_quadratic_benefit-individual_sum_b1_b2_n100_xmax2_hessian-silas_split_mode-forced_once_stop-fixed_time_start_jitter.npz
```

The same works for any two runs on the same grid, e.g. with and without `--flat-benefit`.

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
| reference lines or a prediction | `coop/references.py` | add a branch for the model and grid to `lines` (and `compare`, `cut_lines`) |

## Layout

```
run_simulation.py       simulation entry point
run_analysis.py         analysis entry point
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
