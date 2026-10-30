# Runs a simulation: a payoff model (coop/models.py) over a parameter sweep (coop/sweeps.py) with
# the engine (coop/engine.py), and saves the final state of every run to results_<tag>.npz for
# run_analysis.py. If that file already exists it is loaded instead. See USAGE.md for what to run.
#
# Settings come from, in increasing priority: the engine defaults (Settings in coop/engine.py), the
# model's, the sweep's, and the command line.
#
# Usage:  python run_simulation.py --model quadratic --n 10               (quick test)
#         python run_simulation.py --model quadratic --flat-benefit
#         python run_simulation.py --model exp                            (sweep b_beta, the model's default)
#         python run_simulation.py --model exp --sweep p_q --n 20
#         python run_simulation.py --model quadratic --benefit individual_sum --hessian silas \
#                                  --split-mode forced_once --stop fixed_time --start-jitter
#         in Colab: %run run_simulation.py --model quadratic --n 10

import argparse
import json
import os
import time
from dataclasses import asdict, replace
import numpy as np
import jax

from coop import engine, models, results, sweeps

GRID_N = 100                         # grid size per axis


def settings_for(model, sweep, **changes):
    # engine defaults, then the model's settings, the sweep's, and finally changes
    return replace(engine.Settings(**{**model.settings, **sweep.settings}), **changes)


def make_tag(model, sweep, s, n, base):
    # model (with its options), sweep, grid size, x_max, and every setting changed from the command line
    short = lambda k, v: k if v is True else f"{k}{v:g}" if isinstance(v, float) else f"{k}-{v}"
    changed = [short(k, v) for k, v in asdict(s).items() if k != "x_max" and v != getattr(base, k)]
    return "_".join([model.tag, sweep.name, f"n{n}", f"xmax{s.x_max:g}"] + changed)


def load_or_simulate(model, sweep, s, n, tag):
    name = results.filename(tag)
    meta = dict(tag=tag, model=model.name, model_options=model.options, labels=model.labels,
                **sweeps.describe(sweep, n), n=n, starts=s.starts, x_max=s.x_max, x_min=s.x_min,
                merge_gap=s.merge_gap, max_steps=s.max_steps, settings={k: str(v) for k, v in asdict(s).items()})
    if os.path.exists(name):
        old = results.read_meta(name)
        if {k: v for k, v in old.items() if k != "seconds"} != json.loads(json.dumps(meta)):
            raise SystemExit(f"{name} exists but was made with different settings (e.g. edited sweep values): "
                             f"delete or rename it to re-run")
        print(f"{name} already exists: loading it instead of simulating (delete it to re-run)")
        with np.load(name) as f:
            return {k: f[k] for k in f.files if k != "meta"}, name
    print(f"simulating {np.prod(sweeps.layout(sweep, n, len(s.starts))[0])} runs ({tag}) ...")
    t = time.time()
    arrays = engine.simulate(model, sweep, s, n)
    meta["seconds"] = time.time() - t
    print(f"simulation took {meta['seconds']:.0f}s")
    results.save(name, arrays, meta)
    return arrays, name


def print_summary(arrays, model, sweep, s, n):
    status, steps = arrays["status"], arrays["steps"]
    print(f"\nruns finished: {int((status == 1).sum())} of {status.size}; "
          f"stopped at max_steps = {s.max_steps}: {int((status == 2).sum())}")
    if s.stop == "fixed_time":
        print("  (stop fixed_time: a run hits max_steps if its time step shrank so far that its stages didn't finish)")
    print(f"steps used: median {int(np.median(steps))}, 90% {int(np.percentile(steps, 90))}, "
          f"99% {int(np.percentile(steps, 99))}, max {int(steps.max())}")
    print("runs that needed more slots: " + ", ".join(f"K={K}: {int((arrays['max_k'] >= K).sum())}" for K in s.k_levels)
          + f"; wanted more than {s.k_levels[-1]}: {int(arrays['overflow'].sum())}")
    print(f"runs pushed off a saddle at least once: {int((arrays['pushes'] > 0).sum())}")
    (a, a_values), (b, b_values) = sweep.panels
    for i, va in enumerate(a_values):
        for j, vb in enumerate(b_values):
            capped, pushed = status[:, i, j] == 2, arrays["pushes"][:, i, j] > 0
            if capped.any() or pushed.any():
                print(f"  {a} = {va:g}, {b} = {vb:g}: {int(capped.sum())} runs not converged, "
                      f"{int(pushed.sum())} pushed off a saddle")
    print(f"runs with a group at x_max and a selection gradient of exactly 0 there "
          f"(a flat payoff, e.g. past a flattened peak): {zero_gradient_at_x_max(arrays, model, sweep, s, n)}")


def zero_gradient_at_x_max(arrays, model, sweep, s, n):
    # number of runs with a group at x_max in a good whose selection gradient in that good is exactly 0
    xs, fs = arrays["xs"], arrays["fs"]
    K = fs.shape[-1]
    _, _, values = sweeps.layout(sweep, n, len(s.starts))
    params = np.stack([values[name] for name in model.params], axis=1).astype(xs.dtype)
    grads = engine.selection_gradients(model, params, xs.reshape(-1, K, 2), fs.reshape(-1, K))
    alive = fs.reshape(-1, K, 1) > 0
    corner = alive & (xs.reshape(-1, K, 2) >= s.x_max) & (grads == 0)
    return int(corner.any(axis=(1, 2)).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(models.MODELS), default="quadratic")
    parser.add_argument("--sweep", choices=list(sweeps.SWEEPS), help="default: the model's own")
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    model_options = parser.add_argument_group("model options")
    for name, (choices, help) in models.OPTIONS.items():
        flag = "--" + name.replace("_", "-")
        if choices == [False, True]:
            model_options.add_argument(flag, action="store_true", default=None, help=help)
        else:
            model_options.add_argument(flag, choices=choices, help=help + f" (default {choices[0]})")
    engine_options = parser.add_argument_group("engine options and settings (default: the model's / sweep's)")
    engine_options.add_argument("--hessian", choices=["exact", "silas"])
    engine_options.add_argument("--split-mode", choices=["test_repeat", "forced_once"])
    engine_options.add_argument("--stop", choices=["converged", "fixed_time"])
    engine_options.add_argument("--fixed-time", type=float, help="stage length for --stop fixed_time")
    engine_options.add_argument("--start-jitter", action="store_true", default=None)
    engine_options.add_argument("--jitter-seed", type=int)
    for name in ["x_max", "x_min", "max_dt", "max_steps", "tol"]:
        engine_options.add_argument("--" + name.replace("_", "-"), type=int if name == "max_steps" else float)
    engine_options.add_argument("--x64", action="store_true", default=None, help="64-bit floats")
    args = vars(parser.parse_args())

    # 1. the model, the sweep and the settings
    model = models.make_model(args["model"], **{k: args[k] for k in models.OPTIONS if args[k] is not None})
    sweep = sweeps.SWEEPS[args["sweep"] or model.default_sweep]
    if sweep.params != set(model.params):
        raise SystemExit(f"sweep {sweep.name!r} varies {sorted(sweep.params)}, but model {model.name!r} has "
                         f"{sorted(model.params)}; sweeps for it: "
                         + ", ".join(k for k, v in sweeps.SWEEPS.items() if v.params == set(model.params)))
    base = settings_for(model, sweep)
    s = settings_for(model, sweep, **{k: v for k, v in args.items() if k in asdict(base) and v is not None})
    jax.config.update("jax_enable_x64", s.x64)
    tag = make_tag(model, sweep, s, args["n"], base)

    # 2. checks
    print("running on:", jax.devices())
    print("payoff checks:"); models.check_payoff(model)
    if s.hessian == "silas":
        if model.chain_hessian is None:
            raise SystemExit(f"--hessian silas needs a chain-rule Hessian, which {model.tag} doesn't have")
        print("Hessian checks:"); models.check_chain_hessian(model)

    # 3. simulate (or load), and summarise
    arrays, name = load_or_simulate(model, sweep, s, args["n"], tag)
    print_summary(arrays, model, sweep, s, args["n"])
    print(f"\nnext: python run_analysis.py {name}")
