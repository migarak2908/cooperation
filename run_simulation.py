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
#         python run_simulation.py --model exp --sweep p_q --values beta=0.3,0.5    (other panel values)
#         python run_simulation.py --model quadratic --sweep p_q_quadratic --n 20
#         python run_simulation.py --model quadratic --sweep p_q_near_one --start 0.11,0.05   (one start)
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
    # model (with its options), sweep (with any panel values changed from the command line), grid size,
    # x_max, and every setting changed from the command line
    def short(k, v):
        if k == "start":                                # e.g. start-0.11-0.05
            return "start-" + "-".join(f"{x:g}" for xs in v for x in xs)
        return k if v is True else f"{k}{v:g}" if isinstance(v, float) else f"{k}-{v}"
    changed = [short(k, v) for k, v in asdict(s).items() if k != "x_max" and v != getattr(base, k)]
    return "_".join([model.tag, sweep.name] + sweeps.changed_values(sweep) + [f"n{n}", f"xmax{s.x_max:g}"] + changed)


def load_or_simulate(model, sweep, s, n, tag):
    name = results.filename(tag)
    meta = dict(tag=tag, model=model.name, model_options=model.options, labels=model.labels,
                **sweeps.describe(sweep, n), n=n, starts=s.starts, x_max=s.x_max, x_min=s.x_min,
                merge_gap=s.merge_gap, max_steps=s.max_steps, settings={k: str(v) for k, v in asdict(s).items()})
    if os.path.exists(name):
        old = results.read_meta(name)
        old.pop("seconds", None)
        # options and settings that no longer exist (files made before they were removed) are left out
        for key in ["model_options", "settings"]:
            old[key] = {k: v for k, v in old[key].items() if k in meta[key]}
        if old != json.loads(json.dumps(meta)):
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


def print_summary(arrays, sweep, s):
    status, steps = arrays["status"], arrays["steps"]
    print(f"\nruns finished: {int((status == 1).sum())} of {status.size}; "
          f"stopped at max_steps = {s.max_steps}: {int((status == 2).sum())}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(models.MODELS), default="quadratic")
    parser.add_argument("--sweep", choices=list(sweeps.SWEEPS), help="default: the model's own")
    parser.add_argument("--n", type=int, default=GRID_N, help="grid size per axis")
    parser.add_argument("--values", action="append", default=[], metavar="NAME=V1,V2,...",
                        help="panel values to use instead of the sweep's, e.g. --values b2=0.6,0.8 (repeatable)")
    model_options = parser.add_argument_group("model options")
    for name, help in models.OPTIONS.items():
        model_options.add_argument("--" + name.replace("_", "-"), action="store_true", help=help)
    engine_options = parser.add_argument_group("settings (default: the model's / sweep's)")
    for name in ["x_max", "x_min", "max_dt", "max_steps", "tol"]:
        engine_options.add_argument("--" + name.replace("_", "-"), type=int if name == "max_steps" else float)
    engine_options.add_argument("--x64", action="store_true", default=None, help="64-bit floats")
    engine_options.add_argument("--start", action="append", metavar="X1,X2",
                                help="start every grid point here instead of the four corners (repeatable)")
    args = vars(parser.parse_args())
    if args["start"]:
        args["start"] = tuple(tuple(float(v) for v in item.split(",")) for item in args["start"])

    # 1. the model, the sweep and the settings
    model = models.make_model(args["model"], **{k: True for k in models.OPTIONS if args[k]})
    sweep = sweeps.SWEEPS[args["sweep"] or model.default_sweep]
    values = {name: [float(v) for v in vs.split(",")] for name, vs in (item.split("=") for item in args["values"])}
    sweep = sweeps.with_values(sweep, values)
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

    # 3. simulate (or load), and summarise
    arrays, name = load_or_simulate(model, sweep, s, args["n"], tag)
    print_summary(arrays, sweep, s)
    print(f"\nnext: python run_analysis.py {name}")
