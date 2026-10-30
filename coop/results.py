# Results files: the final state of every run plus everything needed to read it, so the analysis
# needs nothing from the simulation code (and no jax). Arrays are laid out
# (start, panel a, panel b, grid row, grid column, ...); the rest is a JSON string under "meta".

import glob
import json
import os
import numpy as np

PREFIX = "results_"


def filename(tag):
    return f"{PREFIX}{tag}.npz"


def save(name, arrays, meta):
    np.savez(name, **arrays, meta=json.dumps(meta))
    print(f"saved {name}")


def read_meta(name):
    with np.load(name) as f:
        return json.loads(str(f["meta"]))


def newest():
    files = glob.glob(f"{PREFIX}*.npz")
    if not files:
        raise SystemExit(f"no {PREFIX}*.npz here: run run_simulation.py first")
    return max(files, key=os.path.getmtime)


def load(name=None):
    # a results file as a dict r: its arrays, its meta fields, and
    #   params: {parameter: value at each (panel a, panel b, grid row, grid column)}
    name = name or newest()
    with np.load(name) as f:
        r = {k: f[k] for k in f.files if k != "meta"}
        r.update(json.loads(str(f["meta"])))
    r["grid_axes"] = [np.array(axis) for axis in r["grid_axes"]]
    r["params"] = param_grid(r)
    print(f"loaded {name} ({r['model']} model, {r['sweep']} sweep)")
    not_converged = int((r["status"] == 2).sum())
    if not_converged:
        print(f"  note: {not_converged} of {r['status'].size} runs stopped at {r['max_steps']} steps "
              f"without converging")
    return r


def param_grid(r):
    (a, b), (y, x) = r["panel_names"], r["grid_names"]
    (a_values, b_values), (y_axis, x_axis) = r["panel_values"], r["grid_axes"]
    A, B, Y, X = np.meshgrid(a_values, b_values, y_axis, x_axis, indexing="ij")
    return {a: A, b: B, y: Y, x: X}


def panel_value(r, name, i, j):
    # the value of a panel parameter in panel (i, j)
    a, b = r["panel_names"]
    return r["panel_values"][0][i] if name == a else r["panel_values"][1][j]


def panel_index(r, **values):
    # (i, j) of the panel with these panel-parameter values, e.g. panel_index(r, p=1, q=1); None if absent
    (a, b), (a_values, b_values) = r["panel_names"], r["panel_values"]
    if set(values) != {a, b} or values[a] not in a_values or values[b] not in b_values:
        return None
    return a_values.index(values[a]), b_values.index(values[b])
