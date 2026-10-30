# Parameter sweeps: which of a model's parameters vary, and over what. A sweep gives one figure
# panel per pair of values of two panel parameters, and within each panel a grid over two more:
# results arrays are laid out (start, panel a, panel b, grid row, grid column, ...).
#
# To add a sweep: add an entry to SWEEPS. Its four parameters must be the model's parameters.

from dataclasses import dataclass, field, replace
import numpy as np


@dataclass(frozen=True)
class Axis:
    # n evenly spaced values from lo to hi
    lo: float
    hi: float
    skip_lo: bool = False          # True: start one step above lo (e.g. beta = 0 would divide by zero)

    def values(self, n):
        return np.linspace(self.lo + (self.hi - self.lo) / n if self.skip_lo else self.lo, self.hi, n)


@dataclass(frozen=True)
class Sweep:
    name: str
    panels: tuple                  # ((parameter, values), (parameter, values)): panel a, panel b
    grid: tuple                    # ((parameter, Axis), (parameter, Axis)): grid rows (y), grid columns (x)
    panel_rows: str                # the panel parameter that runs down the figure (the other runs across)
    settings: dict = field(default_factory=dict)      # overrides of the engine Settings (after the model's)

    @property
    def params(self):
        return {name for name, _ in self.panels + self.grid}


SWEEPS = {
    # quadratic model: (b1', b2') grid, a panel per (p, q)
    "b1_b2":  Sweep("b1_b2", panels=(("p", [0.8, 1.0, 1.25]), ("q", [0.8, 1.0, 1.25])),
                    grid=(("b2", Axis(0.0, 4.0)), ("b1", Axis(0.0, 4.0))), panel_rows="q"),
    # exponential model: (b', beta') grid, a panel per (p, q); beta' stops below 2 because the
    # singular point grows very large as beta' -> 2
    "b_beta": Sweep("b_beta", panels=(("p", [0.8, 1.0, 1.25]), ("q", [0.8, 1.0, 1.25])),
                    grid=(("beta", Axis(0.0, 1.7, skip_lo=True)), ("b", Axis(0.0, 6.0))), panel_rows="q"),
    # exponential model: (p, q) grid, a panel per (b', beta'), to test the division-of-labour rule
    # (beta' < 1 and b' > beta': q < p -> DC; p < q < (p + 1)/2 -> DC and DC'; q > (p + 1)/2 -> CC).
    # x_min = 1e-3 caps the very steep slopes of the power mean near zero investment when p, q < 1
    # (they caused pixel-to-pixel noise); avoid p, q near 0, where the mean's exponent 1/p blows up.
    "p_q":    Sweep("p_q", panels=(("b", [2.0]), ("beta", [0.5, 0.75, 0.9])),
                    grid=(("q", Axis(0.3, 1.5)), ("p", Axis(0.3, 1.5))), panel_rows="b",
                    settings=dict(x_min=1e-3)),
    # quadratic model: (p, q) grid, a panel per (b1', b2'), to test the division-of-labour rule, which
    # needs b1' > 1 and b2' above both 1 and b1'/2
    "p_q_quadratic": Sweep("p_q_quadratic", panels=(("b1", [2.0]), ("b2", [1.2, 1.5, 2.0])),
                           grid=(("q", Axis(0.3, 1.5)), ("p", Axis(0.3, 1.5))), panel_rows="b1",
                           settings=dict(x_min=1e-3)),
}


def with_values(sweep, values):
    # the sweep with new values for some of its panel parameters, e.g. {"b2": [0.6, 0.8]}
    panels = tuple((name, values.pop(name, vs)) for name, vs in sweep.panels)
    if values:
        raise SystemExit(f"--values: {', '.join(values)} not a panel parameter of sweep {sweep.name!r} "
                         f"(its panel parameters: {', '.join(name for name, _ in sweep.panels)})")
    return replace(sweep, panels=panels)


def changed_values(sweep):
    # the panel parameters whose values differ from the sweep's defaults in SWEEPS, as "b2-0.6-0.8" for tags
    default = dict(SWEEPS[sweep.name].panels)
    return [name + "-" + "-".join(f"{v:g}" for v in vs) for name, vs in sweep.panels if list(vs) != list(default[name])]


def layout(sweep, n, n_starts):
    # every run's start and parameter values, flattened from the results' array order
    # (start, panel a, panel b, grid row, grid column); returns that shape, the start and panel
    # indices of each run, and {parameter: value of each run}
    (a, a_values), (b, b_values) = sweep.panels
    (y, y_axis), (x, x_axis) = [(name, axis.values(n)) for name, axis in sweep.grid]
    I, Ia, Ib, Y, X = np.meshgrid(np.arange(n_starts), np.arange(len(a_values)), np.arange(len(b_values)),
                                  y_axis, x_axis, indexing="ij")
    values = {a: np.array(a_values)[Ia], b: np.array(b_values)[Ib], y: Y, x: X}
    return I.shape, (I.ravel(), Ia.ravel(), Ib.ravel()), {k: v.ravel() for k, v in values.items()}


def describe(sweep, n):
    # everything the analysis needs to know about the sweep, for the results file
    (a, a_values), (b, b_values) = sweep.panels
    return dict(sweep=sweep.name, panel_names=[a, b], panel_values=[list(a_values), list(b_values)],
                grid_names=[name for name, _ in sweep.grid],
                grid_axes=[axis.values(n).tolist() for _, axis in sweep.grid], panel_rows=sweep.panel_rows)
