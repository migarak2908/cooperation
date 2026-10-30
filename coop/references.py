# What theory says, for drawing on the plots and checking the simulations against: the one-good
# singular point x*, reference lines, and analytic predictions. Which apply depends on the model
# and on which parameters are on the grid; each function says what it covers and returns nothing
# (no lines, no comparison) for other combinations. Needs numpy only (no jax).
#
# To add references for a new model or sweep: add a branch to lines (and cut_lines, compare) below.

import numpy as np

from coop.results import panel_index, panel_value

SHADE_VALID_BAND = False             # quadratic: grey out grid cells outside the band between the two lines
FLAT_REGION = (0.5, 1.0)             # quadratic with flat benefit: branching is permissible for b2' in this range


def grid(r):
    # the (grid row, grid column) parameter names, e.g. ("b2", "b1")
    return tuple(r["grid_names"])


# ---------- the one-good singular point ----------

def x_star(r):
    # one-good singular point at every (panel a, panel b, grid row, grid column); NaN where none
    P = r["params"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if r["model"] == "exp":
            # x* = ln(b / beta) / (2 / beta - 1), for b > beta and beta < 2
            b, beta = P["b"], P["beta"]
            return np.where((b > beta) & (beta < 2), np.log(b / beta) / (2 / beta - 1), np.nan)
        return (P["b1"] - 1) / (4 * P["b2"] - 2)        # quadratic: x* = (b1' - 1) / (4 b2' - 2)


def investment_scale(r):
    # what the mean investment is divided by, its name, and the top of the colour scale:
    #   quadratic:                        x_max
    #   exp, b and beta fixed per panel:  that panel's x* (up to 2 x*)
    #   exp, b or beta on the grid:       the largest x* on the grid (x_max is just a wall)
    if r["model"] != "exp":
        return r["x_max"], "x_max", 1
    if {"b", "beta"} <= set(r["panel_names"]):
        return x_star(r), "x*", 2
    return float(np.nanmax(x_star(r))), "max x*", 1


# ---------- reference lines ----------
# a line is dict(x=..., y=..., style=..., label=...), optionally colour and width

def lines(r):
    y_axis, x_axis = r["grid_axes"]
    if grid(r) == ("q", "p"):
        # the division-of-labour rule's boundaries
        p = x_axis
        return [dict(x=p, y=p, style="--", label="q = p"),
                dict(x=p, y=(p + 1) / 2, style=":", label="q = (p + 1)/2"),
                dict(x=[1.0, 1.0], y=[y_axis[0], y_axis[-1]], style="-", colour="grey", width=0.6, label="p = 1")]
    if r["model"] == "quadratic" and grid(r) == ("b2", "b1"):
        b = x_axis
        if r["model_options"].get("flat_benefit"):
            # flat benefit: the region where branching is permissible
            return [dict(x=[b[0], b[-1]], y=[level, level], style="--",
                         label="branching permissible region (between)" if k == 0 else None)
                    for k, level in enumerate(FLAT_REGION)]
        upper, lower = band_lines(b, r["x_max"])
        return [dict(x=b, y=upper, style="--", label="benefit increasing up to cap (below)"),
                dict(x=b, y=lower, style=":", label="x* inside [0, x_max] (above)")]
    if r["model"] == "exp" and grid(r) == ("beta", "b"):
        b, beta = x_axis, y_axis
        out = [dict(x=b, y=b, style="--", label="first unit of investment pays (below)")]
        for level, style, label in [(1, ":", "x* is a valley above, a peak below"), (2, "-.", "x* attracts below")]:
            if beta[0] <= level <= beta[-1]:
                out.append(dict(x=[b[0], b[-1]], y=[level, level], style=style, label=label))
        return out
    return []


def band_lines(b, x_max):
    # quadratic one-good reference lines, from B(z) = b1' z - b2' z^2 and x* = (b1' - 1) / (4 b2' - 2):
    #   upper: benefit still increasing at the largest pair total 2 x_max (valid below it)
    #   lower: x* inside [0, x_max] (valid above it)
    # with x_max = 0.5 both are b2' = b1'/2
    return b / (4 * x_max), (b - 1 + 2 * x_max) / (4 * x_max)


def shade(r):
    # with SHADE_VALID_BAND, the quadratic grid cells outside the band between the two lines (or None)
    if not (SHADE_VALID_BAND and r["model"] == "quadratic" and grid(r) == ("b2", "b1")
            and not r["model_options"].get("flat_benefit")):
        return None
    y_axis, x_axis = r["grid_axes"]
    B1, B2 = np.meshgrid(x_axis, y_axis)
    upper, lower = band_lines(B1, r["x_max"])
    return ~((B2 >= lower) & (B2 <= upper))


def cut_lines(r, x_value):
    # vertical lines for a cut at grid column value x_value, along the grid rows: (position, style, label)
    if grid(r) == ("q", "p"):
        return [(x_value, "--", "q = p"), ((x_value + 1) / 2, ":", "q = (p + 1)/2")]
    return []


# ---------- analytic predictions, compared with the simulated grid labels ----------

def compare(labels, r, division=False):
    # labels: unfiltered grid labels (panel a, panel b, grid row, grid column)
    if grid(r) == ("q", "p") and r["model"] == "exp":
        compare_division_rule(labels, r, division)
    elif set(r["panel_names"]) == {"p", "q"}:
        if panel_index(r, p=1.0, q=1.0) is None:
            print("p = q = 1 check skipped: p and q need to include 1")
        elif r["model"] == "exp":
            compare_one_good_exp(labels, r)
        else:
            compare_one_good(labels, r)
    else:
        print("no analytic prediction for this model and sweep")


def compare_division(labels, r):
    return compare(labels, r, division=True)


def compare_one_good(labels, r):
    # quadratic, p = q = 1: reduces to the one-good model; compare with its analytic diagram
    y_axis, x_axis = r["grid_axes"]
    B1, B2 = np.meshgrid(x_axis, y_axis)
    x_max = r["x_max"]
    D0, Dmax = B1 - 1, (B1 - 1) + (2 - 4 * B2) * x_max
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = D0 / (4 * B2 - 2)
    analytic = np.select([(D0 > 0) & (Dmax > 0), (D0 < 0) & (Dmax < 0), (D0 < 0) & (Dmax > 0),
                          (D0 > 0) & (Dmax < 0) & (B2 < 1), (D0 > 0) & (Dmax < 0) & (B2 > 1)],
                         ["CC", "DD", "CC |or| DD", "CC and DD", np.where(xs > x_max / 2, "CC", "DD")],
                         default="boundary")
    print_comparison(analytic, labels, r)


def compare_one_good_exp(labels, r):
    # exponential, p = q = 1 (one group per run, b = b', beta = beta'):
    #   b > beta, beta < 1:      ESS at x*;  b > beta, 1 < beta < 2: branching;  b > beta, beta > 2: up to the wall
    #   b < beta, beta < 2:      no investment;  b < beta, beta > 2: bistable
    from coop.schemes import invested_threshold
    y_axis, x_axis = r["grid_axes"]
    B, BETA = np.meshgrid(x_axis, y_axis)
    i, j = panel_index(r, p=1.0, q=1.0)
    ess = np.where(x_star(r)[i, j] > invested_threshold(r), "CC", "DD")
    analytic = np.select([(B > BETA) & (BETA < 1), (B > BETA) & (BETA > 1) & (BETA < 2), (B > BETA) & (BETA > 2),
                          (B < BETA) & (BETA < 2), (B < BETA) & (BETA > 2)],
                         [ess, "CC and DD", "CC", "DD", "CC |or| DD"], default="boundary")
    print_comparison(analytic, labels, r)


def print_comparison(analytic, labels, r):
    # simulated p = q = 1 labels vs an analytic diagram, leaving out points next to a region boundary
    pad = np.pad(analytic, 1, mode="edge")
    boundary = analytic == "boundary"
    rows, cols = analytic.shape
    for di, dj in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        boundary |= pad[di:di + rows, dj:dj + cols] != analytic
    sim = labels[panel_index(r, p=1.0, q=1.0)]
    differ = (sim != analytic) & ~boundary
    print(f"p = q = 1 vs one-good analytic, away from boundaries: {differ.sum()} of {(~boundary).sum()} points differ")
    for a, s in sorted(set(zip(analytic[differ], sim[differ]))):
        print(f"  analytic {a!r:14s} simulated {s!r:28s} {int(((analytic == a) & (sim == s) & differ).sum()):4d} points")


def compare_division_rule(labels, r, division=False):
    # exponential, (p, q) grid: the division-of-labour prediction for each panel, away from its two
    # boundary lines:
    #   q < p -> DC;  p < q < (p + 1)/2 -> DC and DC' (DC without division);  q > (p + 1)/2 -> CC (DD if x* is tiny)
    from collections import Counter
    from coop.schemes import invested_threshold
    q_axis, p_axis = r["grid_axes"]
    P, Q = np.meshgrid(p_axis, q_axis)                                    # (q, p)
    cell = max(p_axis[1] - p_axis[0], q_axis[1] - q_axis[0])
    boundary = (np.abs(Q - P) < cell) | (np.abs(Q - (P + 1) / 2) < cell)
    stars = x_star(r)
    print("prediction check (unfiltered labels, points within one grid cell of q = p or q = (p + 1)/2 left out):")
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            b, beta = panel_value(r, "b", i, j), panel_value(r, "beta", i, j)
            if beta >= 1 or b <= beta:
                print(f"  b' = {b:g}, beta' = {beta:g}: skipped, the prediction needs beta < 1 and b > beta")
                continue
            generalists = np.where(stars[i, j] > invested_threshold(r), "CC", "DD")
            predicted = np.select([Q < P, (P < Q) & (Q < (P + 1) / 2), Q > (P + 1) / 2],
                                  ["DC", "DC and DC'" if division else "DC", generalists], default="boundary")
            sim = labels[i, j]
            used = ~boundary & (predicted != "boundary")
            differ = used & (sim != predicted)
            print(f"  b' = {b:g}, beta' = {beta:g}: {int((used & ~differ).sum())} agree, {int(differ.sum())} differ")
            for (a, s), c in Counter(zip(predicted[differ], sim[differ])).most_common(5):
                print(f"    predicted {str(a)!r:14s} simulated {s!r:28s} {c:4d} points")
