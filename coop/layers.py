# Three-layer outcome figure for the quadratic b1_b2 sweep, from the two balanced starts (0 and 3):
#   layer 1, level:          the share-weighted mean total m of each run is defection, in between or
#                            full; a grid point is that if both starts agree, bistable if one start is
#                            defection and the other full, start-dependent otherwise
#   layer 2, specialisation: the individual scheme, sum f |x1 - x2| / sum f t over counted groups,
#                            averaged over the two starts
#                            (shown at "in between" points only)
#   layer 3, branching:      both starts end with 2+ counted groups; how the groups differ (judged on
#                            start 0, "mixed" if the starts disagree), and whether either has 3+ groups
# With each counted group's share f, total t = x1 + x2 and lean L = (x1 - x2) / t, and top = 2 x_max.
# Needs numpy only (no jax).

import numpy as np

from coop.schemes import counted_groups, individual, state

EDGE = 0.01                          # defection: m < EDGE top; full: m > (1 - EDGE) top
SAME_TOTAL = 0.05                    # group totals within SAME_TOTAL top of each other are the same
LEAN_TOL = 0.02                      # a group leans to a good if |L| > LEAN_TOL

BALANCED_STARTS = (0, 3)
LEVELS = ["defection", "in between", "full", "bistable", "start-dependent"]
KINDS = ["none", "between goods", "in total", "mixed"]


def groups(xs, fs, r):
    # per run: counted shares f (0 for groups that don't count), totals t, leans L and counted, (..., K)
    xs, fs, counted = counted_groups(xs, fs, r)
    t = xs[..., 0] + xs[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lean = np.where(t > 0, (xs[..., 0] - xs[..., 1]) / t, 0.0)
    return np.where(counted, fs, 0.0), t, lean, counted


def run_level(xs, fs, r, edge=EDGE):
    # per run: "defection", "in between" or "full", from the share-weighted mean total
    f, t, _, _ = groups(xs, fs, r)
    m = np.sum(f * t, axis=-1) / np.sum(f, axis=-1)
    top = 2 * r["x_max"]
    return np.select([m < edge * top, m > (1 - edge) * top], ["defection", "full"], "in between").astype(object)


def run_kind(xs, fs, r, same_total=SAME_TOTAL, lean_tol=LEAN_TOL):
    # per run: how its groups differ ("none" for fewer than 2 groups), and the number of groups
    _, t, lean, counted = groups(xs, fs, r)
    n = counted.sum(axis=-1)
    spread = np.max(np.where(counted, t, -np.inf), axis=-1) - np.min(np.where(counted, t, np.inf), axis=-1)
    same = spread <= same_total * 2 * r["x_max"]
    opposite = np.any(counted & (lean > lean_tol), axis=-1) & np.any(counted & (lean < -lean_tol), axis=-1)
    kind = np.select([n < 2, same & opposite, ~same & ~opposite], ["none", "between goods", "in total"], "mixed")
    return kind.astype(object), n


def point_level(a, b):
    # a grid point's level from its two starts' levels
    bistable = {"defection", "full"}
    return np.array([str(x) if x == y else "bistable" if {x, y} == bistable else "start-dependent"
                     for x, y in zip(np.ravel(a), np.ravel(b))], dtype=object).reshape(np.shape(a))


def point_kind(a, b):
    # a grid point's kind from its two starts' kinds: none unless both branched, start 0's if they agree
    return np.where((a == "none") | (b == "none"), "none", np.where(a == b, a, "mixed")).astype(object)


def check_balanced_starts(r):
    starts = r["starts"]
    if len(starts) <= max(BALANCED_STARTS) or any(starts[s][0] != starts[s][1] for s in BALANCED_STARTS):
        raise SystemExit("--layers needs the two balanced starts at indices 0 and 3, (lo, lo) and (hi, hi); "
                         "this file has " + ", ".join(f"{s} = ({a:g}, {b:g})" for s, (a, b) in enumerate(starts)))


def layers(r, edge=EDGE, same_total=SAME_TOTAL, lean_tol=LEAN_TOL):
    # all three layers per grid point (panel a, panel b, row, column), from the two balanced starts
    check_balanced_starts(r)
    xs, fs = r["xs"][list(BALANCED_STARTS)], r["fs"][list(BALANCED_STARTS)]
    level = run_level(xs, fs, r, edge)
    kind, n = run_kind(xs, fs, r, same_total, lean_tol)
    return dict(level=point_level(level[0], level[1]),
                specialisation=individual(xs, fs, r).mean(axis=0),          # non-investing groups don't count as balanced
                kind=point_kind(kind[0], kind[1]),
                three=np.any(n >= 3, axis=0) & np.all(n >= 2, axis=0))


def check_layers(r):
    # hand-made final states for each run level and branching kind, and the grid-point rules
    m = r["x_max"]
    level = lambda groups, shares: str(run_level(*state(groups, shares), r))
    kind = lambda groups, shares: str(run_kind(*state(groups, shares), r)[0])
    cases = [
        ("level: one group at (0, 0)",                         level([(0, 0)], [1.0]),                       "defection"),
        (f"level: one group at ({m:g}, {m:g})",                level([(m, m)], [1.0]),                       "full"),
        (f"level: one group at ({m / 2:g}, {m / 2:g})",        level([(m / 2, m / 2)], [1.0]),               "in between"),
        ("point: defection and full",                          point_level(np.array(["defection"]), np.array(["full"]))[0], "bistable"),
        ("point: defection and in between",                    point_level(np.array(["defection"]), np.array(["in between"]))[0], "start-dependent"),
        ("point: full and full",                               point_level(np.array(["full"]), np.array(["full"]))[0], "full"),
        ("kind: one group",                                    kind([(m / 2, 0)], [1.0]),                    "none"),
        (f"kind: ({m / 2:g}, 0) and (0, {m / 2:g})",           kind([(m / 2, 0), (0, m / 2)], [0.5, 0.5]),   "between goods"),
        (f"kind: ({m:g}, {m:g}) and (0.05, 0.05)",             kind([(m, m), (0.05, 0.05)], [0.5, 0.5]),     "in total"),
        (f"kind: ({m:g}, 0) and (0, 0.1)",                     kind([(m, 0), (0, 0.1)], [0.5, 0.5]),         "mixed"),
        ("point: between goods and in total",                  point_kind(np.array(["between goods"]), np.array(["in total"]))[0], "mixed"),
        ("point: between goods and none",                      point_kind(np.array(["between goods"]), np.array(["none"]))[0], "none"),
    ]
    for name, got, expected in cases:
        print(f"  {'ok' if got == expected else 'FAILED':6s} {name:46s} -> {got!r}")
        assert got == expected
    n = run_kind(*state([(m / 2, 0), (0, m / 2), (m / 4, m / 4)], [0.4, 0.4, 0.2]), r)[1]
    print(f"  {'ok' if n == 3 else 'FAILED':6s} {'three groups counted':46s} -> {int(n)}")
    assert n == 3
