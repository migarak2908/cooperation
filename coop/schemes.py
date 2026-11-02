# Labelling schemes: what counts as cooperation. A scheme turns the final state of each run into
# something to plot:
#   categories: a name per run ("CC and DD", ...); a grid point's label is the distinct names over
#               its starts, joined with " |or| "; rare labels are filtered out for a second plot
#   scale:      a number per run, averaged over starts and plotted with a colour bar
# Needs numpy and scipy only (no jax).
#
# To add a scheme: write a function (xs, fs, r) -> name or number per run, like cc_dc_dd below, and
# add it to SCHEMES.

import numpy as np

from coop import references

LABEL_SHARE_MIN = 0.01               # a group counts if its (rounded) share is above this
RARE_FRAC = 0.003                    # labels on fewer than 0.3% of all grid points are rare
INVESTED_MIN = 0.05                  # the "absolute" rule: a good counts as invested above this


def counted_groups(xs, fs, r):
    # the groups that count when labelling: traits rounded to 3 decimals, shares to 2; a group counts
    # if its share is above LABEL_SHARE_MIN and no earlier counted group is within merge_gap of it
    xs, fs = np.round(xs, 3), np.round(fs, 2)
    counted = fs > LABEL_SHARE_MIN
    K = xs.shape[-2]
    dist = np.linalg.norm(xs[..., :, None, :] - xs[..., None, :, :], axis=-1)
    earlier = np.tri(K, k=-1, dtype=bool)                                     # [j, i]: i < j
    counted &= ~np.any(counted[..., None, :] & earlier & (dist < r["merge_gap"]), axis=-1)
    return xs, fs, counted


# ---------- scheme: CC / DC / DD ----------
# A good is "invested" above invested_threshold(r):
#   half_cap: above x_max / 2 (the quadratic model's default: half the cap)
#   absolute: above INVESTED_MIN (the exponential model's default: it has no natural cap, and
#             x_max / 2 would label almost everything DD)
# A group is CC if it invests in both goods, DC in one, DD in neither. A run's outcome is the distinct
# group types, alphabetical, joined with " and ".

def default_rule(r):
    return "absolute" if r["model"] == "exp" else "half_cap"


def invested_rule(r):
    return r.get("invested_rule") or default_rule(r)


def invested_threshold(r):
    return r["x_max"] / 2 if invested_rule(r) == "half_cap" else INVESTED_MIN


def cc_dc_dd(xs, fs, r, division=False):
    # xs: (..., K, 2), fs: (..., K) -> outcome name per run; with division, DC groups investing in
    # different goods make "DC and DC'"
    xs, fs, counted = counted_groups(xs, fs, r)
    invested = xs > invested_threshold(r)
    n_invested = invested.sum(axis=-1)
    has = lambda n: np.any(counted & (n_invested == n), axis=-1)
    dc_in = lambda good: np.any(counted & (n_invested == 1) & invested[..., good], axis=-1)
    division_of_labour = division & dc_in(0) & dc_in(1)

    # each run's combination as bits (1 = CC, 2 = DC, 4 = DD, 8 = division of labour), then its name
    bits = has(2) * 1 + has(1) * 2 + has(0) * 4 + division_of_labour * 8
    return np.array([outcome_name(k) for k in range(16)], dtype=object)[bits]


def outcome_name(bits):
    types = []
    if bits & 1:
        types.append("CC")
    if bits & 2:
        types.append("DC and DC'" if bits & 8 else "DC")
    if bits & 4:
        types.append("DD")
    return " and ".join(types) if types else "none"


def cc_dc_dd_division(xs, fs, r):
    return cc_dc_dd(xs, fs, r, division=True)


# fixed colours: hsv from 0 to 0.9 over these 11 labels in this order (as in the reference);
# "DC and DC'" teal, other labels containing it olive
CC_DC_DD_LABELS = ["CC", "CC and DC", "CC and DD", "CC |or| DC", "CC |or| DC |or| DD", "CC |or| DD",
                   "DC", "DC and DD", "DC and DD |or| DD", "DC |or| DD", "DD"]

def cc_dc_dd_colours(present):
    import matplotlib.pyplot as plt
    names = list(CC_DC_DD_LABELS)
    colours = [tuple(c) for c in plt.cm.hsv(np.linspace(0, 0.9, len(names)))]
    for l in sorted(l for l in present if "DC and DC'" in l):
        names.append(l)
        colours.append("#1b9e77" if l == "DC and DC'" else "#66a61e")
    return names, colours


def check_cc_dc_dd(r):
    # hand-made final states: full investment at x_max, and cases either side of the "invested"
    # threshold, so the checks hold for any x_max and rule
    m, t = r["x_max"], 1e-6
    half = invested_threshold(r)
    hi, lo = 1.2 * half, 0.8 * half                     # 0.6 x_max and 0.4 x_max for the half_cap rule
    rule = "x_max / 2" if invested_rule(r) == "half_cap" else f"{half:g}"
    def outcome(groups, shares, division=False):
        xs = np.full((4, 2), t); fs = np.zeros(4)
        xs[:len(groups)], fs[:len(shares)] = groups, shares
        return cc_dc_dd(xs, fs, r, division)
    cases = [
        (f"one group at ({m:g}, {m:g})",                     outcome([(m, m)], [1.0]),                          "CC"),
        (f"groups at ({m:g}, {m:g}) and (1e-6, 1e-6)",       outcome([(m, m), (t, t)], [0.5, 0.5]),             "CC and DD"),
        (f"groups at ({m:g}, 1e-6) and (1e-6, {m:g})",       outcome([(m, t), (t, m)], [0.5, 0.5]),             "DC"),
        ("same, division of labour on",                      outcome([(m, t), (t, m)], [0.5, 0.5], True),       "DC and DC'"),
        ("full group share 0.995 + (1e-6, 1e-6) share 0.005", outcome([(m, m), (t, t)], [0.995, 0.005]),        "CC"),
        ("three groups CC, DC, DD",                          outcome([(m, m), (m, t), (t, t)], [0.4, 0.3, 0.3]), "CC and DC and DD"),
        (f"one group at ({hi:g}, {hi:g}), above {rule}",     outcome([(hi, hi)], [1.0]),                        "CC"),
        (f"one group at ({lo:g}, {lo:g}), below {rule}",     outcome([(lo, lo)], [1.0]),                        "DD"),
    ]
    if round(half, 3) == half:       # rounding: just above the threshold rounds down onto it
        cases.append((f"one group at ({half + 0.0004:g}, {half + 0.0004:g}) (rounds to {half:g})",
                      outcome([(half + 0.0004, half + 0.0004)], [1.0]), "DD"))
    for name, got, expected in cases:
        print(f"  {'ok' if got == expected else 'FAILED':6s} {name:46s} -> {got!r}")
        assert got == expected
    got = grid_labels(np.array([["DD"], ["CC and DD"], ["DD"], ["CC"]], dtype=object))[0]
    print(f"  {'ok' if got == 'CC |or| CC and DD |or| DD' else 'FAILED':6s} {'grid label from DD, CC and DD, DD, CC':46s} -> {got!r}")
    assert got == "CC |or| CC and DD |or| DD"


# ---------- scheme: mean investment (a sliding scale) ----------

def mean_investment(xs, fs, r):
    # share-weighted average investment over groups and both goods, as a fraction of
    # references.investment_scale (x_max, or the one-good singular point x*)
    return np.sum(fs[..., None] * xs, axis=(-2, -1)) / (2 * references.investment_scale(r)[0])


# ---------- schemes: how the investment is split between goods (sliding scales) ----------
# With group shares f_i, d_i = x1_i - x2_i and t_i = x1_i + x2_i (counted groups only):
#   individual = sum f_i |d_i| / sum f_i t_i      how lopsided individuals are
#   population = |sum f_i d_i| / sum f_i t_i      how lopsided the population is as a whole
#   division_of_labour_index = individual - population
#                                                 0 to 1; 1 = groups specialise in opposite goods
# All three are NaN where nobody invests (nobody_invests): there the split is noise, not "balanced".
NOBODY_INVESTS = 0.01                # share-weighted total below 1% of 2 * investment_scale

def lopsidedness(xs, fs, r):
    xs, fs, counted = counted_groups(xs, fs, r)
    f = np.where(counted, fs, 0.0)
    d, t = xs[..., 0] - xs[..., 1], xs[..., 0] + xs[..., 1]
    total = np.sum(f * t, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        individual = np.sum(f * np.abs(d), axis=-1) / total
        population = np.abs(np.sum(f * d, axis=-1)) / total
    return individual, population, total


def nobody_invests(total, r):
    # share-weighted total below NOBODY_INVESTS of 2 * investment_scale (also catches runs sitting near
    # x_min); a NaN scale (no x*) falls back to a total of about 0
    with np.errstate(invalid="ignore"):
        return (total < 1e-9) | (total < NOBODY_INVESTS * 2 * references.investment_scale(r)[0])


def individual(xs, fs, r):
    individual, _, total = lopsidedness(xs, fs, r)
    return np.where(nobody_invests(total, r), np.nan, individual)


def population(xs, fs, r):
    _, population, total = lopsidedness(xs, fs, r)
    return np.where(nobody_invests(total, r), np.nan, population)


def division_of_labour_index(xs, fs, r):
    individual, population, total = lopsidedness(xs, fs, r)
    return np.where(nobody_invests(total, r), np.nan, individual - population)


def first(value):
    # a hand-made check case's value; the scale may differ per panel (x*), so take the first panel's
    return float(np.ravel(value)[0])


def check_lopsidedness(r):
    def values(groups, shares):
        xs = np.zeros((4, 2)); fs = np.zeros(4)
        xs[:len(groups)], fs[:len(shares)] = groups, shares
        return first(individual(xs, fs, r)), first(population(xs, fs, r))
    cases = [
        ("one group at (1, 0)",                          values([(1, 0)], [1.0]),                  (1.0, 1.0)),
        ("groups at (1, 0) and (0, 1), shares 0.5 each", values([(1, 0), (0, 1)], [0.5, 0.5]),     (1.0, 0.0)),
        ("one group at (0, 0): nobody invests",          values([(0, 0)], [1.0]),                  (np.nan, np.nan)),
    ]
    for name, got, expected in cases:
        ok = np.allclose(got, expected, rtol=0, atol=1e-9, equal_nan=True)
        print(f"  {'ok' if ok else 'FAILED':6s} {name:50s} -> individual {got[0]:g}, population {got[1]:g} "
              f"(expected {expected[0]:g}, {expected[1]:g})")
        assert ok


def check_division_of_labour_index(r):
    def score(groups, shares):
        xs = np.zeros((4, 2)); fs = np.zeros(4)
        xs[:len(groups)], fs[:len(shares)] = groups, shares
        return first(division_of_labour_index(xs, fs, r))
    same = lambda got, expected: abs(got - expected) < 1e-9 or (np.isnan(got) and np.isnan(expected))
    cases = [
        ("one group at (1, 1)",                              score([(1, 1)], [1.0]),                                  0.0),
        ("one group at (1, 0)",                              score([(1, 0)], [1.0]),                                  0.0),
        ("groups at (1, 0) and (0, 1), shares 0.5 each",     score([(1, 0), (0, 1)], [0.5, 0.5]),                     1.0),
        ("(1, 0), (0, 1), (0.5, 0.5), shares 0.4, 0.4, 0.2", score([(1, 0), (0, 1), (0.5, 0.5)], [0.4, 0.4, 0.2]),    0.8),
        ("one group at (0, 0): nobody invests",              score([(0, 0)], [1.0]),                                  np.nan),
    ]
    for name, got, expected in cases:
        print(f"  {'ok' if same(got, expected) else 'FAILED':6s} {name:50s} -> {got:g} (expected {expected:g})")
        assert same(got, expected)


# ---------- scheme: spread of total investment between groups (a sliding scale) ----------
# sqrt(sum f_i (t_i - sum f_j t_j)^2) with t_i = x1_i + x2_i (counted groups only): 0 when every group
# invests the same total, large when the totals branch (e.g. cooperators and defectors)

def spread(xs, fs, r):
    xs, fs, counted = counted_groups(xs, fs, r)
    f = np.where(counted, fs, 0.0)
    t = xs[..., 0] + xs[..., 1]
    mean = np.sum(f * t, axis=-1, keepdims=True)
    return np.sqrt(np.sum(f * (t - mean) ** 2, axis=-1))


def total_spread(xs, fs, r):
    # as a fraction of twice references.investment_scale (x_max, or x*): 0.5 for an even split
    # between investing nothing and the scale in both goods
    return spread(xs, fs, r) / (2 * references.investment_scale(r)[0])


def check_total_spread(r):
    # checked on the unscaled spread, so it holds for any model's scale
    m = r["x_max"]
    xs = np.zeros((4, 2)); fs = np.zeros(4)
    xs[0], fs[:2] = (m, m), (0.5, 0.5)
    got = float(spread(xs, fs, r)) / (2 * m)
    name = f"groups at ({m:g}, {m:g}) and (0, 0), shares 0.5 each"
    print(f"  {'ok' if abs(got - 0.5) < 1e-9 else 'FAILED':6s} {name:50s} -> spread / (2 x_max) {got:g} (expected 0.5)")
    assert abs(got - 0.5) < 1e-9


# ---------- scheme: not converged (a sliding scale) ----------

def not_converged(xs, fs, r):
    # 1 for a run that stopped at max_steps without converging, else 0; averaged over starts, the
    # fraction of starts not converged. r["status"] is laid out like xs without its last two axes
    return (r["status"] == 2).astype(float)


# ---------- scheme: DHK types (by structure, not by the half-cap threshold) ----------
# With each counted group's total t_i = x1_i + x2_i and top = 2 x_max:
#   branching:         two counted groups' totals differ by more than DHK_BRANCH * top
#   otherwise, with the share-weighted mean total m:
#   defection:         m < DHK_EDGE * top;  full cooperation: m > (1 - DHK_EDGE) * top;  else ESS
DHK_BRANCH = 0.05
DHK_EDGE = 0.01
DHK_LABELS = ["defection", "ESS", "full cooperation", "branching", "defection |or| full cooperation"]


def dhk_type(xs, fs, r):
    xs, fs, counted = counted_groups(xs, fs, r)
    top = 2 * r["x_max"]
    t = xs[..., 0] + xs[..., 1]
    hi = np.max(np.where(counted, t, -np.inf), axis=-1)
    lo = np.min(np.where(counted, t, np.inf), axis=-1)
    f = np.where(counted, fs, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.sum(f * t, axis=-1) / np.sum(f, axis=-1)
    return np.select([hi - lo > DHK_BRANCH * top, m < DHK_EDGE * top, m > (1 - DHK_EDGE) * top],
                     ["branching", "defection", "full cooperation"], default="ESS").astype(object)


def dhk_colours(present):
    # defection red, ESS green, full cooperation blue, branching purple, the repeller orange
    return list(DHK_LABELS), ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]


def check_dhk_type(r):
    m = r["x_max"]
    def outcome(groups, shares):
        xs = np.zeros((4, 2)); fs = np.zeros(4)
        xs[:len(groups)], fs[:len(shares)] = groups, shares
        return str(dhk_type(xs, fs, r))
    cases = [
        ("one group at (0, 0)",                           outcome([(0, 0)], [1.0]),                         "defection"),
        (f"one group at ({m:g}, {m:g})",                  outcome([(m, m)], [1.0]),                         "full cooperation"),
        (f"one group at ({m / 2:g}, 0)",                  outcome([(m / 2, 0)], [1.0]),                     "ESS"),
        (f"groups at ({m:g}, 0) and (0, {m:g})",          outcome([(m, 0), (0, m)], [0.5, 0.5]),            "ESS"),
        (f"groups at ({m:g}, {m:g}) and (0, 0)",          outcome([(m, m), (0, 0)], [0.5, 0.5]),            "branching"),
        (f"({m:g}, {m:g}) share 0.995 + (0, 0) share 0.005", outcome([(m, m), (0, 0)], [0.995, 0.005]),     "full cooperation"),
    ]
    for name, got, expected in cases:
        print(f"  {'ok' if got == expected else 'FAILED':6s} {name:46s} -> {got!r}")
        assert got == expected
    got = grid_labels(np.array([["full cooperation"], ["defection"]], dtype=object))[0]
    print(f"  {'ok' if got == DHK_LABELS[-1] else 'FAILED':6s} {'grid label from full cooperation, defection':46s} -> {got!r}")
    assert got == DHK_LABELS[-1]


# ---------- schemes ----------
# kind: "categories" or "scale". Optional: check (run first), and for categories colours (a fixed
# legend) and compare (run on the unfiltered grid labels: the model's analytic prediction); for a
# scale, label (may use {scale}) and range ((vmin, vmax), or a function of r giving it)

SCHEMES = {
    "cc_dc_dd":          dict(kind="categories", outcome=cc_dc_dd, colours=cc_dc_dd_colours,
                              check=check_cc_dc_dd, compare=references.compare),
    "cc_dc_dd_division": dict(kind="categories", outcome=cc_dc_dd_division, colours=cc_dc_dd_colours,
                              check=check_cc_dc_dd, compare=references.compare_division),
    "mean_investment":   dict(kind="scale", outcome=mean_investment, label="mean investment / {scale}",
                              range=lambda r: (0, references.investment_scale(r)[2])),
    "division_of_labour_index": dict(kind="scale", outcome=division_of_labour_index, check=check_division_of_labour_index,
                                     label="division-of-labour score (1 = clean division of labour)", range=(0, 1)),
    "individual":        dict(kind="scale", outcome=individual, check=check_lopsidedness, range=(0, 1),
                              label="individual lopsidedness (0 = invests equally in both goods, 1 = in one only)"),
    "population":        dict(kind="scale", outcome=population, check=check_lopsidedness, range=(0, 1),
                              label="population lopsidedness (0 = goods balanced overall, 1 = one good only)"),
    "total_spread":      dict(kind="scale", outcome=total_spread, check=check_total_spread, range=(0, 0.5),
                              label="spread of total investment between groups / (2 {scale})"),
    "not_converged":     dict(kind="scale", outcome=not_converged, range=(0, 1),
                              label="fraction of starts not converged"),
    "dhk_type":          dict(kind="categories", outcome=dhk_type, colours=dhk_colours,
                              check=check_dhk_type, compare=references.compare_dhk),
}


# ---------- summaries ----------
# one cell per panel: the fraction of the panel's grid points meeting a condition on a scale scheme
# (averaged over starts first, then minus the panel with relative_to's values, if given);
# comparison "abs>" (|value| > threshold), ">" or "<"

SUMMARIES = [
    # title                 scheme                      relative_to         comparison  threshold
    ("totals changed",      "mean_investment",          {"p": 1, "q": 1},   "abs>",     0.05),
    ("branched totals",     "total_spread",             None,               ">",        0.1),
    ("single specialists",  "population",               None,               ">",        0.5),
    ("division of labour",  "division_of_labour_index", None,               ">",        0.5),
    ("balanced",            "individual",               None,               "<",        0.1),
]


# ---------- grid labels ----------

def grid_labels(outcomes):
    # outcomes: (start, ...) -> label per grid point: distinct outcomes, alphabetical, " |or| "
    labels = np.empty(outcomes.shape[1:], dtype=object)
    for idx in np.ndindex(labels.shape):
        labels[idx] = " |or| ".join(sorted(set(outcomes[(slice(None),) + idx])))
    return labels


def replace_rare(labels, r):
    # labels on fewer than RARE_FRAC of all grid points (all panels together) are rare; each rare
    # point takes the label of the nearest non-rare point in the same panel
    from scipy.spatial import cKDTree
    values, counts = np.unique(labels, return_counts=True)
    rare = {v: int(c) for v, c in zip(values, counts) if c < RARE_FRAC * labels.size}
    y_axis, x_axis = r["grid_axes"]
    X, Y = np.meshgrid(x_axis, y_axis)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    out = labels.copy()
    replaced = np.zeros(labels.shape[:2], int)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            flat = labels[i, j].ravel()
            is_rare = np.isin(flat, list(rare))
            if is_rare.any() and not is_rare.all():
                _, nearest = cKDTree(points[~is_rare]).query(points[is_rare])
                flat = flat.copy()
                flat[is_rare] = flat[~is_rare][nearest]
                out[i, j] = flat.reshape(labels.shape[2:])
                replaced[i, j] = is_rare.sum()
    return out, rare, replaced
