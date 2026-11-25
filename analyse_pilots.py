# Pilot runs from run_individual_sim.py --record summary --replicates R: a figure per file (one row per
# replicate: u histogram, v histogram, var u and var v, fraction at the boundaries) and a table per file
# saying in which direction each replicate split first.
#
#   python analyse_pilots.py pilot_*.npz --out-dir pilot_figs
#
# u = (x1 + x2)/sqrt(2) is total investment, v = (x1 - x2)/sqrt(2) allocation between the goods.

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_SD = 0.05            # a direction has split once its sd exceeds this
BEFORE = 10                # records averaged for var_u / var_v


def load(name):
    f = np.load(name)
    if str(f["record"]) != "summary":
        raise SystemExit(f"{name}: not a --record summary file")
    d = {k: f[k] for k in f.files}
    if int(d["replicates"]) == 1:                                  # one replicate: no replicate axis, add it
        for k in ("mean_u", "mean_v", "var_u", "var_v", "cov_uv", "n_zero", "n_max", "hist_u", "hist_v",
                  "pop0", "final"):
            d[k] = d[k][None]
    d["gen"] = int(d["record_every"]) * np.arange(1, d["var_u"].shape[1] + 1)
    return d


def split(var_u, var_v, gen, threshold):
    # (direction, generation, mean var_u/var_v over the BEFORE records before the split or the last BEFORE)
    sd_u, sd_v = np.sqrt(var_u), np.sqrt(var_v)
    over = (sd_u > threshold) | (sd_v > threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = var_u / var_v
    if not over.any():
        return "none", None, np.nanmean(ratio[-BEFORE:])
    k = np.argmax(over)
    direction = "total" if sd_u[k] >= sd_v[k] else "allocation"        # both in the same record: the larger
    return direction, gen[k], (np.nanmean(ratio[max(0, k - BEFORE):k]) if k else np.nan)


def figure(d, filename, threshold):
    R, gen, N, x_max = d["var_u"].shape[0], d["gen"], int(d["pop_size"]), float(d["x_max"])
    step = int(d["record_every"])
    gen_edges = np.append(gen - step / 2, gen[-1] + step / 2)
    fig, axes = plt.subplots(R, 4, figsize=(16, 2.2 * R + 0.6), squeeze=False, sharex=True)
    for r in range(R):
        for c, (h, edges, label) in enumerate([(d["hist_u"], d["u_edges"], "u (total)"),
                                               (d["hist_v"], d["v_edges"], "v (allocation)")]):
            m = axes[r, c].pcolormesh(gen_edges, edges, np.log1p(h[r].T.astype(float)), cmap="viridis", rasterized=True)
            axes[r, c].set_ylabel(label)
            fig.colorbar(m, ax=axes[r, c], label="log(1 + count)")
        ax = axes[r, 2]
        ax.plot(gen, d["var_u"][r], label="var u (total)")
        ax.plot(gen, d["var_v"][r], label="var v (allocation)")
        ax.axhline(threshold**2, color="grey", ls=":", lw=1, label=f"sd = {threshold:g}")
        ax.set_yscale("log")
        ax = axes[r, 3]
        for g in range(2):
            ax.plot(gen, d["n_zero"][r, :, g] / N, label=f"good {g + 1} at 0")
            ax.plot(gen, d["n_max"][r, :, g] / N, ls="--", label=f"good {g + 1} at {x_max:g}")
        ax.set_ylim(-0.02, 1.02)
        axes[r, 0].set_title(f"replicate {r}", loc="left", fontsize=9)
    axes[0, 2].legend(fontsize=7)
    axes[0, 3].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("generation")
    fig.suptitle(f"{os.path.basename(filename)}: b1' = {float(d['b1']):.3g}, b2' = {float(d['b2']):.3g}, "
                 f"p = {float(d['p']):g}, q = {float(d['q']):g}, rho = {float(d['rho']):g}")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def table(d, threshold):
    x_max = float(d["x_max"])
    print(f"{'rep':>4} {'split':>10} {'generation':>11} {'var_u/var_v':>12} {'final at boundary':>18}")
    for r in range(d["var_u"].shape[0]):
        direction, g, ratio = split(d["var_u"][r], d["var_v"][r], d["gen"], threshold)
        final = d["final"][r]
        boundary = ((final == 0) | (final == x_max)).any(1).mean()        # at 0 or x_max in either good
        print(f"{r:>4} {direction:>10} {'-' if g is None else g:>11} {ratio:>12.3g} {boundary:>18.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="figures and split table for --record summary files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--sd-threshold", type=float, default=SPLIT_SD, help="sd at which a direction counts as split")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.files:
        d = load(name)
        out = os.path.join(args.out_dir, os.path.basename(name).replace(".npz", ".pdf"))
        figure(d, out, args.sd_threshold)
        print(f"\n{name} (rho = {float(d['rho']):g}, split at sd > {args.sd_threshold:g}) -> {out}")
        table(d, args.sd_threshold)
