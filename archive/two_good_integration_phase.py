import jax
import jax.numpy as jnp

M = lambda pow, a, b: ((a ** pow + b ** pow)/2)** (1 / pow)

def P (y, x, b1, b2, p, q):
    # y: own traits (goods,), x: others' traits (..., goods) -- [..., k] picks good k
    Mp = M(p, x[..., 0] + y[..., 0], x[..., 1] + y[..., 1])
    Mq = M(q, y[..., 0] , y[..., 1])
    return b1 * Mp - b2 * (Mp ** 2) - Mq + Mq ** 2


def multiple_pop_run(p, q, b1, b2, x0, time_step, steps, x_max=1.0, x_min=1e-6, split_gap = 0.01,
        still = 1e-6, merge_gap = 1e-3, min_share = 1e-4, anneal_frac = 0.2, anneal_to = 1e-3):
    # Same steps as multiple_pop: a population that can split into two groups,
    # for one (p, q, b1', b2') and one starting point x0 = (good 1, good 2).
    # xs has shape (groups, goods) = (2, 2); fs has shape (groups,) = (2,).
    # There are always two groups; before a split (or after a merge) both groups
    # sit at the same point, which behaves exactly like a single group.
    # Traits stay in [x_min, x_max]: x_min > 0 because the power mean's gradient is
    # infinite at 0 when p or q < 1.
    # Python ifs are replaced by jnp.where(condition, value if true, value if false),
    # so that jax can run many of these at once.
    # Over the last anneal_frac of the steps the time step shrinks smoothly to
    # time_step * anneal_to. With p < 1 the push on a good near 0 doesn't fade, so fixed
    # steps overshoot and cycle around a corner at a distance of about dt * gradient;
    # shrinking dt collapses that cycle onto the wall.

    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2, p, q))
    grad_payoff = jax.grad(payoff, argnums=0)        # (goods,)
    curv = jax.hessian(payoff, argnums=0)            # (goods, goods)

    anneal_start = steps - int(steps * anneal_frac)
    anneal_len = max(steps - anneal_start, 1)

    def step(state, t):
        xs, fs = state

        # time step for this step: time_step, then shrinking geometrically to time_step * anneal_to
        dt = time_step * anneal_to ** jnp.clip((t - anneal_start) / anneal_len, 0.0, 1.0)

        # 1. compute group grads
        grads = jnp.array([grad_payoff(xi, xs, fs) for xi in xs])

        # 2. update investments
        new_xs = jnp.clip(xs + dt * grads, x_min, x_max)

        # 3. calculate per group payoffs and average payoff
        payoffs = jnp.array([payoff(xi, xs, fs) for xi in xs])
        avg_payoff = jnp.sum(fs * payoffs)

        # 4. update population shares
        new_fs = fs + dt * fs * (payoffs - avg_payoff)
        new_fs = new_fs / jnp.sum(new_fs)

        moved = jnp.linalg.norm(new_xs - xs, axis=-1)     # distance each group moved
        xs, fs = new_xs, new_fs

        # 5. min check: if a group is too small, the other group becomes the single group
        drop0 = fs[0] < min_share
        drop1 = fs[1] < min_share
        xs = jnp.where(drop0, xs[1], jnp.where(drop1, xs[0], xs))
        moved = jnp.where(drop0, moved[1], jnp.where(drop1, moved[0], moved))
        fs = jnp.where(drop0 | drop1, 0.5, fs)

        # 6. merge check
        gap = jnp.linalg.norm(xs[0] - xs[1])
        merge = (gap > 0) & (gap < merge_gap)
        xs = jnp.where(merge, jnp.sum(fs[:, None] * xs, axis=0), xs)
        fs = jnp.where(merge, 0.5, fs)
        moved = jnp.where(merge, 0.0, moved)

        # 7. split check: split along the direction where the payoff curves upwards
        # (eigenvector of the curvature matrix with the largest eigenvalue), using only
        # the goods that are not at a wall: zeroing a pinned good's row and column
        # removes it, so with good 1 pinned the matrix is [[0, 0], [0, A22]]
        single = jnp.all(xs[0] == xs[1])
        free = (x_min < xs[0]) & (xs[0] < x_max)                  # (goods,)
        H = curv(xs[0], xs, fs) * free[:, None] * free[None, :]

        # largest eigenvalue and its eigenvector of the 2x2 symmetric matrix [[a, b], [b, d]]
        a, b, d = H[0, 0], H[0, 1], H[1, 1]
        lam = (a + d) / 2 + jnp.sqrt(((a - d) / 2)**2 + b**2)
        v = jnp.array([b, lam - a])
        # b = 0: matrix is diagonal, split along the good with the larger curvature
        v = jnp.where(b != 0, v, jnp.where(a >= d, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))
        v = v / jnp.linalg.norm(v)

        # "stopped moving" is judged by speed (distance / dt), so a shrinking dt doesn't
        # make every population look still; at the full time step this is moved < still
        stopped = moved[0] / dt < still / time_step
        split = single & stopped & (lam > 0)
        new_groups = jnp.stack([xs[0] - split_gap * v, xs[0] + split_gap * v])
        new_groups = jnp.clip(new_groups, x_min, x_max)      # a split next to a wall must not cross it
        xs = jnp.where(split, new_groups, xs)
        fs = jnp.where(split, 0.5, fs)

        return (xs, fs), None

    xs = jnp.stack([x0, x0])
    fs = jnp.array([0.5, 0.5])
    (xs, fs), _ = jax.lax.scan(step, (xs, fs), jnp.arange(steps))    # the time loop, t = 0 .. steps-1
    return xs, fs


def outcome(xs, x_max, x_min=1e-6, wall_tol=1e-3, same_tol=1e-3):
    # classify one run by its final state, separately for each good:
    # C = 0, D = 1, C and D = 3, ESS = 4  ->  shape (goods,)
    # within wall_tol of a wall counts as at the wall; groups within same_tol count as one
    same = jnp.abs(xs[0] - xs[1]) < same_tol      # (goods,): do both groups agree on this good?
    x = (xs[0] + xs[1]) / 2
    return jnp.select([same & (x > x_max - wall_tol), same & (x < x_min + wall_tol), same],
                      [0, 1, 4], default=3)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    print("running on:", jax.devices())     # on Colab with a GPU runtime this shows a CUDA device

    x_max = 2.0
    x_min = 1e-6
    b1 = jnp.linspace(0, 4, 30)              # b1' values (across)
    b2 = jnp.linspace(0, 4, 30)              # b2' values (up)
    p = jnp.array([0.8, 1, 1.25])
    q = jnp.array([0.8, 1, 1.25])

    # starting points: every combination of (good 1, good 2) starting values, including
    # the corners, so every bistable point has starts ending on each side
    s = jnp.linspace(x_min, x_max, 5)
    S1, S2 = jnp.meshgrid(s, s)
    starts = jnp.stack([S1.ravel(), S2.ravel()], axis=-1)     # (25, 2)

    # one run for every (start, p, q, b2', b1') combination, as flat lists;
    # I is the index of the start, used to look up its (good 1, good 2) values
    I, Pg, Qg, Y, X = jnp.meshgrid(jnp.arange(len(starts)), p, q, b2, b1, indexing="ij")
    shape = I.shape                          # (start, p, q, b2', b1')
    run_one = lambda p, q, b1, b2, x0: multiple_pop_run(p, q, b1, b2, x0, 0.01, 10000,
                                                        x_max=x_max, x_min=x_min)
    run_all = jax.jit(jax.vmap(run_one))     # vmap: run_one on every entry of the lists at once

    print(f"{I.size} runs ...")
    t = time.time()
    xs, fs = run_all(Pg.ravel(), Qg.ravel(), X.ravel(), Y.ravel(), starts[I.ravel()])
    xs.block_until_ready()                   # jax runs asynchronously: wait for it to finish
    print(f"simulation took {time.time() - t:.0f}s")
    run_code = jax.vmap(lambda xs: outcome(xs, x_max, x_min))(xs).reshape(*shape, 2)

    # back to a (start, p, q, b2', b1') layout
    xs = xs.reshape(*shape, 2, 2)            # (..., group, good)
    fs = fs.reshape(*shape, 2)               # (..., group)

    # average trait per good: over groups (weighted by share), then over starts
    mean_trait = jnp.sum(fs[..., None] * xs, axis=-2).mean(0) / x_max     # (p, q, b2', b1', good)

    # each point takes its most common outcome over starts; C or D if some starts end C and some D
    counts = jnp.stack([(run_code == k).sum(0) for k in range(5)])
    labels = jnp.argmax(counts, 0)                                          # (p, q, b2', b1', good)
    labels = jnp.where((counts[0] > 0) & (counts[1] > 0), 2, labels)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    extent = [b1[0], b1[-1], b2[0], b2[-1]]
    names = ["C", "D", "C or D", "C and D", "ESS"]
    colours = ["#2a9d8f", "#e76f51", "#e9c46a", "#8d6cab", "#457b9d"]

    # one figure per good and per plot type, with a panel for each (p, q)
    for g in range(2):
        # Plot 1: one block colour per region
        fig, axes = plt.subplots(len(p), len(q), sharex=True, sharey=True, figsize=(10, 9))
        for i in range(len(p)):
            for j in range(len(q)):
                ax = axes[i, j]
                ax.imshow(labels[i, j, :, :, g], origin="lower", extent=extent, aspect="auto",
                          cmap=ListedColormap(colours), vmin=-0.5, vmax=4.5)
                ax.set_title(f"p={float(p[i]):g}, q={float(q[j]):g}", fontsize=9)
        fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colours, names)], loc="upper right")
        fig.suptitle(f"good {g + 1}: regions")
        fig.supxlabel("b1'"); fig.supylabel("b2'")

        # Plot 2: average trait / x_max
        fig, axes = plt.subplots(len(p), len(q), sharex=True, sharey=True, figsize=(10, 9))
        for i in range(len(p)):
            for j in range(len(q)):
                ax = axes[i, j]
                im = ax.imshow(mean_trait[i, j, :, :, g], origin="lower", extent=extent, aspect="auto",
                               vmin=0, vmax=1)
                for k in range(5):           # outline each region separately: one line per border
                    ax.contour(labels[i, j, :, :, g] == k, levels=[0.5], extent=extent,
                               colors="white", linewidths=0.6)
                ax.set_title(f"p={float(p[i]):g}, q={float(q[j]):g}", fontsize=9)
        fig.colorbar(im, ax=axes, label="average trait / x_max (simulated)")
        fig.suptitle(f"good {g + 1}: average trait")
        fig.supxlabel("b1'"); fig.supylabel("b2'")

    plt.show()
