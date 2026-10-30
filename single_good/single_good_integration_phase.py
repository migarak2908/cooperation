import jax
import jax.numpy as jnp


P = lambda y, x, b1, b2: b1 * (x + y) - b2 * (x + y)**2 - y + y**2


def multiple_pop_run(b1, b2, x0, time_step, steps, x_max=1.0, split_gap = 0.01,
        still = 1e-6, merge_gap = 1e-3, min_share = 1e-4):
    # Same steps as multiple_pop: a population that can split into two groups,
    # for one (b1', b2') and one starting value.
    # There are always two groups; before a split (or after a merge) both groups
    # sit at the same x, which behaves exactly like a single group.
    # Python ifs are replaced by jnp.where(condition, value if true, value if false),
    # so that jax can run many of these at once.

    payoff = lambda y, xs, fs: jnp.sum(fs * P(y, xs, b1, b2))
    grad_payoff = jax.grad(payoff, argnums=0)
    curv = jax.grad(grad_payoff, argnums=0)

    def step(state, _):
        xs, fs = state

        # 1. compute group grads
        grads = jnp.array([grad_payoff(xi, xs, fs) for xi in xs])

        # 2. update investments
        new_xs = jnp.clip(xs + time_step * grads, 0, x_max)

        # 3. calculate per group payoffs and average payoff
        payoffs = jnp.array([payoff(xi, xs, fs) for xi in xs])
        avg_payoff = jnp.sum(fs * payoffs)

        # 4. update population shares
        new_fs = fs + time_step * fs * (payoffs - avg_payoff)
        new_fs = new_fs / jnp.sum(new_fs)

        moved = jnp.abs(new_xs - xs)
        xs, fs = new_xs, new_fs

        # 5. min check: if a group is too small, the other group becomes the single group
        drop0 = fs[0] < min_share
        drop1 = fs[1] < min_share
        xs = jnp.where(drop0, xs[1], jnp.where(drop1, xs[0], xs))
        moved = jnp.where(drop0, moved[1], jnp.where(drop1, moved[0], moved))
        fs = jnp.where(drop0 | drop1, 0.5, fs)

        # 6. merge check
        gap = jnp.abs(xs[0] - xs[1])
        merge = (gap > 0) & (gap < merge_gap)
        xs = jnp.where(merge, jnp.sum(fs * xs), xs)
        fs = jnp.where(merge, 0.5, fs)
        moved = jnp.where(merge, 0.0, moved)

        # 7. split check
        single = xs[0] == xs[1]
        split = (single & (moved[0] < still) & (0.0 < xs[0]) & (xs[0] < x_max)
                 & (curv(xs[0], xs, fs) > 0))
        xs = jnp.where(split, xs + split_gap * jnp.array([-1.0, 1.0]), xs)
        fs = jnp.where(split, 0.5, fs)

        return (xs, fs), None

    xs = jnp.array([x0, x0])
    fs = jnp.array([0.5, 0.5])
    (xs, fs), _ = jax.lax.scan(step, (xs, fs), None, length=steps)
    # the time loop
    return xs, fs


def outcome(xs, x_max):
    # classify one run by its final state: C = 0, D = 1, C and D = 3, ESS = 4
    single = xs[0] == xs[1]
    return jnp.select([single & (xs[0] == x_max), single & (xs[0] == 0), single],
                      [0, 1, 4], default=3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_max = 2.0
    b1 = jnp.linspace(0, 4, 100)             # b1' values (across)
    b2 = jnp.linspace(0, 4, 100)             # b2' values (up)

    # several starting values, evenly spread over [0, x_max], to average over
    # the C or D region where the outcome depends on the start; including 0 and
    # x_max means every C or D point has at least one start ending C and one ending D
    starts = jnp.linspace(0, x_max, 11)
    shape = (len(starts), len(b2), len(b1))

    # one run for every (start, b2', b1') combination, as flat lists
    S, Y, X = jnp.meshgrid(starts, b2, b1, indexing="ij")    # X = b1' varies across, Y = b2' varies up
    run_one = lambda b1, b2, x0: multiple_pop_run(b1, b2, x0, 0.01, 10000, x_max=x_max)
    run_all = jax.jit(jax.vmap(run_one))     # vmap: run_one on every entry of the lists at once
    xs, fs = run_all(X.ravel(), Y.ravel(), S.ravel())
    run_code = jax.vmap(lambda xs: outcome(xs, x_max))(xs).reshape(shape)

    # back to a (start, b2', b1') layout
    xs = xs.reshape(*shape, 2)
    fs = fs.reshape(*shape, 2)

    # average trait over groups, then over starts
    mean_trait = jnp.sum(fs * xs, -1).mean(0) / x_max

    # each point takes its most common outcome over starts; C or D if some starts end C and some D
    counts = jnp.stack([(run_code == k).sum(0) for k in range(5)])
    labels = jnp.argmax(counts, 0)
    labels = jnp.where((counts[0] > 0) & (counts[1] > 0), 2, labels)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    extent = [b1[0], b1[-1], b2[0], b2[-1]]
    names = ["C", "D", "C or D", "C and D", "ESS"]
    colours = ["#2a9d8f", "#e76f51", "#e9c46a", "#8d6cab", "#457b9d"]

    # Plot 1: one block colour per region
    plt.figure()
    plt.imshow(labels, origin="lower", extent=extent, aspect="auto",
               cmap=ListedColormap(colours), vmin=-0.5, vmax=4.5)
    plt.legend(handles=[Patch(color=c, label=n) for c, n in zip(colours, names)])
    plt.xlabel("b1'"); plt.ylabel("b2'")

    # Plot 2: average trait / x_max
    plt.figure()
    plt.imshow(mean_trait, origin="lower", extent=extent, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="average trait / x_max (simulated)")
    for k in range(5):                       # outline each region separately: one line per border
        plt.contour(labels == k, levels=[0.5], extent=extent, colors="white", linewidths=0.6)
    plt.xlabel("b1'"); plt.ylabel("b2'")
    plt.show()
