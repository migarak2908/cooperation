import jax
import jax.numpy as jnp

b1, b2, c1, c2 = 6, -1.2, 4.56, -1.6

C = lambda y: c1 * y + c2 * y**2
B = lambda y, x: b1 * (x + y) + b2 * (x + y)**2
P = lambda y, x: B(y, x) - C(y)


def multiple_pop(x0, time_step, steps, x_max=1.0, split_gap = 0.01,
        still = 1e-6, merge_gap = 1e-3, min_share = 1e-4, P=P):

    payoff = jax.jit(lambda y, xs, fs: jnp.sum(fs * P(y, xs)))
    grad_payoff = jax.jit(jax.grad(payoff, argnums=0))
    curv = jax.jit(jax.grad(jax.grad(payoff, argnums=0), argnums=0))

    xs = jnp.array([x0])
    fs = jnp.array([1.0])

    for t in range(steps):

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

        # 5. min check

        keep = new_fs > min_share
        moved = jnp.abs(new_xs - xs)

        xs, fs, moved = new_xs[keep], new_fs[keep], moved[keep]
        fs = fs / jnp.sum(fs)

        # 6. merge check

        if len(xs) == 2 and jnp.abs(xs[0]-xs[1]) < merge_gap:
            xs = jnp.array([jnp.sum(fs * xs)])
            fs = jnp.array([1.0])
            moved = jnp.array([0.0])

        # 7. split check

        if len(xs) == 1 and moved[0] < still and 0.0 < xs[0] < x_max:
            if curv(xs[0], xs, fs) > 0:
                x_star = xs[0]
                xs = jnp.array([x_star - split_gap, x_star + split_gap])
                fs = jnp.array([0.5, 0.5])
                print(f"  split at step {t}, x* = {float(x_star):.3f}")

    return xs, fs


if __name__ == "__main__":
    print(multiple_pop(0.1, 0.001, 10000, x_max=1.0, split_gap = 0.01,
            still = 1e-6, merge_gap = 1e-3, min_share = 1e-4))
