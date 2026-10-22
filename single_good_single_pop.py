import jax
import jax.numpy as jnp

b1, b2, c1, c2 = 6, -0.9, 4.56, -1.6

C = lambda y: c1 * y + c2 * y**2
B = lambda y, x: b1 * (x + y) + b2 * (x + y)**2
P = lambda y, x: B(y, x) - C(y)

D = lambda x: jax.grad(P, argnums=0)(x, x)
D = jax.jit(D)

curv = lambda x: jax.grad(jax.grad(P, argnums=0), argnums=0)(x, x)

dDdx = jax.grad(D)


def single_pop(x, time_step, steps, min_step_size, x_max=1.0):
    for _ in steps:
        step = time_step * D(x)
        if abs(step) < min_step_size:
            if dDdx(x) < 0 and curv(x) > 0:
                print("branching")
                break
            else:
                print("finished")
                break
        x = x + step
        x = jnp.clip(x, 0.0, x_max)
        print(x)
    return x


x = single_pop(0.001, 0.001, range(50000), 0.0000001, x_max=1.0)
print(x, D(x), curv(x), dDdx(x))



