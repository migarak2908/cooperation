import jax.numpy as jnp

x_max = 2.0
b1 = jnp.linspace(0, 4, 2000)
b2 = jnp.linspace(0, 4, 2000)
B1, B2 = jnp.meshgrid(b1, b2)

D0   = B1 - 1
Dmax = (B1 - 1) + (2 - 4 * B2) * x_max
curv = 2 - 2 * B2
x_star = (B1 - 1) / ((4 * B2) - 2)

# C
# D
# C or D
# C and D
# ESS

conditions = [
    ((D0 > 0) & (Dmax > 0)),
    ((D0 < 0) & (Dmax < 0)),
    ((D0 < 0) & (Dmax > 0)),
    ((D0 > 0) & (Dmax < 0) & (curv > 0)),
    ((D0 > 0) & (Dmax < 0) & (curv < 0))
]

codes = [0, 1, 2, 3, 4]
names = ["C", "D", "C or D", "C and D", "ESS"]

labels = jnp.select(conditions, codes, default=-1)


# C and D region: cooperator share f, branches at 0 and x_max
m = x_max
f = (B1 - 1 + m - B2 * m) / (2 * B2 * m)

bad = (labels == 3) & ((f < 0) | (f > 1))
print("C and D points with f outside [0, 1]:", int(bad.sum()))


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

extent = [0, 4, 0, 4]

# Plot 1: one block colour per region
colours = ["#2a9d8f", "#e76f51", "#e9c46a", "#8d6cab", "#457b9d"]

plt.figure()
plt.imshow(labels, origin="lower", extent=extent, cmap=ListedColormap(colours), vmin=-0.5, vmax=4.5)
plt.legend(handles=[Patch(color=c, label=n) for c, n in zip(colours, names)])
plt.xlabel("b1'"); plt.ylabel("b2'")


# Plot 2: average trait / x_max: C -> 1, D -> 0, ESS -> x*/x_max, C and D -> f
# C or D: x* is a repeller, starts above it go to x_max, below it to 0.
# With a uniform start on [0, x_max], P(coop) = 1 - x*/x_max
p_coop = 1 - x_star / x_max
mean_trait = jnp.select(
    [labels == 0, labels == 1, labels == 4, labels == 3, labels == 2],
    [jnp.ones_like(B1), jnp.zeros_like(B1), x_star / x_max, f, p_coop],
    default=jnp.nan,
)

plt.figure()
plt.imshow(mean_trait, origin="lower", extent=extent, vmin=0, vmax=1)
plt.colorbar(label="average trait / x_max")
for k in range(5):                           # outline each region separately: one line per border
    plt.contour(B1, B2, labels == k, levels=[0.5], colors="white", linewidths=0.6)
plt.xlabel("b1'"); plt.ylabel("b2'")
plt.show()
