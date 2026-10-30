import jax.numpy as jnp

x_max = 2.0
FLAT_BENEFIT = False      # True: the benefit stays at its maximum past its peak instead of falling (needs x_max <= 0.5)
b1 = jnp.linspace(0, 4, 2000)
b2 = jnp.linspace(0, 4, 2000)
B1, B2 = jnp.meshgrid(b1, b2)

# with x_max > 0.5 the cost x - x^2 falls past x = 0.5, so with a flat benefit the selection gradient
# turns upwards there, adding a repeller the five regions below don't cover
assert not (FLAT_BENEFIT and x_max > 0.5), "FLAT_BENEFIT needs x_max <= 0.5"

# benefit B(z) = b1' z - b2' z^2 of a pair total z, peaking at z = b1' / (2 b2'); b2' can be 0
peak = jnp.where(B2 > 0, B1 / (2 * jnp.where(B2 > 0, B2, 1.0)), jnp.inf)
z_benefit = lambda z: jnp.minimum(z, peak) if FLAT_BENEFIT else z
B = lambda z: B1 * z_benefit(z) - B2 * z_benefit(z) ** 2
B_slope = lambda z: jnp.where(z < peak, B1 - 2 * B2 * z, 0.0) if FLAT_BENEFIT else B1 - 2 * B2 * z
C = lambda x: x - x ** 2
D = lambda x: B_slope(2 * x) - (1 - 2 * x)            # selection gradient of a population all at x

D0   = D(0.0)
# just inside the cap with a flat benefit: past the peak D = 2x - 1, exactly 0 at x_max = 0.5
Dmax = D(x_max - 1e-6) if FLAT_BENEFIT else D(x_max)
curv = 2 - 2 * B2         # x* is below the benefit's peak wherever it is an ESS or branches, so unchanged
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
# (equal payoffs: f B(m) = f B(2m) + (1 - f) B(m) - C(m))
m = x_max
f = (B(m) - C(m)) / (2 * B(m) - B(2 * m))

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
