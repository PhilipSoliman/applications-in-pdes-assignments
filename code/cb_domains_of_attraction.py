import matplotlib.pyplot as plt
import numpy as np
from helper import pyutils

pyutils.add_modules_to_path()
pyutils.set_style()
root = pyutils.get_root()
output_dir = root / "report" / "figures"
data_dir = root / "code" / "data"

#  plot domain of attraction for default parameters
domains = np.load(data_dir / "cb_domains_of_attraction.npy")
equilibria = np.load(data_dir / "cb_equilibria.npy")

fig, ax = plt.subplots()
img = ax.imshow(domains.T, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
cbar = plt.colorbar(ticks=[1, 2, 3], mappable=img)
cbar.ax.set_yticklabels([r"$E_0$", r"$E_1$", r"$E_2$"])
ax.set_xlabel(r"$\hat{R}_a$")
ax.set_ylabel(r"$\hat{\rho}_a$")
for i, root in enumerate(equilibria):
    ax.plot(root[0], root[1], "or", label=f"$E_{i}$")
    # add text to the markers
    ax.text(
        root[0],
        root[1],
        r"$\mathbf{" + f"E_{i}" + "}$",
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
        color="red",
    )
plt.show()

# save the figure
fig.savefig(output_dir / "cb_domains_of_attraction.png", bbox_inches="tight", dpi=400)