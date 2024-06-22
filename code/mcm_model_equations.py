import json
from pprint import pprint

import latextable
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from helper import pyutils

# from tabulate import tabulate
from texttable import Texttable

pyutils.add_modules_to_path()
from continuation import Continuation
from mcm import MCM
from root_finding import RootFinding

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# data folder
data_dir = root / "code" / "data"

# set standard matplotlib style
pyutils.set_style()

############### Setup Non-linear System of Equations #####
mcm = MCM()
sys = mcm.constructSystem()
# mcm.printSystem()
# mcm.printDimensionlessSystem()
# mcm.findStationaryPoints()
stable_points = mcm.getStableStationaryPoints()
print("Stable stationary points:")
pprint(stable_points)
rootfinder = RootFinding()
rootfinder.output = True

############### generate latex table of stationary points #####
print("\nGenerating LaTeX table of stationary points...")
header = [
    r"\textbf{index}",
    r"$\mathbf{x_1}$",
    r"$\mathbf{x_2}$",
    r"$\mathbf{x_3}$",
    r"\textbf{stability}",
]
rows = [header]
stationary_points = mcm.stationaryPoints
coords = stationary_points["coords"]
stabilities = stationary_points["stable"]
for index, (point, stability) in enumerate(zip(coords, stabilities)):
    x1, x2, x3 = point
    row = [index + 1, float(x1), float(x2), float(x3), stability]
    rows.append(row)
table = Texttable()
table.set_cols_align(["c"] * len(header))
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_precision(2)
table.set_cols_dtype(
    ["i", pyutils.scientific_fmt, pyutils.scientific_fmt, pyutils.scientific_fmt, "t"]
)
table.add_rows(rows)
label = f"tab:mcm_stationary_points"
caption = "Stationary points of the MCM model."

# add position specifier & change font size
table_str = latextable.draw_latex(table, caption=caption, label=label)
table_str = table_str.replace(r"\begin{table}", r"\begin{table}[H]")
table_str = table_str.replace(r"{c|", r"{|c|")
table_str = table_str.replace(r"|c}", r"|c|}" + "\n\t\t\t" + r"\hline")
table_str = table_str.replace(
    r"\end{tabular}", "\t" + r"\hline" + "\n\t\t" + r"\end{tabular}"
)

filename = "mcm_stationary_points.tex"
filepath = root / "report" / "tables" / filename
with open(filepath, "w") as f:
    f.write(table_str)
print("Done!")

############### Continuation parameters ##################
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations (per continuation)
continuation = Continuation(tolerance, maxiter)
continuation.output = True
parameter_name = "p1"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0.5, 1)  # range of the parametervalues of interest
stepsize = 0.001  # 0.01  # stepsize (needs to be small, why?)
tune_factor = 0.00001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations

############### Continuation (loop) ##########################
print("\nBuilding bifurcation diagram (continuation loop)...")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))  # sharex=True, sharey=True)

bifurcations = []
for i, stable_point in enumerate(stable_points):
    print(f"\n Continuation loop on {stable_point}")
    ax = axs[i]
    mcm.x = stable_point
    rootfinder.newtonRaphson(mcm)
    solutions = continuation.arclengthLoop(
        mcm,
        parameter_name,
        stepsize,
        tune_factor,
        parameter_range,
        maxContinuations,
    )
    averages = solutions["average"]
    minima = solutions["minimum"]
    maxima = solutions["maximum"]
    p1s = solutions["parameter"]
    stable = solutions["stable"]
    bifs = solutions["bifurcations"]
    eigvals_around_bif = solutions["eigvals_around_bif"]

    # add new bifurcations to list
    bifurcations += bifs

    # plot bifurcations
    for i, bif in enumerate(bifs):
        mean = np.mean(bif["solution"])
        ax.plot(bif["parameter"], mean, "ko", label="bifurcation")
        axs[0].annotate(
            f"H{i}",
            (bif["parameter"], mean),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        # print eigs directly before and after bifurcation
        print("\nEigenvalues around bifurcation:")
        print("\t before: ", eigvals_around_bif[i][0])
        print("\t after: ", eigvals_around_bif[i][1])

    # stable branches
    # ax.plot(p1s[stable], maxima[stable], "r-", label="maximum")
    ax.plot(p1s[stable], averages[stable], "g-", label="average")
    # ax.plot(p1s[stable], minima[stable], "b-", label="minimum")

    # unstable branches
    unstable = ~stable
    # ax.plot(p1s[unstable], maxima[unstable], "r--")
    ax.plot(p1s[unstable], averages[unstable], "r--")
    # ax.plot(p1s[unstable], minima[unstable], "b--")

    ax.set_title(
        rf"$x_0  \approx ({stable_point[0]:.2f}, {stable_point[1]:.1f}, {stable_point[2]:.1f})$"
    )


############### Bifurcation points ##########################
print("\nBifurcation points:")
pprint(bifurcations)

# save bifurcations to file as json object
filename = "mcm_bifurcations.json"
filepath = data_dir / filename
for bif in bifurcations:
    solution = bif["solution"].tolist()
    parameter = bif["parameter"].tolist()
    out = dict(solution=solution, parameter=parameter)
    with open(filepath, "w") as f:
        json.dump(out, f, indent=4)


############### retrieve limit cycles and add to continuation plot ##################
# load limit cycles from file
filename = "mcm_limit_cycles.json"
filepath = data_dir / filename
with open(filepath, "r") as f:
    data = json.load(f)
    solutions_l = data["points"]
    parameters_l = data["parameter"]
    periods_l = data["periods"]
    cycles_l = data["cycles"]
    stable_l = data["stable"]

num_doublings = len(solutions_l)
print(f"\nNumber period doublings: {num_doublings}")
for i in range(num_doublings):
    stable = np.array(stable_l[i]).astype(bool)
    parameters = np.array(parameters_l[i])
    solutions = np.array(solutions_l[i])
    # stable limit cycles
    if i == 0:
        axs[0].plot(
            parameters[stable],
            np.mean(solutions[stable], 1),
            "g-",
            label="stable limit cycle",
            linewidth=3,
        )
    else:
        axs[0].plot(
            parameters[stable],
            np.mean(solutions[stable], 1),
            "g-",
            linewidth=3,
        )

        # unstable limit cycles
    if i == 0:
        axs[0].plot(
            parameters[~stable],
            np.mean(solutions[~stable], 1),
            "r--",
            label="unstable limit cycle",
            linewidth=3,
        )
    else:
        axs[0].plot(
            parameters[~stable],
            np.mean(solutions[~stable], 1),
            "r--",
            linewidth=3,
        )
    
    # plot period doubling bifurcation
    if i >= 1:
        cycle_point = solutions[0]
        cycle_parameter = parameters[0]
        axs[0].plot(cycle_parameter, np.mean(cycle_point), "ko")
        axs[0].annotate(
            f"PD{i}",
            (cycle_parameter, np.mean(cycle_point)),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

axs[0].set_ylabel("$x$")
axs[0].set_xlabel("$p_1$")
axs[0].legend(fontsize=8, loc="lower left")


fig.suptitle("Continuation of the stable stationary point(s)")
fig.tight_layout()
fname = output_dir / "mcm_continuation.png"
plt.savefig(fname, bbox_inches="tight", dpi=300)
