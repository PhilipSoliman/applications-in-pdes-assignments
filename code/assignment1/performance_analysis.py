import numpy as np
import matplotlib.pyplot as plt
from custom_pyutils import pyutils
from non_linear_system import NonLinearSystem
from timeit import timeit
from tqdm import tqdm

root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

pyutils.set_style()

timing_loops = 1000
n_min = 5  # minimum n for performance analysis
n_max = 80  # maximum n for performance analysis (do not set higher than 80 to prevent overflow in scipy.special._orthogonal)
n_values = np.arange(n_min, n_max + 1, 1, dtype=int)
F_times = np.zeros(n_values.size)
F_T_times = np.zeros(n_values.size)
progress_msg = "n = {0:d}, F (s) = {1:2e}, F_T (s) = {2:2e}"
pbar = tqdm(enumerate(n_values), desc="", total=n_max, unit="run", miniters=10)

print("Running NLS evaluation time for different n...")
for i, n in pbar:
    NLS = NonLinearSystem(n, 2 * n, 100)
    F_time = timeit(
        "NLS.evaluate()",
        globals=globals(),
        number=timing_loops,
    )
    F_times[i] = F_time
    F_T_time = timeit(
        "NLS.evaluate_derivative()",
        globals=globals(),
        number=timing_loops,
    )
    F_T_times[i] = F_T_time

    # progress using tqdm
    pbar.set_description(progress_msg.format(n, F_time, F_T_time))

plt.plot(n_values, F_times / timing_loops, label="$F(T)$")
plt.plot(n_values, F_T_times / timing_loops, label="$F_T(T)$")
plt.xlabel("n")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Averaged evaluation time for $F(T)$ and $F_T(T)$")
fn = output_dir / f"function_evaluation_time_n={n_values[-1]}.png"
plt.savefig(fn, dpi=500)
print("Saved function evaluation time plot to:\n  ", fn)
