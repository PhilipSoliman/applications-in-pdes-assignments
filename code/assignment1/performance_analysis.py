import numpy as np
import matplotlib.pyplot as plt
from custom_pyutils import pyutils
from non_linear_system import NonLinearSystem
from timeit import timeit

root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

pyutils.set_style()
timing_loops = 80

# plot function evaluation time for F(T) and F_T(T) for different n
plt.close()
print("Running NLS evaluation time for different n...")
n_values = np.arange(5, 101, 1, dtype=int)
F_times = np.zeros(n_values.size)
F_T_times = np.zeros(n_values.size)
for i, n in enumerate(n_values):
    NLS = NonLinearSystem(n, 2 * n, 100) 
    F_times[i] = timeit(
        "NLS.evaluate()",
        globals=globals(),
        number=timing_loops,
    )
    F_T_times[i] = timeit(
        "NLS.evaluate_derivative()",
        globals=globals(),
        number=timing_loops,
    )
plt.plot(n_values, F_times, label="$F(T)$")
plt.plot(n_values, F_T_times, label="$F_T(T)$")
plt.xlabel("n")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Averaged evaluation time for $F(T)$ and $F_T(T)$")
fn = output_dir / f"function_evaluation_time_n={n_values[-1]}.png"
plt.savefig(fn, dpi=500)
print("Saved function evaluation time plot to:\n  ", fn)
