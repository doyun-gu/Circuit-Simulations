# compare_integrators.py
#
# Compare Explicit Euler, Backward Euler and Trapezoidal
# on a simple ODE x' = -x, x(0)=1.
#
# Run with:
#   python compare_integrators.py
#
# It will pop up two figures:
#   1) Solutions vs exact
#   2) Geometric meaning of trapezoidal vs explicit Euler

import numpy as np
import matplotlib.pyplot as plt


# ODE: x' = -x
def f(t, x):
    return -x


def explicit_euler(f, x0, t_grid):
    """Forward (explicit) Euler method."""
    x = np.zeros_like(t_grid)
    x[0] = x0
    h = t_grid[1] - t_grid[0]
    for n in range(len(t_grid) - 1):
        x[n + 1] = x[n] + h * f(t_grid[n], x[n])
    return x


def backward_euler(x0, t_grid):
    """Backward Euler method for x' = -x (closed-form step)."""
    x = np.zeros_like(t_grid)
    x[0] = x0
    h = t_grid[1] - t_grid[0]
    # For x' = -x, backward Euler step: x_{n+1} = x_n / (1 + h)
    for n in range(len(t_grid) - 1):
        x[n + 1] = x[n] / (1.0 + h)
    return x


def trapezoidal(x0, t_grid):
    """Trapezoidal (Crank–Nicolson) method for x' = -x (closed-form step)."""
    x = np.zeros_like(t_grid)
    x[0] = x0
    h = t_grid[1] - t_grid[0]
    # For x' = -x, trapezoidal step:
    # x_{n+1} = ((1 - h/2) / (1 + h/2)) * x_n
    factor = (1.0 - 0.5 * h) / (1.0 + 0.5 * h)
    for n in range(len(t_grid) - 1):
        x[n + 1] = factor * x[n]
    return x


def main():
    # Time grid (coarse on purpose to show difference)
    t = np.linspace(0.0, 5.0, 11)  # h = 0.5
    x0 = 1.0

    # Exact solution
    x_exact = np.exp(-t)

    # Numerical methods
    x_euler = explicit_euler(f, x0, t)
    x_be = backward_euler(x0, t)
    x_tr = trapezoidal(x0, t)

    # ---------------------------------------------------
    # Figure 1: methods vs exact
    # ---------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, x_exact, label="Exact", linewidth=2)
    plt.plot(t, x_euler, "o-", label="Explicit Euler")
    plt.plot(t, x_be, "s-", label="Backward Euler")
    plt.plot(t, x_tr, "^-", label="Trapezoidal")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Comparison of integration methods for x' = -x (h = 0.5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------------
    # Figure 2: geometric meaning for one step [0,1]
    # ---------------------------------------------------
    t_step = np.linspace(0.0, 1.0, 200)
    f_vals = np.exp(-t_step)  # here f(t) = e^{-t} = -x' with x(0)=1

    plt.figure(figsize=(8, 4.5))
    plt.plot(t_step, f_vals, label="f(t) = e^{-t}", linewidth=2)

    # True area under curve (just for visualization)
    plt.fill_between(t_step, f_vals, alpha=0.2, label="True area")

    # Explicit Euler rectangle: height = f(0) = 1
    plt.fill_between([0, 1], [1, 1], [0, 0],
                     step="pre", alpha=0.2, label="Explicit Euler rectangle")

    plt.xlabel("t over one step [0, 1]")
    plt.ylabel("f(t)")
    plt.title("Area under f(t) vs explicit Euler rectangle\n"
              "(Trapezoidal uses the average of f(0) and f(1))")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
