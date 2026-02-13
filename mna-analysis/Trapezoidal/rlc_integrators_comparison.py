# rlc_integrator_comparison.py
#
# Compare Explicit Euler, Backward Euler, and Trapezoidal
# on a simple series RLC circuit (no source, initial energy in C).
#
# State: x = [i_L, v_C]^T
# Equations:
#   L di_L/dt + R i_L + v_C = 0
#   C dv_C/dt - i_L = 0
#
# Run:
#   python rlc_integrator_comparison.py
#
# You will get two figures:
#   1) Capacitor voltage v_C(t)
#   2) Inductor current i_L(t)
#
# Each figure shows:
#   - Reference solution (trapezoidal with small dt)
#   - Explicit Euler (coarse dt)
#   - Backward Euler (coarse dt)
#   - Trapezoidal (coarse dt)

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Define RLC parameters and system matrix
# ------------------------------------------------------------

# Series RLC parameters
L = 1e-3   # Inductance [H]
C = 1e-6   # Capacitance [F]
R = 5.0    # Resistance [Ohm] (light damping)

# State x = [i_L, v_C]^T, dynamics x' = A x
A = np.array([
    [-R / L,  -1.0 / L],
    [ 1.0 / C, 0.0    ]
])

print("System matrix A =\n", A)


# ------------------------------------------------------------
# 2. Time grids
# ------------------------------------------------------------

# Coarse time step for comparison
dt = 1e-4      # 0.1 ms
t_end = 2e-3   # 2 ms ~ several oscillation periods
t = np.arange(0.0, t_end + dt, dt)

# Small time step for reference solution
dt_ref = dt / 40.0
t_ref = np.arange(0.0, t_end + dt_ref, dt_ref)


# ------------------------------------------------------------
# 3. Step matrices for linear system x' = A x
# ------------------------------------------------------------

def step_matrix_explicit_euler(A, h):
    """
    For x' = A x, explicit Euler:
        x_{n+1} = (I + h A) x_n
    """
    I = np.eye(A.shape[0])
    return I + h * A


def step_matrix_backward_euler(A, h):
    """
    For x' = A x, backward Euler:
        x_{n+1} = (I - h A)^{-1} x_n
    """
    I = np.eye(A.shape[0])
    return np.linalg.inv(I - h * A)


def step_matrix_trapezoidal(A, h):
    """
    For x' = A x, trapezoidal (Crank–Nicolson):
        x_{n+1} = (I - h/2 A)^{-1} (I + h/2 A) x_n
    """
    I = np.eye(A.shape[0])
    return np.linalg.inv(I - 0.5 * h * A) @ (I + 0.5 * h * A)


def simulate_linear_system(x0, t_grid, step_mat):
    """
    Simulate x' = A x using a constant step matrix.

    Parameters
    ----------
    x0 : (2,) array
        Initial state.
    t_grid : (N,) array
        Time grid.
    step_mat : (2,2) array
        Step matrix such that x_{n+1} = step_mat @ x_n.

    Returns
    -------
    x : (N,2) array
        State trajectory over t_grid.
    """
    x = np.zeros((len(t_grid), len(x0)))
    x[0] = x0
    for n in range(len(t_grid) - 1):
        x[n + 1] = step_mat @ x[n]
    return x


# ------------------------------------------------------------
# 4. Run simulations
# ------------------------------------------------------------

# Initial condition: capacitor charged to 1 V, inductor current 0 A
x0 = np.array([0.0, 1.0])

# Step matrices for coarse step
M_eu = step_matrix_explicit_euler(A, dt)
M_be = step_matrix_backward_euler(A, dt)
M_tr = step_matrix_trapezoidal(A, dt)

# Reference step matrix (small dt, trapezoidal)
M_ref = step_matrix_trapezoidal(A, dt_ref)

# Simulations
x_eu = simulate_linear_system(x0, t, M_eu)
x_be = simulate_linear_system(x0, t, M_be)
x_tr = simulate_linear_system(x0, t, M_tr)
x_ref = simulate_linear_system(x0, t_ref, M_ref)


# ------------------------------------------------------------
# 5. Plot capacitor voltage
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))
plt.plot(t_ref * 1e3, x_ref[:, 1], label="Reference (Trapezoidal, small dt)", linewidth=2)
plt.plot(t * 1e3, x_eu[:, 1], "o-", label="Explicit Euler (coarse dt)")
plt.plot(t * 1e3, x_be[:, 1], "s-", label="Backward Euler (coarse dt)")
plt.plot(t * 1e3, x_tr[:, 1], "^-", label="Trapezoidal (coarse dt)")

plt.xlabel("Time [ms]")
plt.ylabel("Capacitor voltage v_C [arb]")
plt.title("Series RLC: v_C(t) with different integrators")
plt.grid(True)
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------
# 6. Plot inductor current
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))
plt.plot(t_ref * 1e3, x_ref[:, 0], label="Reference (Trapezoidal, small dt)", linewidth=2)
plt.plot(t * 1e3, x_eu[:, 0], "o-", label="Explicit Euler (coarse dt)")
plt.plot(t * 1e3, x_be[:, 0], "s-", label="Backward Euler (coarse dt)")
plt.plot(t * 1e3, x_tr[:, 0], "^-", label="Trapezoidal (coarse dt)")

plt.xlabel("Time [ms]")
plt.ylabel("Inductor current i_L [arb]")
plt.title("Series RLC: i_L(t) with different integrators")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
