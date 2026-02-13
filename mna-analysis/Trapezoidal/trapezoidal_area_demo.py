# trapezoidal_area_demo.py
#
# Visual demo of numerical integration:
#   - Rectangle (explicit Euler-style) rule
#   - Trapezoidal rule
#
# Function: f(t) = sin(t) + 1 over [0, T]
#
# Figures:
#   1) Areas for each sub-interval:
#      - Rectangles (left Riemann sum)
#      - Trapezoids (trapezoidal rule)
#   2) Cumulative integral vs t:
#      - Exact integral
#      - Rectangle approximation
#      - Trapezoidal approximation
#
# Run with:
#   python trapezoidal_area_demo.py

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Define function and exact integral
# ------------------------------------------------------------

def f(t):
    """Function to integrate."""
    return np.sin(t) + 1.0


def F_exact(t):
    """
    Exact antiderivative of f(t) = sin(t) + 1:
        ∫ (sin(s) + 1) ds = -cos(s) + s + C
    Here we use F(0) = 0, so C = 1.
    """
    return -np.cos(t) + t + 1.0


# Integration interval
T = 2.0 * np.pi   # one full period
N = 8             # number of sub-intervals
ts = np.linspace(0.0, T, N + 1)   # grid points
h = ts[1] - ts[0]

# Fine grid for plotting the true curve
t_fine = np.linspace(0.0, T, 400)
f_fine = f(t_fine)


# ------------------------------------------------------------
# 2. Rectangle rule and trapezoidal rule values
# ------------------------------------------------------------

# Rectangle rule (left endpoints)
rect_areas = []  # area for each sub-interval
for n in range(N):
    t_left = ts[n]
    rect_areas.append(h * f(t_left))
rect_areas = np.array(rect_areas)

I_rect = rect_areas.sum()

# Trapezoidal rule
trap_areas = []
for n in range(N):
    t_left = ts[n]
    t_right = ts[n + 1]
    trap_areas.append(0.5 * h * (f(t_left) + f(t_right)))
trap_areas = np.array(trap_areas)

I_trap = trap_areas.sum()

# Exact integral over [0,T]
I_exact = F_exact(T) - F_exact(0.0)

print("Rectangle approx  I_rect =", I_rect)
print("Trapezoidal  approx I_trap =", I_trap)
print("Exact integral       I_exact =", I_exact)


# ------------------------------------------------------------
# 3. Plot 1: geometric meaning (areas)
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))

# Plot the true function
plt.plot(t_fine, f_fine, label="f(t)")

# Draw rectangles for rectangle rule (left endpoints)
for n in range(N):
    t_left = ts[n]
    height = f(t_left)
    # use fill_between for the rectangle
    plt.fill_between([t_left, t_left + h], [height, height], [0, 0],
                     alpha=0.2)

# Draw trapezoids for trapezoidal rule (as outlines)
for n in range(N):
    t_left = ts[n]
    t_right = ts[n + 1]
    y_left = f(t_left)
    y_right = f(t_right)
    # outline of trapezoid
    plt.plot([t_left, t_left, t_right, t_right],
             [0, y_left, y_right, 0])

plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Rectangle areas vs trapezoidal areas for ∫ f(t) dt")
plt.grid(True)
plt.tight_layout()


# ------------------------------------------------------------
# 4. Plot 2: cumulative integral vs t
# ------------------------------------------------------------

# Exact cumulative integral
F_fine = F_exact(t_fine) - F_exact(0.0)

# Cumulative rectangle and trapezoidal sums (piecewise constant / linear)
I_rect_cum = np.zeros_like(ts)
I_trap_cum = np.zeros_like(ts)
for n in range(N):
    I_rect_cum[n + 1] = I_rect_cum[n] + rect_areas[n]
    I_trap_cum[n + 1] = I_trap_cum[n] + trap_areas[n]

plt.figure(figsize=(9, 4))

plt.plot(t_fine, F_fine, label="Exact integral", linewidth=2)

# Step-like rectangle cumulative
plt.step(ts, I_rect_cum, where="post", label="Rectangle rule")

# Piecewise linear trapezoidal cumulative
plt.plot(ts, I_trap_cum, "o-", label="Trapezoidal rule")

plt.xlabel("t")
plt.ylabel("Integral from 0 to t")
plt.title("Cumulative integral: exact vs rectangle vs trapezoidal")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
