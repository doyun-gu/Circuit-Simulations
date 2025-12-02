import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# ======================
# 1. Parameters
# ======================
f0 = 50.0                 # fundamental freq [Hz]
w0 = 2 * np.pi * f0
T0 = 1.0 / f0

R = 10.0                  # [ohm]
L = 10e-3                 # [H]
C = 100e-6                # [F]

V_amp = 10.0              # source amplitude

t_end = 0.20              # total simulation time [s]
dt_td = 1e-5              # time-domain step (small → expensive)


# ======================
# 2. Source with "extreme" switching
# ======================
def gate(t):
    """
    Simple on/off gate to mimic extreme switching.
    ON:   0–4T0, 16–20T0
    OFF:  middle part
    """
    on1 = (t >= 0.0) & (t < 4 * T0)
    on2 = (t >= 16 * T0) & (t < 20 * T0)
    return np.where(on1 | on2, 1.0, 0.0)


def v_s(t):
    return V_amp * np.sin(w0 * t) * gate(t)


# ======================
# 3. Time-domain simulation (explicit Euler, RLC)
#    state x = [i_L, v_C]
# ======================
def simulate_time_domain():
    n_steps = int(t_end / dt_td)
    t = np.linspace(0.0, t_end, n_steps + 1)

    x = np.zeros((n_steps + 1, 2))  # [i_L, v_C]

    start = time.perf_counter()
    for k in range(n_steps):
        i_L, v_C = x[k]
        vs = v_s(t[k])

        di_dt = (vs - R * i_L - v_C) / L
        dv_dt = i_L / C

        x[k + 1, 0] = i_L + dt_td * di_dt
        x[k + 1, 1] = v_C + dt_td * dv_dt

    elapsed = time.perf_counter() - start
    return t, x, elapsed


# ======================
# 4. Dynamic phasor (post-processing of i_L)
#    Fundamental component, sliding window of length T0
# ======================
def compute_dynamic_phasor(t, i_L):
    # we don't need phasor at every time step → subsample for animation
    stride = max(1, int(T0 / dt_td / 20))   # ≈ 20 phasor points per period

    window_samples = int(T0 / dt_td)
    if window_samples < 5:
        raise ValueError("dt_td is too large compared to T0.")

    X_list = []
    valid_indices = []

    start = time.perf_counter()
    # dynamic phasor is only defined after first full window
    for idx in range(window_samples, len(t), stride):
        tt = t[idx - window_samples:idx]
        ii = i_L[idx - window_samples:idx]
        integrand = ii * np.exp(-1j * w0 * tt)
        # np.trapz 은 deprecated 경고 → trapezoid로 변경
        Xk = (1.0 / T0) * np.trapezoid(integrand, tt)
        X_list.append(Xk)
        valid_indices.append(idx)

    elapsed = time.perf_counter() - start

    t_dp = t[np.array(valid_indices)]
    X_dp = np.array(X_list, dtype=complex)
    mag_dp = np.abs(X_dp)
    phase_dp = np.angle(X_dp) * 180 / np.pi

    return t_dp, X_dp, mag_dp, phase_dp, elapsed


# ======================
# 5. Run simulations
# ======================
t, x, cost_td = simulate_time_domain()
i_L = x[:, 0]

t_dp, X_dp, mag_dp, phase_dp, cost_dp = compute_dynamic_phasor(t, i_L)

print(f"Time-domain simulation cost   : {cost_td:.4f} s")
print(f"Dynamic phasor processing cost: {cost_dp:.4f} s")

# ----------------------
# 5-1. Reconstruct fundamental from dynamic phasor
#      x1(t) ≈ 2 * Re{ X1(t) e^{j w0 t} }
# ----------------------
# interpolate phasor to full time resolution
Xr = np.interp(t, t_dp, X_dp.real, left=0.0, right=X_dp.real[-1])
Xi = np.interp(t, t_dp, X_dp.imag, left=0.0, right=X_dp.imag[-1])
X_interp = Xr + 1j * Xi

i_rec = 2.0 * np.real(X_interp * np.exp(1j * w0 * t))  # fundamental component


# ======================
# 6. Animation
# ======================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax1, ax2, ax3 = axes

# --- top: time-domain current & reconstructed fundamental ---
ax1.plot(t, i_L, lw=1, label="Time-domain i_L(t)")
ax1.plot(t, i_rec, lw=1, linestyle="--", label="Reconstructed fundamental")
line_td = ax1.axvline(t_dp[0], ls="--")   # moving time marker
ax1.set_ylabel("i_L(t) [A]")
ax1.grid(True)
ax1.legend(loc="upper right")
ax1.set_xlim(t[0], t[-1])

# --- middle: phasor magnitude ---
line_mag, = ax2.plot([], [], lw=2)
ax2.set_ylabel("|I_dyn(t)|")
ax2.grid(True)
ax2.set_xlim(t_dp[0], t_dp[-1])
ax2.set_ylim(0, 1.1 * mag_dp.max())

# --- bottom: phasor phase ---
line_phase, = ax3.plot([], [], lw=2)
ax3.set_ylabel("∠I_dyn(t) [deg]")
ax3.set_xlabel("Time [s]")
ax3.grid(True)
ax3.set_xlim(t_dp[0], t_dp[-1])
ax3.set_ylim(1.1 * min(0, phase_dp.min()),
             1.1 * max(0, phase_dp.max()))

fig.suptitle("Time-domain vs Dynamic Phasor (fundamental component)")


def init():
    # 아래 두 plot 은 처음엔 비워두기
    line_mag.set_data([], [])
    line_phase.set_data([], [])
    # 수직선은 첫 phasor 시점에 맞추기
    line_td.set_xdata([t_dp[0], t_dp[0]])
    return line_td, line_mag, line_phase


def update(frame):
    # frame 번째 phasor까지 표시
    t_now = t_dp[frame]

    # 위쪽 수직선 위치
    line_td.set_xdata([t_now, t_now])

    # magnitude / phase trace
    line_mag.set_data(t_dp[: frame + 1], mag_dp[: frame + 1])
    line_phase.set_data(t_dp[: frame + 1], phase_dp[: frame + 1])

    return line_td, line_mag, line_phase


ani = FuncAnimation(
    fig,
    update,
    frames=len(t_dp),
    init_func=init,
    interval=30,   # ms per frame
    blit=True,
)

plt.tight_layout()
plt.show()

# ======================
# 7. Static snapshot figure (for report/PPT)
# ======================
fig_s, (ax1s, ax2s, ax3s) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# --- top: time-domain + reconstructed fundamental ---
ax1s.plot(t, i_L, lw=1, label="Time-domain i_L(t)")
ax1s.plot(t, i_rec, lw=1, linestyle="--", label="Reconstructed fundamental")
ax1s.set_ylabel("i_L(t) [A]")
ax1s.grid(True)
ax1s.legend(loc="upper right")
ax1s.set_xlim(t[0], t[-1])

# --- middle: dynamic phasor magnitude ---
ax2s.plot(t_dp, mag_dp, lw=2)
ax2s.set_ylabel("|I_dyn(t)|")
ax2s.grid(True)

# --- bottom: dynamic phasor phase ---
ax3s.plot(t_dp, phase_dp, lw=2)
ax3s.set_ylabel("∠I_dyn(t) [deg]")
ax3s.set_xlabel("Time [s]")
ax3s.grid(True)

fig_s.suptitle("Time-domain vs Dynamic Phasor (fundamental component)")

fig_s.tight_layout()
fig_s.savefig("time_vs_dynamic_phasor.png", dpi=300, bbox_inches="tight")
plt.close(fig_s)
