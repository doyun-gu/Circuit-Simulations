"""
Dynamic phasor demo script
--------------------------

Features:
- Generate different test signals:
  1) Pure sinusoid
  2) Slowly varying amplitude
  3) Fast amplitude step
  4) Sinusoid + high-frequency switching
- Compute the fundamental dynamic phasor using a sliding window integral
- Reconstruct the time-domain signal from the dynamic phasor
- Plot:
    - original vs reconstructed waveform
    - |phasor(t)| and angle(phasor(t))
    - error(t) = x(t) - x_rec(t)
- Save all figures into ./fig directory.
- Print RMS and relative errors for each case
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Core utilities
# -------------------------------------------------------------------------

def generate_time_vector(t_end=0.2, dt=1e-5):
    """Generate time vector."""
    t = np.arange(0.0, t_end + dt, dt)
    return t, dt


def generate_signal(case_name, t, f0=50.0):
    """
    Generate test signals for different cases.
    Returns: signal x(t), description string.
    """
    omega0 = 2 * np.pi * f0

    if case_name == "pure_sine":
        # x(t) = sin(ω0 t)
        x = np.sin(omega0 * t)
        desc = "Case 1: Pure sinusoid (constant amplitude)"
    
    elif case_name == "slow_amplitude":
        # Amplitude slowly ramps from 1 to 2 over the full simulation
        A = 1.0 + t / t[-1]  # from 1 to 2
        x = A * np.sin(omega0 * t)
        desc = "Case 2: Slowly varying amplitude"

    elif case_name == "fast_step":
        # Amplitude suddenly steps from 1 to 2 at t_step
        t_step = 0.05
        A = np.ones_like(t)
        A[t >= t_step] = 2.0
        x = A * np.sin(omega0 * t)
        desc = "Case 3: Fast amplitude step at t = 50 ms"

    elif case_name == "switching":
        # Fundamental + high-frequency PWM-like switching
        f_sw = 2000.0  # 2 kHz switching
        rng = np.random.default_rng(0)

        carrier = np.sign(np.sin(2 * np.pi * f_sw * t))
        # Randomly zero-out some switching intervals (like missing pulses)
        mask = (rng.random(len(t)) > 0.2).astype(float)

        base = np.sin(omega0 * t)
        x = base * carrier * mask
        desc = "Case 4: Sinusoid with 2 kHz switching + random pulse drop"
    
    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    return x, desc


def dynamic_phasor(x, t, f0=50.0, T0=None, window_periods=1.0):
    """
    Compute the fundamental dynamic phasor X(t) of a real signal x(t)
    using a sliding window integral over 'window_periods' of the fundamental.

    X(t) ≈ (2/Tw) ∫_{t}^{t+Tw} x(τ) e^{-j ω0 τ} dτ

    where Tw = window_periods * T0.

    Returns:
        t_mid  : time vector at the center of each window
        X      : complex dynamic phasor
    """
    if T0 is None:
        T0 = 1.0 / f0
    omega0 = 2 * np.pi * f0

    dt = t[1] - t[0]
    Tw = window_periods * T0
    Nw = int(round(Tw / dt))  # window length in samples

    if Nw < 2:
        raise ValueError("Window too short; increase window_periods or use smaller dt.")

    N = len(t)
    M = N - Nw  # number of valid windows
    if M <= 0:
        raise ValueError("Signal too short for given window length.")

    X = np.zeros(M, dtype=complex)

    for k in range(M):
        idx = slice(k, k + Nw)
        tau = t[idx]
        x_win = x[idx]
        # Numerical integration approximation
        integrand = x_win * np.exp(-1j * omega0 * tau)
        X[k] = (2.0 / Tw) * np.sum(integrand) * dt

    # Time at the center of each window
    t_mid = t[:M] + 0.5 * Tw

    return t_mid, X


def reconstruct_from_phasor(t_mid, X, f0=50.0):
    """
    Reconstruct the real time-domain signal from its dynamic phasor:
        x_rec(t) = Re{ X(t) e^{j ω0 t} }
    """
    omega0 = 2 * np.pi * f0
    x_rec = np.real(X * np.exp(1j * omega0 * t_mid))
    return x_rec


def compute_error_metrics(x_ref, x_rec):
    """
    Compute RMS error and relative RMS error (in %) between two signals on a common grid.
    Assumes x_ref and x_rec have the same length.
    """
    err = x_ref - x_rec
    rms_err = np.sqrt(np.mean(err**2))
    rms_ref = np.sqrt(np.mean(x_ref**2))
    if rms_ref > 0:
        rel_err = 100.0 * rms_err / rms_ref
    else:
        rel_err = np.nan
    return err, rms_err, rel_err

# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def plot_case(t, x, t_mid, X, x_rec, err, desc, case_name, out_dir="fig"):
    """
    Create and save 3 plots for a given case:
      1) Time-domain: x(t) and x_rec(t)
      2) |X(t)| and angle(X(t))
      3) Error(t)

    All figures are saved into the directory specified by out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --------- 1) Waveform vs reconstructed ----------
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, label="Original x(t)", linewidth=1)
    plt.plot(t_mid, x_rec, label="Reconstructed from dynamic phasor",
             linewidth=1, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"{desc} - Time-domain waveform")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    fig1 = plt.gcf()
    fig1_path = os.path.join(out_dir, f"{case_name}_waveform.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --------- 2) Phasor magnitude & angle ----------
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_mid, np.abs(X))
    plt.ylabel("|X(t)|")
    plt.grid(True)
    plt.title(f"{desc} - Dynamic phasor magnitude & angle")

    plt.subplot(2, 1, 2)
    angle_unwrapped = np.unwrap(np.angle(X))
    plt.plot(t_mid, angle_unwrapped)
    plt.xlabel("Time [s]")
    plt.ylabel("angle(X(t)) [rad]")
    plt.grid(True)

    plt.tight_layout()

    fig2 = plt.gcf()
    fig2_path = os.path.join(out_dir, f"{case_name}_phasor_mag_angle.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # --------- 3) Error ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t_mid, err, linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Error = x(t) - x_rec(t)")
    plt.title(f"{desc} - Reconstruction error")
    plt.grid(True)
    plt.tight_layout()

    fig3 = plt.gcf()
    fig3_path = os.path.join(out_dir, f"{case_name}_error.png")
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)


# -------------------------------------------------------------------------
# Main demo
# -------------------------------------------------------------------------

def main():
    # Fundamental frequency
    f0 = 50.0
    T0 = 1.0 / f0

    # Time grid
    t, dt = generate_time_vector(t_end=0.2, dt=2e-5)  # 200 ms

    # Cases to run
    cases = ["pure_sine", "slow_amplitude", "fast_step", "switching"]

    # Choose window length in terms of fundamental cycles
    window_periods = 1.0  # 1 period; try 0.5, 2.0, etc. to see the effect

    summary = []

    for case_name in cases:
        x, desc = generate_signal(case_name, t, f0=f0)

        # Compute dynamic phasor
        t_mid, X = dynamic_phasor(x, t, f0=f0, T0=T0, window_periods=window_periods)

        # Reconstruct time-domain signal on t_mid grid
        x_rec = reconstruct_from_phasor(t_mid, X, f0=f0)

        # For error, compare x(t) sampled at t_mid vs x_rec(t_mid)
        # Find indices in original t that correspond to t_mid (nearest)
        idx_mid = np.searchsorted(t, t_mid)
        idx_mid = np.clip(idx_mid, 0, len(t) - 1)
        x_ref_mid = x[idx_mid]

        err, rms_err, rel_err = compute_error_metrics(x_ref_mid, x_rec)

        summary.append((case_name, desc, rms_err, rel_err))

        # Plot and save for this case
        plot_case(t, x, t_mid, X, x_rec, err, desc, case_name, out_dir="fig")

    # Print summary to console
    print("=== Dynamic Phasor Reconstruction Error Summary ===")
    print(f"(Window length = {window_periods} * T0, T0 = {T0:.6f} s)")
    for case_name, desc, rms_err, rel_err in summary:
        print(f"{case_name:12s} | RMS error = {rms_err:.4e},  Relative error = {rel_err:7.3f} %")
        print(f"    -> {desc}")


if __name__ == "__main__":
    main()
