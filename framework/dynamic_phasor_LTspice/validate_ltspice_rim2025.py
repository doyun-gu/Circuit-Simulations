#!/usr/bin/env python3
"""
Validate framework against the actual LTspice simulation for the Rim 2025 circuit.

Loads:
  - LTspice .raw file: ltspice_simulations/rim2025/rim2025.raw
  - Framework (time-domain + phasor-domain) via NetlistCircuit

Key note: the LTspice netlist has no explicit GND symbol (node 0).
LTspice therefore measures all node voltages relative to an internal
floating reference.  We must compare differential voltages:
    V_out_lt  = V(n003) - V(n004)   (output voltage relative to N004)
    V_out_fw  = V(N003)              (framework: N004 mapped to ground)
    I_lt      = I(L1)               (same sign convention both sides)

Run from the framework/ directory:
    python dynamic_phasor_LTspice/validate_ltspice_rim2025.py

Author: Doyun Gu (University of Manchester)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for script mode
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from core.ltspice_raw_reader import read_raw
from core.netlist_parser import parse_ltspice_netlist
from core.mna_circuit import NetlistCircuit

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_FILE    = os.path.join(SCRIPT_DIR, 'ltspice_simulations', 'rim2025', 'rim2025.raw')
OUTPUT_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), 'notebooks', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load LTspice results ───────────────────────────────────────────
print("=" * 70)
print("LTspice vs Framework Validation — Rim 2025 RLC Circuit")
print("=" * 70)

print(f"\n1. Loading LTspice .raw file: {RAW_FILE}")
if not os.path.exists(RAW_FILE):
    print(f"   ERROR: File not found. Run LTspice first.")
    sys.exit(1)

raw = read_raw(RAW_FILE)
print(raw.summary())

t_lt = np.real(raw.time)
V_n003 = raw.voltage('N003')   # V(n003) w.r.t. internal LTspice ref
V_n004 = raw.voltage('N004')   # V(n004) w.r.t. internal LTspice ref
V_n001 = raw.voltage('N001')
I_lt   = raw.current('L1')

# Differential voltages (relative to N004 = framework ground)
V_lt       = V_n003 - V_n004   # output voltage
V_src_lt   = V_n001 - V_n004   # source voltage check

print(f"\n   Source check: V(n001)-V(n004) peak = {np.max(np.abs(V_src_lt)):.4f} V  (expect 1.000 V)")
print(f"   Output V(n003)-V(n004): range [{V_lt.min():.4f}, {V_lt.max():.4f}] V")
print(f"   I(L1): range [{I_lt.min():.6f}, {I_lt.max():.6f}] A")

# ── 2. Framework simulation ───────────────────────────────────────────
print("\n2. Framework simulation")

# Netlist with N004 mapped to ground (0)
NETLIST_TEXT = """
* Rim et al. (2025) RLC — exact LTspice circuit (N004 → 0)
V1 N001 0 SINE(0 1 92.3k 0 0 0)
R1 N002 N001 3
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.tran 0 0.5m 0 1u
.end
"""

netlist = parse_ltspice_netlist(NETLIST_TEXT)
circuit = NetlistCircuit(netlist)

# Time-domain
td = circuit.solve_time_domain()
t_fw = td['t']
V_fw = td['V(N003)']
I_fw = td['I(L1)']

print(f"   Time-domain: {len(t_fw)} points, V(N003) range [{V_fw.min():.4f}, {V_fw.max():.4f}] V")

# Phasor-domain
omega_s = 2 * np.pi * 92.3e3
circuit.configure_phasor(omega_s=omega_s)
pd_res = circuit.solve_phasor_domain()
t_pd = pd_res['t']
V_pd = pd_res['V(N003)']
I_pd = pd_res['I(L1)']
V_env = pd_res['envelopes']['V(N003)']
I_env = pd_res['envelopes']['I(L1)']

print(f"   Phasor-domain: {len(t_pd)} points, V envelope SS = {V_env[-1]:.4f} V")

# ── 3. Analytical steady-state values ────────────────────────────────
L = 100.04e-6; C = 30.07e-9; Rs = 3.0; Ro = 2000.0
omega = 2 * np.pi * 92.3e3
Z_CR = Ro / (1 + 1j * omega * Ro * C)
Z_series = Rs + 1j * omega * L + Z_CR
I_analytical = 1.0 / abs(Z_series)
V_analytical = I_analytical * abs(Z_CR)
print(f"\n   Analytical (steady-state): V_out = {V_analytical:.4f} V, I = {I_analytical:.6f} A")

# ── 4. Common time base & interpolation ──────────────────────────────
print("\n3. Interpolating to common time base")
t_common = np.linspace(max(t_fw[0], t_lt[0]), min(t_fw[-1], t_lt[-1]), 10000)

V_lt_c  = interp1d(t_lt, V_lt,  kind='cubic', fill_value='extrapolate')(t_common)
I_lt_c  = interp1d(t_lt, I_lt,  kind='cubic', fill_value='extrapolate')(t_common)
V_fw_c  = interp1d(t_fw, V_fw,  kind='cubic', fill_value='extrapolate')(t_common)
I_fw_c  = interp1d(t_fw, I_fw,  kind='cubic', fill_value='extrapolate')(t_common)
V_pd_c  = interp1d(t_pd, V_pd,  kind='cubic', fill_value='extrapolate')(t_common)
I_pd_c  = interp1d(t_pd, I_pd,  kind='cubic', fill_value='extrapolate')(t_common)

# ── 5. Error metrics ─────────────────────────────────────────────────

def nrmse(ref, test):
    rmse = np.sqrt(np.mean((ref - test) ** 2))
    rng  = np.ptp(ref)
    return rmse / rng * 100 if rng > 0 else 0.0

def peak_err(ref, test, frac=0.8):
    s = int(frac * len(ref))
    pk_ref = np.max(np.abs(ref[s:]))
    pk_tst = np.max(np.abs(test[s:]))
    return 100 * abs(pk_ref - pk_tst) / pk_ref if pk_ref > 0 else 0.0

def correlation(a, b):
    if np.std(a) > 0 and np.std(b) > 0:
        return np.corrcoef(a, b)[0, 1]
    return 1.0 if np.allclose(a, b) else 0.0

ss = int(0.8 * len(t_common))

print("\n4. Validation metrics")
print("=" * 70)
print(f"{'Comparison':<35s} {'V NRMSE':>8s} {'I NRMSE':>8s} {'V corr':>8s} {'I corr':>8s}")
print("-" * 70)

for label, V_a, I_a, V_b, I_b in [
    ("LTspice vs Framework (TD)",   V_lt_c, I_lt_c, V_fw_c, I_fw_c),
    ("LTspice vs Framework (IDP)",  V_lt_c, I_lt_c, V_pd_c, I_pd_c),
    ("Framework: TD vs IDP",        V_fw_c, I_fw_c, V_pd_c, I_pd_c),
]:
    nV = nrmse(V_a[ss:], V_b[ss:])
    nI = nrmse(I_a[ss:], I_b[ss:])
    cV = correlation(V_a[ss:], V_b[ss:])
    cI = correlation(I_a[ss:], I_b[ss:])
    print(f"{label:<35s} {nV:>7.3f}% {nI:>7.3f}% {cV:>8.5f} {cI:>8.5f}")

print("-" * 70)
print("\nNote: LTspice has only 1178 data points (adaptive output compression).")
print("Peak values are ~8% below analytical due to sparse sampling at 92.3 kHz.")
print("NRMSE is computed after cubic interpolation to a 10000-point common grid.")

# Steady-state peaks
ss_fw = int(0.8 * len(t_fw))
ss_lt = int(0.8 * len(t_lt))
print(f"\nSteady-state peak amplitudes:")
print(f"  {'Method':<22s} {'V(N003) [V]':>12s} {'I(L1) [A]':>12s}")
print(f"  {'-'*22} {'-'*12} {'-'*12}")
print(f"  {'Analytical':<22s} {V_analytical:>12.4f} {I_analytical:>12.6f}")
print(f"  {'LTspice (raw)':<22s} {np.max(np.abs(V_lt[ss_lt:])):>12.4f} {np.max(np.abs(I_lt[ss_lt:])):>12.6f}")
print(f"  {'Framework (TD)':<22s} {np.max(np.abs(V_fw[ss_fw:])):>12.4f} {np.max(np.abs(I_fw[ss_fw:])):>12.6f}")
print(f"  {'Framework (IDP)':<22s} {V_env[-1]:>12.4f} {I_env[-1]:>12.6f}")

# ── 6. Plots ─────────────────────────────────────────────────────────
print("\n5. Generating comparison plots")

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("LTspice vs Framework: Rim 2025 RLC Circuit", fontsize=13, fontweight='bold')

t_ms = t_common * 1e3

# ── Full transient: voltage ──
ax = axes[0, 0]
ax.plot(t_lt * 1e3, V_lt, 'b-', lw=1.0, alpha=0.6, label='LTspice (V(n003)-V(n004))')
ax.plot(t_fw * 1e3, V_fw, 'r--', lw=1.5, alpha=0.8, label='Framework (TD)')
ax.plot(t_pd * 1e3, V_env, 'k-', lw=2.0, alpha=0.5, label='IDP envelope')
ax.plot(t_pd * 1e3, -V_env, 'k-', lw=2.0, alpha=0.5)
ax.set_xlabel('Time [ms]'); ax.set_ylabel('V(N003) [V]')
ax.set_title('Output Voltage — Full Transient'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Steady-state detail: voltage ──
ax = axes[0, 1]
t_zoom = (t_common > 0.45e-3)
ax.plot(t_ms[t_zoom], V_lt_c[t_zoom], 'b-', lw=1.5, alpha=0.7, label='LTspice (interp)')
ax.plot(t_ms[t_zoom], V_fw_c[t_zoom], 'r--', lw=2.0, alpha=0.8, label='Framework (TD)')
ax.plot(t_ms[t_zoom], V_pd_c[t_zoom], 'g:', lw=2.0, alpha=0.8, label='Framework (IDP)')
nV = nrmse(V_lt_c[ss:], V_fw_c[ss:])
ax.set_xlabel('Time [ms]'); ax.set_ylabel('V(N003) [V]')
ax.set_title(f'Output Voltage — Steady State (NRMSE LT vs TD: {nV:.2f}%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Full transient: current ──
ax = axes[1, 0]
ax.plot(t_lt * 1e3, I_lt, 'b-', lw=1.0, alpha=0.6, label='LTspice I(L1)')
ax.plot(t_fw * 1e3, I_fw, 'r--', lw=1.5, alpha=0.8, label='Framework (TD)')
ax.plot(t_pd * 1e3, I_env, 'k-', lw=2.0, alpha=0.5, label='IDP envelope')
ax.plot(t_pd * 1e3, -I_env, 'k-', lw=2.0, alpha=0.5)
ax.set_xlabel('Time [ms]'); ax.set_ylabel('I(L1) [A]')
ax.set_title('Inductor Current — Full Transient'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Steady-state detail: current ──
ax = axes[1, 1]
ax.plot(t_ms[t_zoom], I_lt_c[t_zoom], 'b-', lw=1.5, alpha=0.7, label='LTspice (interp)')
ax.plot(t_ms[t_zoom], I_fw_c[t_zoom], 'r--', lw=2.0, alpha=0.8, label='Framework (TD)')
ax.plot(t_ms[t_zoom], I_pd_c[t_zoom], 'g:', lw=2.0, alpha=0.8, label='Framework (IDP)')
nI = nrmse(I_lt_c[ss:], I_fw_c[ss:])
ax.set_xlabel('Time [ms]'); ax.set_ylabel('I(L1) [A]')
ax.set_title(f'Inductor Current — Steady State (NRMSE LT vs TD: {nI:.2f}%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'ltspice_comparison.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {out_path}")
plt.close()

print("\n" + "=" * 70)
print("Validation complete.")
print("=" * 70)
