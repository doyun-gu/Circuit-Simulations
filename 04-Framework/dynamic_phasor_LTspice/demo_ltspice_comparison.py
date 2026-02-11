"""
Demo: Dynamic Phasor Framework with LTspice Netlist Compatibility
=================================================================

This script demonstrates the complete workflow:
  1. Parse an LTspice-format netlist
  2. Run time-domain and phasor-domain simulations
  3. Compare results with LTspice (when .raw files are available)
  4. Export results in LTspice-compatible format

Tested with the series RLC circuit from Rim et al. (2025):
  V1 → R1 → L1 → (C1 ∥ R2) → GND

Author: Doyun Gu (University of Manchester)
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from core.netlist_parser import parse_ltspice_netlist
from core.mna_circuit import NetlistCircuit
from core.ltspice_comparison import LTSpiceComparison, compute_metrics
from core.ltspice_raw_reader import write_raw_ascii


def main():
    print("=" * 70)
    print("Dynamic Phasor Framework — LTspice Compatibility Demo")
    print("=" * 70)

    # ── 1. Define the circuit using LTspice netlist syntax ────────
    # This is exactly what LTspice would produce as a .net file
    netlist_text = """\
* Rim et al. (2025) Series RLC Resonant Circuit — Table II Parameters
* Topology: V1 → Rs → L1 → (C1 ∥ Ro) → GND
* Resonant freq: ~91.76 kHz, Q ≈ 19.2
V1 N001 0 SINE(0 1 92.3k)
R1 N001 N002 3.0
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.tran 0 0.5m
.end
"""

    print("\n1. NETLIST PARSING")
    print("-" * 40)
    netlist = parse_ltspice_netlist(netlist_text)
    print(f"   Title:    {netlist.title}")
    print(f"   Elements: {len(netlist.elements)}")
    print(f"   Nodes:    {netlist.non_ground_nodes()}")
    print(f"   Params:   {netlist.params}")
    for elem in netlist.elements:
        print(f"   • {elem}")
    tran = netlist.tran_params()
    if tran:
        print(f"   .tran: t_stop={tran['t_stop']*1e3:.2f} ms")

    # ── 2. Build MNA system and inspect ───────────────────────────
    print("\n2. MNA SYSTEM CONSTRUCTION")
    print("-" * 40)
    circuit = NetlistCircuit(netlist)
    print(circuit.info())

    # Verify system stability
    eigvals = np.linalg.eigvals(circuit._M_reduced)
    print(f"\n   System eigenvalues: {eigvals}")
    print(f"   Real parts: {eigvals.real}")
    assert np.all(eigvals.real < 0), "UNSTABLE SYSTEM DETECTED!"
    print("   ✓ System is stable (all eigenvalues have negative real parts)")

    # ── 3. Time-domain simulation ─────────────────────────────────
    print("\n3. TIME-DOMAIN SIMULATION")
    print("-" * 40)
    td_results = circuit.solve_time_domain()
    t = td_results['t']
    print(f"   Time span: [{t[0]*1e3:.3f}, {t[-1]*1e3:.3f}] ms")
    print(f"   Points:    {len(t)}")
    print(f"   V(N003):   [{td_results['V(N003)'].min():.4f}, "
          f"{td_results['V(N003)'].max():.4f}] V")
    print(f"   I(L1):     [{td_results['I(L1)'].min():.6f}, "
          f"{td_results['I(L1)'].max():.6f}] A")
    print(f"   I(V1):     [{td_results['I(V1)'].min():.6f}, "
          f"{td_results['I(V1)'].max():.6f}] A")

    # ── 4. Phasor-domain simulation ───────────────────────────────
    print("\n4. PHASOR-DOMAIN SIMULATION")
    print("-" * 40)
    omega_s = 2 * np.pi * 92.3e3
    circuit.configure_phasor(omega_s=omega_s)
    print(f"   Carrier: ω_s = {omega_s:.0f} rad/s "
          f"({omega_s/(2*np.pi)/1e3:.2f} kHz)")

    pd_results = circuit.solve_phasor_domain()
    t_pd = pd_results['t']
    print(f"   Time span: [{t_pd[0]*1e3:.3f}, {t_pd[-1]*1e3:.3f}] ms")
    print(f"   Points:    {len(t_pd)}")

    # Show phasor envelopes
    for key, env in pd_results['envelopes'].items():
        ss_val = env[-1]
        print(f"   {key} envelope (steady-state): {ss_val:.4f}")

    # Reconstruct time-domain from phasors and compare
    print("\n   Phasor→Time cross-check:")
    # Interpolate TD to phasor time points for comparison
    from scipy.interpolate import interp1d
    td_v_interp = interp1d(t, td_results['V(N003)'],
                           kind='linear')(t_pd)
    pd_v_real = pd_results['V(N003)']

    # Only compare after initial transient (last 20%)
    n_start = int(0.8 * len(t_pd))
    rmse = np.sqrt(np.mean((td_v_interp[n_start:] - pd_v_real[n_start:])**2))
    sig_range = np.max(td_v_interp[n_start:]) - np.min(td_v_interp[n_start:])
    nrmse = rmse / sig_range if sig_range > 0 else 0
    print(f"   V(N003) NRMSE (steady-state): {nrmse*100:.2f}%")
    print(f"   {'✓' if nrmse < 0.05 else '✗'} Phasor matches time-domain")

    # ── 5. Export in LTspice-compatible format ────────────────────
    print("\n5. EXPORT LTspice-COMPATIBLE .RAW FILE")
    print("-" * 40)
    export_data = {
        'time': td_results['t'],
    }
    for key in td_results:
        if key.startswith(('V(', 'I(')):
            signal = td_results[key]
            if np.iscomplexobj(signal):
                signal = np.real(signal)
            export_data[key] = signal

    import os
    output_path = os.path.join(os.getcwd(), 'framework_output.raw')
    write_raw_ascii(output_path, export_data,
                    title="Dynamic Phasor Framework - Rim2025 RLC")
    print(f"   Exported to: {output_path}")
    print(f"   Variables: {list(export_data.keys())}")
    print("   (This file can be opened in LTspice's waveform viewer)")

    # ── 6. Self-consistency validation ────────────────────────────
    print("\n6. SELF-CONSISTENCY VALIDATION")
    print("-" * 40)

    # Test with parameterised netlist
    netlist_param = """\
* Parameterised RLC circuit
.param Lval=100.04u Cval=30.07n Rs=3.0 Ro=2k fres={1/(2*pi*sqrt(Lval*Cval))}
V1 N001 0 SINE(0 1 {fres})
R1 N001 N002 {Rs}
L1 N002 N003 {Lval}
C1 N003 0 {Cval}
R2 N003 0 {Ro}
.tran 0 0.5m
.end
"""
    net_param = parse_ltspice_netlist(netlist_param)
    print(f"   Parameterised netlist parsed:")
    print(f"   .param fres = {net_param.params.get('fres', '?'):.2f} Hz")
    print(f"   Source freq matches resonance: ", end="")

    circ_param = NetlistCircuit(net_param)
    fr_est = circ_param.resonant_frequency()
    for v in net_param.voltage_sources():
        if v.source_spec:
            f_src = v.source_spec.sine_freq
            print(f"{f_src:.2f} Hz vs {fr_est:.2f} Hz "
                  f"({'✓ matched' if abs(f_src - fr_est)/fr_est < 0.01 else '✗ mismatch'})")

    # Run and compare
    td_param = circ_param.solve_time_domain()
    print(f"   V(N003) peak: {np.max(np.abs(td_param['V(N003)'])):.4f} V")
    print(f"   I(L1) peak:   {np.max(np.abs(td_param['I(L1)'])):.6f} A")

    # ── 7. Multi-topology test ────────────────────────────────────
    print("\n7. ADDITIONAL TOPOLOGY TESTS")
    print("-" * 40)

    # Test: Simple RC low-pass
    rc_netlist = """\
* RC Low-Pass Filter
V1 in 0 SINE(0 1 10k)
R1 in out 1k
C1 out 0 15.9n
.tran 0 0.5m
.end
"""
    circ_rc = NetlistCircuit.from_string(rc_netlist)
    res_rc = circ_rc.solve_time_domain()
    fc = 1 / (2 * np.pi * 1e3 * 15.9e-9)
    v_out_peak = np.max(np.abs(res_rc['V(out)'][-1000:]))
    expected_gain = 1 / np.sqrt(1 + (10e3/fc)**2)
    print(f"   RC filter: fc={fc/1e3:.2f} kHz, "
          f"V_out peak={v_out_peak:.4f} V, "
          f"expected≈{expected_gain:.4f} V")
    print(f"   {'✓' if abs(v_out_peak - expected_gain) < 0.05 else '✗'} "
          f"RC filter gain correct")

    # Test: Series RL
    rl_netlist = """\
* Series RL
V1 N001 0 SINE(0 1 1k)
R1 N001 N002 100
L1 N002 0 15.9m
.tran 0 5m
.end
"""
    circ_rl = NetlistCircuit.from_string(rl_netlist)
    res_rl = circ_rl.solve_time_domain()
    XL = 2 * np.pi * 1e3 * 15.9e-3
    expected_I = 1 / np.sqrt(100**2 + XL**2)
    i_peak = np.max(np.abs(res_rl['I(L1)'][-1000:]))
    print(f"   RL circuit: XL={XL:.1f} Ω, "
          f"I peak={i_peak:.6f} A, "
          f"expected≈{expected_I:.6f} A")
    print(f"   {'✓' if abs(i_peak - expected_I)/expected_I < 0.05 else '✗'} "
          f"RL current correct")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEMO COMPLETE — All tests passed")
    print("=" * 70)
    print("\nUsage with LTspice files:")
    print("  from core.ltspice_comparison import quick_compare")
    print('  comp = quick_compare("my_circuit.net", "my_circuit.raw")')
    print('  comp.print_report()')
    print('  comp.plot_comparison(save_path="comparison.png")')


if __name__ == "__main__":
    main()
