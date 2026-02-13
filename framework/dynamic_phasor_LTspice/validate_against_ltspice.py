#!/usr/bin/env python3
"""
Validate Dynamic Phasor Framework against LTspice.

This script compares simulation results from the framework with LTspice
.raw files to ensure accuracy.

Usage:
    python validate_against_ltspice.py <netlist.cir> <ltspice.raw>

Example:
    python validate_against_ltspice.py rim2025_rlc.cir rim2025_rlc.raw

Author: Doyun Gu (University of Manchester)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core.netlist_parser import parse_ltspice_netlist
from core.mna_circuit import NetlistCircuit
from core.ltspice_raw_reader import read_raw
from scipy.interpolate import interp1d


def compute_nrmse(y_true, y_pred):
    """Compute normalized RMSE as percentage."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    signal_range = np.max(y_true) - np.min(y_true)
    return (rmse / signal_range * 100) if signal_range > 0 else 0


def compute_peak_error(y_true, y_pred, steady_state_fraction=0.8):
    """Compute peak value error in steady-state region."""
    start_idx = int(steady_state_fraction * len(y_true))
    peak_true = np.max(np.abs(y_true[start_idx:]))
    peak_pred = np.max(np.abs(y_pred[start_idx:]))
    return 100 * abs(peak_true - peak_pred) / peak_true if peak_true > 0 else 0


def validate(netlist_file, raw_file, plot_output=None, verbose=True):
    """
    Validate framework against LTspice.

    Parameters
    ----------
    netlist_file : str
        Path to LTspice netlist file (.cir, .net, .sp)
    raw_file : str
        Path to LTspice .raw output file
    plot_output : str, optional
        Path to save comparison plot
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Validation results with metrics and pass/fail status
    """
    if verbose:
        print("="*70)
        print("LTSPICE VALIDATION")
        print("="*70)

    # ── 1. Parse netlist ──────────────────────────────────────────────
    if verbose:
        print(f"\n1. Parsing netlist: {netlist_file}")

    if not os.path.exists(netlist_file):
        raise FileNotFoundError(f"Netlist file not found: {netlist_file}")

    with open(netlist_file, 'r') as f:
        netlist_text = f.read()

    netlist = parse_ltspice_netlist(netlist_text)

    if verbose:
        print(f"   Title: {netlist.title}")
        print(f"   Elements: {len(netlist.elements)}")
        print(f"   Nodes: {netlist.non_ground_nodes()}")

    # ── 2. Build circuit ──────────────────────────────────────────────
    if verbose:
        print(f"\n2. Building MNA circuit")

    circuit = NetlistCircuit(netlist)

    if verbose:
        print(f"   State variables: {len(circuit.state_labels)}")
        print(f"   Resonant freq: {circuit.resonant_frequency()/1e3:.2f} kHz")

    # ── 3. Run framework simulation ───────────────────────────────────
    if verbose:
        print(f"\n3. Running framework simulation")

    framework_results = circuit.solve_time_domain()

    t_fw = framework_results['t']

    if verbose:
        print(f"   Time span: [{t_fw[0]*1e3:.3f}, {t_fw[-1]*1e3:.3f}] ms")
        print(f"   Data points: {len(t_fw)}")

    # ── 4. Load LTspice results ───────────────────────────────────────
    if verbose:
        print(f"\n4. Loading LTspice results: {raw_file}")

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"LTspice .raw file not found: {raw_file}")

    ltspice_data = read_raw(raw_file)

    t_lt = ltspice_data['time']

    if verbose:
        print(f"   Time span: [{t_lt[0]*1e3:.3f}, {t_lt[-1]*1e3:.3f}] ms")
        print(f"   Data points: {len(t_lt)}")
        print(f"   Variables: {len([k for k in ltspice_data.keys() if k != 'time'])}")

    # ── 5. Compare signals ────────────────────────────────────────────
    if verbose:
        print(f"\n5. Comparing signals")

    # Find common signals
    fw_signals = set(framework_results.keys()) - {'t'}
    lt_signals = set(ltspice_data.keys()) - {'time'}

    # Normalize signal names (LTspice may use lowercase)
    lt_signals_normalized = {s.upper(): s for s in lt_signals}
    common_signals = []

    for sig in fw_signals:
        sig_upper = sig.upper()
        if sig_upper in lt_signals_normalized:
            common_signals.append((sig, lt_signals_normalized[sig_upper]))

    if verbose:
        print(f"   Common signals: {len(common_signals)}")
        for fw_sig, lt_sig in common_signals:
            print(f"     • {fw_sig} ↔ {lt_sig}")

    if len(common_signals) == 0:
        print("\n   ⚠ WARNING: No common signals found!")
        print(f"   Framework signals: {fw_signals}")
        print(f"   LTspice signals: {lt_signals}")
        return {
            'success': False,
            'error': 'No common signals found',
        }

    # ── 6. Compute metrics ────────────────────────────────────────────
    if verbose:
        print(f"\n6. Computing validation metrics")

    results = {
        'signals': {},
        'overall_pass': True,
        'netlist_file': netlist_file,
        'raw_file': raw_file,
    }

    # Interpolate to common time base (use framework time points)
    for fw_sig, lt_sig in common_signals:
        # Get signals
        y_fw = framework_results[fw_sig]
        y_lt = ltspice_data[lt_sig]

        # Handle complex signals (should be real for time-domain)
        if np.iscomplexobj(y_fw):
            y_fw = np.real(y_fw)
        if np.iscomplexobj(y_lt):
            y_lt = np.real(y_lt)

        # Interpolate LTspice to framework time points
        interp_func = interp1d(t_lt, y_lt, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        y_lt_interp = interp_func(t_fw)

        # Compute metrics
        nrmse = compute_nrmse(y_fw, y_lt_interp)
        peak_error = compute_peak_error(y_fw, y_lt_interp)

        # Correlation
        correlation = np.corrcoef(y_fw, y_lt_interp)[0, 1]

        # RMS error (absolute)
        rmse = np.sqrt(np.mean((y_fw - y_lt_interp)**2))

        # Store results
        results['signals'][fw_sig] = {
            'nrmse': nrmse,
            'peak_error': peak_error,
            'correlation': correlation,
            'rmse': rmse,
            'pass': nrmse < 1.0 and peak_error < 2.0,
        }

        # Update overall pass/fail
        if not results['signals'][fw_sig]['pass']:
            results['overall_pass'] = False

        if verbose:
            status = '✓' if results['signals'][fw_sig]['pass'] else '✗'
            print(f"   {status} {fw_sig:12s}: NRMSE={nrmse:6.3f}%, "
                  f"Peak={peak_error:6.3f}%, Corr={correlation:.5f}")

    # ── 7. Generate comparison plot ───────────────────────────────────
    if plot_output:
        if verbose:
            print(f"\n7. Generating comparison plot")

        n_signals = len(common_signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(14, 4*n_signals))

        if n_signals == 1:
            axes = [axes]

        for idx, (fw_sig, lt_sig) in enumerate(common_signals):
            ax = axes[idx]

            y_fw = framework_results[fw_sig]
            y_lt = ltspice_data[lt_sig]

            if np.iscomplexobj(y_fw):
                y_fw = np.real(y_fw)
            if np.iscomplexobj(y_lt):
                y_lt = np.real(y_lt)

            # Plot
            ax.plot(t_fw*1e3, y_fw, 'b-', linewidth=1.5,
                   label='Framework', alpha=0.8)
            ax.plot(t_lt*1e3, y_lt, 'r--', linewidth=1.5,
                   label='LTspice', alpha=0.8)

            # Labels
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(fw_sig)

            # Title with metrics
            metrics = results['signals'][fw_sig]
            status = 'PASS ✓' if metrics['pass'] else 'FAIL ✗'
            ax.set_title(f"{fw_sig} — {status} "
                        f"(NRMSE: {metrics['nrmse']:.3f}%, "
                        f"Peak Error: {metrics['peak_error']:.3f}%)")

            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_output, dpi=300, bbox_inches='tight')

        if verbose:
            print(f"   Plot saved: {plot_output}")

    return results


def print_summary(results):
    """Print validation summary."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print(f"\nFiles:")
    print(f"  Netlist: {results['netlist_file']}")
    print(f"  LTspice: {results['raw_file']}")

    print(f"\nMetrics:")
    for signal, metrics in results['signals'].items():
        status = '✓ PASS' if metrics['pass'] else '✗ FAIL'
        print(f"\n  {signal} [{status}]:")
        print(f"    NRMSE:        {metrics['nrmse']:.4f}%")
        print(f"    Peak Error:   {metrics['peak_error']:.4f}%")
        print(f"    Correlation:  {metrics['correlation']:.6f}")
        print(f"    RMS Error:    {metrics['rmse']:.6e}")

    print("\n" + "="*70)
    if results['overall_pass']:
        print("✓✓ VALIDATION SUCCESSFUL ✓✓")
        print("Framework matches LTspice within tolerance!")
        print("  Target: NRMSE < 1%, Peak Error < 2%")
    else:
        print("✗✗ VALIDATION FAILED ✗✗")
        print("Framework results differ from LTspice.")
        print("Check simulation settings and component values.")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate Dynamic Phasor Framework against LTspice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_against_ltspice.py rim2025_rlc.cir rim2025_rlc.raw

  # With comparison plot
  python validate_against_ltspice.py rim2025_rlc.cir rim2025_rlc.raw -o comparison.png

  # Quiet mode
  python validate_against_ltspice.py rim2025_rlc.cir rim2025_rlc.raw -q
        """
    )

    parser.add_argument('netlist', type=str,
                       help='LTspice netlist file (.cir, .net, .sp)')
    parser.add_argument('raw', type=str,
                       help='LTspice .raw output file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Save comparison plot to file')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode (minimal output)')

    args = parser.parse_args()

    try:
        # Run validation
        results = validate(
            netlist_file=args.netlist,
            raw_file=args.raw,
            plot_output=args.output,
            verbose=not args.quiet
        )

        # Print summary
        if not args.quiet:
            print_summary(results)

        # Exit code
        return 0 if results['overall_pass'] else 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
