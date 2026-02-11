"""
Comparison and validation module for dynamic phasor framework vs LTspice.

Provides tools to:
    1. Run a netlist through the framework and compare with LTspice .raw results
    2. Compute error metrics (RMSE, peak error, normalised error)
    3. Generate publication-quality comparison plots
    4. Export comparison data for APEC paper figures

Typical workflow:
    >>> from dynamic_phasor.core.ltspice_comparison import LTSpiceComparison
    >>> comp = LTSpiceComparison.from_files("circuit.net", "circuit.raw")
    >>> comp.run_comparison()
    >>> comp.print_report()
    >>> comp.plot_comparison()

Author: Doyun Gu (University of Manchester)
"""

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .netlist_parser import parse_ltspice_netlist, ParsedNetlist
from .mna_circuit import NetlistCircuit
from .ltspice_raw_reader import read_raw, LTSpiceRawData


# ──────────────────────────────────────────────────────────────────────
# Error metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonMetrics:
    """Error metrics for a single variable comparison."""
    variable_name: str
    rmse: float = 0.0           # Root mean square error
    peak_error: float = 0.0     # Maximum absolute error
    nrmse: float = 0.0          # RMSE normalised by signal range
    peak_nrmse: float = 0.0     # Peak error normalised by signal range
    correlation: float = 0.0    # Pearson correlation coefficient
    signal_range: float = 0.0   # Range of the reference (LTspice) signal
    n_points: int = 0           # Number of comparison points

    def is_good(self, nrmse_threshold: float = 0.05) -> bool:
        """Check if comparison is within acceptable error."""
        return self.nrmse < nrmse_threshold

    def summary_line(self) -> str:
        """One-line summary."""
        status = "✓" if self.is_good() else "✗"
        return (f"{status} {self.variable_name:>12s}: "
                f"NRMSE={self.nrmse:.4f} ({self.nrmse*100:.2f}%), "
                f"Peak={self.peak_error:.6g}, "
                f"R²={self.correlation**2:.6f}")


def compute_metrics(ref: np.ndarray, test: np.ndarray,
                    name: str = "") -> ComparisonMetrics:
    """
    Compute comparison metrics between reference and test signals.

    Parameters
    ----------
    ref : ndarray
        Reference signal (LTspice)
    test : ndarray
        Test signal (framework)
    name : str
        Variable name for labelling

    Returns
    -------
    ComparisonMetrics
    """
    assert len(ref) == len(test), "Signals must have the same length"
    n = len(ref)

    error = test - ref
    rmse = np.sqrt(np.mean(error**2))
    peak_error = np.max(np.abs(error))

    sig_range = np.max(ref) - np.min(ref)
    if sig_range > 0:
        nrmse = rmse / sig_range
        peak_nrmse = peak_error / sig_range
    else:
        nrmse = 0.0 if rmse == 0 else float('inf')
        peak_nrmse = 0.0 if peak_error == 0 else float('inf')

    # Correlation
    if np.std(ref) > 0 and np.std(test) > 0:
        corr = np.corrcoef(ref, test)[0, 1]
    else:
        corr = 1.0 if np.allclose(ref, test) else 0.0

    return ComparisonMetrics(
        variable_name=name,
        rmse=rmse,
        peak_error=peak_error,
        nrmse=nrmse,
        peak_nrmse=peak_nrmse,
        correlation=corr,
        signal_range=sig_range,
        n_points=n,
    )


# ──────────────────────────────────────────────────────────────────────
# Main comparison class
# ──────────────────────────────────────────────────────────────────────

class LTSpiceComparison:
    """
    Compare dynamic phasor framework simulation with LTspice results.

    This class orchestrates:
    1. Parsing the LTspice netlist
    2. Running the framework's time-domain and/or phasor-domain simulation
    3. Loading LTspice .raw results
    4. Interpolating to common time base
    5. Computing error metrics
    6. Generating comparison plots

    Parameters
    ----------
    netlist : ParsedNetlist or str
        Parsed netlist or netlist string/filepath
    ltspice_data : LTSpiceRawData, optional
        Pre-loaded LTspice results
    """

    def __init__(self, netlist: ParsedNetlist,
                 ltspice_data: Optional[LTSpiceRawData] = None):
        self.netlist = netlist
        self.circuit = NetlistCircuit(netlist)
        self.ltspice_data = ltspice_data

        # Results storage
        self.framework_td: Optional[Dict] = None   # time-domain results
        self.framework_pd: Optional[Dict] = None   # phasor-domain results
        self.metrics_td: Dict[str, ComparisonMetrics] = {}
        self.metrics_pd: Dict[str, ComparisonMetrics] = {}

        # Common time base for comparison
        self._t_common: Optional[np.ndarray] = None

    # ── Factory constructors ──────────────────────────────────────

    @classmethod
    def from_files(cls, netlist_path: str,
                   raw_path: str = None) -> 'LTSpiceComparison':
        """
        Create comparison from file paths.

        Parameters
        ----------
        netlist_path : str
            Path to .net/.cir file
        raw_path : str, optional
            Path to .raw file. If None, auto-detects by replacing extension.
        """
        netlist = parse_ltspice_netlist(netlist_path)

        ltspice_data = None
        if raw_path:
            ltspice_data = read_raw(raw_path)
        else:
            # Auto-detect .raw file
            net_path = Path(netlist_path)
            for ext in ['.raw', '.Raw', '.RAW']:
                candidate = net_path.with_suffix(ext)
                if candidate.exists():
                    ltspice_data = read_raw(str(candidate))
                    break

        return cls(netlist, ltspice_data)

    @classmethod
    def from_strings(cls, netlist_text: str,
                     raw_path: str = None) -> 'LTSpiceComparison':
        """Create comparison from a netlist string."""
        netlist = parse_ltspice_netlist(netlist_text)
        ltspice_data = read_raw(raw_path) if raw_path else None
        return cls(netlist, ltspice_data)

    # ── Run simulations ───────────────────────────────────────────

    def run_time_domain(self, **solver_kwargs) -> Dict:
        """
        Run framework time-domain simulation.

        Returns the results dict and stores it internally.
        """
        self.framework_td = self.circuit.solve_time_domain(**solver_kwargs)
        return self.framework_td

    def run_phasor_domain(self, omega_s: float = None,
                          **solver_kwargs) -> Dict:
        """
        Run framework phasor-domain simulation.

        Parameters
        ----------
        omega_s : float, optional
            Carrier frequency. Auto-detected if None.
        """
        self.circuit.configure_phasor(omega_s=omega_s)
        self.framework_pd = self.circuit.solve_phasor_domain(**solver_kwargs)
        return self.framework_pd

    def run_comparison(self, run_phasor: bool = True,
                       omega_s: float = None,
                       **solver_kwargs) -> Dict[str, ComparisonMetrics]:
        """
        Run full comparison: simulate in framework, compare with LTspice.

        Parameters
        ----------
        run_phasor : bool
            Also run phasor-domain comparison
        omega_s : float, optional
            Carrier frequency for phasor analysis
        **solver_kwargs : passed to solvers

        Returns
        -------
        dict of ComparisonMetrics
            Metrics for each compared variable
        """
        # Run time-domain
        self.run_time_domain(**solver_kwargs)

        # Run phasor if requested
        if run_phasor:
            try:
                self.run_phasor_domain(omega_s=omega_s, **solver_kwargs)
            except (ValueError, RuntimeError) as e:
                print(f"  Phasor simulation skipped: {e}")

        # Compare with LTspice if available
        if self.ltspice_data is not None and self.framework_td is not None:
            self.metrics_td = self._compare_results(
                self.framework_td, self.ltspice_data, label="TD"
            )

        return self.metrics_td

    def _compare_results(self, framework_results: Dict,
                         ltspice_data: LTSpiceRawData,
                         label: str = "") -> Dict[str, ComparisonMetrics]:
        """Compare framework results with LTspice on a common time base."""
        metrics = {}

        t_fw = framework_results['t']
        t_lt = ltspice_data.time

        if t_lt is None:
            return metrics

        # Ensure t_lt is real (sometimes stored as complex in .raw)
        t_lt = np.real(t_lt)

        # Create common time base
        t_start = max(t_fw[0], t_lt[0])
        t_end = min(t_fw[-1], t_lt[-1])
        n_common = min(len(t_fw), len(t_lt), 10000)
        self._t_common = np.linspace(t_start, t_end, n_common)

        # Find matching variables
        for var_name in framework_results:
            if var_name in ('t', 'x', 'node_voltages', 'branch_currents',
                            'source_voltages', 'phasor_voltages',
                            'phasor_currents', 'envelopes', 'x_phasor'):
                continue

            # Find corresponding LTspice variable
            lt_signal = _find_ltspice_match(var_name, ltspice_data)
            if lt_signal is None:
                continue

            fw_signal = framework_results[var_name]
            if np.iscomplexobj(fw_signal):
                fw_signal = np.real(fw_signal)
            if np.iscomplexobj(lt_signal):
                lt_signal = np.real(lt_signal)

            # Interpolate both to common time base
            try:
                fw_interp = interp1d(t_fw, fw_signal,
                                     kind='linear',
                                     fill_value='extrapolate')(self._t_common)
                lt_interp = interp1d(t_lt, lt_signal,
                                     kind='linear',
                                     fill_value='extrapolate')(self._t_common)
            except (ValueError, IndexError):
                continue

            key = f"{label}:{var_name}" if label else var_name
            metrics[key] = compute_metrics(lt_interp, fw_interp, var_name)

        return metrics

    # ── Reporting ─────────────────────────────────────────────────

    def print_report(self):
        """Print a formatted comparison report."""
        print("=" * 70)
        print("Dynamic Phasor Framework vs LTspice Comparison Report")
        print("=" * 70)
        print(f"\nCircuit: {self.netlist.title}")
        print(f"Elements: {len(self.netlist.elements)}")
        print(f"Nodes: {len(self.netlist.non_ground_nodes())}")

        fr = self.circuit.resonant_frequency()
        if fr:
            print(f"Resonant frequency: {fr/1e3:.2f} kHz")

        Q = self.circuit.quality_factor()
        if Q:
            print(f"Quality factor: {Q:.2f}")

        if self.metrics_td:
            print(f"\n{'─'*70}")
            print("Time-Domain Comparison:")
            print(f"{'─'*70}")
            all_good = True
            for key, m in self.metrics_td.items():
                print(f"  {m.summary_line()}")
                if not m.is_good():
                    all_good = False
            if all_good:
                print("\n  ✓ All variables within 5% NRMSE tolerance")
            else:
                print("\n  ✗ Some variables exceed 5% NRMSE tolerance")

        if self.metrics_pd:
            print(f"\n{'─'*70}")
            print("Phasor-Domain Comparison:")
            print(f"{'─'*70}")
            for key, m in self.metrics_pd.items():
                print(f"  {m.summary_line()}")

        print(f"\n{'='*70}")

    def get_comparison_data(self, variable: str) -> Optional[Dict]:
        """
        Get interpolated comparison data for a specific variable.

        Returns dict with 't', 'framework', 'ltspice', 'error' arrays
        on the common time base.
        """
        if self._t_common is None or self.framework_td is None:
            return None
        if self.ltspice_data is None:
            return None

        t_fw = self.framework_td['t']
        t_lt = np.real(self.ltspice_data.time)

        fw_signal = self.framework_td.get(variable)
        if fw_signal is None:
            return None

        lt_signal = _find_ltspice_match(variable, self.ltspice_data)
        if lt_signal is None:
            return None

        if np.iscomplexobj(fw_signal):
            fw_signal = np.real(fw_signal)
        if np.iscomplexobj(lt_signal):
            lt_signal = np.real(lt_signal)

        fw_interp = interp1d(t_fw, fw_signal, kind='linear',
                             fill_value='extrapolate')(self._t_common)
        lt_interp = interp1d(t_lt, lt_signal, kind='linear',
                             fill_value='extrapolate')(self._t_common)

        return {
            't': self._t_common,
            'framework': fw_interp,
            'ltspice': lt_interp,
            'error': fw_interp - lt_interp,
        }

    # ── Plotting ──────────────────────────────────────────────────

    def plot_comparison(self, variables: List[str] = None,
                        show_error: bool = True,
                        save_path: str = None,
                        figsize: Tuple[float, float] = (12, 8)):
        """
        Generate comparison plots.

        Parameters
        ----------
        variables : list of str, optional
            Variables to plot. If None, plots all available.
        show_error : bool
            Include error subplot
        save_path : str, optional
            Save figure to file
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt

        if variables is None:
            # Auto-detect plottable variables
            variables = []
            if self.framework_td:
                for k in self.framework_td:
                    if k.startswith(('V(', 'I(')) and self.ltspice_data:
                        lt_match = _find_ltspice_match(k, self.ltspice_data)
                        if lt_match is not None:
                            variables.append(k)

        if not variables:
            print("No matching variables found for comparison plotting.")
            return

        n_vars = len(variables)
        n_rows = n_vars * (2 if show_error else 1)

        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
        if n_rows == 1:
            axes = [axes]

        plot_idx = 0
        for var in variables:
            comp_data = self.get_comparison_data(var)
            if comp_data is None:
                continue

            t_ms = comp_data['t'] * 1e3  # Convert to ms

            # Main comparison plot
            ax = axes[plot_idx]
            ax.plot(t_ms, comp_data['ltspice'], 'b-', linewidth=1.2,
                    label='LTspice', alpha=0.8)
            ax.plot(t_ms, comp_data['framework'], 'r--', linewidth=1.0,
                    label='Framework', alpha=0.8)
            ax.set_ylabel(var)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{var} Comparison", fontsize=10)
            plot_idx += 1

            # Error subplot
            if show_error:
                ax_err = axes[plot_idx]
                ax_err.plot(t_ms, comp_data['error'], 'g-', linewidth=0.8)
                ax_err.set_ylabel(f"Error")
                ax_err.grid(True, alpha=0.3)

                # Add NRMSE annotation
                key = f"TD:{var}"
                if key in self.metrics_td:
                    m = self.metrics_td[key]
                    ax_err.set_title(
                        f"Error (NRMSE={m.nrmse*100:.2f}%, "
                        f"Peak={m.peak_error:.4g})", fontsize=9
                    )
                plot_idx += 1

        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle(f"Framework vs LTspice: {self.netlist.title}",
                     fontsize=12, fontweight='bold')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Figure saved to {save_path}")

        return fig

    def plot_framework_only(self, variables: List[str] = None,
                            save_path: str = None,
                            figsize: Tuple[float, float] = (12, 6)):
        """
        Plot framework results only (when no LTspice data available).

        Useful for quick verification of the framework simulation.
        """
        import matplotlib.pyplot as plt

        if self.framework_td is None:
            self.run_time_domain()

        results = self.framework_td
        t_ms = results['t'] * 1e3

        if variables is None:
            variables = [k for k in results
                         if k.startswith(('V(', 'I('))]

        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
        if n_vars == 1:
            axes = [axes]

        for i, var in enumerate(variables):
            if var in results:
                signal = results[var]
                if np.iscomplexobj(signal):
                    signal = np.real(signal)
                axes[i].plot(t_ms, signal, 'b-', linewidth=0.8)
                axes[i].set_ylabel(var)
                axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle(f"Framework Simulation: {self.netlist.title}",
                     fontsize=12, fontweight='bold')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# ──────────────────────────────────────────────────────────────────────
# Helper: match variable names between framework and LTspice
# ──────────────────────────────────────────────────────────────────────

def _find_ltspice_match(fw_name: str,
                        ltspice_data: LTSpiceRawData) -> Optional[np.ndarray]:
    """
    Find the matching LTspice variable for a framework variable name.

    LTspice may use different casing or naming conventions:
        Framework: V(N003)  →  LTspice: V(n003) or V(N003)
        Framework: I(L1)    →  LTspice: I(L1) or Ix(L1:1)
    """
    # Direct match
    if fw_name in ltspice_data.data:
        return ltspice_data.data[fw_name]

    # Case-insensitive match
    fw_lower = fw_name.lower()
    for lt_name, lt_data in ltspice_data.data.items():
        if lt_name.lower() == fw_lower:
            return lt_data

    # Try variant formats
    # V(N003) → V(n003), v(N003), etc.
    if fw_name.startswith(('V(', 'I(')):
        prefix = fw_name[0]
        inner = fw_name[2:-1]

        variants = [
            f"{prefix}({inner})",
            f"{prefix}({inner.lower()})",
            f"{prefix}({inner.upper()})",
            f"{prefix.lower()}({inner})",
            f"{prefix.lower()}({inner.lower()})",
        ]

        # LTspice sometimes uses I(Rn) instead of I(rn), etc.
        for v in variants:
            if v in ltspice_data.data:
                return ltspice_data.data[v]

    return None


# ──────────────────────────────────────────────────────────────────────
# Quick comparison function
# ──────────────────────────────────────────────────────────────────────

def quick_compare(netlist_source: str,
                  raw_path: str = None,
                  plot: bool = True,
                  run_phasor: bool = False) -> LTSpiceComparison:
    """
    One-line comparison of framework vs LTspice.

    Parameters
    ----------
    netlist_source : str
        Netlist file path or netlist text string
    raw_path : str, optional
        Path to LTspice .raw file
    plot : bool
        Generate plots
    run_phasor : bool
        Also run phasor-domain analysis

    Returns
    -------
    LTSpiceComparison
        Full comparison object with results and metrics

    Examples
    --------
    >>> comp = quick_compare("circuit.net", "circuit.raw")
    >>> comp = quick_compare(netlist_text)  # framework-only
    """
    import os

    if os.path.isfile(netlist_source):
        comp = LTSpiceComparison.from_files(netlist_source, raw_path)
    else:
        comp = LTSpiceComparison.from_strings(netlist_source, raw_path)

    if comp.ltspice_data is not None:
        comp.run_comparison(run_phasor=run_phasor)
        comp.print_report()
        if plot:
            comp.plot_comparison()
    else:
        comp.run_time_domain()
        if run_phasor:
            try:
                comp.run_phasor_domain()
            except (ValueError, RuntimeError):
                pass
        print("No LTspice data — framework results only:")
        print(comp.circuit.info())
        if plot:
            comp.plot_framework_only()

    return comp
