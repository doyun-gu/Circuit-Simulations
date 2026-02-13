"""
Benchmark validation against Rim et al. (IEEE TPEL 2025).

This module provides tools to reproduce and validate against the
experimental and simulation results from:
    "General Instantaneous Dynamic Phasor"
    IEEE Trans. Power Electronics, Vol. 40, No. 11, Nov. 2025

Key validation points:
    - Table II: Circuit parameters
    - Figures 5-6: Simulation waveforms (ωs = 580k, 650k rad/s)
    - Figures 8-9: Experimental waveforms
    - <3% accuracy target between simulation and experiment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import warnings

# Import from our framework
from ..core.circuit import RLCCircuit, CircuitParameters


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    amplitude_error_is: float = 0.0      # Current amplitude error (%)
    amplitude_error_vo: float = 0.0      # Voltage amplitude error (%)
    envelope_correlation: float = 0.0     # Envelope correlation coefficient
    transient_timing_error: float = 0.0   # Transient timing error (%)
    rmse_is: float = 0.0                  # Current RMSE
    rmse_vo: float = 0.0                  # Voltage RMSE
    passed: bool = False                   # Overall pass/fail
    
    def __repr__(self):
        status = "PASS ✓" if self.passed else "FAIL ✗"
        return (f"ValidationMetrics [{status}]\n"
                f"  Current amplitude error: {self.amplitude_error_is:.2f}%\n"
                f"  Voltage amplitude error: {self.amplitude_error_vo:.2f}%\n"
                f"  Envelope correlation: {self.envelope_correlation:.4f}\n"
                f"  Transient timing error: {self.transient_timing_error:.2f}%\n"
                f"  Current RMSE: {self.rmse_is:.4e}\n"
                f"  Voltage RMSE: {self.rmse_vo:.4e}")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    omega_s: float = 580e3          # Operating frequency (rad/s)
    t_end: float = 0.2e-3           # Simulation end time (s)
    Ve: float = 1.0                 # Step envelope amplitude (V)
    phi_s: float = 0.0              # Source phase (rad)
    
    # Validation thresholds
    amplitude_threshold: float = 3.0     # Max amplitude error (%)
    correlation_threshold: float = 0.95  # Min envelope correlation
    timing_threshold: float = 10.0       # Max timing error (%)
    
    # FM parameters (for extended tests)
    omega_0: float = None           # FM base frequency
    omega_1: float = 200e3          # FM modulation frequency
    alpha: float = 0.0              # FM modulation index (0 = no FM)


class Rim2025Benchmark:
    """
    Benchmark validation against Rim et al. (2025).
    
    Provides methods to:
    1. Create the exact circuit from Table II
    2. Define source signals matching the paper
    3. Run simulations in both time and phasor domains
    4. Compare results with experimental data
    5. Compute validation metrics
    
    Examples
    --------
    >>> benchmark = Rim2025Benchmark()
    >>> results = benchmark.run_validation()
    >>> print(results['metrics'])
    """
    
    # Experimental reference values from Figures 8-9
    # These are approximate peak values read from the figures
    EXPERIMENTAL_REFERENCE = {
        580e3: {  # omega_s = 580 krad/s (Figure 8)
            'is_peak': 0.35,      # Peak current ~0.35 A
            'vo_peak': 16.0,      # Peak voltage ~16 V
            'transient_time': 0.08e-3,  # Time to reach ~63% of steady state
        },
        650e3: {  # omega_s = 650 krad/s (Figure 9)
            'is_peak': 0.18,      # Peak current ~0.18 A
            'vo_peak': 7.0,       # Peak voltage ~7 V
            'transient_time': 0.06e-3,
        }
    }
    
    def __init__(self, config: BenchmarkConfig = None):
        """
        Initialize benchmark.
        
        Parameters
        ----------
        config : BenchmarkConfig, optional
            Benchmark configuration. Uses defaults if None.
        """
        self.config = config or BenchmarkConfig()
        self.circuit = None
        self._results = {}
    
    def create_circuit(self) -> RLCCircuit:
        """
        Create circuit with Table II parameters.
        
        Returns
        -------
        RLCCircuit
            Configured circuit ready for simulation
        """
        params = CircuitParameters(
            L=100.04e-6,   # 100.04 µH
            C=30.07e-9,    # 30.07 nF
            Rs=3.0,        # 3.0 Ω (includes inverter internal resistance)
            Ro=2000.0      # 2.00 kΩ
        )
        
        self.circuit = RLCCircuit(params)
        
        # Configure phasor transformation
        if self.config.alpha != 0:
            # FM case
            self.circuit.configure_phasor(
                omega_s=self.config.omega_s,
                omega_0=self.config.omega_0 or self.config.omega_s,
                omega_1=self.config.omega_1,
                alpha=self.config.alpha
            )
        else:
            # Standard case
            self.circuit.configure_phasor(omega_s=self.config.omega_s)
        
        return self.circuit
    
    def source_voltage_time_domain(self, t: float) -> float:
        """
        Time-domain source voltage for step envelope.
        
        vs(t) = Ve · u(t) · cos(ωs·t + φs)
        
        where u(t) is unit step function.
        
        Matches Eq. (22):
            vs(t) = ve(t)·cos(ωs·t + φs)
        """
        if t < 0:
            return 0.0
        return (self.config.Ve * 
                np.cos(self.config.omega_s * t + self.config.phi_s))
    
    def source_voltage_phasor(self, t: float) -> complex:
        """
        Phasor-domain source voltage.
        
        For standard case θ(t) = ωs·t:
            ṽs(t) = ve(t)·e^(jφs) = Ve·e^(jφs)  [Eq. 24]
            
        For FM case θ(t) = ω₀t + α·sin(ω₁t):
            ṽs = ve(t)·e^(j(ωs·t + φs - θ(t)))  [Eq. 38b]
        """
        if t < 0:
            return 0.0 + 0.0j
        
        if self.config.alpha != 0:
            # FM case - Eq. (38b)
            theta = (self.config.omega_0 * t + 
                    self.config.alpha * np.sin(self.config.omega_1 * t))
            phase_diff = self.config.omega_s * t + self.config.phi_s - theta
            return self.config.Ve * np.exp(1j * phase_diff)
        else:
            # Standard case - Eq. (24)
            return self.config.Ve * np.exp(1j * self.config.phi_s)
    
    def run_time_domain(self) -> Dict:
        """
        Run time-domain simulation.
        
        Returns
        -------
        dict
            Time-domain results with keys: t, is_t, vo_t, vs_t
        """
        if self.circuit is None:
            self.create_circuit()
        
        t_span = (0, self.config.t_end)
        
        results = self.circuit.solve_time_domain(
            vs_func=self.source_voltage_time_domain,
            t_span=t_span,
            rtol=1e-8,
            atol=1e-10
        )
        
        self._results['time_domain'] = results
        return results
    
    def run_phasor_domain(self) -> Dict:
        """
        Run phasor-domain simulation (instantaneous method).
        
        Returns
        -------
        dict
            Phasor-domain results including both phasors and 
            reconstructed time-domain signals
        """
        if self.circuit is None:
            self.create_circuit()
        
        t_span = (0, self.config.t_end)
        
        results = self.circuit.solve_phasor_domain(
            vs_phasor_func=self.source_voltage_phasor,
            t_span=t_span,
            rtol=1e-8,
            atol=1e-10
        )
        
        self._results['phasor_domain'] = results
        return results
    
    def run_analytical(self) -> Dict:
        """
        Compute analytical solution (Eq. 39).
        
        Only valid for standard case (no FM).
        
        Returns
        -------
        dict
            Analytical phasor response
        """
        if self.circuit is None:
            self.create_circuit()
        
        if self.config.alpha != 0:
            warnings.warn("Analytical solution only valid for α=0 (no FM)")
        
        t = np.linspace(0, self.config.t_end, 5000)
        results = self.circuit.analytical_phasor_response(
            t, Ve=self.config.Ve, phi_s=self.config.phi_s
        )
        
        self._results['analytical'] = results
        return results
    
    def run_validation(self, include_analytical: bool = True) -> Dict:
        """
        Run complete validation suite.
        
        Parameters
        ----------
        include_analytical : bool
            Whether to compute analytical solution
            
        Returns
        -------
        dict with keys:
            - time_domain: Time-domain simulation results
            - phasor_domain: Phasor-domain simulation results
            - analytical: Analytical solution (if requested)
            - metrics: ValidationMetrics object
            - comparison: Method comparison data
        """
        # Run simulations
        td_results = self.run_time_domain()
        pd_results = self.run_phasor_domain()
        
        results = {
            'time_domain': td_results,
            'phasor_domain': pd_results,
        }
        
        if include_analytical and self.config.alpha == 0:
            results['analytical'] = self.run_analytical()
        
        # Compute metrics
        results['metrics'] = self.compute_metrics(td_results, pd_results)
        
        # Method comparison
        results['comparison'] = self.compare_methods(td_results, pd_results)
        
        return results
    
    def compute_metrics(self, td_results: Dict, pd_results: Dict) -> ValidationMetrics:
        """
        Compute validation metrics comparing time and phasor domain results.
        
        Also compares against experimental reference values if available.
        """
        metrics = ValidationMetrics()
        
        # Interpolate phasor results to time-domain time points
        from scipy.interpolate import interp1d
        
        t_td = td_results['t']
        is_td = td_results['is_t']
        vo_td = td_results['vo_t']
        
        # Get reconstructed time-domain from phasor
        is_pd_interp = interp1d(pd_results['t'], pd_results['is_t'], 
                                fill_value='extrapolate')
        vo_pd_interp = interp1d(pd_results['t'], pd_results['vo_t'],
                                fill_value='extrapolate')
        
        is_pd = is_pd_interp(t_td)
        vo_pd = vo_pd_interp(t_td)
        
        # Peak amplitude comparison (use second half for steady-state)
        mid_idx = len(t_td) // 2
        
        is_peak_td = np.max(np.abs(is_td[mid_idx:]))
        is_peak_pd = np.max(np.abs(is_pd[mid_idx:]))
        vo_peak_td = np.max(np.abs(vo_td[mid_idx:]))
        vo_peak_pd = np.max(np.abs(vo_pd[mid_idx:]))
        
        # Amplitude errors
        if is_peak_td > 0:
            metrics.amplitude_error_is = 100 * abs(is_peak_pd - is_peak_td) / is_peak_td
        if vo_peak_td > 0:
            metrics.amplitude_error_vo = 100 * abs(vo_peak_pd - vo_peak_td) / vo_peak_td
        
        # RMSE
        metrics.rmse_is = np.sqrt(np.mean((is_td - is_pd)**2))
        metrics.rmse_vo = np.sqrt(np.mean((vo_td - vo_pd)**2))
        
        # Envelope correlation
        env_td = np.abs(self.circuit.phasor.to_phasor(vo_td, t_td))
        env_pd = pd_results['vo_envelope']
        env_pd_interp = interp1d(pd_results['t'], env_pd, fill_value='extrapolate')
        
        correlation = np.corrcoef(env_td, env_pd_interp(t_td))[0, 1]
        metrics.envelope_correlation = correlation if not np.isnan(correlation) else 0.0
        
        # Transient timing (time to reach 63% of final value)
        final_env = np.mean(env_pd[-100:])
        target = 0.63 * final_env
        
        try:
            idx_td = np.where(env_td >= target)[0][0]
            idx_pd = np.where(env_pd >= target)[0][0]
            t_63_td = t_td[idx_td]
            t_63_pd = pd_results['t'][idx_pd]
            
            if t_63_td > 0:
                metrics.transient_timing_error = 100 * abs(t_63_pd - t_63_td) / t_63_td
        except IndexError:
            metrics.transient_timing_error = float('nan')
        
        # Overall pass/fail
        metrics.passed = (
            metrics.amplitude_error_is < self.config.amplitude_threshold and
            metrics.amplitude_error_vo < self.config.amplitude_threshold and
            metrics.envelope_correlation > self.config.correlation_threshold
        )
        
        return metrics
    
    def compare_methods(self, td_results: Dict, pd_results: Dict) -> Dict:
        """
        Generate method comparison data for plotting.
        """
        return {
            'time_vector_td': td_results['t'],
            'time_vector_pd': pd_results['t'],
            'current': {
                'time_domain': td_results['is_t'],
                'phasor_reconstructed': pd_results['is_t'],
                'phasor_envelope': pd_results['is_envelope'],
            },
            'voltage': {
                'time_domain': td_results['vo_t'],
                'phasor_reconstructed': pd_results['vo_t'],
                'phasor_envelope': pd_results['vo_envelope'],
            }
        }
    
    def compare_with_experiment(self) -> Dict:
        """
        Compare simulation results with experimental reference values.
        
        Note: Full experimental waveforms are not available, only
        approximate peak values from Figures 8-9.
        """
        if not self._results:
            raise RuntimeError("Run validation first")
        
        omega_s = self.config.omega_s
        if omega_s not in self.EXPERIMENTAL_REFERENCE:
            return {'error': f'No experimental data for omega_s={omega_s}'}
        
        ref = self.EXPERIMENTAL_REFERENCE[omega_s]
        pd = self._results.get('phasor_domain', {})
        
        if not pd:
            return {'error': 'Phasor domain results not available'}
        
        # Get simulation peaks
        is_peak_sim = np.max(np.abs(pd['is_t']))
        vo_peak_sim = np.max(np.abs(pd['vo_t']))
        
        # Compute errors vs experiment
        is_error = 100 * abs(is_peak_sim - ref['is_peak']) / ref['is_peak']
        vo_error = 100 * abs(vo_peak_sim - ref['vo_peak']) / ref['vo_peak']
        
        return {
            'omega_s': omega_s,
            'experimental': ref,
            'simulated': {
                'is_peak': is_peak_sim,
                'vo_peak': vo_peak_sim,
            },
            'errors': {
                'is_error_percent': is_error,
                'vo_error_percent': vo_error,
            },
            'within_3_percent': is_error < 3.0 and vo_error < 3.0
        }


def run_full_benchmark(frequencies: List[float] = None) -> Dict:
    """
    Run benchmark validation at multiple frequencies.
    
    Parameters
    ----------
    frequencies : list of float, optional
        Frequencies to test (default: [580e3, 650e3])
        
    Returns
    -------
    dict
        Results for each frequency
    """
    if frequencies is None:
        frequencies = [580e3, 650e3]
    
    all_results = {}
    
    for omega_s in frequencies:
        print(f"\n{'='*60}")
        print(f"Running benchmark at ωs = {omega_s/1e3:.0f} krad/s")
        print('='*60)
        
        config = BenchmarkConfig(omega_s=omega_s)
        benchmark = Rim2025Benchmark(config)
        results = benchmark.run_validation()
        
        print(results['metrics'])
        
        exp_comparison = benchmark.compare_with_experiment()
        if 'error' not in exp_comparison:
            print(f"\nExperimental comparison:")
            print(f"  Current error: {exp_comparison['errors']['is_error_percent']:.2f}%")
            print(f"  Voltage error: {exp_comparison['errors']['vo_error_percent']:.2f}%")
            print(f"  Within 3%: {exp_comparison['within_3_percent']}")
        
        all_results[omega_s] = {
            'results': results,
            'experimental_comparison': exp_comparison
        }
    
    return all_results
