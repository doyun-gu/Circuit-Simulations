"""
Basic tests for dynamic phasor framework.

Run with: pytest tests/test_phasor.py -v
"""

import numpy as np
import os
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dynamic_phasor.core.phasor import (
    PhasorConfig, InstantaneousPhasor, GeneralizedAveraging
)
from dynamic_phasor.core.circuit import (
    CircuitParameters, RLCCircuit, create_rim2025_circuit
)


class TestPhasorTransform:
    """Test phasor transformation operations."""
    
    def test_instantaneous_roundtrip(self):
        """Test that to_phasor -> to_real recovers original signal."""
        omega = 10e3
        t = np.linspace(0, 1e-3, 1000)
        x_real = np.cos(omega * t)
        
        config = PhasorConfig(omega=omega)
        phasor = InstantaneousPhasor(config)
        
        x_phasor = phasor.to_phasor(x_real, t)
        x_recovered = phasor.to_real(x_phasor, t)
        
        np.testing.assert_allclose(x_real, x_recovered, rtol=1e-10)
    
    def test_envelope_extraction(self):
        """Test that phasor magnitude equals envelope."""
        omega = 10e3
        t = np.linspace(0, 1e-3, 1000)
        envelope = np.exp(-t * 1000)  # Decaying envelope
        x_real = envelope * np.cos(omega * t)
        
        config = PhasorConfig(omega=omega)
        phasor = InstantaneousPhasor(config)
        
        x_phasor = phasor.to_phasor(x_real, t)
        extracted_env = np.abs(x_phasor)
        
        np.testing.assert_allclose(extracted_env, envelope, rtol=1e-10)
    
    def test_reactance_values(self):
        """Test time-varying reactance calculations."""
        omega = 1000
        L = 100e-6
        C = 30e-9
        t = 0.0
        
        config = PhasorConfig(omega=omega)
        phasor = InstantaneousPhasor(config)
        
        XL = phasor.reactance_L(L, t)
        XC = phasor.reactance_C(C, t)
        
        assert np.isclose(XL, omega * L)
        assert np.isclose(XC, -1 / (omega * C))


class TestCircuit:
    """Test circuit simulation."""
    
    def test_rim2025_parameters(self):
        """Test that parameters match Table II."""
        params = CircuitParameters()
        
        assert np.isclose(params.L, 100.04e-6, rtol=0.01)
        assert np.isclose(params.C, 30.07e-9, rtol=0.01)
        assert np.isclose(params.Rs, 3.0)
        assert np.isclose(params.Ro, 2000.0)
    
    def test_resonant_frequency(self):
        """Test computed resonant frequency."""
        params = CircuitParameters()
        expected_omega_r = 1.0 / np.sqrt(params.L * params.C)
        
        assert np.isclose(params.omega_r, expected_omega_r)
    
    def test_time_domain_simulation(self):
        """Test basic time domain simulation runs."""
        circuit = create_rim2025_circuit(omega_s=580e3)
        
        def vs(t):
            return np.cos(580e3 * t) if t >= 0 else 0.0
        
        results = circuit.solve_time_domain(vs, t_span=(0, 0.1e-3))
        
        assert 't' in results
        assert 'is_t' in results
        assert 'vo_t' in results
        assert len(results['t']) > 0
    
    def test_phasor_domain_simulation(self):
        """Test basic phasor domain simulation runs."""
        circuit = create_rim2025_circuit(omega_s=580e3)
        
        def vs_phasor(t):
            return 1.0 + 0.0j if t >= 0 else 0.0 + 0.0j
        
        results = circuit.solve_phasor_domain(vs_phasor, t_span=(0, 0.1e-3))
        
        assert 't' in results
        assert 'is_phasor' in results
        assert 'vo_phasor' in results
        assert 'is_envelope' in results


class TestBenchmark:
    """Test benchmark validation."""
    
    def test_benchmark_runs(self):
        """Test that benchmark runs without errors."""
        from dynamic_phasor.benchmarks.rim2025 import (
            Rim2025Benchmark, BenchmarkConfig
        )
        
        config = BenchmarkConfig(omega_s=580e3, t_end=0.05e-3)
        benchmark = Rim2025Benchmark(config)
        results = benchmark.run_validation()
        
        assert 'metrics' in results
        assert 'time_domain' in results
        assert 'phasor_domain' in results
    
    def test_validation_metrics(self):
        """Test that validation produces reasonable metrics."""
        from dynamic_phasor.benchmarks.rim2025 import (
            Rim2025Benchmark, BenchmarkConfig
        )
        
        config = BenchmarkConfig(omega_s=580e3, t_end=0.1e-3)
        benchmark = Rim2025Benchmark(config)
        results = benchmark.run_validation()
        
        metrics = results['metrics']
        
        # Amplitude errors should be small (phasor should match time-domain)
        assert metrics.amplitude_error_is < 5.0  # Less than 5%
        assert metrics.amplitude_error_vo < 5.0
        
        # Envelope correlation should be high
        assert metrics.envelope_correlation > 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
