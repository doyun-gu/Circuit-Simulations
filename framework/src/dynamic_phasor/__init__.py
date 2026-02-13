"""
Dynamic Phasor Simulation Framework
====================================

Open-Source Framework for Rapid Simulation of Switching Resonant Converters
using Instantaneous Dynamic Phasor and Generalized Averaging methods.

Quick Start
-----------
>>> from dynamic_phasor import Rim2025Benchmark
>>> benchmark = Rim2025Benchmark()
>>> results = benchmark.run_validation()
>>> print(results['metrics'])

Or for direct circuit simulation:
>>> from dynamic_phasor import create_rim2025_circuit
>>> circuit = create_rim2025_circuit(omega_s=580e3)
>>> results = circuit.solve_phasor_domain(vs_phasor, t_span=(0, 0.2e-3))

Reference
---------
This framework validates against:
    C. T. Rim et al., "General Instantaneous Dynamic Phasor,"
    IEEE Trans. Power Electron., vol. 40, no. 11, Nov. 2025.

Author: Doyun Gu (University of Manchester)
"""

__version__ = "0.1.0"
__author__ = "Doyun Gu"

# Core components
from .core.phasor import (
    PhasorConfig,
    PhasorMethod,
    InstantaneousPhasor,
    GeneralizedAveraging,
    HybridPhasor,
    create_phasor_transform,
    compare_methods
)

from .core.circuit import (
    CircuitParameters,
    RLCCircuit,
    create_rim2025_circuit
)

# Benchmarks
from .benchmarks.rim2025 import (
    Rim2025Benchmark,
    BenchmarkConfig,
    ValidationMetrics,
    run_full_benchmark
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    # Phasor transforms
    'PhasorConfig',
    'PhasorMethod',
    'InstantaneousPhasor',
    'GeneralizedAveraging',
    'HybridPhasor',
    'create_phasor_transform',
    'compare_methods',
    # Circuits
    'CircuitParameters',
    'RLCCircuit',
    'create_rim2025_circuit',
    # Benchmarks
    'Rim2025Benchmark',
    'BenchmarkConfig',
    'ValidationMetrics',
    'run_full_benchmark',
]
