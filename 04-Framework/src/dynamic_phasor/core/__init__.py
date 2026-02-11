# Core module init
from .phasor import (
    PhasorConfig,
    PhasorMethod,
    InstantaneousPhasor,
    GeneralizedAveraging,
    HybridPhasor,
    create_phasor_transform,
    compare_methods
)

from .circuit import (
    CircuitParameters,
    RLCCircuit,
    create_rim2025_circuit
)

__all__ = [
    'PhasorConfig',
    'PhasorMethod', 
    'InstantaneousPhasor',
    'GeneralizedAveraging',
    'HybridPhasor',
    'create_phasor_transform',
    'compare_methods',
    'CircuitParameters',
    'RLCCircuit',
    'create_rim2025_circuit',
]
