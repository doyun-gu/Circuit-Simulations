# Dynamic Phasor Simulation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Open-Source Dynamic Phasor Framework for Rapid Simulation of Switching Resonant Converters**

## Overview

This framework implements both **Instantaneous Dynamic Phasor (IDP)** and **Generalized Averaging** methods for power electronics circuit simulation, with a novel **hybrid approach** that automatically selects the optimal method based on operating conditions.

### Key Features

- **Dual-Method Support**: Switch between instantaneous and averaged phasor representations
- **<3% Accuracy**: Validated against published experimental results
- **Hybrid Solver**: Automatically selects optimal method for each operating region
- **Comprehensive Benchmarks**: Includes Rim et al. (IEEE TPEL 2025) validation
- **Publication-Ready Plots**: Generate figures suitable for IEEE publications

## Theoretical Background

### Instantaneous Dynamic Phasor (Eq. 1 from Rim et al.)

```
x(t) = Re{(1/√m) · x̃(t) · e^(jθ(t))}
```

Where:
- `x(t)` is the real-space variable
- `x̃(t)` is the complex phasor
- `θ(t)` is an arbitrary phase angle (can be time-varying)
- `m` = 1 for single-phase, 3 for three-phase

### Generalized Averaging (Eq. 13 from Sanders et al.)

```
⟨x⟩_k(t) = (1/T) ∫₀ᵀ x(t-T+s) · e^(-jkωs(t-T+s)) ds
```

### Hybrid Approach (This Work)

Automatically switches between methods based on:
- Signal bandwidth relative to carrier frequency
- Switching event detection
- Envelope variation rate

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dynamic-phasor-sim.git
cd dynamic-phasor-sim

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from dynamic_phasor import RLCCircuit, HybridSolver
from dynamic_phasor.benchmarks import Rim2025Benchmark

# Load benchmark circuit (Table II from Rim et al.)
benchmark = Rim2025Benchmark()
circuit = benchmark.create_circuit()

# Create hybrid solver
solver = HybridSolver(circuit, omega_s=580e3)

# Simulate step response
t, results = solver.simulate(t_end=0.2e-3)

# Compare with experimental data
benchmark.validate(t, results)
```

## Benchmark Validation

### Rim et al. (IEEE TPEL 2025) - Table II Parameters

| Parameter | Value |
|-----------|-------|
| L | 100.04 µH |
| C | 30.07 nF |
| Rs | 3.0 Ω |
| Ro | 2.00 kΩ |
| ωs | 580, 650 krad/s |

### Validation Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Current amplitude | <3% error | ✓ |
| Voltage amplitude | <3% error | ✓ |
| Envelope correlation | >0.95 | ✓ |
| Transient timing | <10% error | ✓ |

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{gu2026dynamic,
  title={Open-Source Dynamic Phasor Framework for Rapid Simulation of 
         Switching Resonant Converters: Implementation and Multi-Benchmark Validation},
  author={Gu, Doyun},
  journal={IEEE Applied Power Electronics Conference (APEC)},
  year={2026}
}
```

### Reference Implementation

This framework validates against:

```bibtex
@article{rim2025general,
  title={General Instantaneous Dynamic Phasor},
  author={Rim, Chun T. and Shah, Syed Ahson Ali and Park, Hyo J. and 
          Routray, Abhinandan and Jeong, Seog Y. and Chung, Youngjoo},
  journal={IEEE Transactions on Power Electronics},
  volume={40},
  number={11},
  pages={16953--16962},
  year={2025}
}
```

## License

<!-- MIT License - see [LICENSE](LICENSE) for details. -->

## Contributing

<!-- Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first. -->
