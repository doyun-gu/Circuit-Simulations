# Dynamic Phasor Simulation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Open-Source Dynamic Phasor Framework for Rapid Simulation of Switching Resonant Converters**

## Overview

This framework implements the **Instantaneous Dynamic Phasor (IDP)** method for power electronics circuit simulation, with **LTspice integration** for cross-validation.

### Key Features

- **Instantaneous Dynamic Phasor**: Phasor-domain simulation without period averaging
- **MNA-Based Circuit Construction**: Automatic state-space from LTspice netlists
- **<3% Accuracy**: Validated against Rim et al. (IEEE TPEL 2025) benchmark
- **LTspice Integration**: Parse netlists, compare with SPICE results

## Theoretical Background

### Instantaneous Dynamic Phasor (Eq. 1 from Rim et al.)

$$x(t) = \text{Re}\left\lbrace\frac{1}{\sqrt{m}}\,\tilde{x}(t)\,e^{j\theta(t)}\right\rbrace$$

Where:
- $x(t)$ is the real-space variable
- $\tilde{x}(t)$ is the complex phasor
- $\theta(t)$ is an arbitrary phase angle (can be time-varying)
- $m = 1$ for single-phase, $m = 3$ for three-phase

The key advantage of IDP over classical generalized averaging (Sanders et al., 1991) is that it defines the phasor at every instant without requiring period-averaged integration, enabling accurate tracking of fast transients and switching events.

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
| $L$ | 100.04 µH |
| $C$ | 30.07 nF |
| $R_s$ | 3.0 $\Omega$ |
| $R_o$ | 2.00 k$\Omega$ |
| $\omega_s$ | 580, 650 krad/s |

### Validation Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Current amplitude | <3% error | ✓ |
| Voltage amplitude | <3% error | ✓ |
| Envelope correlation | >0.95 | ✓ |
| Transient timing | <10% error | ✓ |

## Citation

If you use this framework in your research, please cite:

### Reference

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
