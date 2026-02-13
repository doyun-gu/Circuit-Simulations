<p align="left">
  <img src="./images/uom-logo.png" alt="The University of Manchester" width="160">
</p>

# Dynamic Phasor Circuit Simulation

Open-source circuit simulation framework using **Modified Nodal Analysis (MNA)** and **Instantaneous Dynamic Phasors** for rapid simulation of switching resonant converters.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository provides a complete implementation of dynamic phasor theory for power electronics circuit simulation, including:

- **MNA-based circuit simulation** with multiple numerical integration methods
- **Instantaneous Dynamic Phasor (IDP)** and **Generalized Averaging** solvers
- **LTspice integration** for cross-validation against industry-standard SPICE
- Validated against Rim et al. (IEEE TPEL 2025) benchmark results (<3% error)

## Repository Structure

| Folder | Description |
|--------|-------------|
| [`theory/`](theory/) | Dynamic phasor theory derivations and mathematical background |
| [`circuit-library/`](circuit-library/) | Basic MNA circuit simulation library built from scratch |
| [`mna-analysis/`](mna-analysis/) | Deep dive into MNA: trapezoidal integration, error analysis, switching circuits |
| [`framework/`](framework/) | Production framework with LTspice integration, benchmarks, and notebooks |

### Suggested Reading Order

1. **Theory** (`theory/`) — Dynamic phasor fundamentals and state-space models
2. **Circuit Library** (`circuit-library/`) — MNA matrix construction and transient analysis
3. **MNA Analysis** (`mna-analysis/`) — Integration methods (Euler, trapezoidal) and dynamic phasor error analysis
4. **Framework** (`framework/`) — Production framework with LTspice netlist parsing and validation

## Quick Start

```bash
cd framework
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_quickstart.ipynb
```

## Framework Notebooks

| Notebook | Description |
|----------|-------------|
| `01_quickstart.ipynb` | Basic phasor transformation and RLC simulation |
| `02_rim2025_validation.ipynb` | Full validation against Rim et al. (2025) at two operating frequencies |
| `03_circuitSim.ipynb` | LTspice netlist parsing, MNA construction, time/phasor domain comparison |
| `04_comparison.ipynb` | Three-way comparison: LTspice vs framework (time-domain) vs framework (phasor) |

## Citation

This framework validates against:

> C. T. Rim et al., "General Instantaneous Dynamic Phasor," *IEEE Trans. Power Electron.*, vol. 40, no. 11, pp. 16953-16962, Nov. 2025.

## Author

**Doyun Gu**, University of Manchester

Supervisor: **Dr. Gus Cheng Zhang**
- Email: [cheng.zhang@manchester.ac.uk](mailto:cheng.zhang@manchester.ac.uk)
- Research profile: [University of Manchester Research Explorer](https://research.manchester.ac.uk/en/persons/cheng.zhang)

## License

MIT License
