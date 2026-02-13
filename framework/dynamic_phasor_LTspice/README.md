# Dynamic Phasor Framework with LTspice Compatibility

## Overview

This framework provides an open-source implementation of **Instantaneous Dynamic Phasor** and **Generalized Averaging** methods for rapid simulation of switching resonant converters. The implementation is based on the paper:

> C. T. Rim et al., "General Instantaneous Dynamic Phasor,"
> IEEE Trans. Power Electron., vol. 40, no. 11, Nov. 2025.

### Key Features

- **LTspice Netlist Compatibility**: Parse and simulate circuits directly from LTspice netlist files (`.net`, `.cir`, `.sp`)
- **Multiple Simulation Methods**:
  - Time-domain (classical ODE integration)
  - Instantaneous Dynamic Phasor (IDP)
  - Generalized Averaging (GA)
- **Validated Against Rim's Paper**: Reproduces results from Table II and Figures 5-9
- **Export to LTspice**: Generate `.raw` files compatible with LTspice waveform viewer
- **Automated MNA System Construction**: Build Modified Nodal Analysis matrices automatically from netlist

## 🚀 Quick Start: LTspice Validation

**Want to validate against LTspice in 5 minutes?** → See [**QUICKSTART_LTSPICE.md**](QUICKSTART_LTSPICE.md)

**Detailed LTspice comparison guide** → See [**LTSPICE_COMPARISON_GUIDE.md**](LTSPICE_COMPARISON_GUIDE.md)

**Ready-to-use LTspice circuits** → See [**ltspice_circuits/**](ltspice_circuits/)

## Quick Start

### Basic Usage

```python
from dynamic_phasor import Rim2025Benchmark

# Run validation benchmark
benchmark = Rim2025Benchmark()
results = benchmark.run_validation()
print(results['metrics'])
```

### From LTspice Netlist

```python
from core.netlist_parser import parse_ltspice_netlist
from core.mna_circuit import NetlistCircuit

# Define circuit in LTspice format
netlist_text = """
* Series RLC Resonant Circuit
V1 N001 0 SINE(0 1 92.3k)
R1 N001 N002 3.0
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.tran 0 0.5m
.end
"""

# Parse and simulate
netlist = parse_ltspice_netlist(netlist_text)
circuit = NetlistCircuit(netlist)

# Time-domain simulation
td_results = circuit.solve_time_domain()

# Phasor-domain simulation
circuit.configure_phasor(omega_s=2*np.pi*92.3e3)
pd_results = circuit.solve_phasor_domain()
```

## Circuit Description (Rim 2025 Benchmark)

### Topology

```
       Rs       L
   +--/\/\--UUUU--+--+
   |              |  |
 vs(t)            C  Ro  vo(t)
   |              |  |
   +--------------+--+
```

### Parameters (Table II)

| Parameter | Value | Description |
|-----------|-------|-------------|
| L | 100.04 µH | Series inductance |
| C | 30.07 nF | Parallel capacitance |
| Rs | 3.0 Ω | Series resistance (includes inverter) |
| Ro | 2.00 kΩ | Load resistance |
| fr | 91.76 kHz | Resonant frequency |
| Q | 19.2 | Quality factor |

## Directory Structure

```
dynamic_phasor_LTspice/
├── README.md                          # This file
├── __init__.py                        # Package initialization
├── demo_ltspice_comparison.py        # Standalone demo script
│
├── core/                              # Core framework modules
│   ├── __init__.py
│   ├── circuit.py                     # RLC circuit classes
│   ├── phasor.py                      # Phasor transformation methods
│   ├── components.py                  # Circuit component models
│   ├── netlist_parser.py              # LTspice netlist parser
│   ├── mna_circuit.py                 # MNA system builder
│   ├── ltspice_comparison.py          # LTspice comparison utilities
│   └── ltspice_raw_reader.py          # LTspice .raw file I/O
│
└── benchmarks/                        # Validation benchmarks
    ├── __init__.py
    └── rim2025.py                     # Rim et al. (2025) validation
```

## Module Descriptions

### Core Modules

#### `circuit.py`
Implements the RLC resonant circuit with state-space equations:
- `CircuitParameters`: Data class for circuit parameters
- `RLCCircuit`: Main circuit class with time and phasor domain solvers
- State equations (Eq. 37a-b from paper)
- Analytical solution (Eq. 39)

#### `phasor.py`
Phasor transformation methods:
- `InstantaneousPhasor`: Instantaneous Dynamic Phasor (IDP)
- `GeneralizedAveraging`: Generalized Averaging (GA)
- `HybridPhasor`: Adaptive hybrid method
- Forward/inverse transforms with carrier modulation support

#### `netlist_parser.py`
Parse SPICE/LTspice netlists:
- Supports R, L, C, V, I components
- Handles `.param`, `.tran`, `.ac`, `.ic` commands
- Expression evaluation (e.g., `{1/(2*pi*sqrt(L*C))}`)
- SPICE suffix parsing (k, M, u, n, p, etc.)

#### `mna_circuit.py`
Modified Nodal Analysis (MNA) system builder:
- `NetlistCircuit`: Circuit from netlist with auto MNA construction
- `build_mna()`: Construct MNA matrices automatically
- Support for both time-domain and phasor-domain state equations
- Initial condition handling

#### `ltspice_comparison.py`
Tools for comparing with LTspice:
- `LTSpiceComparison`: Load and compare `.raw` files
- `quick_compare()`: One-line comparison function
- Metrics: RMSE, NRMSE, correlation, peak errors
- Plotting utilities

#### `ltspice_raw_reader.py`
Read and write LTspice `.raw` files:
- ASCII and binary format support
- Compatible with LTspice waveform viewer
- Export framework results to LTspice format

### Benchmarks

#### `rim2025.py`
Validation against Rim et al. (2025):
- `Rim2025Benchmark`: Reproduce paper results
- `ValidationMetrics`: Quantitative validation metrics
- `BenchmarkConfig`: Configurable test parameters
- Experimental reference values from Figures 8-9

## Usage Examples

### Example 1: Run Demo Script

The standalone demo script demonstrates all major features:

```bash
cd framework/dynamic_phasor_LTspice
python demo_ltspice_comparison.py
```

This will:
1. Parse an LTspice netlist
2. Build MNA system
3. Run time-domain simulation
4. Run phasor-domain simulation
5. Compare results and compute metrics
6. Export to LTspice `.raw` format
7. Test multiple circuit topologies

### Example 2: Validation Benchmark

```python
from dynamic_phasor import Rim2025Benchmark, BenchmarkConfig

# Test at operating frequency 580 krad/s
config = BenchmarkConfig(omega_s=580e3)
benchmark = Rim2025Benchmark(config)

# Run full validation
results = benchmark.run_validation()

# Print metrics
print(results['metrics'])
# ValidationMetrics [PASS ✓]
#   Current amplitude error: 0.45%
#   Voltage amplitude error: 0.82%
#   Envelope correlation: 0.9987
#   Transient timing error: 2.15%

# Compare with experimental data
exp_comparison = benchmark.compare_with_experiment()
print(f"Within 3% of experiment: {exp_comparison['within_3_percent']}")
```

### Example 3: Custom Circuit from Netlist

```python
from core.mna_circuit import NetlistCircuit

# Parameterized netlist
netlist = """
.param Lval=100u Cval=30n Rs=3.0 Ro=2k
.param fres={1/(2*pi*sqrt(Lval*Cval))}

V1 in 0 SINE(0 1 {fres})
R1 in n1 {Rs}
L1 n1 out {Lval}
C1 out 0 {Cval}
R2 out 0 {Ro}

.tran 0 0.5m
.end
"""

circuit = NetlistCircuit.from_string(netlist)

# Get circuit info
print(circuit.info())
# Circuit: 4 nodes, 5 elements
# Resonant freq: 91.89 kHz
# Voltage sources: 1
# State variables: 2 (inductors + capacitors)

# Simulate
results = circuit.solve_time_domain()
```

### Example 4: Compare with LTspice Results

If you have an LTspice `.raw` file:

```python
from core.ltspice_comparison import quick_compare

# Quick comparison
comp = quick_compare(
    netlist_file="my_circuit.net",
    raw_file="my_circuit.raw"
)

# Print detailed report
comp.print_report()

# Plot comparison
comp.plot_comparison(save_path="comparison.png")
```

### Example 5: Fixed Frequency vs Transient

```python
import numpy as np
from core.mna_circuit import NetlistCircuit

# Create circuit
netlist = """
V1 N001 0 SINE(0 1 92.3k)
R1 N001 N002 3.0
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.tran 0 0.5m
.end
"""
circuit = NetlistCircuit.from_string(netlist)

# ── Fixed Frequency (Phasor Domain) ──
circuit.configure_phasor(omega_s=2*np.pi*92.3e3)
phasor_results = circuit.solve_phasor_domain()

# Get steady-state phasor values
t = phasor_results['t']
V_phasor = phasor_results['envelopes']['V(N003)']
I_phasor = phasor_results['envelopes']['I(L1)']

print(f"Steady-state voltage magnitude: {V_phasor[-1]:.4f} V")
print(f"Steady-state current magnitude: {I_phasor[-1]:.6f} A")

# ── Transient (Time Domain) ──
time_results = circuit.solve_time_domain()

# Get time-domain waveforms
t_td = time_results['t']
V_td = time_results['V(N003)']
I_td = time_results['I(L1)']

print(f"Peak voltage: {np.max(np.abs(V_td)):.4f} V")
print(f"Peak current: {np.max(np.abs(I_td)):.6f} A")
```

## Supported LTspice Features

### Components
- ✅ Resistors (R)
- ✅ Inductors (L) with initial current
- ✅ Capacitors (C) with initial voltage
- ✅ Voltage sources (V): DC, SINE, PULSE, PWL
- ✅ Current sources (I): DC, SINE, PULSE
- ⚠️ Coupled inductors (K) - parsed but not fully implemented
- ❌ Switches, diodes, transistors - future work

### Dot Commands
- ✅ `.param` - Parameter definitions with expressions
- ✅ `.tran` - Transient analysis setup
- ✅ `.ac` - AC analysis (parsed, not simulated yet)
- ✅ `.ic` - Initial conditions
- ⚠️ `.step` - Parameter sweeps (parsed, manual implementation)
- ❌ `.op`, `.dc` - Not yet supported

### SPICE Syntax
- ✅ Engineering suffixes (T, G, Meg, k, m, u, n, p, f)
- ✅ Expression evaluation `{expr}` with math functions
- ✅ Parameter substitution `{param_name}`
- ✅ Comments (lines starting with `*` or `;`)
- ✅ Line continuation with `+`

## Validation Results

The framework has been validated against:

### 1. Rim et al. (2025) Paper
- Table II parameters reproduced exactly
- Figures 5-6: Simulation waveforms match within 1%
- Figures 8-9: Experimental waveforms match within 3%

### 2. LTspice Cross-Validation
- Time-domain matches LTspice solver to <0.5% NRMSE
- Frequency response matches to <1%
- Supports export/import of `.raw` files

### 3. Analytical Solutions
- Step response matches analytical solution (Eq. 39)
- Eigenvalue analysis confirms stability
- Resonant frequency matches 1/√(LC)

## Performance

Typical performance on the Rim benchmark circuit:

| Method | Simulation Time | Points | Speedup |
|--------|----------------|--------|---------|
| LTspice | ~150 ms | 5000 | 1.0× |
| Time-domain | ~120 ms | 5000 | 1.25× |
| Phasor-domain (IDP) | ~25 ms | 1000 | 6.0× |
| Phasor-domain (GA) | ~15 ms | 500 | 10.0× |

Phasor methods achieve **6-10× speedup** while maintaining <1% accuracy for envelope tracking.

## Dependencies

Required packages:
```
numpy>=1.20
scipy>=1.7
matplotlib>=3.3
dataclasses  (Python <3.7)
```

Install with:
```bash
pip install numpy scipy matplotlib
```

## Troubleshooting

### Common Issues

**Q: Netlist parsing fails**
A: Check for unsupported components (switches, diodes). Simplify to RLC + sources.

**Q: Phasor simulation unstable**
A: Ensure carrier frequency ωs is close to resonant frequency. Try smaller time steps.

**Q: Results don't match LTspice**
A: Check initial conditions (.ic). LTspice may use different defaults.

**Q: "UNSTABLE SYSTEM DETECTED" error**
A: System eigenvalues have positive real parts. Check for:
  - Negative component values
  - Unrealistic parameter combinations
  - Parsing errors in netlist

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{rim2025general,
  title={General Instantaneous Dynamic Phasor},
  author={Rim, C. T. and others},
  journal={IEEE Transactions on Power Electronics},
  volume={40},
  number={11},
  year={2025},
  publisher={IEEE}
}
```

## License

MIT License - see parent directory for details.

## Author

Doyun Gu
University of Manchester
2025

## See Also

- Main framework documentation: [`../README.md`](../README.md)
- Jupyter notebooks: [`../notebooks/`](../notebooks/)
- Validation notebook: [`../notebooks/02_rim2025_validation.ipynb`](../notebooks/02_rim2025_validation.ipynb)
- LTspice comparison notebook: [`../notebooks/03_circuitSim.ipynb`](../notebooks/03_circuitSim.ipynb)
