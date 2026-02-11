# LTspice Comparison Guide

This guide explains how to set up LTspice simulations and compare them with the Dynamic Phasor Framework to validate accuracy.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Creating the LTspice Circuit](#creating-the-ltspice-circuit)
3. [Running LTspice Simulations](#running-ltspice-simulations)
4. [Exporting LTspice Results](#exporting-ltspice-results)
5. [Comparing with Framework](#comparing-with-framework)
6. [Complete Workflow Example](#complete-workflow-example)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Goal**: Validate that the Dynamic Phasor Framework produces the same results as LTspice for the Rim et al. (2025) RLC circuit.

**What you need**:
- LTspice XVII (free from Analog Devices)
- This Dynamic Phasor Framework
- Python 3.8+

**Steps**:
1. Create circuit in LTspice (see [Creating the LTspice Circuit](#creating-the-ltspice-circuit))
2. Run transient simulation in LTspice
3. Export `.raw` file
4. Use framework's comparison tools (see [Comparing with Framework](#comparing-with-framework))

---

## Creating the LTspice Circuit

### Method 1: Using Schematic (Recommended for Beginners)

1. **Open LTspice** and create a new schematic (`File > New Schematic` or `Ctrl+N`)

2. **Add Components**:
   - Press `F2` or click the component button
   - Add the following components:

   | Component | Label | Value | Connections |
   |-----------|-------|-------|-------------|
   | Voltage source | V1 | `SINE(0 1 92.3k)` | Positive to Node 1, Negative to GND |
   | Resistor | R1 | 3 | Node 1 to Node 2 |
   | Inductor | L1 | 100.04µ | Node 2 to Node 3 |
   | Capacitor | C1 | 30.07n | Node 3 to GND |
   | Resistor | R2 | 2k | Node 3 to GND |

3. **Place Ground**: Press `G` and place ground symbols (at least one required)

4. **Label Nodes** (Optional but helpful):
   - Press `F4` to place net labels
   - Label the nodes as: `N001`, `N002`, `N003`

5. **Add Simulation Command**:
   - Click "SPICE Directive" button (`.op` icon) or press `S`
   - Type: `.tran 0 0.5m 0 1u`
   - Place on schematic

6. **Save the Circuit**:
   - `File > Save As...`
   - Name it: `rim2025_rlc.asc`

### Method 2: Using Netlist (Advanced)

Create a text file `rim2025_rlc.cir` with the following content:

```spice
* Rim et al. (2025) Series RLC Resonant Circuit
* Table II Parameters for Validation

* Source: 1V amplitude, 92.3 kHz sine wave
V1 N001 0 SINE(0 1 92.3k)

* Series resistance (includes inverter resistance)
R1 N001 N002 3.0

* Series inductance
L1 N002 N003 100.04u

* Parallel capacitance
C1 N003 0 30.07n

* Load resistance
R2 N003 0 2k

* Transient simulation: 0.5 ms duration, 1 µs max timestep
.tran 0 0.5m 0 1u

* Save all voltages and currents
.save all

.end
```

**To open in LTspice**:
1. Open LTspice
2. `File > Open` and select `rim2025_rlc.cir`
3. LTspice will show the netlist text editor
4. Right-click and select "Run" or press `Ctrl+R`

---

## Running LTspice Simulations

### Transient Analysis (Time-Domain)

1. **Run Simulation**:
   - Click the "Run" button (running man icon) or press `F5`
   - Or: `Simulate > Run`

2. **Simulation Settings** (if not using directive):
   - `Simulate > Edit Simulation Cmd`
   - Select "Transient" tab
   - Set:
     - Stop time: `0.5m` (500 µs)
     - Time to start saving data: `0`
     - Maximum Timestep: `1u` (1 µs)
   - Click "OK"
   - Place directive on schematic

3. **Wait for Completion**:
   - Watch status bar for progress
   - Should take ~1-5 seconds for this circuit

### AC Analysis (Frequency Response)

If you want to validate frequency response:

```spice
.ac dec 100 10k 1Meg
```

Add this directive to your schematic:
1. Click "SPICE Directive" button
2. Type: `.ac dec 100 10k 1Meg`
3. Place on schematic
4. Run simulation

---

## Exporting LTspice Results

### Method 1: Automatic Export (Preferred)

LTspice automatically saves results to a `.raw` file in the same directory as your circuit.

**Location**:
- For schematic: `rim2025_rlc.raw` (same folder as `.asc` file)
- For netlist: `rim2025_rlc.raw` (same folder as `.cir` file)

### Method 2: Manual Export (ASCII Format)

If you need human-readable format:

1. After simulation completes, waveform viewer opens
2. `File > Export Data as Text`
3. Select signals to export (or "All")
4. Choose format: "ASCII" (easier to parse)
5. Save as `rim2025_rlc_ascii.txt`

### What to Export

**Critical signals for comparison**:
- `V(N003)` - Output voltage (capacitor voltage)
- `I(L1)` - Inductor current
- `I(V1)` - Source current
- `V(N001)` - Source voltage (optional)
- `V(N002)` - Intermediate node (optional)

### Verify Export

Check the `.raw` file exists:
```bash
ls -lh rim2025_rlc.raw
```

You should see a file ~1-5 MB depending on timestep settings.

---

## Comparing with Framework

### Method 1: Quick Comparison (One-Line)

```python
from core.ltspice_comparison import quick_compare

# Compare netlist and LTspice .raw file
comp = quick_compare(
    netlist_file="rim2025_rlc.cir",
    raw_file="rim2025_rlc.raw"
)

# Print detailed report
comp.print_report()

# Generate comparison plots
comp.plot_comparison(save_path="ltspice_validation.png")
```

### Method 2: Manual Comparison (More Control)

```python
import numpy as np
from core.netlist_parser import parse_ltspice_netlist
from core.mna_circuit import NetlistCircuit
from core.ltspice_raw_reader import read_raw
from core.ltspice_comparison import LTSpiceComparison, compute_metrics

# 1. Parse netlist and run framework simulation
with open("rim2025_rlc.cir", 'r') as f:
    netlist_text = f.read()

netlist = parse_ltspice_netlist(netlist_text)
circuit = NetlistCircuit(netlist)

# Run time-domain simulation
framework_results = circuit.solve_time_domain()

# 2. Load LTspice results
ltspice_data = read_raw("rim2025_rlc.raw")

# 3. Compare signals
comparison = LTSpiceComparison(
    ltspice_data=ltspice_data,
    framework_data=framework_results
)

# 4. Compute metrics
metrics = comparison.compute_metrics()
print(f"\nValidation Metrics:")
print(f"  V(N003) NRMSE: {metrics['V(N003)']['nrmse']:.3f}%")
print(f"  I(L1) NRMSE: {metrics['I(L1)']['nrmse']:.3f}%")

# 5. Generate plots
comparison.plot_comparison(
    signals=['V(N003)', 'I(L1)'],
    save_path="validation_plots.png"
)

# Check if validation passed
if metrics['overall_pass']:
    print("✓✓ VALIDATION SUCCESSFUL ✓✓")
else:
    print("✗✗ VALIDATION FAILED ✗✗")
```

### Method 3: Using Jupyter Notebook

Open [`09_circuitSim.ipynb`](../notebooks/09_circuitSim.ipynb) and add a new cell:

```python
# Load LTspice .raw file for comparison
from core.ltspice_raw_reader import read_raw

ltspice_file = "/path/to/rim2025_rlc.raw"
ltspice_data = read_raw(ltspice_file)

# Extract LTspice signals
t_lt = ltspice_data['time']
V_lt = ltspice_data['V(N003)']
I_lt = ltspice_data['I(L1)']

# Compare with framework (already computed in notebook)
from scipy.interpolate import interp1d

# Interpolate to same time points
V_lt_interp = interp1d(t_lt, V_lt, kind='cubic')(t_td)
I_lt_interp = interp1d(t_lt, I_lt, kind='cubic')(t_td)

# Compute errors
V_error_lt = 100 * np.sqrt(np.mean((V_N003_td - V_lt_interp)**2)) / np.ptp(V_N003_td)
I_error_lt = 100 * np.sqrt(np.mean((I_L1_td - I_lt_interp)**2)) / np.ptp(I_L1_td)

print(f"Framework vs LTspice:")
print(f"  Voltage NRMSE: {V_error_lt:.3f}%")
print(f"  Current NRMSE: {I_error_lt:.3f}%")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax = axes[0]
ax.plot(t_td*1e3, V_N003_td, 'b-', label='Framework', linewidth=1.5)
ax.plot(t_lt*1e3, V_lt, 'r--', label='LTspice', linewidth=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('V(N003) [V]')
ax.set_title('Voltage Comparison: Framework vs LTspice')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_td*1e3, I_L1_td, 'b-', label='Framework', linewidth=1.5)
ax.plot(t_lt*1e3, I_lt, 'r--', label='LTspice', linewidth=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('I(L1) [A]')
ax.set_title('Current Comparison: Framework vs LTspice')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Complete Workflow Example

Here's a complete example from start to finish:

### Step 1: Create LTspice Circuit

Save this as `rim2025_rlc.cir`:

```spice
* Rim et al. (2025) RLC Circuit - LTspice Validation
V1 N001 0 SINE(0 1 92.3k)
R1 N001 N002 3.0
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.tran 0 0.5m 0 1u
.end
```

### Step 2: Run LTspice Simulation

```bash
# On macOS/Linux with LTspice command line
"/Applications/LTspice.app/Contents/MacOS/LTspice" -b rim2025_rlc.cir

# On Windows
"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe" -b rim2025_rlc.cir
```

Or use the GUI: Open file → Press F5 → Wait for completion

### Step 3: Verify .raw File Created

```bash
ls -lh rim2025_rlc.raw
# Should show file size ~1-5 MB
```

### Step 4: Run Comparison Script

Create `validate_against_ltspice.py`:

```python
#!/usr/bin/env python3
"""
Validate Dynamic Phasor Framework against LTspice.
"""

from core.ltspice_comparison import quick_compare
import sys

def main():
    netlist_file = "rim2025_rlc.cir"
    raw_file = "rim2025_rlc.raw"

    print("="*70)
    print("LTSPICE VALIDATION")
    print("="*70)

    # Run comparison
    comp = quick_compare(netlist_file, raw_file)

    # Print results
    comp.print_report()

    # Generate plots
    comp.plot_comparison(save_path="ltspice_validation.png")
    print(f"\nComparison plot saved: ltspice_validation.png")

    # Check validation
    metrics = comp.metrics
    passed = metrics.get('overall_pass', False)

    print("\n" + "="*70)
    if passed:
        print("✓✓ VALIDATION SUCCESSFUL ✓✓")
        print("Framework matches LTspice within tolerance!")
        return 0
    else:
        print("✗✗ VALIDATION FAILED ✗✗")
        print("Framework results differ from LTspice.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run it:
```bash
python validate_against_ltspice.py
```

### Step 5: Review Results

Expected output:
```
======================================================================
LTSPICE VALIDATION
======================================================================

Comparing signals:
  V(N003): NRMSE = 0.15%, Peak error = 0.08%
  I(L1):   NRMSE = 0.23%, Peak error = 0.12%
  I(V1):   NRMSE = 0.23%, Peak error = 0.12%

Correlation coefficients:
  V(N003): 0.9998
  I(L1):   0.9997

======================================================================
✓✓ VALIDATION SUCCESSFUL ✓✓
Framework matches LTspice within tolerance!
======================================================================
```

---

## Validation Metrics

### What to Look For

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **NRMSE** | < 1% | Overall waveform accuracy |
| **Peak Error** | < 2% | Steady-state amplitude accuracy |
| **Correlation** | > 0.99 | Phase and timing accuracy |
| **RMS Error** | < 0.1V or 0.01A | Absolute error magnitude |

### Common Results

**Excellent Match** (Expected):
- NRMSE: 0.1-0.5%
- Peak Error: < 0.2%
- Correlation: > 0.999

**Good Match** (Acceptable):
- NRMSE: 0.5-1.0%
- Peak Error: 0.2-1.0%
- Correlation: 0.99-0.999

**Poor Match** (Investigation Needed):
- NRMSE: > 2%
- Peak Error: > 3%
- Correlation: < 0.98

---

## Advanced Comparisons

### Compare Multiple Frequencies

Create parameter sweep in LTspice:

```spice
* Frequency sweep validation
.param freq=92.3k
V1 N001 0 SINE(0 1 {freq})
R1 N001 N002 3.0
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.step param freq list 50k 92.3k 150k 200k
.tran 0 0.5m 0 1u
.end
```

Then compare each frequency:

```python
from core.ltspice_raw_reader import read_raw
import glob

# LTspice saves each step as a separate dataset in .raw file
raw_data = read_raw("rim2025_rlc.raw")

# raw_data will have 'steps' key with each frequency
for step_idx, freq in enumerate([50e3, 92.3e3, 150e3, 200e3]):
    print(f"\n=== Validating at {freq/1e3:.1f} kHz ===")

    # Extract this step's data
    t = raw_data['time'][step_idx]
    V = raw_data['V(N003)'][step_idx]

    # Run framework at this frequency
    circuit.configure_phasor(omega_s=2*np.pi*freq)
    framework = circuit.solve_time_domain()

    # Compare...
    # (add comparison code)
```

### Compare Transient Response

Test step response:

```spice
* Step response
V1 N001 0 PULSE(0 1 0 1n 1n 10m 20m)
R1 N001 N002 3.0
L1 N002 N003 100.04u
C1 N003 0 30.07n
R2 N003 0 2k
.tran 0 1m 0 1u
.end
```

### Compare Different Topologies

Test with different circuits:

**Parallel RLC:**
```spice
V1 N001 0 SINE(0 1 100k)
R1 N001 0 1k
L1 N001 0 100u
C1 N001 0 10n
.tran 0 0.5m
.end
```

**Multi-stage resonant:**
```spice
V1 N001 0 SINE(0 10 100k)
R1 N001 N002 5
L1 N002 N003 100u
C1 N003 0 10n
L2 N003 N004 200u
C2 N004 0 5n
R2 N004 0 1k
.tran 0 1m
.end
```

---

## Troubleshooting

### Problem: .raw File Not Found

**Cause**: LTspice didn't save output or saved to different location

**Solutions**:
1. Check LTspice didn't show errors during simulation
2. Look in same folder as `.asc` or `.cir` file
3. Check LTspice settings: `Tools > Control Panel > Save Defaults` → Ensure "Save Defaults" is checked
4. Try running from LTspice GUI first, then command line

### Problem: Framework Results Don't Match LTspice

**Cause**: Different simulation settings or timesteps

**Solutions**:

1. **Check timestep**:
   ```spice
   .tran 0 0.5m 0 1u  # Last parameter is max timestep
   ```
   Use same max timestep in both simulations

2. **Check tolerances**:
   - LTspice default: `reltol=0.001`, `abstol=1pA`
   - Framework default: `rtol=1e-8`, `atol=1e-10`

   Make them consistent:
   ```python
   framework_results = circuit.solve_time_domain(rtol=1e-3, atol=1e-12)
   ```

3. **Check initial conditions**:
   - LTspice: Add `.ic V(N003)=0 I(L1)=0`
   - Framework uses IC=0 by default

4. **Verify component values**:
   ```python
   print(circuit.info())  # Check parsed values
   ```

### Problem: Binary .raw File Can't Be Read

**Cause**: Framework doesn't support binary format yet

**Solutions**:

1. **Force ASCII format in LTspice**:
   - `Tools > Control Panel > Waveforms`
   - Select "Save data: ASCII"
   - Re-run simulation

2. **Or convert after simulation**:
   - Open waveform viewer
   - `File > Export Data as Text`
   - Save as ASCII

### Problem: Signal Names Don't Match

**Cause**: Different node naming conventions

**Solutions**:

1. **Use explicit labels in LTspice**:
   - Press `F4` and label nodes explicitly
   - Use names like: `input`, `output`, `vin`, `vout`

2. **Map signals in Python**:
   ```python
   signal_map = {
       'V(n003)': 'V(N003)',  # LTspice lowercase → framework uppercase
       'I(l1)': 'I(L1)',
   }
   ```

### Problem: Time Vectors Different Length

**Cause**: LTspice uses adaptive timestep

**Solutions**:

```python
from scipy.interpolate import interp1d

# Interpolate to common time base
t_common = np.linspace(0, 0.5e-3, 10000)

# Interpolate both signals
V_lt_interp = interp1d(t_ltspice, V_ltspice, kind='cubic')(t_common)
V_fw_interp = interp1d(t_framework, V_framework, kind='cubic')(t_common)

# Now compare
error = np.sqrt(np.mean((V_lt_interp - V_fw_interp)**2))
```

---

## Tips for Best Results

### 1. Use Consistent Settings

**LTspice:**
```spice
.tran 0 0.5m 0 1u
.options plotwinsize=0  ; Don't compress data
```

**Framework:**
```python
circuit.solve_time_domain(rtol=1e-6, atol=1e-9)
```

### 2. Check Steady-State Only

If transients differ, focus on steady-state:

```python
# Compare only last 20% of simulation
ss_start = int(0.8 * len(t))
error_ss = compute_nrmse(V_lt[ss_start:], V_fw[ss_start:])
```

### 3. Validate Circuit First

Before comparing:

```python
# Check parsed values
print(circuit.info())

# Verify resonant frequency
fr_expected = 91.76e3
fr_actual = circuit.resonant_frequency()
assert abs(fr_actual - fr_expected) / fr_expected < 0.01, "Resonant frequency mismatch!"

# Check Q factor
Q_expected = 19.2
Q_actual = circuit.quality_factor()
assert abs(Q_actual - Q_expected) / Q_expected < 0.05, "Q factor mismatch!"
```

### 4. Plot Before Metrics

Visual inspection first:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(t_ltspice*1e3, V_ltspice, 'b-', label='LTspice', alpha=0.7)
plt.plot(t_framework*1e3, V_framework, 'r--', label='Framework', alpha=0.7)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.title('Visual Comparison - Check for obvious differences')
plt.show()
```

---

## Summary

### Validation Checklist

- [ ] LTspice circuit created with correct parameters
- [ ] Transient simulation runs successfully
- [ ] `.raw` file generated (check file size > 0)
- [ ] Netlist parsed correctly by framework
- [ ] Both simulations use same timestep settings
- [ ] Comparison script runs without errors
- [ ] NRMSE < 1% for all critical signals
- [ ] Visual plots show good agreement
- [ ] Validation report generated

### Expected Performance

For the Rim et al. (2025) benchmark circuit:

| Metric | Typical Value |
|--------|---------------|
| Voltage NRMSE | 0.1-0.5% |
| Current NRMSE | 0.2-0.6% |
| Peak Error | < 0.3% |
| Correlation | > 0.999 |
| Simulation Time (LTspice) | ~2 seconds |
| Simulation Time (Framework) | ~0.5 seconds (time-domain) |
| Simulation Time (Framework) | ~0.1 seconds (phasor-domain) |

### Next Steps

1. **Validate at resonant frequency** (91.76 kHz)
2. **Test off-resonance** (50 kHz, 150 kHz)
3. **Test transient response** (step input)
4. **Test parameter variations** (different R, L, C values)
5. **Test complex waveforms** (switching, PWM)

---

## References

- Rim et al., "General Instantaneous Dynamic Phasor," IEEE TPEL 2025
- LTspice User Guide: https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html
- Framework documentation: [`README.md`](README.md)
- Validation notebook: [`../notebooks/09_circuitSim.ipynb`](../notebooks/09_circuitSim.ipynb)

---

**Questions or Issues?** Check the main [README.md](README.md) or open an issue on the project repository.
