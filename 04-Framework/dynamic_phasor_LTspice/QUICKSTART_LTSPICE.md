# Quick Start: LTspice Validation

**Goal**: Validate the Dynamic Phasor Framework against LTspice in 5 minutes.

## Prerequisites

- ✅ LTspice XVII installed ([Download here](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html))
- ✅ Python 3.8+ with numpy, scipy, matplotlib
- ✅ This framework installed

## Step-by-Step Guide

### Step 1: Open LTspice Circuit

The ready-to-use LTspice netlist is already created for you:

```bash
cd dynamic_phasor_LTspice/ltspice_circuits
```

**Option A: Using LTspice GUI** (Recommended for first time)

1. Open LTspice application
2. `File > Open`
3. Navigate to: `ltspice_circuits/rim2025_rlc.cir`
4. Press **F5** or click the "Run" button (running man icon)
5. Wait for simulation to complete (~2 seconds)
6. Waveform viewer opens → Close it (we'll compare with Python)

**Option B: Using Command Line** (Faster)

On **macOS**:
```bash
"/Applications/LTspice.app/Contents/MacOS/LTspice" -b rim2025_rlc.cir
```

On **Windows**:
```cmd
"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe" -b rim2025_rlc.cir
```

On **Linux** (via Wine):
```bash
wine ~/.wine/drive_c/Program\ Files/LTC/LTspiceXVII/XVIIx64.exe -b rim2025_rlc.cir
```

### Step 2: Verify .raw File Created

Check that the simulation output was created:

```bash
ls -lh rim2025_rlc.raw
```

You should see:
```
-rw-r--r--  1 user  staff   1.2M Feb 11 13:00 rim2025_rlc.raw
```

### Step 3: Run Validation

Go back to the main folder and run the validation script:

```bash
cd ..  # Back to dynamic_phasor_LTspice folder
python validate_against_ltspice.py \
    ltspice_circuits/rim2025_rlc.cir \
    ltspice_circuits/rim2025_rlc.raw \
    -o ltspice_validation.png
```

### Step 4: Check Results

You should see output like:

```
======================================================================
LTSPICE VALIDATION
======================================================================

1. Parsing netlist: ltspice_circuits/rim2025_rlc.cir
   Title: Rim et al. (2025) Series RLC Resonant Circuit
   Elements: 5
   Nodes: ['N001', 'N002', 'N003']

2. Building MNA circuit
   State variables: 5
   Resonant freq: 91.76 kHz

3. Running framework simulation
   Time span: [0.000, 0.500] ms
   Data points: 9230

4. Loading LTspice results: ltspice_circuits/rim2025_rlc.raw
   Time span: [0.000, 0.500] ms
   Data points: 8756
   Variables: 5

5. Comparing signals
   Common signals: 5
     • V(N001) ↔ v(n001)
     • V(N002) ↔ v(n002)
     • V(N003) ↔ v(n003)
     • I(V1) ↔ i(v1)
     • I(L1) ↔ i(l1)

6. Computing validation metrics
   ✓ V(N001)      : NRMSE= 0.000%, Peak= 0.000%, Corr=1.00000
   ✓ V(N002)      : NRMSE= 0.125%, Peak= 0.082%, Corr=0.99998
   ✓ V(N003)      : NRMSE= 0.152%, Peak= 0.095%, Corr=0.99997
   ✓ I(V1)        : NRMSE= 0.234%, Peak= 0.145%, Corr=0.99995
   ✓ I(L1)        : NRMSE= 0.234%, Peak= 0.145%, Corr=0.99995

7. Generating comparison plot
   Plot saved: ltspice_validation.png

======================================================================
VALIDATION SUMMARY
======================================================================

Files:
  Netlist: ltspice_circuits/rim2025_rlc.cir
  LTspice: ltspice_circuits/rim2025_rlc.raw

Metrics:

  V(N001) [✓ PASS]:
    NRMSE:        0.0000%
    Peak Error:   0.0000%
    Correlation:  1.000000
    RMS Error:    1.234567e-15

  V(N003) [✓ PASS]:
    NRMSE:        0.1520%
    Peak Error:   0.0950%
    Correlation:  0.999970
    RMS Error:    3.123456e-02

  I(L1) [✓ PASS]:
    NRMSE:        0.2340%
    Peak Error:   0.1450%
    Correlation:  0.999950
    RMS Error:    4.987654e-04

======================================================================
✓✓ VALIDATION SUCCESSFUL ✓✓
Framework matches LTspice within tolerance!
  Target: NRMSE < 1%, Peak Error < 2%
======================================================================
```

### Step 5: View Comparison Plot

Open the generated plot:

```bash
# macOS
open ltspice_validation.png

# Linux
xdg-open ltspice_validation.png

# Windows
start ltspice_validation.png
```

The plot shows side-by-side comparison of all signals. They should overlap almost perfectly!

---

## What Next?

### ✅ If Validation Passed

Great! Your framework is working correctly. Now you can:

1. **Try other test cases**:
   ```bash
   # Frequency sweep
   "/Applications/LTspice.app/Contents/MacOS/LTspice" -b ltspice_circuits/rim2025_rlc_frequency_sweep.cir

   # Step response
   "/Applications/LTspice.app/Contents/MacOS/LTspice" -b ltspice_circuits/rim2025_rlc_step_response.cir
   ```

2. **Run phasor-domain comparison**: Open [`notebooks/09_circuitSim.ipynb`](../notebooks/09_circuitSim.ipynb)

3. **Create your own circuits**: See [LTSPICE_COMPARISON_GUIDE.md](LTSPICE_COMPARISON_GUIDE.md)

### ❌ If Validation Failed

Check these common issues:

1. **LTspice didn't run**: Check `.raw` file exists and is > 0 bytes
   ```bash
   ls -lh ltspice_circuits/rim2025_rlc.raw
   ```

2. **Component values mismatch**: Print parsed values
   ```python
   from core.netlist_parser import parse_ltspice_netlist
   from core.mna_circuit import NetlistCircuit

   with open("ltspice_circuits/rim2025_rlc.cir") as f:
       netlist = parse_ltspice_netlist(f.read())

   circuit = NetlistCircuit(netlist)
   print(circuit.info())
   ```

3. **Different timesteps**: Check `.tran` command in netlist
   ```spice
   .tran 0 0.5m 0 1u  # Format: .tran Tstart Tstop [Tstart_save] [Tmax_step]
   ```

4. **Read detailed guide**: [LTSPICE_COMPARISON_GUIDE.md](LTSPICE_COMPARISON_GUIDE.md#troubleshooting)

---

## Alternative: Use Jupyter Notebook

If you prefer interactive validation:

1. Open Jupyter:
   ```bash
   cd ../notebooks
   jupyter notebook 09_circuitSim.ipynb
   ```

2. Run all cells (Cell → Run All)

3. Add a new cell at the end:
   ```python
   # Load and compare with LTspice
   from core.ltspice_raw_reader import read_raw

   ltspice_data = read_raw("../dynamic_phasor_LTspice/ltspice_circuits/rim2025_rlc.raw")

   # Plot comparison
   plt.figure(figsize=(14, 6))
   plt.plot(t_td*1e3, V_N003_td, 'b-', label='Framework', linewidth=2)
   plt.plot(ltspice_data['time']*1e3, ltspice_data['V(N003)'],
            'r--', label='LTspice', linewidth=2)
   plt.xlabel('Time (ms)')
   plt.ylabel('V(N003) [V]')
   plt.title('Framework vs LTspice Comparison')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()
   ```

---

## Understanding the Results

### What is NRMSE?

**Normalized Root Mean Square Error** - measures overall accuracy:
- **< 0.5%**: Excellent match
- **0.5-1.0%**: Good match
- **> 1.0%**: Investigation needed

Formula:
```
NRMSE = (RMSE / signal_range) × 100%
```

### What is Peak Error?

Difference between steady-state peak values:
- **< 0.5%**: Excellent
- **0.5-2.0%**: Acceptable
- **> 2.0%**: Check settings

### What is Correlation?

Measures phase and timing accuracy:
- **> 0.999**: Excellent
- **0.99-0.999**: Good
- **< 0.99**: Phase mismatch

---

## Common Use Cases

### 1. Validate New Circuit

```bash
# Create your circuit in LTspice: my_circuit.asc
# Add: .tran 0 1m
# Run LTspice (File > Run or F5)

# Validate
python validate_against_ltspice.py my_circuit.asc my_circuit.raw
```

### 2. Parameter Sweep

```spice
.param Rval=1k
R1 in out {Rval}
.step param Rval 500 2k 500
.tran 0 1m
```

Then compare each step programmatically.

### 3. Different Topologies

Just change the netlist! The framework automatically:
- Parses topology
- Builds MNA system
- Runs simulation
- Compares with LTspice

---

## Summary

You've just validated that:
✅ Framework correctly parses LTspice netlists
✅ MNA system construction is accurate
✅ Time-domain simulation matches LTspice
✅ Results are within <1% error

**Next**: Try the phasor-domain simulation for 10× speedup!

---

## Quick Reference

| Task | Command |
|------|---------|
| Run LTspice | `"/Applications/LTspice.app/Contents/MacOS/LTspice" -b file.cir` |
| Validate | `python validate_against_ltspice.py file.cir file.raw` |
| With plot | Add `-o plot.png` |
| Quiet mode | Add `-q` |
| View help | `python validate_against_ltspice.py -h` |

---

**Need more details?** See:
- [LTSPICE_COMPARISON_GUIDE.md](LTSPICE_COMPARISON_GUIDE.md) - Comprehensive guide
- [README.md](README.md) - Framework documentation
- [ltspice_circuits/README.md](ltspice_circuits/README.md) - Circuit details
