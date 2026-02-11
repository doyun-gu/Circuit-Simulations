# LTspice Circuit Files

This directory contains LTspice netlist files for validating the Dynamic Phasor Framework.

## Files

### 1. `rim2025_rlc.cir`
**Basic validation circuit** - Exact implementation of Rim et al. (2025) Table II parameters.

**Usage:**
```bash
# Open in LTspice GUI
open rim2025_rlc.cir

# Or run from command line (macOS)
"/Applications/LTspice.app/Contents/MacOS/LTspice" -b rim2025_rlc.cir

# Or run from command line (Windows)
"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe" -b rim2025_rlc.cir
```

**Parameters:**
- L = 100.04 µH
- C = 30.07 nF
- Rs = 3.0 Ω
- Ro = 2.00 kΩ
- Source: 1V @ 92.3 kHz
- Simulation: 0.5 ms

**Output:** `rim2025_rlc.raw`

---

### 2. `rim2025_rlc_parametric.cir`
**Parametric version** - Uses `.param` for easy modification.

**Features:**
- All parameters defined at top
- Computed resonant frequency
- Easy to modify for different test cases

**How to modify:**
```spice
.param Lval=150u      ; Change inductance
.param fsrc=100k      ; Change source frequency
```

---

### 3. `rim2025_rlc_frequency_sweep.cir`
**Multi-frequency validation** - Sweeps source frequency.

**Test frequencies:**
- 50 kHz (below resonance)
- 70 kHz
- 92.3 kHz (near resonance)
- 120 kHz
- 150 kHz
- 200 kHz (above resonance)

**Usage:**
Run simulation. LTspice will generate one dataset per frequency in the `.raw` file.

**Output:** Contains 6 sweeps in `rim2025_rlc_frequency_sweep.raw`

---

### 4. `rim2025_rlc_step_response.cir`
**Transient envelope test** - Step input at resonant frequency.

**Purpose:**
Validate framework's ability to track envelope dynamics during transients.

**Input:** Step from 0→1V at t=0.1ms
**Simulation:** 1 ms (longer to capture full transient)

---

## Quick Validation Workflow

### Step 1: Run LTspice Simulation

```bash
cd ltspice_circuits
"/Applications/LTspice.app/Contents/MacOS/LTspice" -b rim2025_rlc.cir
```

### Step 2: Validate with Framework

```bash
cd ..
python validate_against_ltspice.py \
    ltspice_circuits/rim2025_rlc.cir \
    ltspice_circuits/rim2025_rlc.raw \
    -o validation_plot.png
```

### Step 3: Check Results

Expected output:
```
✓✓ VALIDATION SUCCESSFUL ✓✓
Framework matches LTspice within tolerance!

Metrics:
  V(N003): NRMSE = 0.15%
  I(L1):   NRMSE = 0.23%
```

---

## Validation with Python

### Method 1: Using Validation Script

```python
from validate_against_ltspice import validate

results = validate(
    netlist_file="ltspice_circuits/rim2025_rlc.cir",
    raw_file="ltspice_circuits/rim2025_rlc.raw",
    plot_output="comparison.png"
)

print(f"Validation passed: {results['overall_pass']}")
```

### Method 2: Manual Comparison

```python
from core.ltspice_raw_reader import read_raw
from core.netlist_parser import parse_ltspice_netlist
from core.mna_circuit import NetlistCircuit

# Load LTspice results
ltspice_data = read_raw("ltspice_circuits/rim2025_rlc.raw")

# Run framework simulation
with open("ltspice_circuits/rim2025_rlc.cir", 'r') as f:
    netlist = parse_ltspice_netlist(f.read())

circuit = NetlistCircuit(netlist)
framework_results = circuit.solve_time_domain()

# Compare
import matplotlib.pyplot as plt
plt.plot(ltspice_data['time'], ltspice_data['V(N003)'], label='LTspice')
plt.plot(framework_results['t'], framework_results['V(N003)'], label='Framework')
plt.legend()
plt.show()
```

---

## Expected Results

### Basic Circuit (rim2025_rlc.cir)

| Signal | LTspice Peak | Framework Peak | Error |
|--------|--------------|----------------|-------|
| V(N003) | 12.20 V | 12.20 V | < 0.3% |
| I(L1) | 0.213 A | 0.213 A | < 0.5% |

**NRMSE:** < 0.5%
**Correlation:** > 0.999

### Frequency Sweep (rim2025_rlc_frequency_sweep.cir)

| Frequency | V(N003) Peak | Q-factor Visibility |
|-----------|--------------|---------------------|
| 50 kHz | ~3.2 V | Off-resonance |
| 92.3 kHz | ~12.2 V | Near resonance (max) |
| 200 kHz | ~1.8 V | Off-resonance |

---

## Troubleshooting

### Problem: .raw file not generated

**Solution:**
- Check LTspice completed without errors
- Look in same directory as `.cir` file
- Check LTspice log for warnings

### Problem: Framework and LTspice don't match

**Check:**
1. Component values parsed correctly
   ```python
   print(circuit.info())
   ```

2. Same simulation time
   ```python
   tran = netlist.tran_params()
   print(f"Duration: {tran['t_stop']} s")
   ```

3. Same timestep
   ```spice
   .tran 0 0.5m 0 1u  ; Last parameter is max timestep
   ```

### Problem: Signal names don't match

LTspice may use lowercase node names. Map them:

```python
signal_map = {
    'V(n003)': 'V(N003)',
    'I(l1)': 'I(L1)',
}
```

---

## Adding Your Own Circuits

Create a new `.cir` file:

```spice
* My Custom Circuit
V1 in 0 SINE(0 1 100k)
R1 in out 1k
C1 out 0 10n
.tran 0 1m
.end
```

Save and run:
```bash
"/Applications/LTspice.app/Contents/MacOS/LTspice" -b my_circuit.cir
python validate_against_ltspice.py my_circuit.cir my_circuit.raw
```

---

## Next Steps

1. ✅ Validate basic circuit (rim2025_rlc.cir)
2. ✅ Test frequency sweep
3. ✅ Test transient response
4. 🔄 Create your own test cases
5. 🔄 Compare phasor-domain vs time-domain

---

## See Also

- [Main README](../README.md) - Framework documentation
- [LTspice Comparison Guide](../LTSPICE_COMPARISON_GUIDE.md) - Detailed guide
- [Validation Script](../validate_against_ltspice.py) - Automated validation
- [Notebook](../../notebooks/09_circuitSim.ipynb) - Interactive validation
