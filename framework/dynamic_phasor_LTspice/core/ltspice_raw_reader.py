"""
LTspice .raw file reader for the dynamic phasor framework.

Reads LTspice simulation output files (.raw) in both binary and ASCII
formats, enabling direct comparison between framework results and LTspice.

Supported analysis types:
    - Transient (.tran)
    - AC analysis (.ac)
    - DC sweep (.dc)
    - Operating point (.op)

LTspice .raw file structure:
    1. Header (text, often UTF-16LE for Windows LTspice):
       - Title, Date, Plotname, Flags, No. Variables, No. Points
       - Variable list (index, name, type)
       - "Binary:" or "Values:" marker
    2. Data block (binary doubles or ASCII text)

Author: Doyun Gu (University of Manchester)
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class RawVariable:
    """A single variable from a .raw file."""
    index: int
    name: str          # e.g. "V(N003)", "I(L1)", "time"
    var_type: str      # e.g. "voltage", "current", "time", "frequency"


@dataclass
class LTSpiceRawData:
    """
    Parsed LTspice .raw file data.

    Attributes
    ----------
    title : str
        Circuit title from the netlist
    date : str
        Simulation date
    plotname : str
        Analysis type description (e.g. "Transient Analysis")
    flags : str
        Flags (e.g. "real", "complex", "stepped")
    variables : list of RawVariable
        Variable metadata
    n_variables : int
        Number of variables
    n_points : int
        Number of data points
    data : dict
        {variable_name: ndarray} mapping
    is_complex : bool
        True for AC analysis (complex data)
    is_stepped : bool
        True for stepped parameter sweeps
    """
    title: str = ""
    date: str = ""
    plotname: str = ""
    flags: str = ""
    variables: List[RawVariable] = field(default_factory=list)
    n_variables: int = 0
    n_points: int = 0
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    is_complex: bool = False
    is_stepped: bool = False

    # ── Convenience accessors ────────────────────────────────────

    @property
    def time(self) -> Optional[np.ndarray]:
        """Return the time vector (transient analysis)."""
        for key in ('time', 'Time'):
            if key in self.data:
                return self.data[key]
        return None

    @property
    def frequency(self) -> Optional[np.ndarray]:
        """Return the frequency vector (AC analysis)."""
        for key in ('frequency', 'Frequency'):
            if key in self.data:
                return self.data[key]
        return None

    def voltage(self, node: str) -> Optional[np.ndarray]:
        """
        Get voltage at a node.

        Parameters
        ----------
        node : str
            Node name, e.g. "N003" or "out". Tries V(node) variants.
        """
        candidates = [
            f"V({node})", f"v({node})",
            f"V({node.lower()})", f"V({node.upper()})",
            node,
        ]
        for c in candidates:
            if c in self.data:
                return self.data[c]
        return None

    def current(self, element: str) -> Optional[np.ndarray]:
        """
        Get current through an element.

        Parameters
        ----------
        element : str
            Element name, e.g. "L1", "V1". Tries I(element) variants.
        """
        candidates = [
            f"I({element})", f"i({element})",
            f"I({element.lower()})", f"I({element.upper()})",
            f"Ix({element})", element,
        ]
        for c in candidates:
            if c in self.data:
                return self.data[c]
        return None

    def variable_names(self) -> List[str]:
        """Return all variable names."""
        return list(self.data.keys())

    def node_voltages(self) -> Dict[str, np.ndarray]:
        """Return all node voltage variables."""
        return {k: v for k, v in self.data.items()
                if k.lower().startswith('v(')}

    def branch_currents(self) -> Dict[str, np.ndarray]:
        """Return all branch current variables."""
        return {k: v for k, v in self.data.items()
                if k.lower().startswith('i(')}

    def summary(self) -> str:
        """Print a readable summary."""
        lines = [
            f"LTspice Raw Data: {self.title}",
            f"  Analysis: {self.plotname}",
            f"  Points: {self.n_points}",
            f"  Variables ({self.n_variables}):",
        ]
        for v in self.variables:
            arr = self.data.get(v.name)
            if arr is not None and len(arr) > 0:
                if np.iscomplexobj(arr):
                    lines.append(f"    {v.name} ({v.var_type}): "
                                 f"|range| = [{np.abs(arr).min():.6g}, "
                                 f"{np.abs(arr).max():.6g}]")
                else:
                    lines.append(f"    {v.name} ({v.var_type}): "
                                 f"[{arr.min():.6g}, {arr.max():.6g}]")
        return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────
# Reader implementation
# ──────────────────────────────────────────────────────────────────────

def read_raw(filepath: str) -> LTSpiceRawData:
    """
    Read an LTspice .raw file (binary or ASCII format).

    Automatically detects encoding (UTF-16LE from Windows LTspice
    or UTF-8/ASCII from LTspice on macOS/Linux) and data format.

    Parameters
    ----------
    filepath : str or Path
        Path to the .raw file

    Returns
    -------
    LTSpiceRawData
        Parsed data with all variables accessible by name

    Examples
    --------
    >>> data = read_raw("circuit.raw")
    >>> print(data.summary())
    >>> t = data.time
    >>> v_out = data.voltage("N003")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw file not found: {filepath}")

    raw_bytes = filepath.read_bytes()

    # Detect encoding: LTspice on Windows uses UTF-16LE
    header_text, data_offset, encoding = _detect_and_decode_header(raw_bytes)

    # Parse header
    result = _parse_header(header_text)

    # Determine if data is binary or ASCII
    is_binary = 'Binary:' in header_text or 'Binary:\n' in header_text

    if is_binary:
        _read_binary_data(raw_bytes, data_offset, encoding, result)
    else:
        _read_ascii_data(header_text, result)

    return result


def _detect_and_decode_header(raw_bytes: bytes) -> Tuple[str, int, str]:
    """
    Detect encoding and decode the header portion.

    Returns
    -------
    header_text : str
        Decoded header text
    data_offset : int
        Byte offset where binary data begins (after "Binary:\\n")
    encoding : str
        Detected encoding ('utf-16-le' or 'utf-8')
    """
    # Check for UTF-16LE BOM or characteristic pattern
    if raw_bytes[:2] == b'\xff\xfe':
        encoding = 'utf-16-le'
        text = raw_bytes.decode('utf-16-le', errors='replace')
    elif b'T\x00i\x00t\x00l\x00e\x00' in raw_bytes[:100]:
        # UTF-16LE without BOM (common for LTspice Windows)
        encoding = 'utf-16-le'
        text = raw_bytes.decode('utf-16-le', errors='replace')
    else:
        encoding = 'utf-8'
        text = raw_bytes.decode('utf-8', errors='replace')

    # Find binary data marker
    if encoding == 'utf-16-le':
        # Search for "Binary:\n" in UTF-16LE
        marker_bytes = 'Binary:\n'.encode('utf-16-le')
        marker_pos = raw_bytes.find(marker_bytes)
        if marker_pos >= 0:
            data_offset = marker_pos + len(marker_bytes)
        else:
            # Try without BOM offset
            marker_bytes_no_bom = 'Binary:\n'.encode('utf-16-le')
            marker_pos = raw_bytes.find(marker_bytes_no_bom)
            data_offset = marker_pos + len(marker_bytes_no_bom) if marker_pos >= 0 else len(raw_bytes)
    else:
        marker = b'Binary:\n'
        marker_pos = raw_bytes.find(marker)
        if marker_pos >= 0:
            data_offset = marker_pos + len(marker)
        else:
            data_offset = len(raw_bytes)

    return text, data_offset, encoding


def _parse_header(header_text: str) -> LTSpiceRawData:
    """Parse the header section into metadata."""
    result = LTSpiceRawData()
    lines = header_text.splitlines()

    in_variables = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('Title:'):
            result.title = line[6:].strip()
        elif line.startswith('Date:'):
            result.date = line[5:].strip()
        elif line.startswith('Plotname:'):
            result.plotname = line[9:].strip()
        elif line.startswith('Flags:'):
            result.flags = line[6:].strip().lower()
            result.is_complex = 'complex' in result.flags
            result.is_stepped = 'stepped' in result.flags
        elif line.startswith('No. Variables:'):
            result.n_variables = int(line[14:].strip())
        elif line.startswith('No. Points:'):
            result.n_points = int(line[11:].strip())
        elif line.startswith('Variables:'):
            in_variables = True
            continue
        elif line.startswith('Binary:') or line.startswith('Values:'):
            break
        elif in_variables:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                idx = int(parts[0])
                name = parts[1]
                var_type = parts[2]
                result.variables.append(RawVariable(idx, name, var_type))

    return result


def _read_binary_data(raw_bytes: bytes, data_offset: int,
                      encoding: str, result: LTSpiceRawData):
    """Read binary data section."""
    n_vars = result.n_variables
    n_pts = result.n_points

    if n_vars == 0 or n_pts == 0:
        return

    data_bytes = raw_bytes[data_offset:]

    if result.is_complex:
        # AC analysis: all variables are complex (two doubles each)
        # Time/frequency variable is also stored as double
        bytes_per_point = n_vars * 16  # 16 bytes per complex double
        _read_binary_complex(data_bytes, n_vars, n_pts, bytes_per_point, result)
    else:
        # Transient / DC: first variable (time) is double (8 bytes),
        # remaining variables are float (4 bytes) in older LTspice
        # or double (8 bytes) in newer versions

        # Try double precision first (8 bytes per variable)
        bytes_per_point_d = n_vars * 8
        expected_d = bytes_per_point_d * n_pts

        # Try mixed: first var double, rest float
        bytes_per_point_m = 8 + (n_vars - 1) * 4
        expected_m = bytes_per_point_m * n_pts

        if len(data_bytes) >= expected_d:
            _read_binary_real_double(data_bytes, n_vars, n_pts, result)
        elif len(data_bytes) >= expected_m:
            _read_binary_real_mixed(data_bytes, n_vars, n_pts, result)
        else:
            # Try to infer from available data
            if n_pts > 0:
                bytes_per_point = len(data_bytes) // n_pts
                if bytes_per_point == n_vars * 8:
                    _read_binary_real_double(data_bytes, n_vars, n_pts, result)
                elif bytes_per_point == 8 + (n_vars - 1) * 4:
                    _read_binary_real_mixed(data_bytes, n_vars, n_pts, result)
                else:
                    raise ValueError(
                        f"Cannot determine binary format: {len(data_bytes)} bytes "
                        f"for {n_pts} points × {n_vars} variables "
                        f"({bytes_per_point} bytes/point)"
                    )


def _read_binary_real_double(data_bytes: bytes, n_vars: int,
                              n_pts: int, result: LTSpiceRawData):
    """Read binary data where all variables are 8-byte doubles."""
    arrays = [np.zeros(n_pts) for _ in range(n_vars)]

    offset = 0
    for pt in range(n_pts):
        for v in range(n_vars):
            if offset + 8 <= len(data_bytes):
                arrays[v][pt] = struct.unpack('<d', data_bytes[offset:offset+8])[0]
                offset += 8

    for v, var_meta in enumerate(result.variables):
        result.data[var_meta.name] = arrays[v]


def _read_binary_real_mixed(data_bytes: bytes, n_vars: int,
                             n_pts: int, result: LTSpiceRawData):
    """Read binary data: first variable double, rest float."""
    arrays = [np.zeros(n_pts) for _ in range(n_vars)]

    offset = 0
    for pt in range(n_pts):
        # First variable (time) is double
        if offset + 8 <= len(data_bytes):
            arrays[0][pt] = struct.unpack('<d', data_bytes[offset:offset+8])[0]
            offset += 8
        # Remaining variables are float
        for v in range(1, n_vars):
            if offset + 4 <= len(data_bytes):
                arrays[v][pt] = struct.unpack('<f', data_bytes[offset:offset+4])[0]
                offset += 4

    for v, var_meta in enumerate(result.variables):
        result.data[var_meta.name] = arrays[v]


def _read_binary_complex(data_bytes: bytes, n_vars: int,
                          n_pts: int, bytes_per_point: int,
                          result: LTSpiceRawData):
    """Read complex binary data (AC analysis)."""
    arrays = [np.zeros(n_pts, dtype=complex) for _ in range(n_vars)]

    offset = 0
    for pt in range(n_pts):
        for v in range(n_vars):
            if offset + 16 <= len(data_bytes):
                real = struct.unpack('<d', data_bytes[offset:offset+8])[0]
                imag = struct.unpack('<d', data_bytes[offset+8:offset+16])[0]
                arrays[v][pt] = complex(real, imag)
                offset += 16

    for v, var_meta in enumerate(result.variables):
        result.data[var_meta.name] = arrays[v]


def _read_ascii_data(header_text: str, result: LTSpiceRawData):
    """Read ASCII format data section."""
    n_vars = result.n_variables
    n_pts = result.n_points

    # Find the "Values:" marker
    marker = 'Values:'
    marker_pos = header_text.find(marker)
    if marker_pos < 0:
        return

    data_text = header_text[marker_pos + len(marker):]
    lines = data_text.strip().splitlines()

    arrays = [np.zeros(n_pts) for _ in range(n_vars)]

    pt = -1
    var_idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Check if this starts a new data point (index number followed by value)
        if len(parts) >= 2 and parts[0].isdigit():
            pt += 1
            var_idx = 0
            if pt >= n_pts:
                break
            try:
                arrays[0][pt] = float(parts[1])
                var_idx = 1
            except ValueError:
                pass
        elif len(parts) >= 1:
            # Continuation line with just values
            for val_str in parts:
                if var_idx < n_vars and pt >= 0:
                    try:
                        val_str_clean = val_str.replace(',', '')
                        if result.is_complex and ',' in val_str:
                            re_im = val_str.split(',')
                            arrays[var_idx][pt] = complex(float(re_im[0]),
                                                          float(re_im[1]))
                        else:
                            arrays[var_idx][pt] = float(val_str_clean)
                    except ValueError:
                        pass
                    var_idx += 1

    for v, var_meta in enumerate(result.variables):
        result.data[var_meta.name] = arrays[v][:pt+1] if pt >= 0 else arrays[v]


# ──────────────────────────────────────────────────────────────────────
# LTspice raw file writer (for testing / round-trip)
# ──────────────────────────────────────────────────────────────────────

def write_raw_ascii(filepath: str, data: Dict[str, np.ndarray],
                    title: str = "Dynamic Phasor Framework Output",
                    plotname: str = "Transient Analysis"):
    """
    Write simulation data in LTspice-compatible ASCII .raw format.

    This allows the framework's output to be opened in LTspice's
    waveform viewer for visual comparison.

    Parameters
    ----------
    filepath : str
        Output file path
    data : dict
        {variable_name: ndarray} mapping. Must include 'time'.
    title : str
        Title line
    plotname : str
        Analysis type
    """
    filepath = Path(filepath)

    # Determine variable types
    var_names = list(data.keys())
    # Put time first
    if 'time' in var_names:
        var_names.remove('time')
        var_names.insert(0, 'time')
    elif 't' in var_names:
        var_names.remove('t')
        var_names.insert(0, 't')

    n_vars = len(var_names)
    n_pts = len(next(iter(data.values())))

    with open(filepath, 'w') as f:
        f.write(f"Title: {title}\n")
        f.write(f"Date: (framework output)\n")
        f.write(f"Plotname: {plotname}\n")
        f.write(f"Flags: real\n")
        f.write(f"No. Variables: {n_vars}\n")
        f.write(f"No. Points: {n_pts}\n")
        f.write("Variables:\n")

        for i, name in enumerate(var_names):
            if 'time' in name.lower() or name == 't':
                vtype = 'time'
            elif name.lower().startswith('v(') or name.lower().startswith('v_'):
                vtype = 'voltage'
            elif name.lower().startswith('i(') or name.lower().startswith('i_'):
                vtype = 'current'
            else:
                vtype = 'voltage'
            f.write(f"\t{i}\t{name}\t{vtype}\n")

        f.write("Values:\n")
        for pt in range(n_pts):
            f.write(f"{pt}\t{data[var_names[0]][pt]:.15e}\n")
            for v in range(1, n_vars):
                f.write(f"\t{data[var_names[v]][pt]:.15e}\n")
