"""
Generic MNA-based circuit for dynamic phasor simulation.

Builds Modified Nodal Analysis (MNA) state equations automatically from a
parsed LTspice netlist, then solves them in both time domain and phasor domain
using the existing dynamic phasor framework.

The MNA formulation:
    [G  B] [v]   [C_v  0 ] d [v]   [i_s(t)]
    [B' 0] [j] + [0    L ] dt[j] = [v_s(t)]

where:
    v = node voltages (unknowns)
    j = voltage-source / inductor branch currents (unknowns)
    G = conductance matrix (from R, C, I sources)
    B = incidence of voltage sources & inductors
    C_v = capacitance contribution matrix
    L = inductance matrix
    i_s(t) = current source excitations
    v_s(t) = voltage source excitations

State vector x = [v_nodes; i_branches], dynamics: E·dx/dt = A·x + b(t)

Reference:
    Chung-Wen Ho, A. Ruehli, P. Brennan, "The Modified Nodal Approach to
    Network Analysis," IEEE TCAS, 1975.

Author: Doyun Gu (University of Manchester)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import warnings

from .netlist_parser import (
    ParsedNetlist, NetlistElement, SourceSpec, SourceType,
    parse_ltspice_netlist, parse_spice_value,
)
from .phasor import PhasorConfig, InstantaneousPhasor


# ──────────────────────────────────────────────────────────────────────
# Node indexing helper
# ──────────────────────────────────────────────────────────────────────

class NodeMap:
    """
    Maps node names to integer indices for matrix construction.
    Ground node (0, GND, gnd, ground) always maps to index -1 (excluded).
    """
    GROUND_NAMES = {'0', 'gnd', 'GND', 'ground', 'GROUND'}

    def __init__(self, nodes: set):
        non_ground = sorted(nodes - self.GROUND_NAMES)
        self._name_to_idx = {name: i for i, name in enumerate(non_ground)}
        self._idx_to_name = {i: name for name, i in self._name_to_idx.items()}
        self.n = len(non_ground)

    def idx(self, name: str) -> int:
        """Return matrix index for node. Returns -1 for ground."""
        if name in self.GROUND_NAMES:
            return -1
        return self._name_to_idx[name]

    def name(self, idx: int) -> str:
        return self._idx_to_name[idx]

    def is_ground(self, name: str) -> bool:
        return name in self.GROUND_NAMES

    def names(self) -> List[str]:
        return [self._idx_to_name[i] for i in range(self.n)]


# ──────────────────────────────────────────────────────────────────────
# MNA builder
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MNASystem:
    """
    The MNA system matrices: E · dx/dt = A · x + b(t)

    State vector x = [v1, v2, ..., vN, iV1, iV2, ..., iL1, iL2, ...]
    where the first N entries are node voltages and the remaining are
    branch currents through voltage sources and inductors.

    Attributes
    ----------
    A : ndarray (n_total × n_total)
        System matrix (contains -G and branch contributions)
    E : ndarray (n_total × n_total)
        Mass matrix (contains C and L)
    b_func : callable
        Returns b(t) excitation vector at time t
    n_nodes : int
        Number of non-ground nodes
    n_vsrc : int
        Number of voltage source branches
    n_ind : int
        Number of inductor branches
    n_total : int
        Total system size = n_nodes + n_vsrc + n_ind
    node_map : NodeMap
        Mapping of node names to indices
    vsrc_names : list of str
        Voltage source branch names (in order)
    ind_names : list of str
        Inductor branch names (in order)
    state_labels : list of str
        Human-readable label for each state variable
    """
    A: np.ndarray
    E: np.ndarray
    b_func: Callable[[float], np.ndarray]
    n_nodes: int
    n_vsrc: int
    n_ind: int
    n_total: int
    node_map: NodeMap
    vsrc_names: List[str] = field(default_factory=list)
    ind_names: List[str] = field(default_factory=list)
    state_labels: List[str] = field(default_factory=list)


def build_mna(netlist: ParsedNetlist) -> MNASystem:
    """
    Build the MNA system matrices from a parsed netlist.

    Strategy:
    - Resistors → stamp into G (conductance matrix)
    - Capacitors → stamp into C_v (capacitance part of E)
    - Inductors → add branch current variable, stamp into L part of E
    - Voltage sources → add branch current variable, KVL constraint
    - Current sources → stamp into b(t) excitation vector

    Parameters
    ----------
    netlist : ParsedNetlist
        Parsed netlist from the LTspice parser

    Returns
    -------
    MNASystem
        Complete MNA system ready for ODE solving
    """
    node_map = NodeMap(netlist.nodes)
    n_nodes = node_map.n

    # Count additional branch variables
    vsrc_list = netlist.voltage_sources()
    ind_list = netlist.inductors()
    n_vsrc = len(vsrc_list)
    n_ind = len(ind_list)
    n_total = n_nodes + n_vsrc + n_ind

    # Build indices for branch variables
    vsrc_names = [e.name for e in vsrc_list]
    ind_names = [e.name for e in ind_list]

    # Branch current index offset
    vsrc_offset = n_nodes
    ind_offset = n_nodes + n_vsrc

    # Initialise matrices
    # A · x is the "static" part:  -(G·v + B·j) for node equations
    #                                B'·v        for branch equations
    # E · dx/dt is the "dynamic" part: C·dv/dt, L·di/dt
    A = np.zeros((n_total, n_total))
    E = np.zeros((n_total, n_total))

    # ─── Stamp resistors into G (rows/cols 0..n_nodes-1) ─────────
    for elem in netlist.resistors():
        if elem.value == 0:
            warnings.warn(f"Zero resistance in {elem.name}, skipping")
            continue
        g = 1.0 / elem.value  # conductance
        np_idx = node_map.idx(elem.nodes[0])
        nm_idx = node_map.idx(elem.nodes[1])

        # Stamp conductance: G[i,i] += g, G[j,j] += g, G[i,j] -= g, G[j,i] -= g
        if np_idx >= 0:
            A[np_idx, np_idx] -= g
        if nm_idx >= 0:
            A[nm_idx, nm_idx] -= g
        if np_idx >= 0 and nm_idx >= 0:
            A[np_idx, nm_idx] += g
            A[nm_idx, np_idx] += g

    # ─── Stamp capacitors into C part of E ───────────────────────
    for elem in netlist.capacitors():
        c = elem.value
        np_idx = node_map.idx(elem.nodes[0])
        nm_idx = node_map.idx(elem.nodes[1])

        if np_idx >= 0:
            E[np_idx, np_idx] += c
        if nm_idx >= 0:
            E[nm_idx, nm_idx] += c
        if np_idx >= 0 and nm_idx >= 0:
            E[np_idx, nm_idx] -= c
            E[nm_idx, np_idx] -= c

    # ─── Stamp voltage sources ────────────────────────────────────
    for k, elem in enumerate(vsrc_list):
        branch_idx = vsrc_offset + k
        np_idx = node_map.idx(elem.nodes[0])
        nm_idx = node_map.idx(elem.nodes[1])

        # KCL: branch current enters n+ and leaves n-
        if np_idx >= 0:
            A[np_idx, branch_idx] += 1.0
            A[branch_idx, np_idx] += 1.0  # KVL: V(n+) - V(n-) = vs(t)
        if nm_idx >= 0:
            A[nm_idx, branch_idx] -= 1.0
            A[branch_idx, nm_idx] -= 1.0

    # ─── Stamp inductors (as branch variables like V sources) ────
    for k, elem in enumerate(ind_list):
        branch_idx = ind_offset + k
        np_idx = node_map.idx(elem.nodes[0])
        nm_idx = node_map.idx(elem.nodes[1])

        # KCL contribution: inductor branch current flows n+ → n−
        # (leaves n+, enters n−) — opposite sign to voltage sources
        if np_idx >= 0:
            A[np_idx, branch_idx] -= 1.0
        if nm_idx >= 0:
            A[nm_idx, branch_idx] += 1.0

        # KVL: V(n+) - V(n-) = L · di/dt
        if np_idx >= 0:
            A[branch_idx, np_idx] += 1.0
        if nm_idx >= 0:
            A[branch_idx, nm_idx] -= 1.0

        # Inductance in mass matrix
        E[branch_idx, branch_idx] = elem.value

    # ─── Build excitation vector b(t) ────────────────────────────

    # Pre-build static current source stamps
    isrc_list = netlist.current_sources()
    isrc_info = []
    for elem in isrc_list:
        np_idx = node_map.idx(elem.nodes[0])
        nm_idx = node_map.idx(elem.nodes[1])
        func = elem.source_spec.time_function() if elem.source_spec else (lambda t: 0.0)
        isrc_info.append((np_idx, nm_idx, func))

    # Pre-build voltage source functions
    vsrc_funcs = []
    for elem in vsrc_list:
        func = elem.source_spec.time_function() if elem.source_spec else (lambda t: 0.0)
        vsrc_funcs.append(func)

    def b_func(t: float) -> np.ndarray:
        b = np.zeros(n_total)
        # Current sources: positive current flows from n+ to n-
        # In MNA sign convention, current INTO node is positive
        for np_idx, nm_idx, func in isrc_info:
            val = func(t)
            if np_idx >= 0:
                b[np_idx] -= val  # current leaves n+
            if nm_idx >= 0:
                b[nm_idx] += val  # current enters n-
        # Voltage sources: KVL row gets the source voltage
        for k, func in enumerate(vsrc_funcs):
            b[vsrc_offset + k] = func(t)
        return b

    # Build state labels
    state_labels = [f"V({node_map.name(i)})" for i in range(n_nodes)]
    state_labels += [f"I({name})" for name in vsrc_names]
    state_labels += [f"I({name})" for name in ind_names]

    return MNASystem(
        A=A, E=E, b_func=b_func,
        n_nodes=n_nodes, n_vsrc=n_vsrc, n_ind=n_ind, n_total=n_total,
        node_map=node_map,
        vsrc_names=vsrc_names, ind_names=ind_names,
        state_labels=state_labels,
    )


# ──────────────────────────────────────────────────────────────────────
# NetlistCircuit: the main solver class
# ──────────────────────────────────────────────────────────────────────

class NetlistCircuit:
    """
    Generic circuit built from an LTspice netlist.

    Provides time-domain and phasor-domain simulation using MNA,
    compatible with the dynamic_phasor framework for comparison with
    LTspice results.

    Usage
    -----
    >>> from dynamic_phasor.core.netlist_parser import parse_ltspice_netlist
    >>> from dynamic_phasor.core.mna_circuit import NetlistCircuit
    >>>
    >>> # From a netlist string
    >>> netlist_text = '''
    ... * Series RLC
    ... V1 N001 0 SINE(0 1 92.3k)
    ... R1 N001 N002 3.0
    ... L1 N002 N003 100.04u
    ... C1 N003 0 30.07n
    ... R2 N003 0 2k
    ... .tran 0 0.2m
    ... .end
    ... '''
    >>> circuit = NetlistCircuit.from_string(netlist_text)
    >>> results = circuit.solve_time_domain()

    For phasor-domain simulation:
    >>> circuit.configure_phasor(omega_s=2*pi*92.3e3)
    >>> phasor_results = circuit.solve_phasor_domain()
    """

    def __init__(self, netlist: ParsedNetlist):
        """
        Build circuit from a parsed netlist.

        Parameters
        ----------
        netlist : ParsedNetlist
            Output from LTSpiceNetlistParser
        """
        self.netlist = netlist
        self.mna = build_mna(netlist)
        self.phasor: Optional[InstantaneousPhasor] = None
        self.omega_s: float = 0.0

        # Cache the decomposed system for ODE solving
        self._setup_ode_system()

    @classmethod
    def from_file(cls, filepath: str) -> 'NetlistCircuit':
        """Create circuit from a netlist file."""
        netlist = parse_ltspice_netlist(filepath)
        return cls(netlist)

    @classmethod
    def from_string(cls, text: str) -> 'NetlistCircuit':
        """Create circuit from a netlist string."""
        netlist = parse_ltspice_netlist(text)
        return cls(netlist)

    def _setup_ode_system(self):
        """
        Decompose E·dx/dt = A·x + b(t) into a proper ODE by eliminating
        algebraic variables.

        The MNA system is a semi-explicit index-1 DAE:
            E_d · dx_d/dt = A_dd · x_d + A_da · x_a + b_d(t)    (diff eqs)
            0             = A_ad · x_d + A_aa · x_a + b_a(t)    (alg eqs)

        From the algebraic equations:
            x_a = -A_aa^{-1} · (A_ad · x_d + b_a(t))

        Substituting into the differential equations gives a reduced ODE
        in only the differential state variables x_d.
        """
        E = self.mna.E
        A = self.mna.A
        n = self.mna.n_total

        # Identify differential (d) and algebraic (a) rows
        # A row i is algebraic if E[i,:] is all zeros
        self._diff_idx = []
        self._alg_idx = []
        for i in range(n):
            if np.any(np.abs(E[i, :]) > 1e-30):
                self._diff_idx.append(i)
            else:
                self._alg_idx.append(i)

        self._diff_idx = np.array(self._diff_idx, dtype=int)
        self._alg_idx = np.array(self._alg_idx, dtype=int)
        self._n_diff = len(self._diff_idx)
        self._n_alg = len(self._alg_idx)
        self._is_stiff = self._n_alg > 0

        if self._n_alg == 0:
            # Pure ODE – all rows are differential
            self._E_inv = np.linalg.inv(E)
            self._M = self._E_inv @ A
            self._use_reduced = False
            return

        # Extract sub-matrices
        d = self._diff_idx
        a = self._alg_idx

        E_dd = E[np.ix_(d, d)]  # should be the only non-zero block
        A_dd = A[np.ix_(d, d)]
        A_da = A[np.ix_(d, a)]
        A_ad = A[np.ix_(a, d)]
        A_aa = A[np.ix_(a, a)]

        # Check A_aa is invertible
        det_Aaa = np.linalg.det(A_aa)
        if abs(det_Aaa) < 1e-30:
            # If A_aa is singular, add small regularisation (GMIN in SPICE)
            gmin = 1e-12
            for i in range(len(a)):
                A_aa[i, i] += gmin

        self._E_dd_inv = np.linalg.inv(E_dd)
        self._A_aa_inv = np.linalg.inv(A_aa)

        # Reduced system: E_dd · dx_d/dt = A_reduced · x_d + b_reduced(t)
        # where A_reduced = A_dd - A_da · A_aa^{-1} · A_ad
        self._A_reduced = A_dd - A_da @ self._A_aa_inv @ A_ad
        self._M_reduced = self._E_dd_inv @ self._A_reduced

        # For the excitation:
        # b_reduced(t) = b_d(t) - A_da · A_aa^{-1} · b_a(t)
        self._A_da = A_da
        self._A_ad = A_ad
        self._A_aa_inv_cached = self._A_aa_inv

        self._use_reduced = True

    def _recover_algebraic(self, x_d: np.ndarray, t: float) -> np.ndarray:
        """
        Recover algebraic variables from differential variables.
        x_a = -A_aa^{-1} · (A_ad · x_d + b_a(t))
        """
        b_full = self.mna.b_func(t)
        b_a = b_full[self._alg_idx]
        return -self._A_aa_inv_cached @ (self._A_ad @ x_d + b_a)

    def _recover_full_state(self, x_d: np.ndarray, t: float) -> np.ndarray:
        """Reconstruct full state vector from differential variables."""
        x_full = np.zeros(self.mna.n_total)
        x_full[self._diff_idx] = x_d
        if self._n_alg > 0:
            x_full[self._alg_idx] = self._recover_algebraic(x_d, t)
        return x_full

    # ──────────────────────────────────────────────────────────────
    # Phasor configuration
    # ──────────────────────────────────────────────────────────────

    def configure_phasor(self, omega_s: float = None,
                         omega_0: float = None,
                         omega_1: float = None,
                         alpha: float = 0.0):
        """
        Configure phasor transformation.

        If omega_s is not given, tries to auto-detect from the first
        SINE voltage source in the netlist.

        Parameters
        ----------
        omega_s : float, optional
            Carrier angular frequency (rad/s)
        omega_0, omega_1, alpha : float
            FM parameters (see PhasorConfig)
        """
        if omega_s is None:
            omega_s = self._detect_carrier_frequency()

        self.omega_s = omega_s
        config = PhasorConfig(
            omega=omega_s,
            omega_0=omega_0 or omega_s,
            omega_1=omega_1 or 0,
            alpha=alpha,
        )
        self.phasor = InstantaneousPhasor(config)

    def _detect_carrier_frequency(self) -> float:
        """Auto-detect carrier frequency from SINE sources."""
        for elem in self.netlist.voltage_sources():
            if elem.source_spec and elem.source_spec.source_type == SourceType.SINE:
                return elem.source_spec.omega()
        for elem in self.netlist.current_sources():
            if elem.source_spec and elem.source_spec.source_type == SourceType.SINE:
                return elem.source_spec.omega()
        raise ValueError("No SINE source found; please specify omega_s manually")

    # ──────────────────────────────────────────────────────────────
    # Time-domain solver
    # ──────────────────────────────────────────────────────────────

    def time_domain_ode(self, t: float, x_d: np.ndarray) -> np.ndarray:
        """
        ODE right-hand side for the reduced differential system.

        If no algebraic elimination was needed:
            dx/dt = E_inv @ (A·x + b(t))
        If reduced:
            dx_d/dt = E_dd_inv @ (A_reduced · x_d + b_d(t) - A_da · A_aa^{-1} · b_a(t))
        """
        if not self._use_reduced:
            return self._M @ x_d + self._E_inv @ self.mna.b_func(t)

        b_full = self.mna.b_func(t)
        b_d = b_full[self._diff_idx]
        b_a = b_full[self._alg_idx]

        b_reduced = b_d - self._A_da @ self._A_aa_inv_cached @ b_a
        return self._M_reduced @ x_d + self._E_dd_inv @ b_reduced

    def solve_time_domain(self, t_span: Tuple[float, float] = None,
                          t_eval: np.ndarray = None,
                          x0: np.ndarray = None,
                          **solver_kwargs) -> Dict:
        """
        Solve the circuit in the time domain.

        Parameters
        ----------
        t_span : tuple (t_start, t_end), optional
            If None, uses .tran command from netlist
        t_eval : ndarray, optional
            Specific times to evaluate
        x0 : ndarray, optional
            Initial state vector. If None, uses .ic or zeros
        **solver_kwargs :
            Additional arguments for scipy.integrate.solve_ivp

        Returns
        -------
        dict
            't' : time vector
            'x' : full state matrix (n_total × n_points)
            'node_voltages' : dict {node_name: voltage_array}
            'branch_currents' : dict {branch_name: current_array}
            For each voltage source: 'I(V1)', etc.
            For each inductor: 'I(L1)', etc.
            For each node: 'V(N001)', etc.
        """
        # Determine time span
        if t_span is None:
            tran = self.netlist.tran_params()
            if tran:
                t_start = tran.get('t_start', 0.0)
                t_stop = tran['t_stop']
                t_span = (t_start, t_stop)
            else:
                raise ValueError("No .tran command found; provide t_span")

        # Initial conditions
        if x0 is None:
            x0_full = np.zeros(self.mna.n_total)
            # Apply .ic
            for node_name, val in self.netlist.initial_conditions.items():
                idx = self.mna.node_map.idx(node_name)
                if idx >= 0:
                    x0_full[idx] = val
            # Apply IC= on components
            for k, elem in enumerate(self.netlist.inductors()):
                if elem.ic is not None:
                    x0_full[self.mna.n_nodes + self.mna.n_vsrc + k] = elem.ic
            for elem in self.netlist.capacitors():
                if elem.ic is not None:
                    np_idx = self.mna.node_map.idx(elem.nodes[0])
                    nm_idx = self.mna.node_map.idx(elem.nodes[1])
                    if np_idx >= 0:
                        x0_full[np_idx] = elem.ic
        else:
            x0_full = x0

        # Extract differential initial conditions
        if self._use_reduced:
            x0_d = x0_full[self._diff_idx]
        else:
            x0_d = x0_full

        # Time eval points
        if t_eval is None:
            # Aim for ~200 points per carrier period if there's a SINE source
            try:
                omega = self._detect_carrier_frequency()
                period = 2 * np.pi / omega
                n_periods = (t_span[1] - t_span[0]) / period
                n_points = max(2000, min(int(200 * n_periods), 100000))
            except ValueError:
                n_points = 5000
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # Solve – use implicit method if stiff
        default_method = 'Radau' if self._is_stiff else 'RK45'
        sol = solve_ivp(
            self.time_domain_ode,
            t_span, x0_d, t_eval=t_eval,
            method=solver_kwargs.pop('method', default_method),
            rtol=solver_kwargs.pop('rtol', 1e-8),
            atol=solver_kwargs.pop('atol', 1e-10),
            **solver_kwargs,
        )

        if not sol.success:
            warnings.warn(f"ODE solver warning: {sol.message}")

        # Reconstruct full state at each time point
        if self._use_reduced:
            x_full = np.zeros((self.mna.n_total, len(sol.t)))
            for j in range(len(sol.t)):
                x_full[:, j] = self._recover_full_state(sol.y[:, j], sol.t[j])
        else:
            x_full = sol.y

        return self._package_results(sol.t, x_full)

    def _package_results(self, t: np.ndarray, x: np.ndarray) -> Dict:
        """Package raw ODE solution into a labelled dictionary."""
        mna = self.mna
        results = {'t': t, 'x': x}

        # Node voltages
        node_voltages = {}
        for i in range(mna.n_nodes):
            name = mna.node_map.name(i)
            key = f"V({name})"
            node_voltages[name] = x[i]
            results[key] = x[i]
        results['node_voltages'] = node_voltages

        # Branch currents
        branch_currents = {}
        for k, name in enumerate(mna.vsrc_names):
            key = f"I({name})"
            branch_currents[name] = x[mna.n_nodes + k]
            results[key] = x[mna.n_nodes + k]
        for k, name in enumerate(mna.ind_names):
            key = f"I({name})"
            branch_currents[name] = x[mna.n_nodes + mna.n_vsrc + k]
            results[key] = x[mna.n_nodes + mna.n_vsrc + k]
        results['branch_currents'] = branch_currents

        # Source voltage waveform for convenience
        results['source_voltages'] = {}
        for elem in self.netlist.voltage_sources():
            if elem.source_spec:
                func = elem.source_spec.time_function()
                results['source_voltages'][elem.name] = np.array([func(ti) for ti in t])

        return results

    # ──────────────────────────────────────────────────────────────
    # Phasor-domain solver
    # ──────────────────────────────────────────────────────────────

    def _build_phasor_b_func(self) -> Callable[[float], np.ndarray]:
        """
        Build the phasor-domain excitation vector.

        For each source, convert its time-domain waveform to phasor form:
          SINE source: ṽ_s(t) = V_amp · e^(j·φ_s) (constant phasor for
                       standard case where θ(t) = ω_s·t)
          DC source:   contributes a rotating phasor at -ω_s
          PULSE/PWL:   use instantaneous phasor transform
        """
        mna = self.mna
        phasor = self.phasor
        omega_s = self.omega_s

        # Pre-compute phasor source functions
        vsrc_phasor_funcs = []
        for elem in self.netlist.voltage_sources():
            spec = elem.source_spec
            if spec is None:
                vsrc_phasor_funcs.append(lambda t: 0.0 + 0.0j)
                continue

            if spec.source_type == SourceType.SINE:
                # For SINE(Voff, Vamp, freq, td, theta, phi):
                # If freq matches omega_s, the phasor is constant
                amp = spec.sine_amplitude
                phi_rad = np.radians(spec.sine_phase)
                offset = spec.sine_offset
                freq = spec.sine_freq
                td = spec.sine_delay
                omega_src = 2 * np.pi * freq

                if abs(omega_src - omega_s) < 1.0:
                    # Source at carrier frequency – constant phasor
                    # v(t) = Voff + Vamp·sin(ωt + φ)
                    #       = Voff + Vamp·cos(ωt + φ - π/2)
                    # Phasor of cos part: Vamp · e^(j(φ - π/2))
                    # Note: SPICE SINE uses sin(), our phasor uses cos() as reference
                    phasor_val = amp * np.exp(1j * (phi_rad - np.pi / 2))

                    def _sine_phasor(t, pv=phasor_val, off=offset, td_=td):
                        if t < td_:
                            return 0.0 + 0.0j
                        # The offset contributes a rotating term in phasor domain
                        # For simplicity and typical use, offset is usually 0
                        return pv
                    vsrc_phasor_funcs.append(_sine_phasor)
                else:
                    # Source frequency ≠ carrier – need instantaneous transform
                    time_func = spec.time_function()
                    def _general_phasor(t, f=time_func, p=phasor):
                        val = f(t)
                        theta = p.theta(t)
                        return val * np.exp(-1j * theta)
                    vsrc_phasor_funcs.append(_general_phasor)

            elif spec.source_type == SourceType.DC:
                # DC in phasor domain: rotating at -ω_s
                dc = spec.dc_value
                def _dc_phasor(t, dc_=dc, ws=omega_s):
                    return dc_ * np.exp(-1j * ws * t)
                vsrc_phasor_funcs.append(_dc_phasor)

            else:
                # General case: instantaneous phasor transform
                time_func = spec.time_function()
                def _gen_phasor(t, f=time_func, p=phasor):
                    val = f(t)
                    theta = p.theta(t)
                    return val * np.exp(-1j * theta)
                vsrc_phasor_funcs.append(_gen_phasor)

        # Current source phasors
        isrc_list = self.netlist.current_sources()
        isrc_phasor_info = []
        for elem in isrc_list:
            np_idx = mna.node_map.idx(elem.nodes[0])
            nm_idx = mna.node_map.idx(elem.nodes[1])
            spec = elem.source_spec
            if spec and spec.source_type == SourceType.SINE:
                amp = spec.sine_amplitude
                phi_rad = np.radians(spec.sine_phase)
                phasor_val = amp * np.exp(1j * (phi_rad - np.pi / 2))
                func = lambda t, pv=phasor_val: pv
            elif spec:
                time_func = spec.time_function()
                func = lambda t, f=time_func, p=phasor: f(t) * np.exp(-1j * p.theta(t))
            else:
                func = lambda t: 0.0 + 0.0j
            isrc_phasor_info.append((np_idx, nm_idx, func))

        n_total = mna.n_total
        vsrc_offset = mna.n_nodes

        def b_phasor_func(t: float) -> np.ndarray:
            b = np.zeros(n_total, dtype=complex)
            # Current sources
            for np_idx, nm_idx, func in isrc_phasor_info:
                val = func(t)
                if np_idx >= 0:
                    b[np_idx] -= val
                if nm_idx >= 0:
                    b[nm_idx] += val
            # Voltage sources
            for k, func in enumerate(vsrc_phasor_funcs):
                b[vsrc_offset + k] = func(t)
            return b

        return b_phasor_func

    def phasor_domain_ode(self, t: float, x_d_ri: np.ndarray,
                          b_phasor_func: Callable) -> np.ndarray:
        """
        Phasor-domain ODE (reduced system).

        The phasor transform modifies the MNA:
            E · (dX̃/dt + jω·X̃) = A·X̃ + b̃(t)
        →   E · dX̃/dt = (A - jωE)·X̃ + b̃(t)

        With algebraic elimination, the effective system matrix
        becomes (A - jωE) instead of A, but only E_dd has nonzero
        entries so the jω term only affects the differential block.

        State is interleaved: [Re(x_d[0]), Im(x_d[0]), Re(x_d[1]), ...]
        """
        # Get theta_dot
        theta_dot = self.phasor.theta_dot(t)

        if not self._use_reduced:
            n = self.mna.n_total
            x_complex = x_d_ri[0::2] + 1j * x_d_ri[1::2]
            b = b_phasor_func(t)
            rhs = self.mna.A @ x_complex + b - 1j * theta_dot * self.mna.E @ x_complex
            dx_complex = self._E_inv @ rhs
            result = np.zeros(2 * n)
            result[0::2] = np.real(dx_complex)
            result[1::2] = np.imag(dx_complex)
            return result

        # Reduced system
        nd = self._n_diff
        x_d = x_d_ri[0::2] + 1j * x_d_ri[1::2]  # (n_diff,)

        b_full = b_phasor_func(t)
        b_d = b_full[self._diff_idx]
        b_a = b_full[self._alg_idx]

        # The phasor-modified system matrix: A_phasor = A - jω·E
        # Since E is only nonzero in the differential block:
        #   A_phasor_dd = A_dd - jω·E_dd  →  modifies A_reduced
        #   A_phasor_da = A_da             (unchanged, E_da = 0)
        #   A_phasor_ad = A_ad             (unchanged)
        #   A_phasor_aa = A_aa             (unchanged, E_aa = 0)
        #
        # So: A_reduced_phasor = (A_dd - jωE_dd) - A_da · A_aa^{-1} · A_ad
        #                      = A_reduced - jω · E_dd

        E_dd = self.mna.E[np.ix_(self._diff_idx, self._diff_idx)]

        # dx_d/dt = E_dd^{-1} · ((A_reduced - jω·E_dd)·x_d + b_reduced)
        # Note: A_reduced·x_d - jω·E_dd·x_d = (A_reduced)·x_d - jω·E_dd·x_d
        # And E_dd^{-1} · (- jω·E_dd·x_d) = -jω·x_d

        b_reduced = b_d - self._A_da @ self._A_aa_inv_cached @ b_a
        dx_d = self._M_reduced @ x_d + self._E_dd_inv @ b_reduced - 1j * theta_dot * x_d

        result = np.zeros(2 * nd)
        result[0::2] = np.real(dx_d)
        result[1::2] = np.imag(dx_d)
        return result

    def solve_phasor_domain(self, t_span: Tuple[float, float] = None,
                            t_eval: np.ndarray = None,
                            x0: np.ndarray = None,
                            **solver_kwargs) -> Dict:
        """
        Solve the circuit in the phasor domain.

        Parameters
        ----------
        t_span, t_eval, x0 : same as solve_time_domain
        **solver_kwargs : passed to solve_ivp

        Returns
        -------
        dict
            Same keys as time-domain results, plus:
            'phasor_voltages' : dict {node: complex phasor array}
            'phasor_currents' : dict {branch: complex phasor array}
            'envelopes' : dict {label: envelope (magnitude) array}
        """
        if self.phasor is None:
            self.configure_phasor()

        n = self.mna.n_total

        # Time span
        if t_span is None:
            tran = self.netlist.tran_params()
            if tran:
                t_start = tran.get('t_start', 0.0)
                t_stop = tran['t_stop']
                t_span = (t_start, t_stop)
            else:
                raise ValueError("No .tran command; provide t_span")

        # Initial conditions (complex, stored as interleaved real/imag)
        if self._use_reduced:
            n_state = self._n_diff
        else:
            n_state = self.mna.n_total

        if x0 is None:
            x0_ri = np.zeros(2 * n_state)
        else:
            if np.iscomplexobj(x0):
                # x0 is full complex state – extract differential part
                if self._use_reduced:
                    x0_d = x0[self._diff_idx]
                else:
                    x0_d = x0
                x0_ri = np.zeros(2 * n_state)
                x0_ri[0::2] = np.real(x0_d)
                x0_ri[1::2] = np.imag(x0_d)
            else:
                x0_ri = np.zeros(2 * n_state)
                x0_ri[0::2] = x0[:n_state] if len(x0) >= n_state else x0

        # Time eval
        if t_eval is None:
            n_points = max(2000, min(10000, int(50000 * (t_span[1] - t_span[0]))))
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # Build phasor excitation
        b_phasor_func = self._build_phasor_b_func()

        # Solve – use implicit method if stiff
        default_method = 'Radau' if self._is_stiff else 'RK45'
        sol = solve_ivp(
            lambda t, y: self.phasor_domain_ode(t, y, b_phasor_func),
            t_span, x0_ri, t_eval=t_eval,
            method=solver_kwargs.pop('method', default_method),
            rtol=solver_kwargs.pop('rtol', 1e-8),
            atol=solver_kwargs.pop('atol', 1e-10),
            **solver_kwargs,
        )

        if not sol.success:
            warnings.warn(f"Phasor ODE solver warning: {sol.message}")

        # Reconstruct full complex phasor state at each time point
        n_total = self.mna.n_total
        x_phasor_full = np.zeros((n_total, len(sol.t)), dtype=complex)

        for j in range(len(sol.t)):
            x_d = sol.y[0::2, j] + 1j * sol.y[1::2, j]
            if self._use_reduced:
                # Recover algebraic variables in phasor domain
                # Same formula but with complex-valued b_phasor
                b_full = b_phasor_func(sol.t[j])
                b_a = b_full[self._alg_idx]
                x_a = -self._A_aa_inv_cached @ (self._A_ad @ x_d + b_a)
                x_phasor_full[self._diff_idx, j] = x_d
                x_phasor_full[self._alg_idx, j] = x_a
            else:
                x_phasor_full[:, j] = x_d

        return self._package_phasor_results(sol.t, x_phasor_full)

    def _package_phasor_results(self, t: np.ndarray, x_phasor: np.ndarray) -> Dict:
        """
        Package phasor ODE results.

        Parameters
        ----------
        t : ndarray (n_points,)
        x_phasor : ndarray (n_total, n_points) - complex
        """
        mna = self.mna

        results = {'t': t}

        # Phasor quantities
        phasor_voltages = {}
        phasor_currents = {}
        envelopes = {}

        for i in range(mna.n_nodes):
            name = mna.node_map.name(i)
            phasor_v = x_phasor[i]
            phasor_voltages[name] = phasor_v

            # Reconstruct time-domain
            real_v = self.phasor.to_real(phasor_v, t)
            results[f"V({name})"] = real_v
            envelopes[f"V({name})"] = np.abs(phasor_v)

        for k, bname in enumerate(mna.vsrc_names):
            phasor_i = x_phasor[mna.n_nodes + k]
            phasor_currents[bname] = phasor_i
            results[f"I({bname})"] = self.phasor.to_real(phasor_i, t)
            envelopes[f"I({bname})"] = np.abs(phasor_i)

        for k, bname in enumerate(mna.ind_names):
            phasor_i = x_phasor[mna.n_nodes + mna.n_vsrc + k]
            phasor_currents[bname] = phasor_i
            results[f"I({bname})"] = self.phasor.to_real(phasor_i, t)
            envelopes[f"I({bname})"] = np.abs(phasor_i)

        results['phasor_voltages'] = phasor_voltages
        results['phasor_currents'] = phasor_currents
        results['envelopes'] = envelopes
        results['x_phasor'] = x_phasor

        return results

    # ──────────────────────────────────────────────────────────────
    # Derived quantities
    # ──────────────────────────────────────────────────────────────

    def resonant_frequency(self) -> Optional[float]:
        """
        Estimate resonant frequency from L and C values.
        Returns f_r in Hz, or None if no LC pair found.
        """
        inductors = self.netlist.inductors()
        capacitors = self.netlist.capacitors()
        if inductors and capacitors:
            L = inductors[0].value
            C = capacitors[0].value
            return 1.0 / (2 * np.pi * np.sqrt(L * C))
        return None

    def quality_factor(self) -> Optional[float]:
        """Estimate Q factor for series RLC."""
        inductors = self.netlist.inductors()
        capacitors = self.netlist.capacitors()
        resistors = self.netlist.resistors()
        if inductors and capacitors and resistors:
            L = inductors[0].value
            C = capacitors[0].value
            # Use the smallest resistor as the series resistance
            R = min(r.value for r in resistors if r.value > 0)
            return (1.0 / R) * np.sqrt(L / C)
        return None

    def info(self) -> str:
        """Print circuit summary."""
        lines = [self.netlist.summary()]
        lines.append(f"\nMNA system size: {self.mna.n_total}")
        lines.append(f"  Node voltages: {self.mna.n_nodes}")
        lines.append(f"  V-source branches: {self.mna.n_vsrc}")
        lines.append(f"  Inductor branches: {self.mna.n_ind}")
        lines.append(f"  State labels: {self.mna.state_labels}")

        fr = self.resonant_frequency()
        if fr:
            lines.append(f"\nEstimated resonant freq: {fr/1e3:.2f} kHz")
        Q = self.quality_factor()
        if Q:
            lines.append(f"Estimated Q factor: {Q:.2f}")

        if self.phasor:
            lines.append(f"\nPhasor configured: ω_s = {self.omega_s:.0f} rad/s "
                         f"({self.omega_s/(2*np.pi)/1e3:.2f} kHz)")

        return '\n'.join(lines)

    def __repr__(self):
        return (f"NetlistCircuit({self.netlist.title}, "
                f"{self.mna.n_total} states, "
                f"{len(self.netlist.elements)} elements)")
