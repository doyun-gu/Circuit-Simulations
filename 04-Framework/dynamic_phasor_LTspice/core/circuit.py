"""
Circuit classes for dynamic phasor simulation.

Implements the RLC resonant circuit from Figure 1 of Rim et al. (2025)
with both time-domain and phasor-domain state equations.

Reference:
    Rim et al., "General Instantaneous Dynamic Phasor," IEEE TPEL 2025
    - Figure 1: Circuit topology
    - Table II: Circuit parameters
    - Equations (37a-b): Phasor-space state equations
    - Equation (39): Analytical solution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict
from scipy.integrate import solve_ivp
from .phasor import PhasorConfig, InstantaneousPhasor


@dataclass
class CircuitParameters:
    """
    Circuit parameters matching Table II from Rim et al. (2025).
    
    Default values are from the benchmark paper:
        L = 100.04 µH
        C = 30.07 nF  
        Rs = 3.0 Ω (includes inverter resistance)
        Ro = 2.00 kΩ
    """
    L: float = 100.04e-6    # Inductance [H]
    C: float = 30.07e-9     # Capacitance [F]
    Rs: float = 3.0         # Series resistance [Ω]
    Ro: float = 2000.0      # Load resistance [Ω]
    
    def __post_init__(self):
        self._validate()
        self._compute_derived()
    
    def _validate(self):
        """Validate all parameters are positive."""
        if self.L <= 0:
            raise ValueError("Inductance must be positive")
        if self.C <= 0:
            raise ValueError("Capacitance must be positive")
        if self.Rs < 0:
            raise ValueError("Series resistance must be non-negative")
        if self.Ro <= 0:
            raise ValueError("Load resistance must be positive")
    
    def _compute_derived(self):
        """Compute derived parameters."""
        # Resonant frequency
        self.omega_r = 1.0 / np.sqrt(self.L * self.C)
        self.f_r = self.omega_r / (2 * np.pi)
        
        # Quality factor
        self.Q = (1.0 / self.Rs) * np.sqrt(self.L / self.C)
        
        # Characteristic impedance
        self.Z0 = np.sqrt(self.L / self.C)
        
        # Damping factor
        self.zeta = self.Rs / (2 * np.sqrt(self.L / self.C))
    
    def __repr__(self) -> str:
        return (f"CircuitParameters(L={self.L*1e6:.2f}µH, C={self.C*1e9:.2f}nF, "
                f"Rs={self.Rs}Ω, Ro={self.Ro/1e3:.2f}kΩ)\n"
                f"  Resonant freq: {self.f_r/1e3:.2f} kHz ({self.omega_r/1e3:.2f} krad/s)\n"
                f"  Q factor: {self.Q:.2f}")


class RLCCircuit:
    r"""
    Series RLC resonant circuit with parallel load.

    Topology (Figure 1 from Rim et al.):

        Rs      L
    +--/\/\--UUUU--+--+
    |              |  |
    vs(t)          C  Ro  vo(t)
    |              |  |
    +--------------+--+
    
    State variables:
        is(t) = source/inductor current
        vo(t) = output/capacitor voltage
        
    Time-domain equations:
        L·dis/dt = vs - Rs·is - vo
        C·dvo/dt = is - vo/Ro
        
    Phasor-space equations (Eq. 37a-b):
        L·dĩs/dt = ṽs - {Rs + jXL(t)}·ĩs - ṽo
        C·dṽo/dt = ĩs - ṽo/Ro - ṽo/(jXc(t))
    """
    
    def __init__(self, params: CircuitParameters = None):
        """
        Initialize RLC circuit.
        
        Parameters
        ----------
        params : CircuitParameters, optional
            Circuit parameters. Uses Table II defaults if None.
        """
        self.params = params or CircuitParameters()
        self.phasor = None  # Set when using phasor methods
    
    def configure_phasor(self, omega_s: float, 
                         omega_0: float = None,
                         omega_1: float = None,
                         alpha: float = 0.0):
        """
        Configure phasor transformation parameters.
        
        Parameters
        ----------
        omega_s : float
            Source angular frequency (rad/s)
        omega_0, omega_1, alpha : float
            FM parameters for θ(t) = ω₀t + α·sin(ω₁t)
            If not specified, uses standard θ(t) = ωs·t
        """
        self.omega_s = omega_s
        
        config = PhasorConfig(
            omega=omega_s,
            omega_0=omega_0 or omega_s,
            omega_1=omega_1 or 0,
            alpha=alpha
        )
        self.phasor = InstantaneousPhasor(config)
    
    # =========================================================================
    # Time-Domain Methods
    # =========================================================================
    
    def time_domain_ode(self, t: float, state: np.ndarray, 
                        vs_func: Callable[[float], float]) -> np.ndarray:
        """
        Time-domain ODE for RLC circuit.
        
        State: [is, vo]
        
        L·dis/dt = vs - Rs·is - vo
        C·dvo/dt = is - vo/Ro
        
        Parameters
        ----------
        t : float
            Current time
        state : array [is, vo]
            Current state
        vs_func : callable
            Source voltage function vs(t)
            
        Returns
        -------
        array [dis/dt, dvo/dt]
        """
        is_t, vo_t = state
        vs_t = vs_func(t)
        
        p = self.params
        
        dis_dt = (vs_t - p.Rs * is_t - vo_t) / p.L
        dvo_dt = (is_t - vo_t / p.Ro) / p.C
        
        return np.array([dis_dt, dvo_dt])
    
    def solve_time_domain(self, vs_func: Callable[[float], float],
                          t_span: Tuple[float, float],
                          t_eval: np.ndarray = None,
                          initial_state: np.ndarray = None,
                          **solver_kwargs) -> Dict:
        """
        Solve circuit in time domain.
        
        Parameters
        ----------
        vs_func : callable
            Source voltage function vs(t)
        t_span : tuple
            (t_start, t_end)
        t_eval : array, optional
            Times at which to evaluate solution
        initial_state : array [is0, vo0], optional
            Initial conditions (default zeros)
        **solver_kwargs :
            Additional arguments for scipy.integrate.solve_ivp
            
        Returns
        -------
        dict with keys: t, is_t, vo_t, vs_t
        """
        if initial_state is None:
            initial_state = np.array([0.0, 0.0])
        
        if t_eval is None:
            # Default: 1000 points per period at resonant frequency
            n_points = int(1000 * (t_span[1] - t_span[0]) * self.params.omega_r / (2*np.pi))
            n_points = max(1000, min(n_points, 100000))
            t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.time_domain_ode(t, y, vs_func),
            t_span,
            initial_state,
            t_eval=t_eval,
            method=solver_kwargs.pop('method', 'RK45'),
            **solver_kwargs
        )
        
        return {
            't': sol.t,
            'is_t': sol.y[0],
            'vo_t': sol.y[1],
            'vs_t': np.array([vs_func(ti) for ti in sol.t])
        }
    
    # =========================================================================
    # Phasor-Domain Methods (Instantaneous)
    # =========================================================================
    
    def phasor_domain_ode(self, t: float, state: np.ndarray,
                          vs_phasor_func: Callable[[float], complex]) -> np.ndarray:
        """
        Phasor-domain ODE for RLC circuit.
        
        State: [Re(is), Im(is), Re(vo), Im(vo)]
        
        Implements Eq. (37a-b):
            L·dĩs/dt = ṽs - {Rs + jXL(t)}·ĩs - ṽo
            C·dṽo/dt = ĩs - ṽo/Ro - ṽo/(jXc(t))
            
        Note: jXc = -j/|Xc| so ṽo/(jXc) = j·|Xc|·ṽo = j·ṽo/(ω̇C)
        
        Parameters
        ----------
        t : float
            Current time
        state : array [Re(is), Im(is), Re(vo), Im(vo)]
            Current phasor state (real/imag parts)
        vs_phasor_func : callable
            Phasor source voltage function ṽs(t)
        """
        # Unpack state
        is_re, is_im, vo_re, vo_im = state
        is_phasor = is_re + 1j * is_im
        vo_phasor = vo_re + 1j * vo_im
        
        vs_phasor = vs_phasor_func(t)
        
        p = self.params
        
        # Time-varying reactances
        XL = self.phasor.reactance_L(p.L, t)  # XL = dθ/dt · L
        Xc = self.phasor.reactance_C(p.C, t)  # Xc = -1/(dθ/dt · C)
        
        # State equations (37a-b)
        # L·dĩs/dt = ṽs - (Rs + jXL)·ĩs - ṽo
        dis_phasor = (vs_phasor - (p.Rs + 1j * XL) * is_phasor - vo_phasor) / p.L
        
        # C·dṽo/dt = ĩs - ṽo/Ro - ṽo/(jXc)
        # Note: ṽo/(jXc) = ṽo·j/(-1/(dθ/dt·C)) = -j·dθ/dt·C·ṽo
        dvo_phasor = (is_phasor - vo_phasor / p.Ro - vo_phasor / (1j * Xc)) / p.C
        
        return np.array([
            np.real(dis_phasor),
            np.imag(dis_phasor),
            np.real(dvo_phasor),
            np.imag(dvo_phasor)
        ])
    
    def solve_phasor_domain(self, vs_phasor_func: Callable[[float], complex],
                            t_span: Tuple[float, float],
                            t_eval: np.ndarray = None,
                            initial_state: np.ndarray = None,
                            **solver_kwargs) -> Dict:
        """
        Solve circuit in phasor domain.
        
        Parameters
        ----------
        vs_phasor_func : callable
            Phasor source voltage function ṽs(t)
        t_span : tuple
            (t_start, t_end)
        t_eval : array, optional
            Times at which to evaluate solution
        initial_state : array [is_re, is_im, vo_re, vo_im], optional
            Initial phasor conditions (default zeros)
            
        Returns
        -------
        dict with keys: t, is_phasor, vo_phasor, vs_phasor
        """
        if self.phasor is None:
            raise RuntimeError("Must call configure_phasor() first")
        
        if initial_state is None:
            initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        
        if t_eval is None:
            n_points = max(1000, int(50000 * (t_span[1] - t_span[0])))
            t_eval = np.linspace(t_span[0], t_span[1], min(n_points, 10000))
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.phasor_domain_ode(t, y, vs_phasor_func),
            t_span,
            initial_state,
            t_eval=t_eval,
            method=solver_kwargs.pop('method', 'RK45'),
            **solver_kwargs
        )
        
        # Reconstruct complex phasors
        is_phasor = sol.y[0] + 1j * sol.y[1]
        vo_phasor = sol.y[2] + 1j * sol.y[3]
        vs_phasor = np.array([vs_phasor_func(ti) for ti in sol.t])
        
        return {
            't': sol.t,
            'is_phasor': is_phasor,
            'vo_phasor': vo_phasor,
            'vs_phasor': vs_phasor,
            # Also return real-space conversions
            'is_t': self.phasor.to_real(is_phasor, sol.t),
            'vo_t': self.phasor.to_real(vo_phasor, sol.t),
            'vs_t': self.phasor.to_real(vs_phasor, sol.t),
            # Envelopes
            'is_envelope': np.abs(is_phasor),
            'vo_envelope': np.abs(vo_phasor)
        }
    
    # =========================================================================
    # Analytical Solution (Eq. 39)
    # =========================================================================
    
    def analytical_phasor_response(self, t: np.ndarray, 
                                   Ve: float = 1.0,
                                   phi_s: float = 0.0) -> Dict:
        """
        Analytical phasor response for step envelope input.
        
        Implements Eq. (39) from Rim et al.:
            ṽo(t) = K·{1 - p1(t)·p2(t)}
            
        Valid for θ(t) = ωs·t (standard case, no FM)
        
        Parameters
        ----------
        t : array
            Time vector
        Ve : float
            Step envelope amplitude
        phi_s : float
            Source phase angle
            
        Returns
        -------
        dict with analytical phasor response
        """
        if self.phasor is None:
            raise RuntimeError("Must call configure_phasor() first")
        
        p = self.params
        omega_s = self.omega_s
        
        # Eq. (39c): Constants
        K_num = p.Ro * np.exp(1j * phi_s)
        K_den = (p.Ro + p.Rs - p.L * p.C * p.Ro * omega_s**2 + 
                 1j * omega_s * (p.L + p.C * p.Ro * p.Rs))
        K = K_num / K_den * Ve
        
        # p1(t)
        decay = -(1.0 / (2 * p.C * p.Ro) + p.Rs / (2 * p.L))
        p1 = np.exp(decay * t - 1j * omega_s * t)
        
        # omega_p (natural frequency in phasor domain)
        omega_p_sq = ((4 * p.L * p.C * p.Ro**2 - (p.L - p.C * p.Ro * p.Rs)**2) / 
                      (4 * p.L**2 * p.C**2 * p.Ro**2))
        
        if omega_p_sq > 0:
            omega_p = np.sqrt(omega_p_sq)
        else:
            # Overdamped case
            omega_p = 1j * np.sqrt(-omega_p_sq)
        
        # p2(t)
        coeff = (1.0 / (2 * p.C * p.Ro) + p.Rs / (2 * p.L) + 1j * omega_s)
        
        if np.isreal(omega_p):
            p2 = 1 - (np.cos(omega_p * t) + coeff * np.sin(omega_p * t) / omega_p)
        else:
            # Overdamped: use sinh/cosh
            omega_p_real = np.imag(omega_p)  # omega_p was set to j*sqrt(...)
            p2 = 1 - (np.cosh(omega_p_real * t) + coeff * np.sinh(omega_p_real * t) / omega_p_real)
        
        # Final result
        vo_phasor = K * (1 - p1 * p2)
        
        return {
            't': t,
            'vo_phasor': vo_phasor,
            'vo_envelope': np.abs(vo_phasor),
            'K': K,
            'omega_p': omega_p
        }
    
    # =========================================================================
    # Transfer Functions
    # =========================================================================
    
    def transfer_function_real(self, s: complex) -> complex:
        """
        Real-space transfer function Gv(s) = Vo(s)/Vs(s).
        
        Eq. (25):
            Gv(s) = Ro / ((Rs + sL)(1 + sCRo) + Ro)
        """
        p = self.params
        return p.Ro / ((p.Rs + s * p.L) * (1 + s * p.C * p.Ro) + p.Ro)
    
    def transfer_function_phasor(self, s: complex) -> complex:
        """
        Phasor-space transfer function Gv(s) = Gv(s + jωs).
        
        Eq. (34-35): Frequency-shifted transfer function
        """
        return self.transfer_function_real(s + 1j * self.omega_s)
    
    def frequency_response(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response (magnitude and phase).
        
        Parameters
        ----------
        f : array
            Frequency vector in Hz
            
        Returns
        -------
        magnitude : array
            |Gv(jω)| in dB
        phase : array
            ∠Gv(jω) in degrees
        """
        omega = 2 * np.pi * f
        H = np.array([self.transfer_function_real(1j * w) for w in omega])
        
        magnitude = 20 * np.log10(np.abs(H))
        phase = np.angle(H, deg=True)
        
        return magnitude, phase


def create_rim2025_circuit(omega_s: float = 580e3) -> RLCCircuit:
    """
    Create RLC circuit with exact parameters from Rim et al. (2025) Table II.
    
    Parameters
    ----------
    omega_s : float
        Operating frequency (default 580 krad/s)
        Paper tests at 580 and 650 krad/s
        
    Returns
    -------
    RLCCircuit configured for benchmark validation
    """
    params = CircuitParameters(
        L=100.04e-6,   # 100.04 µH
        C=30.07e-9,    # 30.07 nF
        Rs=3.0,        # 3.0 Ω
        Ro=2000.0      # 2.00 kΩ
    )
    
    circuit = RLCCircuit(params)
    circuit.configure_phasor(omega_s=omega_s)
    
    return circuit
