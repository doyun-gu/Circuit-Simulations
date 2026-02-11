"""
Core circuit components for dynamic phasor simulation.

This module implements fundamental circuit elements (R, L, C) with support
for both time-domain and phasor-domain representations.

Reference:
    Rim et al., "General Instantaneous Dynamic Phasor," IEEE TPEL 2025
    Equations (7)-(11) for component phasor transformations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
import numpy as np


@dataclass
class ComponentBase(ABC):
    """Abstract base class for all circuit components."""
    
    name: str = ""
    
    @abstractmethod
    def impedance(self, omega: float) -> complex:
        """Return complex impedance at angular frequency omega."""
        pass
    
    @abstractmethod
    def time_domain_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return time-domain differential equation contribution."""
        pass
    
    @abstractmethod
    def phasor_impedance(self, omega: float, s: complex = 0) -> complex:
        """
        Return phasor-domain impedance.
        
        For dynamic phasor, this includes the frequency shift:
        Z_phasor(s) = Z(s + jω)  [Eq. 34 in Rim et al.]
        
        Parameters
        ----------
        omega : float
            Angular frequency for phasor transformation
        s : complex
            Laplace variable (default 0 for steady-state)
        """
        pass


@dataclass
class Resistor(ComponentBase):
    """
    Linear time-invariant resistor.
    
    Phasor transformation (Eq. 10):
        R·i_R(t) = v_R(t) ↔ R·i_R(t) = v_R(t)
    
    Parameters
    ----------
    R : float
        Resistance in Ohms
    """
    R: float = 1.0
    
    def __post_init__(self):
        if self.R < 0:
            raise ValueError("Resistance must be non-negative")
    
    def impedance(self, omega: float) -> complex:
        """Resistance is frequency-independent."""
        return complex(self.R, 0)
    
    def time_domain_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """Ohm's law: v = R·i"""
        return self.R * state
    
    def phasor_impedance(self, omega: float, s: complex = 0) -> complex:
        """Resistor phasor impedance is unchanged."""
        return complex(self.R, 0)
    
    def __repr__(self) -> str:
        return f"Resistor({self.name}, R={self.R}Ω)"


@dataclass
class Inductor(ComponentBase):
    """
    Linear time-invariant inductor.
    
    Time-domain (Eq. 7a):
        L·di_L/dt = v_L
        
    Phasor transformation (Eq. 7c):
        L·di_L/dt + jX_L(t)·i_L = v_L
        where X_L(t) = dθ/dt · L = ω·L for θ(t) = ωt
        
    Laplaced phasor (Eq. 19):
        (s + jω)L·I_L(s) = V_L(s)
        
    Approximation for slow dynamics (Eq. 21a):
        I_L(s) ≈ V_L(s)/(jωL) + s·C_e·V_L(s)
        where C_e = 1/(ω²L) - equivalent capacitance
    
    Parameters
    ----------
    L : float
        Inductance in Henries
    """
    L: float = 100e-6  # Default 100 µH
    
    def __post_init__(self):
        if self.L <= 0:
            raise ValueError("Inductance must be positive")
    
    def impedance(self, omega: float) -> complex:
        """Standard inductive impedance jωL."""
        return complex(0, omega * self.L)
    
    def reactance(self, omega: float) -> float:
        """Inductive reactance X_L = ωL."""
        return omega * self.L
    
    def time_varying_reactance(self, theta_dot: float) -> float:
        """
        Time-varying reactance for instantaneous dynamic phasor.
        
        X_L(t) = dθ/dt · L  [Eq. 8]
        
        For θ(t) = ω₀t + α·sin(ω₁t):
            X_L(t) = (ω₀ + α·ω₁·cos(ω₁t))·L  [Eq. 36a]
        """
        return theta_dot * self.L
    
    def time_domain_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """di_L/dt = v_L / L"""
        return state / self.L
    
    def phasor_impedance(self, omega: float, s: complex = 0) -> complex:
        """
        Phasor impedance with frequency shift.
        
        Z_L(s) = (s + jω)L  [from Eq. 19]
        """
        return (s + 1j * omega) * self.L
    
    def equivalent_capacitance(self, omega: float) -> float:
        """
        Equivalent capacitance for slow dynamics approximation.
        
        C_e = 1/(ω²L)  [Eq. 21a]
        """
        return 1.0 / (omega**2 * self.L)
    
    def __repr__(self) -> str:
        return f"Inductor({self.name}, L={self.L*1e6:.2f}µH)"


@dataclass
class Capacitor(ComponentBase):
    """
    Linear time-invariant capacitor.
    
    Time-domain (Eq. 9a):
        C·dv_c/dt = i_c
        
    Phasor transformation (Eq. 9b):
        C·dv_c/dt + v_c/(jX_c(t)) = i_c
        where X_c(t) = -1/(dθ/dt · C) = -1/(ωC) for θ(t) = ωt
        
    Laplaced phasor (Eq. 20a):
        (s + jω)C·V_c(s) = I_c(s)
        
    Approximation for slow dynamics (Eq. 21b):
        V_c(s) ≈ I_c(s)/(jωC) + s·L_e·I_c(s)
        where L_e = 1/(ω²C) - equivalent inductance
    
    Parameters
    ----------
    C : float
        Capacitance in Farads
    """
    C: float = 30e-9  # Default 30 nF
    
    def __post_init__(self):
        if self.C <= 0:
            raise ValueError("Capacitance must be positive")
    
    def impedance(self, omega: float) -> complex:
        """Standard capacitive impedance -j/(ωC)."""
        if omega == 0:
            return complex(float('inf'), 0)
        return complex(0, -1.0 / (omega * self.C))
    
    def reactance(self, omega: float) -> float:
        """Capacitive reactance X_c = -1/(ωC)."""
        if omega == 0:
            return float('-inf')
        return -1.0 / (omega * self.C)
    
    def time_varying_reactance(self, theta_dot: float) -> float:
        """
        Time-varying reactance for instantaneous dynamic phasor.
        
        X_c(t) = -1/(dθ/dt · C)  [Eq. 9c]
        
        For θ(t) = ω₀t + α·sin(ω₁t):
            X_c(t) = -1/((ω₀ + α·ω₁·cos(ω₁t))·C)  [Eq. 36b]
        """
        if theta_dot == 0:
            return float('-inf')
        return -1.0 / (theta_dot * self.C)
    
    def time_domain_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """dv_c/dt = i_c / C"""
        return state / self.C
    
    def phasor_impedance(self, omega: float, s: complex = 0) -> complex:
        """
        Phasor impedance with frequency shift.
        
        Z_c(s) = 1/((s + jω)C)  [from Eq. 20a]
        """
        return 1.0 / ((s + 1j * omega) * self.C)
    
    def equivalent_inductance(self, omega: float) -> float:
        """
        Equivalent inductance for slow dynamics approximation.
        
        L_e = 1/(ω²C)  [Eq. 21b]
        """
        return 1.0 / (omega**2 * self.C)
    
    def __repr__(self) -> str:
        return f"Capacitor({self.name}, C={self.C*1e9:.2f}nF)"


@dataclass
class VoltageSource(ComponentBase):
    """
    Voltage source with time-varying envelope.
    
    General form (Eq. 22):
        v_s(t) = v_e(t)·cos(ω_s·t + φ_s)
               = Re{v_e(t)·e^(jω_s·t + jφ_s)}
    
    Dynamic phasor (Eq. 24):
        v_s(t) = v_e(t)·e^(jφ_s)
    
    Parameters
    ----------
    amplitude : float or callable
        Peak amplitude or envelope function v_e(t)
    omega : float
        Angular frequency ω_s in rad/s
    phase : float
        Phase angle φ_s in radians
    """
    amplitude: Union[float, Callable[[float], float]] = 1.0
    omega: float = 0.0
    phase: float = 0.0
    
    def envelope(self, t: float) -> float:
        """Return envelope value at time t."""
        if callable(self.amplitude):
            return self.amplitude(t)
        return self.amplitude
    
    def value(self, t: float) -> float:
        """
        Return instantaneous voltage value.
        
        v_s(t) = v_e(t)·cos(ω_s·t + φ_s)
        """
        return self.envelope(t) * np.cos(self.omega * t + self.phase)
    
    def phasor(self, t: float) -> complex:
        """
        Return dynamic phasor value.
        
        v_s(t) = v_e(t)·e^(jφ_s)  [Eq. 24]
        """
        return self.envelope(t) * np.exp(1j * self.phase)
    
    def phasor_for_theta(self, t: float, theta: float) -> complex:
        """
        Return dynamic phasor for arbitrary phase θ(t).
        
        v_s = v_e(t)·e^(j(ω_s·t + φ_s - θ(t)))  [Eq. 38b]
        
        This is for the extended phasor transformation with
        θ(t) ≠ ω_s·t (e.g., FM case).
        """
        phase_diff = self.omega * t + self.phase - theta
        return self.envelope(t) * np.exp(1j * phase_diff)
    
    def impedance(self, omega: float) -> complex:
        """Ideal voltage source has zero impedance."""
        return complex(0, 0)
    
    def time_domain_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """Voltage source directly provides voltage."""
        return np.array([self.value(t)])
    
    def phasor_impedance(self, omega: float, s: complex = 0) -> complex:
        """Ideal voltage source has zero impedance."""
        return complex(0, 0)
    
    def __repr__(self) -> str:
        amp = self.amplitude if not callable(self.amplitude) else "f(t)"
        return f"VoltageSource({self.name}, V={amp}, ω={self.omega:.0f}rad/s)"


@dataclass  
class CurrentSource(ComponentBase):
    """
    Current source with time-varying envelope.
    
    General form (Eq. 11b):
        i_o(t) = Re{i_o(t)·e^(jω·t)}
    
    Parameters
    ----------
    amplitude : float or callable
        Peak amplitude or envelope function
    omega : float
        Angular frequency in rad/s
    phase : float
        Phase angle in radians
    """
    amplitude: Union[float, Callable[[float], float]] = 1.0
    omega: float = 0.0
    phase: float = 0.0
    
    def envelope(self, t: float) -> float:
        """Return envelope value at time t."""
        if callable(self.amplitude):
            return self.amplitude(t)
        return self.amplitude
    
    def value(self, t: float) -> float:
        """Return instantaneous current value."""
        return self.envelope(t) * np.cos(self.omega * t + self.phase)
    
    def phasor(self, t: float) -> complex:
        """Return dynamic phasor value."""
        return self.envelope(t) * np.exp(1j * self.phase)
    
    def impedance(self, omega: float) -> complex:
        """Ideal current source has infinite impedance."""
        return complex(float('inf'), 0)
    
    def time_domain_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """Current source directly provides current."""
        return np.array([self.value(t)])
    
    def phasor_impedance(self, omega: float, s: complex = 0) -> complex:
        """Ideal current source has infinite impedance."""
        return complex(float('inf'), 0)
    
    def __repr__(self) -> str:
        amp = self.amplitude if not callable(self.amplitude) else "f(t)"
        return f"CurrentSource({self.name}, I={amp}, ω={self.omega:.0f}rad/s)"


# Utility functions for component calculations
def resonant_frequency(L: float, C: float) -> float:
    """
    Calculate resonant angular frequency.
    
    ω_r = 1/√(LC)
    """
    return 1.0 / np.sqrt(L * C)


def quality_factor(L: float, C: float, R: float) -> float:
    """
    Calculate quality factor for series RLC circuit.
    
    Q = (1/R)·√(L/C)
    """
    if R == 0:
        return float('inf')
    return (1.0 / R) * np.sqrt(L / C)


def characteristic_impedance(L: float, C: float) -> float:
    """
    Calculate characteristic impedance.
    
    Z_0 = √(L/C)
    """
    return np.sqrt(L / C)
