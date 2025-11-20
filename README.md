# Dynamic Phasors Study Repo

Authored by `doyun-gu` @ the University of Manchester

---

This repository contains simulation-based studies on dynamic phasors, supervised by **Dr. Gus**.

The project goes a bit beyond the standard university coursework and is run as an individual study, with the aim of digging deeper into dynamic phasors, which are one of the key concepts in modern circuit and power-electronics analysis.

The code and reports explore:

- How circuit simulation can be formulated using **time-domain differential equations** and **complex matrix representations**.
- How **dynamic phasor models** are derived from these equations and how they relate to classical phasor and steady-state analysis.
- How different power transfer functions and node equations can be used to analyse and compare circuits in both the time and phasor domains.

For more details, please refer to the reports and notebooks included in this repository.

## Reports
- [Report 1: Instantaneous vs Classical Dynamic Phasor](https://drive.google.com/file/d/1N8srI0JEFsgYhiBxKU9C63Dn7HJH6IQw/view?usp=drive_link)

# Series RLC Circuit and Node Equations

This simulation will consider a simple RLC circuit.

- Input: Voltage source $u(t)$
- Element: Resistor $R$, inductor $L$, Capacitor $C$ in series
- Output: Capacitor voltage $v_c(t)$

We choose as state variables:

- Inductor current: $i_L(t)$
- Capacitor voltage: $v_C(t)$

Using KVL and the element relations:

- $v_R(t) = Ri_L(t)$
- $v_L(t) = L{di_L\over dt}$
- $i_C(t) = C{dv_c\over dt}$ and $i_C = i_L$ in series

We obtain the time-domain equations:

$$
L {di_L \over dt} = u(t) - R i_L(t) - v_C(t)
$$

$$
C {dv_C\over dt} = i_L(t)
$$

These two-order equations describe the dynamics of the series RLC circuit.

---

# Transfer Function $G(s) = {V_C(s) \over U(s)}$

In the Laplace domain, the impedances of the elements are

- $Z_R = R$
- $Z_L = sL$
- $Z_C = {1\over sC}$

Using voltage division for the series circuit, the transfer function from input $u(t)$ to capacitor voltage $v_C(t)$ is:

$$
G(s) = {V_C(t)\over U(s)} = {Z_C\over Z_R + Z_L + Z_C} = {{1\over sC}\over R+sL+{1\over sC}} = {1\over LCs^2+RCs+1}
$$

This is the frequency-domain description of the circuit.

---

# Differential Equation from the Transfer Function

Starting from

$$
{V_C(s)\over U(s)} = {1\over LCs^2 + RC s + 1'}
$$

we multiply both sides by $LCs^2 + RCs + 1$:

$$
(LCs^2 + RCs + 1)V_C(s) = U(s)
$$

Taking the inverse Laplace Transform gives the second-order differential equation:

$$
LC {d^2v_C\over dt^2} + RC {dv_C \over dt} + v_C(t) = u(t)
$$

This equation is equivalent to the first-order system written in the terms of $i_L(t)$ and $v_C(t)$.

---

# State-Space (Matrix) Form

We define the state vector:

$$
\mathbf{x}(t) =
\begin{bmatrix}
i_L(t) \\
v_C(t)
\end{bmatrix},
\qquad
\dot{\mathbf{x}}(t) =
\begin{bmatrix}
\dot{i}_L(t) \\
\dot{v}_C(t)
\end{bmatrix}.
$$

Using the first-order equations, we can write:

$$
\dot {\mathbf{x}}(t) = A\mathbf{x}(t) + Bu(t)
$$

with 

$$
A =
\begin{bmatrix}
-\dfrac{R}{L} & -\dfrac{1}{L} \\
\dfrac{1}{C}  & 0
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
\dfrac{1}{L} \\
0
\end{bmatrix}.
$$

If we choose the output to be the capacitor voltage $y(t) = v_C(t)$, the output equation is 

$$
y(t) = C\mathbf{x}(t) + Du(t)
$$

with

$$
C = [0 \; 1], \qquad D = 0
$$

$$
\dot{\mathbf{x}} = A\mathbf{x} + B u, \qquad
y = C\mathbf{x} + D u
$$

This compact form

$$ 
\dot{\mathbf{x}}=A\mathbf{x} + Bu, \qquad
y = C\mathbf{x} + Du
$$

is the **matrix differential equation** (state-space model) used in simulation.

---

# Sine and Square Wave Inputs

We can excite the RLC circuit with different input waveforms:

## Single Sine Wave:

$$
u(t) = \hat{U} \sin(\omega t)
$$

which, in steady state, produces a sinusoidal output at the same frequency with gain and phase given by $G(j\omega)$

## Square Wave

$$
u(t) = \hat {U} \text{sgn} (\sin(\omega t))
$$

which can be seen as a sum of many sine waves (Fourier series). The state-space model handles this input naturally in the time domain by integrating the same differential equations.

In both cases, the underlying dynamics are governed by the same state-space system; only the input function $u(t)$ changes.

---

## References

> [!NOTE]
> Most of the material in this repo is based on the textbook Circuit Simulation, and extended with a few papers on dynamic phasors

For phasor and dynamic phasor theory, I also referred to:

[1] C. K. Alexander and M. N. O. Sadiku, *Fundamentals of Electric Circuits*, 5th ed. New York, NY, USA: McGraw–Hill, 2013, ch. 9.  

[2] S. R. Sanders, J. M. Noworolski, X. Z. Liu, and G. C. Verghese, “Generalized averaging method for power conversion circuits,” *IEEE Trans. Power Electron.*, vol. 6, no. 2, pp. 251–259, Apr. 1991.  

[3] C. T. Rim, S. A. A. Shah, H. Park, *et al*., “General Instantaneous Dynamic Phasor,” *IEEE Trans. Power Electron.*, vol. 40, no. 11, pp. 16953–16962, Nov. 2025.
