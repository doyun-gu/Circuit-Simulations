# Circuit Simulation Library

This Python library designed for creating, connecting, and analysing electrical circuits, featuring both DC/AC operating point calculation and dynamic Transient Analysis.

## Overview

This library provides an object-oriented framework for circuit modeling, leveraging the **Modified Nodal Analysis (MNA)** technique and the **Backward Euler (BE)** method for time-domain simulation.

The design is split into three main layers: **Component Modeling**, **Circuit Building**, and **Mathematical Analysis**.

## Features

- **Clean API**: Easy component creation via a Factory Pattern (`createComps.resistor(value)`).

- **Component Set**: Supports standard R, L, C, and independent Voltage Sources.

- **DC Analysis**: Calculates the steady-state node voltages and component currents.

- **Transient Analysis**: Simulates circuit behavior over time, modeling energy storage elements (L and C).

- **Visualisation**: Integrated plotting functions using `matplotlib` for time-series results.

- **Data Output**: Tabular results displayed via Pandas DataFrames for clear notebook output.

## Mathematical Approaches: Modified Nodal Analysis (MNA)

The library uses Modified Nodal Analysis (MNA) as its primary solution technique. MNA is an extension of Nodal Analysis designed to handle components that cannot be modeled solely by conductance, such as voltage sources and inductors.

The entire circuit is converted into a linear system of equations of the form:

$$\mathbf{A} \mathbf{x} = \mathbf{z}$$

|Variable|Description|Components|
|:------:|:---------:|:--------:|
|$\mathbf{A}$ (MNA Matrix)|Contains the relationship between node voltages and component stamps.|Node conductances (from $R$, $C$, $L$) and constraint entries (from $V$, $L$).|
|$\mathbf{x}$ (Solution Vector)|Contains all unknown voltages and currents.|Node voltages ($V_1, V_2, \dots$) and auxiliary currents ($I_{V_1}, I_{L_1}, \dots$)|
|$\mathbf{z}$ (Source Vector)|Contains known independent sources.|Voltage source values and equivalent current source values.|

# Introduction to MNA sovler Vectors

The vector $\mathbf{x}$ is known as the Solution Vector. It contains every unknown quantity we are trying to find in the circuit. Since standard Nodal Analysis only solves for voltages, MNA adds extra rows (and corresponding columns) to the matrix to solve for crucial unknown currents.The full MNA solution vector $\mathbf{x}$ is always structured as two stacked sub-vectors:

$$
\mathbf{x} = \begin{bmatrix} \mathbf{V} \\ \mathbf{I} \end{bmatrix}
$$

## The Voltage Subvector ($\mathbf{V}$)

This sub-vector contains the primary unknowns: the potential difference (voltage) at every non-ground node in the circuit.

- **Definition**: The voltage at every named node (e.g., N1, N_mid, N_start) relative to the reference ground node (GND).

- **Composition**:

    $$
    \mathbf{V} = \begin{bmatrix} V_{N1} \\ V_{N2} \\ \vdots \\ V_{N_n} \end{bmatrix}
    $$

    where $N_n$ is the total number of non-ground nodes.

## The Auxiliary Current Subvector ($\mathbf{I}$)

This sub-vector contains the currents through the components that required extra rows in the MNA matrix to be solved. These are often called auxiliary variables.

- **Definition**: Currents that are not easily calculated via Ohm's Law (like the current through a resistor), but are necessary constraints in the MNA framework.

- **Composition**:

    $$
    \mathbf{I} = \begin{bmatrix} I_{V1} \\ I_{L1} \\ \vdots \\ I_{L_m} \end{bmatrix}
    $$

    where $m$ is the total count of components that require a current variable—in your case, Voltage Sources and Inductors.

- **Significance**: This allows the solver to determine the current flowing out of a Voltage Source (which is unknown) or the current flowing through an Inductor (which is an energy state variable).

### DC Analysis

Goal: Calculate the voltages and currents at the steady-state operating point ($t \to \infty$).

Component Model:

Capacitor (C): Treated as an open circuit (infinite resistance).

Inductor (L): Treated as a short circuit (zero resistance, approximated in code as $R_{approx} = 10^{-9}\Omega$).

Equations: Uses standard admittance stamps for Resistors ($G=1/R$) and constraint rows for Voltage Sources.

Transient Analysis (Time Domain)

Goal: Calculate the voltages and currents $V(t)$ over a finite time period.

Method: Backward Euler (BE) integration. This is a robust numerical method that transforms the circuit's differential equations into algebraic equations solvable by MNA at each small time step ($\Delta t$).

Component Modeling: Backward Euler Stamping

BE transforms the energy storage elements (C and L) into equivalent static circuits based on the previous solution, $V(t-\Delta t)$, and the current time step, $\Delta t$.

|Component|Physical Equation|BE Equivalent Model|Effective Value (MNA Stamp)|
|:-------:|:---------------:|:-----------------:|:-------------------------:|
|Capacitor ($C$)|$I(t) = C \frac{dV}{dt}$|Equivalent Conductance $G_{eq}$ and Current Source $I_{eq}$.|$G_{eq} = \frac{C}{\Delta t}$|
|Inductor ($L$)|$V(t) = L \frac{dI}{dt}$|Equivalent Resistor $R_{eq}$ and Voltage Source $V_{eq}$.|$R_{eq} = \frac{L}{\Delta t}$|

Stamping Equations added to the Source Vector ($\mathbf{z}$):

Capacitor Source ($I_{eq}$):


$$I_{eq} = G_{eq} \cdot (V_{i}(t-\Delta t) - V_{j}(t-\Delta t))$$


(This represents the current injected by the capacitor's energy stored at the previous time step)

Inductor Source ($V_{eq}$):


$$V_{eq} = R_{eq} \cdot I_{L}(t-\Delta t)$$


(This represents the voltage generated by the inductor's current at the previous time step)

