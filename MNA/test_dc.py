import numpy as np

class RLCCircuitMNA:
    def __init__(self, R, L, C, V_source, frequency=0):
        self.R = R
        self.L = L
        self.C = C
        self.V_source = V_source
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.s = 1j * self.omega
        self.Z_L = self.s * L if frequency > 0 else 0
        self.Z_C = 1 / (self.s * C) if frequency > 0 else float('inf')

    def build_mna_matrix(self):
        if self.frequency == 0:
            # DC Analysis
            n = 4
            A = np.zeros((n, n), dtype=complex)
            b = np.zeros(n, dtype=complex)

            # Node 1: Voltage source connection
            A[0, 0] = 1/self.R
            A[0, 1] = -1/self.R
            A[0, 3] = 1
            b[0] = 0

            # Node 2: Between R and L (L is short at DC)
            A[1, 0] = -1/self.R
            A[1, 1] = 1/self.R
            b[1] = 0

            # Node 3: V3 = V2 constraint
            A[2, 1] = -1
            A[2, 2] = 1
            b[2] = 0

            # Voltage source constraint
            A[3, 0] = 1
            b[3] = self.V_source

        else:
            # AC Analysis
            n = 4
            A = np.zeros((n, n), dtype=complex)
            b = np.zeros(n, dtype=complex)

            A[0, 0] = 1/self.R
            A[0, 1] = -1/self.R
            A[0, 3] = 1
            b[0] = 0

            A[1, 0] = -1/self.R
            A[1, 1] = 1/self.R + 1/self.Z_L
            A[1, 2] = -1/self.Z_L
            b[1] = 0

            A[2, 1] = -1/self.Z_L
            A[2, 2] = 1/self.Z_L + 1/self.Z_C
            b[2] = 0

            A[3, 0] = 1
            A[3, 3] = 0
            b[3] = self.V_source

        return A, b

    def solve(self):
        A, b = self.build_mna_matrix()
        print(f"Matrix rank: {np.linalg.matrix_rank(A)}")
        print(f"Matrix shape: {A.shape}")
        print(f"Determinant: {np.linalg.det(A)}")
        x = np.linalg.solve(A, b)
        return x, A, b

# Test DC
print("Testing DC analysis:")
R = 100
L = 0.1
C = 1e-6
V_source = 10
frequency = 0

circuit = RLCCircuitMNA(R, L, C, V_source, frequency)
x, A, b = circuit.solve()

print("\nSolution:")
print(f"V1 = {x[0].real:.6f} V")
print(f"V2 = {x[1].real:.6f} V")
print(f"V3 = {x[2].real:.6f} V")
print(f"I_vs = {x[3].real:.6f} A")
print("\nSuccess!")
