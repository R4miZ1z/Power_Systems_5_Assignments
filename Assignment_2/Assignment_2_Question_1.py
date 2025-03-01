#%% md
# # Imports
#%%
import numpy as np
from tabulate import tabulate
#%% md
# # Some helper functions
#%%
def pol_to_rec(radii: float, angle: float) -> complex:
    """
    Convert polar coordinates to rectangular coordinates.

    Parameters
    ----------
    radii: float
        Magnitude of the vector/phasor
    angle: float
        Angle of the vector/phasor in degrees

    Returns
    -------
    complex
        Rectangular coordinates of the vector/phasor
    """
    angle = angle * np.pi / 180
    return radii * (np.cos(angle) + 1j * np.sin(angle))

def rec_to_pol(z: complex) -> list[float]:
    """
    Convert rectangular coordinates to polar coordinates.

    Parameters
    ----------
    z: complex
        Rectangular coordinates of the vector/phasor

    Returns
    -------
    list[float]
        Polar coordinates of the vector/phasor in the format [radii, degree angle]
    """
    return [np.abs(z), np.angle(z, deg = True)]
#%% md
# # Question 1.1
#%% md
# ## Provided impedances from the question
#%%
imp_1_2: complex = 0.0 + 0.125j     # Impedance from bus 1 to bus 2
imp_1_3: complex = 0.0 + 0.250j     # Impedance from bus 1 to bus 3
imp_1_4: complex = 0.0 + 0.400j     # Impedance from bus 1 to bus 4
imp_2_3: complex = 0.0 + 0.250j     # Impedance from bus 2 to bus 3
imp_2_4: complex = 0.0 + 0.200j     # Impedance from bus 2 to bus 4
imp_3_0: complex = 0.0 + 1.250j     # Impedance from bus 3 to bus 0
imp_4_0: complex = 0.0 + 1.250j     # Impedance from bus 4 to bus 0

add_1_2: complex = 1 / imp_1_2      # admittance of line 1-2
add_1_3: complex = 1 / imp_1_3      # admittance of line 1-3
add_1_4: complex = 1 / imp_1_4      # admittance of line 1-4
add_2_3: complex = 1 / imp_2_3      # admittance of line 2-3
add_2_4: complex = 1 / imp_2_4      # admittance of line 2-4
add_3_0: complex = 1 / imp_3_0      # admittance of line 3-0
add_4_0: complex = 1 / imp_4_0      # admittance of line 4-0
#%% md
# ## Calculate the mutual and self admittances
#%%
# Mutual admittances
Y_12: complex = -add_1_2    # mutual admittance of line 1-2
Y_13: complex = -add_1_3    # mutual admittance of line 1-3
Y_14: complex = -add_1_4    # mutual admittance of line 1-4
Y_23: complex = -add_2_3    # mutual admittance of line 2-3
Y_24: complex = -add_2_4    # mutual admittance of line 2-4
Y_30: complex = -add_3_0    # mutual admittance of line 3-0
Y_40: complex = -add_4_0    # mutual admittance of line 4-0
Y_21: complex = -add_1_2    # mutual admittance of line 2-1
Y_31: complex = -add_1_3    # mutual admittance of line 3-1
Y_41: complex = -add_1_4    # mutual admittance of line 4-1
Y_32: complex = -add_2_3    # mutual admittance of line 3-2
Y_42: complex = -add_2_4    # mutual admittance of line 4-2

Y_34: complex = 0.0         # mutual admittance of line 3-4
Y_43: complex = 0.0         # mutual admittance of line 4-3

# Self admittances
Y_11: complex = add_1_2 + add_1_3 + add_1_4     # self admittance of bus 1
Y_22: complex = add_1_2 + add_2_3 + add_2_4     # self admittance of bus 2
Y_33: complex = add_1_3 + add_2_3 + add_3_0     # self admittance of bus 3
Y_44: complex = add_1_4 + add_2_4 + add_4_0     # self admittance of bus 4

# Construct admittance matrix
Y_matrix: np.ndarray = np.array(
    [
        [Y_11, Y_12, Y_13, Y_14],
        [Y_21, Y_22, Y_23, Y_24],
        [Y_31, Y_32, Y_33, Y_34],
        [Y_41, Y_42, Y_43, Y_44],
    ]
)

Y_matrix.imag
#%% md
# ## Define the current phasors
#%%
I_1: complex = pol_to_rec(0.00, 0.00)      # Current at bus 1
I_2: complex = pol_to_rec(0.00, 0.00)      # Current at bus 2
I_3: complex = pol_to_rec(1.00, -90.00)    # Current at bus 3
I_4: complex = pol_to_rec(0.68, -135.00)   # Current at bus 4

# Construct current vector
I_vector: np.ndarray = np.array(
    [
        [I_1],
        [I_2],
        [I_3],
        [I_4]
    ]
)
#%% md
# # Question 1.2
#%% md
# ## Solve the system of linear equations to calculate the bus voltages
#%%
# Using triangular factorisation to solve the system of linear equations
V_vector = np.linalg.solve(Y_matrix, I_vector)

# Round the values to 3 decimal places
V_vector_rounded = np.round(V_vector, 3)

V_vector_header = ["Bus", "Rectangular Voltage", "Polar Voltage"]
V_vector_data = []
for index, value in enumerate(V_vector_rounded):
    V_vector_data.append(
        [
            f"V_{index + 1}",
            f"{"-" if value[0].real < 0 else "+"} {abs(value[0].real):^7.3f}"
            f"{"-" if value[0].imag < 0 else "+"} {abs(value[0].imag):^7.3f}j",
            f"{rec_to_pol(value[0])[0]:^7.5f} V " u"\N{ANGLE}" f" {rec_to_pol(value[0])[1]:^7.5f}" u"\N{DEGREE SIGN}",

        ]
    )

print(tabulate(V_vector_data, headers=V_vector_header, tablefmt = "pipe", numalign="center", stralign="center"))
#%% md
# ## Compute branch currents
#%%
# Calculate branch currents
I_12: complex = add_1_2 * (V_vector[0][0] - V_vector[1][0])
I_13: complex = add_1_3 * (V_vector[0][0] - V_vector[2][0])
I_14: complex = add_1_4 * (V_vector[0][0] - V_vector[3][0])
I_23: complex = add_2_3 * (V_vector[1][0] - V_vector[2][0])
I_24: complex = add_2_4 * (V_vector[1][0] - V_vector[3][0])

I_30: complex = add_3_0 * (V_vector[2][0])
I_40: complex = add_4_0 * (V_vector[3][0])

I_21: complex = add_1_2 * (V_vector[1][0] - V_vector[0][0])
I_31: complex = add_1_3 * (V_vector[2][0] - V_vector[0][0])
I_41: complex = add_1_4 * (V_vector[3][0] - V_vector[0][0])
I_32: complex = add_2_3 * (V_vector[2][0] - V_vector[1][0])
I_42: complex = add_2_4 * (V_vector[3][0] - V_vector[1][0])

I_branch_vector: dict[str, complex] = {
    "I_12": I_12,
    "I_13": I_13,
    "I_14": I_14,
    "I_23": I_23,
    "I_24": I_24,
    "I_30": I_30,
    "I_40": I_40,
    "I_21": I_21,
    "I_31": I_31,
    "I_41": I_41,
    "I_32": I_32,
    "I_42": I_42
}

I_branch_header = ["Branch", "Rectangular Current", "Polar Current"]
I_branch_data = []
for key, value in I_branch_vector.items():
    sign_real = "+" if value.real >= 0 else "-"
    sign_imag = "+" if value.imag >= 0 else "-"
    I_branch_data.append(
        [
            key,
            f"{sign_real:^1} {abs(value.real):^7.5f} {sign_imag:^1} {abs(value.imag):^7.5f}j",
            f"{abs(value):^7.5f} A " u"\N{ANGLE}" f" {np.angle(value, deg=True):>10.5f}" u"\N{DEGREE SIGN}"
        ]
    )
print(tabulate(I_branch_data, headers=I_branch_header, tablefmt = "pipe", numalign="center", stralign="center"))
#%% md
# ## Compute branch power flows
#%%
# Calculate the power flows in each line
S_12: complex = V_vector[0][0] * np.conj(I_12)  # Power flow in line 1-2
S_13: complex = V_vector[0][0] * np.conj(I_13)  # Power flow in line 1-3
S_14: complex = V_vector[0][0] * np.conj(I_14)  # Power flow in line 1-4
S_23: complex = V_vector[1][0] * np.conj(I_23)  # Power flow in line 2-3
S_24: complex = V_vector[1][0] * np.conj(I_24)  # Power flow in line 2-4

S_30: complex = V_vector[2][0] * np.conj(I_30)  # Power flow in line 3-0
S_40: complex = V_vector[3][0] * np.conj(I_40)  # Power flow in line 4-0

S_21: complex = V_vector[1][0] * np.conj(I_21)  # Power flow in line 2-1
S_31: complex = V_vector[2][0] * np.conj(I_31)  # Power flow in line 3-1
S_41: complex = V_vector[3][0] * np.conj(I_41)  # Power flow in line 4-1
S_32: complex = V_vector[2][0] * np.conj(I_32)  # Power flow in line 3-2
S_42: complex = V_vector[3][0] * np.conj(I_42)  # Power flow in line 4-2

S_vector: dict[str, complex] = {
    "S_12": S_12,
    "S_13": S_13,
    "S_14": S_14,
    "S_23": S_23,
    "S_24": S_24,
    "S_30": S_30,
    "S_40": S_40,
    "S_21": S_21,
    "S_31": S_31,
    "S_41": S_41,
    "S_32": S_32,
    "S_42": S_42
}

# Print out the power flows in each line
S_branch_header = ["Branch", "Rectangular Apparent Power", "Polar Apparent Power"]
S_branch_data = []
for key, value in S_vector.items():
    sign_real = "+" if value.real >= 0 else "-"
    sign_imag = "+" if value.imag >= 0 else "-"
    S_branch_data.append(
        [
            key,
            f"{sign_real:^1} {abs(value.real):>8.5f} {sign_imag:^1} {abs(value.imag):>8.5f}j",
            f"{abs(value):>10.5f} " "MVA " u"\N{ANGLE}" f" {np.angle(value, deg=True):>10.5f}" u"\N{DEGREE SIGN}"
        ]
    )
print(tabulate(S_branch_data, headers=S_branch_header, tablefmt = "pipe", numalign="center", stralign="center"))
#%% md
# # Question 1.3
#%% md
# ## Shunt admittance calculation
# Each bus has 3 connections hence each will have $3 \times 0.1 \text{ p.u.}$ shunt admittance.
#%%
# defining shunt admittance
shunt_admittance = 3 * 0.2j / 2  # with 3 lines per bus and half of 0.2 p.u. per line

# Add shunt admittance to diagonal elements
Y_matrix[np.diag_indices_from(Y_matrix)] += shunt_admittance

Y_matrix_rounded = np.round(Y_matrix, 3)

# Print final matrix
Y_matrix_rounded.imag
#%%
