# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 16:40:10 2025

@author: joejo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

R = 8.314462618  # J/molK

def G_species(T, specie):

    if specie == "H2":
        row = 2 if T > 1000 else 3
    elif specie == "CO":
        row = 12 if T > 1000 else 13
    elif specie == "CH3OH":
        row = 6 if T > 1000 else 7
    else:
        raise ValueError("Unknown species")

    coeff = np.array([
        [0.29328305E+01, 0.8265980E-03, -0.14640057E-06, 0.15409851E-10, -0.68879615E-15, -0.81305582E+3, -0.10243164E+1],  # H2 high
        [0.23443029E+01, 0.79804248E-02, -0.19477917E-04, 0.20156967E-07, -0.73760289E-11, -0.91792413E+03, 0.68300218E+00],  # H2 low
        [0.30484859E+01, 0.13517281E-02, -0.48579405E-06, 0.78853644E-10,-0.46980746E-14, -0.14266117E+05, 0.60170977E+01],  # CO high
        [0.35795335E+01, -0.61035369E-03, 0.10168143E-05,  0.90700586E-09, -0.90442449E-12, -0.14344086E+05, 0.35084093E+01],  # CO low
        [3.52726795E+00, 1.03178783E-02, -3.62892944E-06, 5.77448016E-10, -3.42182632E-14, -2.60028834E+04, 5.16758693E+00], # CH3OH high
        [5.65851051E+00, -1.62983419E-02, 6.91938156E-05, -7.58372926E-08, 2.80427550E-11, -2.56119736E+04, -8.97330508E-01], # CH3OH low
    ])

    # remap rows
    idx = {"H2": (0,1), "CO": (2,3), "CH3OH": (4,5)}[specie]
    a = coeff[idx[0] if T > 1000 else idx[1]]

    G_RT = (
        a[0]*(1 - np.log(T))
        - a[1]*T/2
        - a[2]*T**2/6
        - a[3]*T**3/12
        - a[4]*T**4/20
        + a[5]/T
        - a[6]
    )

    return G_RT * R * T

def deltaG_rxn(T):
    return (
        G_species(T,"CO")
        + 2*G_species(T,"H2")
        - G_species(T,"CH3OH")
    )

def Kp(T):
    return np.exp(-deltaG_rxn(T) / (R*T))


def equilibrium_conversion(K):
    x_test = np.linspace(1e-8, 0.999999, 1000)
    Kmax = np.max(4*x_test**3 / ((1-x_test)*(1+2*x_test)**2))

    if K >= Kmax:
        return 0.999999
    elif K <= 0:
        return 0.0
    else:
        from scipy.optimize import brentq
        f = lambda x: 4*x**3 / ((1-x)*(1+2*x)**2) - K
        return brentq(f, 1e-8, 0.999999)

T = np.linspace(300, 1000, 200)  # K
x = np.array([equilibrium_conversion(Kp(Ti)) for Ti in T])

plt.plot(T, x)
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium conversion of CH$_3$OH")
plt.title("CH$_3$OH → CO + 2H$_2$ at 1 atm")
plt.grid(True)
plt.show()


