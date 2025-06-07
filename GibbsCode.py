'''
Function for calculating fugacity coefficients needs to be implemented, and then called
within the def Gibbs() function, just prior to the total_gibbs calculation. Additionally, 
provide option for user to select 'ideal gas' hence fug coeffs = 1, or 'real gas' to allow
for fug coeffs to be calculated based on data and EoSs. 
'''


import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd

num_species_in = int(input("How many input species are present? (e.g., 2 for ethanol and water): "))

moles_of_species_in = {}
C = 0
H = 0
O = 0
total_moles = 0
accessing_C = 0
for _ in range(num_species_in):
    add_c = 0
    add_h = 0
    add_o = 0
    if _ == 0:
        print('Please input using the following format: ')
        print('e.g. Ethanol -> c2h6o1')
        print('e.g. Methane -> C1H4')
        print('e.g. Hydrogen -> H2o1')
    
    species_name = str(input(f"Input the chemical formula for species {_+1}: " ))
    moles = float(input(f"How many moles of species {_+1} is present? " ))
    moles_of_species_in[species_name] = moles
    
    
    for Z in range(len(species_name)):

        if species_name[Z].isdigit():
            continue
        
        elif species_name[Z] == 'c' or species_name[Z] == 'C':
            print('in c')
            if Z!= len(species_name)-1:
                add_c = float(species_name[Z+1])*moles
            else:
                add_c = moles
        
        elif species_name[Z] == 'h' or species_name[Z] == 'H':
            print('in h')
            if Z!= len(species_name)-1:
                add_h = float(species_name[Z+1])*moles
            else:
                add_h = moles
            
        elif species_name[Z] == 'o' or species_name[Z] == 'O':
            print(' in o')
            if Z!= len(species_name)-1:
                add_o = float(species_name[Z+1])*moles
            else:
                add_o = moles
        
        else:
            print("Chemical formula has invalid input. Exciting programme.")
            exit()
            
    C += add_c
    H += add_h
    O += add_o
print('Oxygen: ', O)
print('Carbon: ', C)
print('Hydrogen: ', H)
    
    
    

def delGf(T,P):
    coeff = np.array([[8.05532035898967E-14,  -4.54258761039957E-10, 9.74514532917984E-07, -9.81657565169609E-04,  +5.11923575045929E-01, -3.24372890991510E+02], # H2O 1 bar
                      [-3.41544901192528E-18, 5.94902651736064E-14,  -3.78177322557548E-10, 1.78077091726027E-06, - 3.96194591894066E-03,  - 3.93364345454672E+02],  # CO2 1 bar
                      [-1.19480733517904E-16, 2.05212901316876E-12, -1.30585665160995E-08, +3.77285316359721E-05, +6.34019284941587E-02,  -7.14155077385076E+01],   # CH4
                      [                    0,                    0,                     0,                     0,                      0,                      0 ],  # H2
                      [2.26110992549836E-18, -2.49784774284562E-14, -2.96507088667142E-11, +2.59877014328724E-06, -9.30966445312590E-02,  -1.09676600595313E+02],    # CO
                      [                    0,                    0,                     0,                     0,                      0,                      0 ], # C
                      [-1.11515065129641E-17, +2.11447930436933E-13, -1.52766168149359E-09, +5.78567116482972E-06,  - 6.19869714803181E-02, +2.27175463481463E+02], # C2H2
                      ])
    
    T_poly = np.array([T**5, T**4, T**3, T**2, T**1, 1]) 
    
    coeff_T = np.matmul(T_poly, np.transpose(coeff)) * 1000 # J/mol
    
    return coeff_T


def phi_vect(n, T, P):
    # Using cubic equation of state - 
    
    
    
    return phis

def Gibbs(n, T, p):
    # function currently takes the fugacity coefficient as 1
    
    gibbs_form = delGf(T,p)
    
    n_gas = np.delete(n, 5) # index for removing solid carbon. 
    
    for i in range(n.shape[0]): # protect from negative log
        if n[i] <= 0:
            n[i] = 1e-49
            
    
    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1 # standard pressure in bar
    
    total_gibbs = np.dot(n, gibbs_form) + R*T*np.sum(n*np.log((p*n)/(p0*np.sum(n_gas))))
    return total_gibbs

# n0 is the element input based on moles of species and their corresponding elemental stoichometry
def element_balance(n, e0):
    
    # Species-element stoichometry 
    #             #C #H #O
    A = np.array([[0, 2, 1],  #H2O
                  [1, 0, 2],  #CO2
                  [1, 4, 0],  #CH4
                  [0, 2, 0],  #H2 
                  [1, 0, 1],  #CO 
                  [1, 0, 0],  #C
                  [2, 2, 0],  #C2H2
                  ])
    
    resid = np.dot(np.transpose(A), n ) - e0
    
    return resid

e0 = np.array([2, 16, 6]) # C, H, O
n0 = np.ones(7)
n0[-1] = 0.5
#n0[-1] = 1e-10
ps = np.array([1.01325])
Ts = np.linspace(500, 1400, 200)

cons = {'type': 'eq', 'fun': element_balance, 'args':[e0]}
bnds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0,np.inf), (0,np.inf), (0,np.inf)) # number of bounds needs to match the number of species. e.g. 2 species, 2 bounds. 
bnds = [(0, np.inf)]*7
result_y = np.ones((ps.shape[0], Ts.shape[0], n0.shape[0])) # what is n0?? and what dimensions is this giving.
h2 = []
co2 = []
ch4 = []
co = []
c = []
h2o  = []
c2h2 = []
s_o_f = []
for i in range(ps.shape[0]):
    for j in range(Ts.shape[0]):
        # if np.sum(x0) !=1:
        #     print('please check initial guess for the mole fraction compoistion')
        
        # else:
        res = minimize(Gibbs, n0, args = (Ts[j], ps[i]), method = 'SLSQP', bounds = bnds, constraints = cons,options = {'maxiter': 500} )
        y = res.x
        result_y[i, j] = y/ np.sum(y)
        h2.append(y[3]/np.sum(y))
        h2o.append(y[0]/np.sum(y))
        co2.append(y[1]/np.sum(y))
        ch4.append(y[2]/np.sum(y))
        co.append(y[4]/np.sum(y))
        c.append(y[5]/np.sum(y))
        c2h2.append(y[6]/np.sum(y))
        
        s_o_f.append(res.success)
            
       # print(result_y[i, j], '@: ', ps[i], 'Bar ,', Ts[j] , 'K')
       # print('Success or Fail: ', res.success)
            

test = delGf(500, 1)

P_1bar = -192.593

P_10bar = -173.468

Calc_P_10bar = P_1bar + 0.008314 * 1000 * np.log(10 / 1)

plt.plot(Ts, h2, label = 'H2')
plt.plot(Ts, h2o, label = 'H2O')
plt.plot(Ts, co2, label = 'CO2')
plt.plot(Ts, co, label = 'CO')
plt.plot(Ts, ch4, label = 'CH4')
plt.plot(Ts, c, label = 'C')
plt.plot(Ts, c2h2, label = 'C2H2')
plt.ylabel('Mole Fraction')
plt.xlabel('Temperature (K)')
plt.legend()
plt.show()

