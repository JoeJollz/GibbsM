# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:19:03 2025

@author: jrjol
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd


def Abs_G(T, specie):
    
    R = 8.31446 # J/mol.K
    
    if specie == "CH4" and T > 1000:
        row = 0
    
    elif specie == "CH4" and T <= 1000:
        row = 1
        
    elif specie == "H2" and T > 1000:
        row = 2
        
    elif specie == "H2" and T <= 1000:
        row = 3
        
    elif specie == "C" and T > 1000:
        row = 4
    
    elif specie == "C" and T <= 1000:
        row = 5
    
    elif specie == "CH3OHg" and T > 1000:
        row = 6
    
    elif specie == "CH3OHg" and T <= 1000:
        row = 7
    
    elif specie == "O2" and T > 1000:
        row = 8
        
    elif specie == "O2" and T <= 1000:
        row = 9
        
    elif specie == "CH3OHl" and T > 1000:
        row = 10
    
    elif specie == "CH3OHl" and T <= 1000:
        row = 11
        
    coeff = np.array([
                      [1.91178600E+00, 9.60267960E-03, -3.38387841E-06, 5.38797240E-10, -3.19306807E-14, -1.00992136E+04, 8.48241861E+00],  # CH4 at T=[1000,6000]K
                      [5.14825732E+00, -1.37002410E-02, 4.93749414E-05, -4.91952339E-08, 1.70097299E-11, -1.02453222E+04, -4.63322726E+00],  # CH4 at T=[200,1000]K
                      [0.29328305E+01, 0.8265980E-03, -0.14640057E-06, 0.15409851E-10, -0.68879615E-15, -0.81305582E+3, -0.10243164E+1],  # H2, T=[1000,6000]K
                      [0.23443029E+01, 0.79804248E-02, -0.19477917E-04, 0.20156967E-07, -0.73760289E-11, -0.91792413E+03, 0.68300218E+00],  # H2 T=[200,1000]K
                      [ 0.14556924E+01, 0.17170638E-02,-0.69758410E-06, 0.13528316E-09,-0.96764905E-14, -0.69512804E+03, -0.85256842E+01],  # C at T=[1000,6000]K
                      [-0.31087207E+00, 0.44035369E-02, 0.19039412E-05, -0.63854697E-08, 0.29896425E-11, -0.10865079E+03, 0.11138295E+01],  # C at T=[200,1000]K
                      [3.52726795E+00, 1.03178783E-02, -3.62892944E-06, 5.77448016E-10, -3.42182632E-14, -2.60028834E+04, 5.16758693E+00], # CH3OH gas at T = [1000,6000]K
                      [5.65851051E+00, -1.62983419E-02, 6.91938156E-05, -7.58372926E-08, 2.80427550E-11, -2.56119736E+04, -8.97330508E-01], # CH3OH gas at T = [200,1000]K
                      [ 3.66096083E+00, 6.56365523E-04, -1.41149485E-07, 2.05797658E-11, -1.29913248E-15, -1.21597725E+03, 3.41536184E+00], # O2 at T=[1000,6000]K
                      [3.78245636E+00, -2.99673415E-03, 9.84730200E-06, -9.68129508E-09, 3.24372836E-12, -1.06394356E+03, 3.65767573E+00], # O2 at T=[200,1000]K
                      [0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00 ], # CH3OH liquid at T = [1000,6000]K
                      [1.21754995E+01, -4.19673868E-02, 1.42400437E-04,-1.60999972E-07, 2.14794684E-10, -3.15401115E+04, -4.68827360E+01], # CH3OH liquid at T = [200,1000]K
                      [0.30484859E+01, 0.13517281E-02, -0.48579405E-06, 0.78853644E-10,-0.46980746E-14, -0.14266117E+05, 0.60170977E+01], # CO at T =[1000,6000]K
                      [0.35795335E+01, -0.61035369E-03, 0.10168143E-05,  0.90700586E-09, -0.90442449E-12, -0.14344086E+05, 0.35084093E+01], # CO at T =[200, 1000]K
                      [0.46365111E+01, 0.27414569E-02,-0.99589759E-06, 0.16038666E-09, -0.91619857E-14, -0.49024904E+05, -0.19348955E+01], # CO2 T = [1000,6000]K
                      [0.23568130E+01, 0.89841299E-02, -0.71220632E-05, 0.24573008E-08, -0.14288548E-12, -0.48371971E+05, 0.99009035E+01], # CO2 T = [200,1000]K
                      [0.26770389E+01, 0.29731816E-02, -0.77376889E-06, 0.94433514E-10, -0.42689991E-14, -0.29885894E+05, 0.68825500E+01], # H2O at T=[1000,6000]K
                      [0.41986352E+01, -0.20364017E-02, 0.65203416E-05, -0.54879269E-08, 0.17719680E-11, -0.30293726E+05, -0.84900901E+00], # H2O at T=[200,1000]K
                      ])
    
    a = coeff[row]
    ## 9 constant polynomial ##
    # G_T_RT = (-a[0]/(2*T**(2))   (2*a[1]*(1-np.log(T)))/T  +  a[2]*(1-np.log(T)) 
    #           - (a[3]*T))/2 - (a[4]*T**2)/6  - (a[5]*T**3)/12  -  (a[6]*T**4) /20  
    #           + a[7]/T  -a[8])
    ## 7 constant polynoial ##
    G_T_RT =( a[0]*(1-np.log(T)) - (a[1]*T)/2 - (a[2]*T**2)/6  -  (a[3]*T**3)/12
             - (a[4]*T**4)/20 + a[5]/T - a[6])
        
        
    G_T = G_T_RT*R*T # absolute gibbs energy for specie / element of interest.
        
    
    return G_T

def Gabs_molar(T):
    
    return gibbs_form

# def delGf(T,P):
#     coeff = np.array([[8.05532035898967E-14,  -4.54258761039957E-10, 9.74514532917984E-07, -9.81657565169609E-04,  +5.11923575045929E-01, -3.24372890991510E+02], # H2O 1 bar
#                       [-3.41544901192528E-18, 5.94902651736064E-14,  -3.78177322557548E-10, 1.78077091726027E-06, - 3.96194591894066E-03,  - 3.93364345454672E+02],  # CO2 1 bar
#                       [-1.19480733517904E-16, 2.05212901316876E-12, -1.30585665160995E-08, +3.77285316359721E-05, +6.34019284941587E-02,  -7.14155077385076E+01],   # CH4
#                       [                    0,                    0,                     0,                     0,                      0,                      0 ],  # H2
#                       [2.26110992549836E-18, -2.49784774284562E-14, -2.96507088667142E-11, +2.59877014328724E-06, -9.30966445312590E-02,  -1.09676600595313E+02],    # CO
#                       [                    0,                    0,                     0,                     0,                      0,                      0 ], # C
#                       ])
    
#     T_poly = np.array([T**5, T**4, T**3, T**2, T**1, 1]) 
    
#     coeff_T = np.matmul(T_poly, np.transpose(coeff)) * 1000 # J/mol
    
#     #methanol gibbs form#
#     if T >338: #ch3oh is vapor
#         gf_ch3oh = Abs_G(T, "CH3OHg") - (Abs_G(T, "C") + 2*Abs_G(T, "H2") + 0.5 * Abs_G(T, "O2"))
#     else:
#         gf_ch3oh = Abs_G(T, "CH3OHl") - (Abs_G(T, "C") + 2*Abs_G(T, "H2") + 0.5 * Abs_G(T, "O2"))
#     coeff_T = np.append(coeff_T, gf_ch3oh)
    
#     return coeff_T

store_new_gf = []
store_old_gf =[]
store_ch3oh = []
error = []
T_s = []

for T in range(298,3000):
    gf_ch4_new = Abs_G(T, "CH4") - (Abs_G(T, "C") + 2*Abs_G(T, "H2"))
    gf_ch3oh_new = Abs_G(T, "CH3OHg") - (Abs_G(T, "C") + 2*Abs_G(T, "H2") + 0.5*Abs_G(T, "O2"))
   # gf_ch4_old = delGf(T,1)[2]
   # gf_ch4_old_list = delGf(T,1)
  #  e = gf_ch4_new - gf_ch4_old
  #  error.append(e)
    store_new_gf.append(gf_ch4_new)
  #  store_old_gf.append(gf_ch4_old)
    store_ch3oh.append(gf_ch3oh_new)
    T_s.append(T)
    
plt.plot(T_s,error)
plt.title('Gibbs_f error for CH4. NIST JANAF VS ARGONNE')
plt.xlabel('Temperature (K)')
plt.ylabel('Error (J/mol)')
plt.show()

plt.plot(T_s, store_new_gf , label='New model')
plt.plot(T_s, store_old_gf , label ='Old model')
plt.legend()
plt.show()

plt.plot(T_s, store_ch3oh)
plt.title('Gibbs E of Form CH3OH')
plt.xlabel('Temperature (K)')
plt.ylabel('Gibbs form (J/mol)')
plt.show()

def Gibbs(n, T, p):
    # function currently takes the fugacity coefficient as 1
    
   # gibbs_form = delGf(T,p)
   gibbs_form = Gabs_molar(T)
    
    
    gas_indicies = list(range(len(n)))
    
    gas_indicies.remove(5)
    if T<338:
        gas_indicies.remove(6)
    n_gas = n[gas_indicies]
    
    # n_gas = np.delete(n, 5) # index for removing solid carbon.
    # if T<338: # CH3OH is a liquid, hence remove.
    #     n_gas = np.delete(n,6)

    for i in range(n.shape[0]): # protect from negative log
        if n[i] <= 0:
            n[i] = 1e-20
            
    
    R  = 8.314 # universal gas constant in J / mol K
    p0 = 3 # standard pressure in bar
    
    total_gibbs = np.dot(n, gibbs_form) + R*T*np.sum(n_gas*np.log((p*n_gas)/(p0*np.sum(n_gas))))  # changed so no longer calculating gas phase entropy for non gas terms.
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
                  [1, 4, 1],  #CH3OH
                  ])
    
    resid = np.dot(np.transpose(A), n ) - e0
    
    return resid

def retry(y, res):
    
    retry_c = 0
    max_retries = 5
    
    while not res.success and retry_c < max_retries:
      #  if retry_c ==0:        
            #n0_perturbed = y
            res = minimize(Gibbs, y, args = (Ts[j], ps[i]), method = 'SLSQP', bounds = bnds, constraints = cons)
            print(f'Retry {retry_c+1}: ', res.success)
            retry_c +=1
            y = y * (1 + 1e-2 * np.random.randn(len(y)))

    
    return res
    
    

e0 = np.array([1, 5.92, 2.46]) # C, H, O
N_MeOH = 1 # moles of MeOH in
y = n0 = np.ones(7)
ps = np.array([1.01325])
Ts = np.linspace(473.15, 1173.15, 200)

cons = {'type': 'eq', 'fun': element_balance, 'args':[e0]}
bnds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0,np.inf), (0,np.inf), (0,np.inf)) # number of bounds needs to match the number of species. e.g. 2 species, 2 bounds. 

result_y = np.ones((ps.shape[0], Ts.shape[0], n0.shape[0])) # what is n0?? and what dimensions is this giving.
h2 = []
co2 = []
ch4 = []
co = []
c = []
h2o  = []
ch3oh = []
s_o_f = []
X_MeOH = []
f = 0
s = 0
for i in range(ps.shape[0]):
    for j in range(Ts.shape[0]):
        # if np.sum(x0) !=1:
        #     print('please check initial guess for the mole fraction compoistion')
        
        # else:
        res_o = res = minimize(Gibbs, n0, args = (Ts[j], ps[i]), method = 'SLSQP', bounds = bnds, constraints = cons, options={'ftol':1e-12, 'maxiter':500})
        print('--------------------------------------------------------------')
        print('Resdiual element balance Pre: ', element_balance(y, e0))
        if not res.success:
            res = retry(y, res)
        
        if np.linalg.norm(element_balance(res_o.x, e0), ord =1) <= np.linalg.norm(element_balance(res.x, e0),1): # check which equalit constraint is greater breached. 
            y = res_o.x
            if not res_o.success:
                f +=1
            else: 
                s+=1
            s_o_f.append(res_o.success)
            
        else:
            y = res.x
            if not res.success:
                f +=1
            else: 
                s+=1
            s_o_f.append(res.success)
        
        result_y[i, j] = y/ np.sum(y)
        h2.append(y[3]/np.sum(y))
        h2o.append(y[0]/np.sum(y))
        co2.append(y[1]/np.sum(y))
        ch4.append(y[2]/np.sum(y))
        co.append(y[4]/np.sum(y))
        c.append(y[5]/np.sum(y))
        ch3oh.append(y[6]/np.sum(y))
        X_MeOH.append( ((y[1] + y[2] + y[4] + y[5]) / N_MeOH )*100)
        
        s_o_f.append(res.success)
        
        
        
            
        print(result_y[i, j], '@: ', ps[i], 'Bar ,', Ts[j] , 'K')
        print('Success or Fail: ', res.success)
        print('Resdiual element balance Post: ', element_balance(y, e0))
        print('--------------------------------------------------------------')
            

#test = delGf(500, 1)

P_1bar = -192.593

P_10bar = -173.468

Calc_P_10bar = P_1bar + 0.008314 * 1000 * np.log(10 / 1)
Tc = Ts -273.15
plt.plot(Tc, h2, label = 'H$_2$')
plt.plot(Tc, h2o, label = 'H$_2$O')
plt.plot(Tc, co2, label = 'CO$_2$')
plt.plot(Tc, co, label = 'CO')
plt.plot(Tc, ch4, label = 'CH$_4$')
plt.plot(Tc, c, label = 'C')
plt.plot(Tc, ch3oh, label = 'CH$_3$OH')
plt.ylabel('Mole Fraction')
plt.xlabel('Temperature (°C)')
plt.title('4CO$_2$:1H$_2$')
plt.legend()
plt.show()

print('Percentage failed: ', f/(s+f)*100, '%')

plt.plot(Tc, h2, label='H$_2$')
plt.plot(Tc, h2o, label='H$_2$O')
plt.plot(Tc, co2, label='CO$_2$')
plt.plot(Tc, co, label='CO')
plt.plot(Tc, ch4, label='CH$_4$')
plt.plot(Tc, c, label='C')
plt.plot(Tc, ch3oh, label='CH$_3$OH')

plt.ylabel('Mole Fraction')
plt.xlabel('Temperature (°C)')
plt.title('4CO$_2$:1H$_2$')

# === ADD THIS SECTION ===
for T, success in zip(Tc, s_o_f):
    if not success:        # if solver failed
        plt.axvline(T, color='red', alpha=0.3, linewidth=1)

# OPTIONAL: add legend entry for failure markers
plt.axvline(Tc[0], color='red', alpha=0.3, linewidth=1, label='Failed convergence')

plt.legend()
plt.show()

