# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:19:03 2025

@author: jrjol

Seems to be converging well at all points.

TODO:
    - Check the formula for the ratio of success v fail convergences. CHECKED and satisfactory.
    - Methanol product mole fraction is still negligable. consider gibbs E of 
    carbon. Issue could be due to consideration of C, yet CH4 is still moderate.
    
Temporarily removing carbon from the product species. 
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
    
    elif specie == "CO" and T > 1000:
        row = 12
    
    elif specie == "CO" and T <= 1000:
        row = 13
        
    elif specie == "CO2" and T > 1000:
        row = 14
    
    elif specie == "CO2" and T <= 1000:
        row = 15
    
    elif specie == "H2O" and T > 1000:
        row = 16
    
    elif specie == "H2O" and T <= 1000:
        row = 17
    
        
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
    
    # species_of_interest = ["H2O", "CO2", "CH4", "H2", "CO", "C", "CH3OHg"] 
    species_of_interest = ["H2O", "CO2", "CH4", "H2", "CO", "CH3OHg"] # removing C
    gibbs_form = np.zeros(len(species_of_interest))
    
    for idx, name in enumerate(species_of_interest):
        gibbs_form[idx] = Abs_G(T, name)
    
    return gibbs_form

def Gibbs(n, T, p):
    # function currently takes the fugacity coefficient as 1
    
   # gibbs_form = delGf(T,p)
    gibbs_form = Gabs_molar(T)
    
    
    gas_indicies = list(range(len(n)))
    
    #gas_indicies.remove(5) # removing C
    if T<338:
        gas_indicies.remove(5)
    n_gas = n[gas_indicies]
    

    for i in range(n.shape[0]): # protect from negative log
        if n[i] <= 0:
            n[i] = 1e-20
            
    
    R  = 8.314 # universal gas constant in J / mol K
    p0 = 1 # standard pressure in bar
    
    total_gibbs = np.dot(n, gibbs_form) + R*T*np.sum(n*np.log((p*n)/(p0*np.sum(n_gas))))  # changed so no longer calculating gas phase entropy for non gas terms.
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
                 # [1, 0, 0],  #C
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
#y = n0 = np.array([1, 1, 1, 1, 1, 1, 5])
y = n0 = np.array([1, 1, 1, 1, 1, 5])
ps = np.array([1.01325])
Ts = np.linspace(473.15, 1173.15, 200)

cons = {'type': 'eq', 'fun': element_balance, 'args':[e0]}
bnds = ((1e-10, np.inf), (1e-10, np.inf), (1e-10, np.inf), (1e-10, np.inf), (1e-10,np.inf), (1e-10,np.inf)) # number of bounds needs to match the number of species. e.g. 2 species, 2 bounds. 

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
        
        ### Has the retried attempts achieved 'convergence'? checker.
        if np.linalg.norm(element_balance(res_o.x, e0), ord =1) <= np.linalg.norm(element_balance(res.x, e0),1): # check which equality constraint is greater breached. 
            y = res_o.x # original result before retry is satisfactory. 
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
       # c.append(y[5]/np.sum(y))
        ch3oh.append(y[5]/np.sum(y))
      #  X_MeOH.append( ((y[1] + y[2] + y[4] + y[5]) / N_MeOH )*100)
        
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
#plt.plot(Tc, c, label = 'C')
plt.plot(Tc, ch3oh, label = 'CH$_3$OH')
plt.ylabel('Mole Fraction')
plt.xlabel('Temperature (°C)')
plt.title('1CH$_3$OH:1.5H$_2$O')
plt.legend()
plt.show()

print('Percentage failed: ', f/(s+f)*100, '%')

plt.plot(Tc, h2, label='H$_2$')
plt.plot(Tc, h2o, label='H$_2$O')
plt.plot(Tc, co2, label='CO$_2$')
plt.plot(Tc, co, label='CO')
plt.plot(Tc, ch4, label='CH$_4$')
#plt.plot(Tc, c, label='C')
plt.plot(Tc, ch3oh, label='CH$_3$OH')

plt.ylabel('Mole Fraction')
plt.xlabel('Temperature (°C)')
plt.title('1CH$_3$OH:1.5H$_2$O')

# === ADD THIS SECTION ===
for T, success in zip(Tc, s_o_f):
    if not success:        # if solver failed
        plt.axvline(T, color='red', alpha=0.3, linewidth=1)

# OPTIONAL: add legend entry for failure markers
plt.axvline(Tc[0], color='red', alpha=0.3, linewidth=1, label='Failed convergence')

plt.legend()
plt.show()


ch4_gibbs = []
ch3oh_gibbs = []
co2_gibbs = []
h2o_gibbs = []
for T in Ts:
    gibbs_form = Gabs_molar(T)
    
    ch4_gibbs.append(gibbs_form[2])
    ch3oh_gibbs.append(gibbs_form[5])
    h2o_gibbs.append(gibbs_form[0])
    co2_gibbs.append(gibbs_form[1])
    
plt.plot(Ts, ch4_gibbs, label='CH4 Gibbs')
plt.plot(Ts, ch3oh_gibbs, label = 'CH3OH Gibbs')
plt.plot(Ts, h2o_gibbs, label = 'H2O Gibbs')
plt.plot(Ts, co2_gibbs, label = 'CO2 Gibbs')
plt.legend()
plt.show()
    

