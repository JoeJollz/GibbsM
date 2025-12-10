# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:30:05 2025

@author: jrjol
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# x = np.linspace(-10, 10, 40)
# y = np.linspace(-10, 10, 40)
# X, Y = np.meshgrid(x, y)
# Z = fun(X, Y)

# fig = plt.figure(figsize=(10, 8))
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)

# ax.set_title('3D Contour Plot')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# plt.show()

def delGf(T,P):
    coeff = np.array([[8.05532035898967E-14,  -4.54258761039957E-10, 9.74514532917984E-07, -9.81657565169609E-04,  +5.11923575045929E-01, -3.24372890991510E+02], # H2O 1 bar
                      [-3.41544901192528E-18, 5.94902651736064E-14,  -3.78177322557548E-10, 1.78077091726027E-06, - 3.96194591894066E-03,  - 3.93364345454672E+02],  # CO2 1 bar
                      [-1.19480733517904E-16, 2.05212901316876E-12, -1.30585665160995E-08, +3.77285316359721E-05, +6.34019284941587E-02,  -7.14155077385076E+01],   # CH4
                      [                    0,                    0,                     0,                     0,                      0,                      0 ],  # H2
                      [2.26110992549836E-18, -2.49784774284562E-14, -2.96507088667142E-11, +2.59877014328724E-06, -9.30966445312590E-02,  -1.09676600595313E+02],    # CO
                      [                    0,                    0,                     0,                     0,                      0,                      0 ], # C
                      ])
    
    T_poly = np.array([T**5, T**4, T**3, T**2, T**1, 1]) 
    
    coeff_T = np.matmul(T_poly, np.transpose(coeff)) * 1000 # J/mol
    
    return coeff_T

def Gibbs(n, T, p):
    # function currently takes the fugacity coefficient as 1
    
    gibbs_form = delGf(T,p)
    
    n_gas = np.delete(n, 5) # index for removing solid carbon. 
    
    for i in range(n.shape[0]): # protect from negative log
        if n[i] <= 0:
            n[i] = 1e-20
            
    
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
                  ])
    
    resid = np.dot(np.transpose(A), n ) - e0
    
    return resid




def G_solver(e0, T):
    
    n0 = np.ones(6)
    ps = np.array([1.01325])
    Ts = T

    cons = {'type': 'eq', 'fun': element_balance, 'args':[e0]}
    bnds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0,np.inf), (0,np.inf)) # number of bounds needs to match the number of species. e.g. 2 species, 2 bounds. 

    result_y = np.ones((ps.shape[0], Ts.shape[0], n0.shape[0])) # what is n0?? and what dimensions is this giving.
    h2 = []
    co2 = []
    ch4 = []
    co = []
    c = []
    h2o  = []
    s_o_f = []
    for i in range(ps.shape[0]):
        for j in range(Ts.shape[0]):
            # if np.sum(x0) !=1:
            #     print('please check initial guess for the mole fraction compoistion')
            
            # else:
            res = minimize(Gibbs, n0, args = (Ts[j], ps[i]), method = 'SLSQP', bounds = bnds, constraints = cons)
            y = res.x
            result_y[i, j] = y/ np.sum(y)
            h2.append(y[3]/np.sum(y))
            h2o.append(y[0]/np.sum(y))
            co2.append(y[1]/np.sum(y))
            ch4.append(y[2]/np.sum(y))
            co.append(y[4]/np.sum(y))
            c.append(y[5]/np.sum(y))
            
            s_o_f.append(res.success)
                
            print(result_y[i, j], '@: ', ps[i], 'Bar ,', Ts[j] , 'K')
            print('Success or Fail: ', res.success)
    
    return h2, h2o, co2, ch4, co, c
    
    


ratios = np.linspace(0.5, 5, 30)
temp_range = np.linspace(473.15, 1173.15, 300)
# Z_h2 = np.zeros((len(ratios), len(temp_range)))
# Z_h2o = np.zeros((len(ratios), len(temp_range)))
# Z_co2 = np.zeros((len(ratios), len(temp_range)))
# Z_ch4 = np.zeros((len(ratios), len(temp_range)))
# Z_co = np.zeros((len(ratios), len(temp_range)))
# Z_c = np.zeros((len(ratios), len(temp_range)))
Z_h2 = np.zeros((len(temp_range), len(ratios)))
Z_h2o = np.zeros((len(temp_range), len(ratios)))
Z_co2 = np.zeros((len(temp_range), len(ratios)))
Z_ch4 = np.zeros((len(temp_range), len(ratios)))
Z_co = np.zeros((len(temp_range), len(ratios)))
Z_c = np.zeros((len(temp_range), len(ratios)))
x = ratios
y = temp_range
X, Y = np.meshgrid(x, y)

for i in range(0, len(ratios)):
    r = ratios[i]
    mole_meth = 1 # always 1 mole of methanol, because using this, and the h2o/methanol ratio, we can then calc moles of h2o
    mole_h2o = r/mole_meth # ratio inputted from ratio mesh arry
    Oin = mole_h2o*1+mole_meth*1
    Cin = mole_h2o*0 + mole_meth*1
    Hin = mole_h2o*2+mole_meth*4
    
    
    e0 = np.array([Cin, Hin, Oin])
    
    h2, h2o, co2, ch4, co, c = G_solver(e0, temp_range)
    Z_h2[:, i] = h2
    Z_h2o[:, i] = h2o
    Z_co2[:, i] = co2
    Z_ch4[:, i] = ch4
    Z_co[:, i] = co
    Z_c[:, i] = c
    
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_h2, cmap='cool', alpha=0.8, edgecolor='none')

#ax.set_title('H$_2$ Mole Fraction', pad = 1)
# ax.text(
#     10, 600, 1.05,               # x, y, z in axis coordinates (approx)
#     'CO Mole Fraction',
#     transform=ax.transAxes,
#     ha='center',
#     fontsize=12
# )

ax.set_xlabel('Ratio (H$_2$O/CH$_3$OH)', fontsize =13)
ax.set_ylabel('Temperature (K)', fontsize =13, labelpad = 11.7)
ax.set_zlabel('Product Mole Fraction', fontsize =13, labelpad =11.1)
ax.set_zlim(0, 0.6)
ax.set_yticks([600, 800,1000,1200])
ax.set_zticks([0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='z', labelsize=13)
ax.dist = 11.5
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_h2o, cmap='cool', alpha=0.8, edgecolor='none')

#ax.set_title('H$_2$O Mole Fraction', pad = 1)
ax.set_xlabel('Ratio (H$_2$O/CH$_3$OH)', fontsize =13)
ax.set_ylabel('Temperature (K)', fontsize =13, labelpad = 11.7)
ax.set_zlabel('Product Mole Fraction', fontsize =13, labelpad = 11.1)
ax.set_zlim(0, 0.6)
ax.set_yticks([600, 800,1000,1200])
ax.set_zticks([0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='z', labelsize=13)
ax.dist = 11.5
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_co2, cmap='cool', alpha=0.8, edgecolor='none')

#ax.set_title('CO$_2$ Mole Fraction', pad = 1)
ax.set_xlabel('Ratio (H$_2$O/CH$_3$OH)', fontsize = 13)
ax.set_ylabel('Temperature (K)', fontsize = 13, labelpad=11.7)
ax.set_zlabel('Product Mole Fraction', fontsize =13, labelpad = 11.1)
ax.set_zlim(0, 0.6)
ax.set_yticks([600, 800,1000,1200])
ax.set_zticks([0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='z', labelsize=13)
ax.dist = 11.5
plt.show()
    

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_co, cmap='cool', alpha=0.8, edgecolor='none')

#ax.set_title('CO Mole Fraction', pad = 1)
ax.set_xlabel('Ratio (H$_2$O/CH$_3$OH)', fontsize =13)
ax.set_ylabel('Temperature (K)', fontsize =13, labelpad = 11.7)
ax.set_zlabel('Product Mole Fraction', fontsize =13, labelpad=11.1)
ax.set_zlim(0, 0.6)
ax.set_yticks([600, 800,1000,1200])
ax.set_zticks([0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='z', labelsize=13)
ax.dist = 11.5
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_ch4, cmap='cool', alpha=0.8, edgecolor='none')

#ax.set_title('CH$_4$ Mole Fraction', pad = 1)
ax.set_xlabel('Ratio (H$_2$O/CH$_3$OH)', fontsize =13)
ax.set_ylabel('Temperature (K)', fontsize =13, labelpad=11.7)
ax.set_zlabel('Product Mole Fraction', fontsize =13, labelpad = 11.1)
ax.set_zlim(0, 0.6)
ax.set_yticks([600, 800,1000,1200])
ax.set_zticks([0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='z', labelsize=13)
ax.dist = 11.5
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_c, cmap='cool', alpha=0.8, edgecolor='none')

#ax.set_title('C Mole Fraction', pad = 15)
ax.set_xlabel('Ratio (H$_2$O/CH$_3$OH)', fontsize =13)
ax.set_ylabel('Temperature (K)', fontsize =13, labelpad = 11.7)
ax.set_zlabel('Product Mole Fraction', fontsize = 13, labelpad = 11.1)
ax.set_zlim(0, 0.6)
ax.set_yticks([600, 800,1000,1200])
ax.set_zticks([0.0, 0.2, 0.4, 0.6])
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='z', labelsize=13)
ax.dist = 11.5
plt.show()
    