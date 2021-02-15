# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:45:44 2019

@author: ez23
"""

import os
import numpy as np
# data and variable for initializing the methane project


#outdir_data = 'C:\\Users\\ez23\\Documents\\methane_project\\output_data'
#outdir_plot = 'C:\\Users\\ez23\\Documents\\methane_project\\output_plot'
#datapath = os.path.join('C:\\','Users','ez23','Documents',
#                        'methane_project','data_methane')

# better use relative path only:
outdir_data  = os.path.join('..', 'output_data')
outdir_plot  = os.path.join('..', 'output_plot')
datapath     = os.path.join('..', 'methane_data')
outfolder_df = os.path.join(datapath, 'Siikaneva_2013_cleaned')

# Global variables::
#########################
kv    = 0.4 # Von Karman constant
g     = 9.806 # [m/s^2] # grav acc
nu = 0.156*10**(-4) # kinematic viscosity
rho = 1.225 # ave air density: Kg/m**3
Cp = 1005 # Air Heat capacity J/Kg/K
###############################################################################

# site-specific constants for Siikaneva1
z_ref = 2.8 # [m] - Siikaneva1
fs = 10    # [Hz] - Siikaneva1
dt = 1/fs  # [s] - Siikaneva1


# limits for lag corrction to gas conc time series
laglim = {'CH4':[1, 4], # for lag correction
          'CO2':[0, 3],
          'H2O':[0, 10]}

# molecular diffusivities at 0 celsius
# DM = {'H2O':0.2178*10**(-4),
#       'CO2':0.1381*10**(-4),
#       'CH4':0.1952*10**(-4),
#       'T':0.1862*10**(-4),
#       'u':0.1328*10**(-4)}







def Dm(T, scalar = 'u'):
# compute diffusivities of different species in air
# data obrained from Brutsaert or Massman, 1998
# INPUT:
# one of the following scalars: 'u', 'T', 'H2O', 'CH4', 'CO2'
# temperature in Kelvin
# all at standard pressure 1 atmosphere
    isscalar = np.isscalar(T)
    if isscalar:
        T = np.array([T])
    mt = np.size(T)
    DIFF = np.zeros(mt)
    for i in range(mt):
        Ti = T[i]
        # values from Brutsaert Evaporation into the atmosphere:
        TempsC = np.array([-20, -10, 0, 10, 20, 30, 40])
        Temps = TempsC + 273.15
        Dt = np.array([1.628, 1.747, 1.865, 1.994, 2.122, 2.250, 2.388]) * 10 ** (-5)  # m^2 s^-1
        nu = np.array([1.158, 1.243, 1.328, 1.418, 1.509, 1.602, 1.700]) * 10 ** (-5)  # m^2 s^-1
        Dw = np.array([1.944, 2.082, 2.230, 2.378, 2.536, 2.694, 2.852]) * 10 ** (-5)  # m^2 s^-1
        interp = False
        T0 = 273.15
        if scalar == 'T':
            valz = Dt
            interp = True
        elif scalar == 'u':
            valz = nu
            interp = True
        elif scalar == 'H2O':
            valz = Dw
            interp = True
        elif scalar == 'H2O_':
            # diff = 0.2090*(Ti/T0)**(2.072) * 10**(-4)
            diff = 0.2178*(Ti/T0)**(1.81) * 10**(-4)
        elif scalar == 'CO2':
            # diff = 0.2020*np.exp(-0.3738/(Ti/T0))*(Ti/T0)**(1.590)* 10**(-4)
            diff = 0.1381*(Ti/T0)**(1.81) * 10**(-4)
        elif scalar == 'CH4':
            # diff = 0.1859*(Ti/T0)**(1.747)* 10**(-4)
            diff = 0.1952 * (Ti / T0) ** (1.81) * 10 ** (-4)
        else:
            print('Dm error: insert a valid scalar')
        if interp: # interpolation
            if Ti < Temps[0]:
                print('warning: max T exceeded to compute molecular diffusivities')
                print('using the value for -20 celsius instead')
                diff = valz[0]
            elif Ti > Temps[-1]:
                print('warning: max T exceeded to compute molecular diffusivities')
                print('using the value for +40 celsius instead')
                diff = valz[-1]
            else:
                difft = np.argmax(Temps > Ti)
                lowerT = Temps[difft-1]
                # print(lowerT)
                upperT = Temps[difft]
                # print(upperT)
                dx1 = np.abs(Ti - lowerT)
                dx2 = np.abs(upperT - Ti)
                w1 = dx1/(dx1+dx2)
                # print(w1)
                w2 = dx2/(dx1+dx2)
                # print(w1)
                diff = w2*valz[difft-1] + w1*valz[difft]
                # print(valz[difft-1])
                # print(diff)
                # print(valz[difft])
        DIFF[i] = diff
    res = DIFF[0] if isscalar else DIFF
    return res



scalars = ['H2O', 'CO2', 'CH4', 'T', 'u']
myT = 20 + 273.15
dms = [Dm(myT, scalar=scal) for scal in scalars]
DM = {scal:val for scal, val in zip(scalars, dms)}


SC = {key: DM['u'] / val for key, val in DM.items()}

# test temperature function
#
# Ti = np.linspace(0, 30, 100) + 273.15
#
# # Ti = 22 + 273.15
#
# Dwi = Dm(scalar='H2O', T=Ti)
# Dwi2 = Dm(scalar='H2O_', T=Ti)
# Dme = Dm(scalar='CH4', T=Ti)
# Dco2 = Dm(scalar='CO2', T=Ti)
# Du = Dm(scalar='u', T=Ti)
# DT = Dm(scalar='T', T=Ti)
#
# plt.figure()
# plt.plot(Ti, Dwi)
# # plt.plot(Ti, Dwi)
# plt.plot(Ti, Dme)
# plt.plot(Ti, Dco2)
# plt.plot(Ti, Du)
# plt.plot(Ti, DT)
# plt.show()


# variables to save for each turbulent quantity
#turb_vars = [ 'date', 'exists', 'windir', 'is_stationary'] + [
#        
#        'Ubar', 'Tbar', 'ustar', 'H', 'Cov_wT', 'Lmo',                  # from turb_quant
#          'stab',   'Tu',   'Tw',    'Tt', 'phim', 'phis', 'T_stdv', 
#          'U_stdv', 'W_stdv', 'T_star', 'T_skew', 'Re_star', 'n_obs', 
#          'Iu', 'z0corr', 'epsilon', 'tau_eta', 'eta', 'z0', 'stab0', 
#          'myphim0', 'epsilon0', 'tau_eta0', 'eta0', 'Re0', 'h0', 't0'
#        ]

## variables to save for each scalar quantity
#scal_vars = ['mu', 'stdv', 'flux_ec', 'cstar', 'Tc', 'Lc',                   # scalar_quant
#             'Rcw', 'M03', 'M30', 'M21', 'M12', 'ICEM', 'CEM', 'd_Time_CEM', # mixed_moments
#             'dSo', 'dQ_Time',                                               # Delta_So
#             'eT', 'eT_GAUSS',                                               # Transport_Eff
#             'Beta_p','Slope_REA','Slope_REA_up', 'Slope_REA_dn',            # REA_Beta
#             'Beta_REA','Beta_Milne', 'M04', 'M13',                          # REA_Beta_Milne
#              
#             ]
