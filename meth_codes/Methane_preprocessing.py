'''Main code with the analysis for the Methane project'''


import os
import time
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import methfun as mf # module for this project
import methdata as md
import spectral_analysis as sa
from scipy.stats import ks_2samp
# import stats as st


# TODOs ###
# - check writing / reading of dataframe with multi index
# - check stationarity measure vs Matlab code
# - check partition and intermittence surface renewals
# - check IWATA partitioning & wavelet coeffients
###



####################### move all this is methdata at some point ###########
# 1. Read turbulence data
fs = 10 # sampling rate [Hz]
dt = 1/fs # delta time between two observations
z_ref = 2.8 #[m]
# refscal = 'H2O' # for ebullition detection

# do not include in the analysis runs with less than minsamplesize observations
minsamplesize = 10000 # keep only complete records
nobs_min = 2**14 # after max-correlation shift
################################################################################

# read atmospheric pressure if needed to convert concentration values
dfpr = pd.read_csv(os.path.join(md.datapath, 'SMEARII_p.csv'))
dfpr['datetime'] = pd.to_datetime(dfpr.iloc[:,:6])
dfpr.rename(columns = {'HYY_META.Pamb0':'patm_hPa'}, inplace = True)
dfpr30 = dfpr.resample('30min', on = 'datetime').agg({'patm_hPa':np.mean})
dfpr30['patm'] = dfpr30['patm_hPa']*100  # in Pascal
################################################################################


# counters needed below
first_time1 = True
first_time2 = True
init_time = time.time()



# to save the cleaned data in csv files for further analyses
save_cleaned_data = True
   
# ordered list of variables to read on Siikaneva_1 files
site = 'Siikaneva1'
myvars = ['u', 'v', 'w', 'T', 'CO2', 'H2O', 'CH4']

# scalar quantities to analyze and save in csv
scalars = ['u', 'T', 'CO2', 'H2O', 'CH4']
# scalars = ['H2O', 'CH4']

# range of date-times when observations were collected
# date_times = pd.date_range(start='2013-06-01 00:00:00',
#                  end='2013-08-31 23:30:00', freq='30min')

# date_times = pd.date_range(start='2013-07-17 10:00:00',
#                            end='2013-07-17 10:00:00', freq='30min')
# date_times = pd.date_range(start='2013-06-01 10:00:00',
#                      end='2013-06-1 10:00:00', freq='30min')

# olli's code is one date ahead (compare previous day here)
date_times = pd.date_range(start='2013-06-1 18:00:00',
                           end='2013-08-31 20:00:00', freq='30min')



# date_times = pd.date_range(start='2013-07-09 14:00:00',
#                            end='2013-07-09 14:00:00', freq='30min')
# remove a corrupted file:
if np.size(np.where(date_times ==  '2013-08-27 01:00:00')) > 0:
        date_times = date_times.delete(
        np.where(date_times ==  '2013-08-27 01:00:00')[0][0])
    
# create folders for output data if they do not exist already::
if not os.path.exists(md.outfolder_df):
    os.makedirs(md.outfolder_df)
if not os.path.exists(md.outdir_data):
    os.makedirs(md.outdir_data)
if not os.path.exists(md.outdir_plot):
    os.makedirs(md.outdir_plot)


# select relevant dates
# data available are from June 1st (DOY 152) to August 31st (DOY 243)
#date_times = pd.date_range(start='2013-06-01 00:30:00',
#                     end='2013-06-09 23:30:00', freq='30min')

# date_times = date_times[:100]
###########################################################################

# loop on runs
nruns = len(date_times)
# for ir, element in enumerate(date_times):
# EXS = np.zeros(len(date_times), dtype= bool)
# SIZES = np.zeros(len(date_times))


for ir, mydate in enumerate(date_times):
    print(ir)

    # df0, exists = me.load_data_Siikaneva(mydate, md.datapath, myvars)
    df0, exists = mf.load_data_Siikaneva(date_times[ir], md.datapath, myvars)
    # EXS[ir] = exists
    if exists:
        # keep only the longest stretch without missing data:
        df01 = mf.remove_missing_data(df0, nobs_min=nobs_min)[0]

    if exists and (df01.shape[0] < minsamplesize):
        exists = False


    # df1, nspikes = mf.despike(df1, myvars, plot = True)


    if exists:
        df1, nspikes = mf.despike(df01, myvars)
        # compute mean wind direction and rotate axes:
        windir = mf.WindDir(np.mean(df1['u'].values), np.mean(df1['v'].values))
        df2, angles = mf.coordrot(df1)

        # correct gas concentration to be relative to dry air
        # skip correction for CH4 for now
        df2['CO2'] = df2['CO2'].values/(1-df2['H2O'].values*1e-3) # check it does not affect df0
        df2['H2O'] = df2['H2O'].values/(1-df2['H2O'].values*1e-3)
        df2['CH4'] = df2['CH4'].values/(1-df2['H2O']*1e-3) ## ADDED OCT

        # compute concentrations per unit volume
        # using average surface atm pressure pa = 101325 Pa and dry air
        TmeanK = np.mean(df2['T'])
        # use average or local atmospheric pressure
        localp = True
        if localp:
            print('using local atm pressure')
            # read from the csv the pressure value for this date time
            patm = dfpr30['patm'].loc[mydate]
        else:
            print('using average atm pressure')
            patm = 101325
        univ_gas_const = 8.314
        corr_conc = patm/univ_gas_const/TmeanK
        df2['CO2'] = df2['CO2']*corr_conc
        df2['CH4'] = df2['CH4']*corr_conc
        df2['H2O'] = df2['H2O']*corr_conc


        # apply spectral correction - SKIP

        # further check for remaining CH4 spikes:
        cn = (df2['CH4'].values - np.mean(df2['CH4'].values))/np.std(df2['CH4'].values)
        cn0 = (df01['CH4'].values - np.mean(df01['CH4'].values))/np.std(df01['CH4'].values)
        if np.max(cn) > 60:
            print('Possible CH4 spike')
            plt.figure()
            plt.title('run {}'.format(ir))
            plt.plot(cn)
            plt.plot(cn0)
            # plt.plot(cn, 'o')
            plt.show()


        # compute the lag between gas and velocity time series
        df3, lag_dict = mf.correct_lags(df2, md.laglim, fs)

        # remove spikes in excess of 12 stdv
        # df4, nspikes, totspikes = mf.simple_despiking(df3, lim = 12)
        # df4, nspikes = mf.despike(df3, myvars)
        # df4 = df3

        # keep only the longest stretch without missing data for all variables
        # (we repeat this after despiking and translating data)
        df, enough_data = mf.remove_missing_data(df3, nobs_min = nobs_min)
        if not enough_data:
            exists = False

    if exists:
        ##################### save df with clean data if so ##############################
        # mydate = date_times[ir] # use element instead
        # name for output csv file in format YYYYMMDD_HHMM
        csv_name = str(mydate.year) + mf.dbs(mydate.month) \
                   + mf.dbs(mydate.day) + '_' \
                   + mf.dbs(mydate.hour) + mf.dbs(mydate.minute)
        if save_cleaned_data:
            df.to_csv( os.path.join(md.outfolder_df,
                    '{}.csv'.format(csv_name)), index = False)
            # print(df.shape[0])
            print(mydate)
        ########################################################

        # compute turbulent quantities
        turb_stats = mf.turb_quant(df, fs, z_ref)

        # modified: only check for u, T, H2O stationarity, not CH4
        fst, is_stationary = mf.flux_stationarity(df,fs,ws = 300,lim_diff=0.3)
        for mykey, myval in fst.items():
            turb_stats['stat_foken_{}'.format(mykey)] = myval

        # add other relevant quantities to then summary statistics::
        turb_stats['windir'] = windir
        turb_stats['angles_theta'] = angles[0]
        turb_stats['angles_phi'] = angles[1]
        turb_stats['angles_psi'] = angles[2]
        turb_stats['date'] = date_times[ir]
        turb_stats['exists'] = exists # file is there and has at least 2**14 data points
        # turb_stats['enough_data'] = enough_data # included in exists
        turb_stats['length'] = df.shape[0]
        turb_stats['is_stationary']= is_stationary
        turb_stats['csv_name'] = str(csv_name)

#
        if first_time1:
            print('initializing dataframe with turbulent quantities')
            turb_variables = list(turb_stats.keys())
            init_data0 = np.zeros((nruns, len(turb_variables)))*np.nan
            tdf = pd.DataFrame(init_data0, index=np.arange(nruns),
                                           columns = turb_variables)
            # tdf['exists'] =             tdf['exists'].astype('bool')
            # tdf['is_stationary'] =      tdf['is_stationary'].astype('bool')
            # # tdf['enough_data'] =        tdf['enough_data'].astype('bool')
            tdf['csv_name'] =           tdf['csv_name'].astype('str')
            tdf['exists'] =             np.zeros(nruns).astype('bool')
            tdf['is_stationary'] =      np.zeros(nruns).astype('bool')
            # tdf['csv_name'] =           np.zeros(nruns).astype('bool')
            first_time1 = False

        for varx in turb_stats.keys():
            tdf.at[ir, varx] = turb_stats[varx]



        # repeat the main analysis in Gaby's paper for all the scalars
        for var in scalars:


            squants = mf.scalar_quant(var, df, turb_stats, fs, z_ref)
            # print(var, squants['flux_ec'])


            mixmom = mf.mixed_moments(df[var].values, df['w'].values)


            me_DSo = mf.Delta_So(df[var].values, df['w'].values)


            res_eT = mf.Transport_Eff(df[var].values, df['w'].values)


            betaREAres = mf.REA_Beta(df[var].values, df['w'].values)


            betaREA_Milne = mf.REA_Beta_Milne(df[var].values, df['w'].values)

            ###################################################################
            # merge all results for the scalar I wanna keep in a single dictionary

            sdict = {**squants, **mixmom, **me_DSo, **res_eT, **betaREAres,
                     **betaREA_Milne}
                     # **flux_wa}

            if first_time2 and var == scalars[0]:

                print('initializing multi index data frame with results')
                variables = list(sdict.keys())
                scal_cols = pd.MultiIndex.from_product([variables, scalars],
                                               names=['variable', 'scalar'])
                init_data1 = np.zeros((nruns, len(variables)*len(scalars)))*np.nan
                sdf = pd.DataFrame(init_data1, index=np.arange(nruns),
                                               columns = scal_cols)
                first_time2 = False

            for variable in sdict.keys():
                sdf[variable, var].iloc[ir] = sdict[variable]
#
#

#
# # save results for each scalar for each run in a dataframe
# # first remove missing datasets

tdf3 = tdf[tdf['exists']==True].copy()
sdf3 = sdf[tdf['exists']==True].copy()


tdf.to_csv( os.path.join(md.outdir_data, 'all_results_tdf.csv'))
sdf.to_csv( os.path.join(md.outdir_data, 'all_scalars_sdf.csv'))


# test for non locality of H2O records:
h2o_is_local = mf.test_non_locality(tdf3, sdf3, plot = False)
tdf3['h2o_is_local'] = h2o_is_local
# tdf3.assign(h2o_is_local = h2o_is_local)

tdf3.to_csv( os.path.join(md.outdir_data, 'results_tdf.csv'))
sdf3.to_csv( os.path.join(md.outdir_data, 'scalars_sdf.csv'))
plot = False
if plot:
    os.system('python meth_plots_old.py')

final_time = time.time()
exec_time = (final_time - init_time)/60.0 # in minutes
print('execution time was {} minutes'.format(exec_time))

# tdf2 = tdf[tdf['exists'] == True]
#

# plt.plot(df['CH4'])
