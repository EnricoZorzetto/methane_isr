'''Main code with the analysis for the Methane project
Check the performance of EC vs SR vs WAVELET vs Flux-Variance
Also check flux variance relations for all runs
'''

# load dataframes with results
# and all stations
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methdata as md
import methfun as mf
from scipy.stats import gaussian_kde
# from scipy.stats import gaussian_kde
# from sklearn.neighbors import KernelDensity
# import pywt
# import spectral_analysis as sa
# import matplotlib.cm as cm
# import scipy as sc
# import matplotlib as mpl
# from scipy.optimize import curve_fit

# matplotlib_update_settings()
# mf.matplotlib_update_settings()

################################ PARAMETERS ####################################

N = 2 ** 14 # min number obs in each run
wavelet = 'haar'
mysc = 'CH4'
refsc = 'H2O'
# tscale = 'tau_eta'
tscale = 'tau_eta0' # roughness tke dissipation rate
# minsize = 4 # filter out scales with less than 4 wavelet coefficients
minscale = 0.5 # normalized to Tw
maxscale = 200 # normalized to Tw
# maxsize = 2**11 # alternatively
nstdvs = 3 # this is equiv to 3 stdv computed excluding the smalles scales
mystdv_c = 0.3 # wc # 0.3 - 0.2 for 1 - 0.5 min Iw

################################################################################




# define how to compute DC for surface renewal
# currently using m1m5 scales instead
minsize_sr = 4 # for computing the source strength only
useminsize_sr = False # if true use minsize_sr, otherwise maxscale_sr
minscale_sr = 0.5 # relative to Iw
maxscale_sr = 5 # for computing the source strength only


# analyze_a_single_run = False
analyze_a_single_run = True
# nbins = 10 # for computing histograms

# nstdvs_flux = 3
# mystdv = 0.20*4
# mystdv_flux = 0.8*4 # *4 values obtained from stdv from max scale of 2**11
# mystdv_quad_en = 0.6
# mystdv_quad_fl = 0.3

# mystdv2 = 10 # w2
# mystdv = 0.55 # wc
# mystdv_flux = 3 # wf # *4 values obtained from stdv from max scale of 2**11
# mystdv_quad_en = 0.65 # qc
# mystdv_quad_fl = 0.28 # qf

###########################################################
# values corresponding to a correlation H2o- CH4 of 0.98 -
###########################################################
# mystdv_c2 = 3.0 # w2 # 4.5 - 4.0 for 1 - 0.5 min Iw
# # mystdv_c = 0.2 # wc # 0.3 - 0.2 for 1 - 0.5 min Iw
# mystdv_flux = 0.8 # wf # 1.2 - 1.0
# mystdv_quad_en = 0.3 # qc
# mystdv_quad_fl = 0.15 # qf
###########################################################


# load data from previous analysis
tdf0 = pd.read_csv(os.path.join(md.outdir_data, 'results_tdf.csv'),index_col=0)
sdf0 = pd.read_csv(os.path.join(md.outdir_data, 'scalars_sdf.csv'),
                     header=[0,1], index_col = 0)

#
cond1 = (tdf0['ustar'] > 0.2) \
        & (tdf0['is_stationary'] == True) \
        & (tdf0['exists'] == True) \
         & (tdf0['windir'] > 230) \
         & (tdf0['windir'] < 270)\
         & (tdf0['length'] >= N)\
         & (tdf0['h2o_is_local'] == True) \
         & (sdf0['flux_ec', 'H2O'] > 0)
#

# cond1 = (tdf0['ustar'] > 0.2) \
#         & (tdf0['is_stationary'] == True) \
#         & (tdf0['exists'] == True) \
#         & (tdf0['windir'] < 275) \
#         & (tdf0['windir'] > 25) \
#         & (tdf0['length'] >= N) \
#         & (tdf0['h2o_is_local'] == True) \
#         & (sdf0['flux_ec', 'H2O'] > 0)



# cond1 = (tdf0['ustar'] > 0.2) \
#         & (tdf0['exists'] == True) \
#         & (tdf0['is_stationary'] == True) \
#         & (tdf0['z0'] < 0.2) \
#         & (tdf0['length'] >= N) \
#         & (sdf0['flux_ec', 'H2O'] > 0.001)

tdf = tdf0[cond1].copy()
sdf = sdf0[cond1].copy()

sdfM30 = pd.DataFrame(sdf['M30', 'CH4'])
sdfM30['csv_name'] = tdf['csv_name']

####################### select a given date-time ###############################
save_outdata = True
################################################################################
if analyze_a_single_run:
    save_outdata = False
    # mydatetime = '2013-07-09 11:00:00' # low ebullition # skip these
    # mydatetime = '2013-06-03 14:00:00' # low ebullition # skip these
    # mydatetime = '2013-06-03 17:30:00' # low ebullition # skip these
    # mydatetime = '2013-08-18 15:00:00' # low ebullition # skip these
    ## ## Gaby paper:
    mydatetime = '2013-07-09 13:30:00' # also high ebullition (morning)
    ############# ############# ##############
    # mydatetime = '2013-07-09 14:00:00' # high ebullition
    # mydatetime = '201
    # mydatetime = '2013-06-17 12:00:00' # medium/high ebullition
    tdf = tdf0[np.logical_and(cond1, tdf0['date'] == mydatetime)].copy()
    sdf = sdf0[np.logical_and(cond1, tdf0['date'] == mydatetime)].copy()
################################################################################

# tdf = tdf[:100]
# sdf = sdf[:100]

init_time = time.time()
nruns = tdf.shape[0]

testplot = True
if nruns > 3:
    testplot = False

# initialize df with pdf data
colnames = ['x_wa', 'x_me', 'x_en', 'x_ba', 'p_wa', 'p_me', 'p_en', 'p_ba']
pdf = pd.DataFrame(data=None, index=range(nruns),
                   columns=colnames, dtype=object)

first_run = True
print('NUMBER OF RUNS = {}'.format(nruns))
for i in range(nruns):
    df = pd.read_csv( os.path.join(md.datapath,
           'Siikaneva_2013_cleaned', '{}.csv'.format(tdf['csv_name'].iloc[i])))
    print(i)
    df = df[:N]
    # cp0 = df[mysc].copy() - np.mean(df[mysc].copy() )


    turb_quants = mf.turb_quant(df, md.fs, md.z_ref, nobs_min=6000)


    # filter time series (remove largest scales and / or smallest):
    # df['w'] = sa.wave_filter(df['w'].values, wavelet=wavelet,
    #                          minsize=minsize, maxsize = maxsize)
    # df[mysc] = sa.wave_filter(df[mysc].values, wavelet=wavelet,
    #                           minsize=minsize, maxsize = maxsize)
    # df[refsc] = sa.wave_filter(df[refsc].values, wavelet=wavelet,
    #                            minsize=minsize, maxsize = maxsize)

    # wp = (df['w'] - np.mean(df['w']))
    # cp = (df[mysc] - np.mean(df[mysc]))
    # rp = (df[refsc] - np.mean(df[refsc]))
    #
    # stdvw = np.std(df['w'])
    # stdvc = np.std(df[mysc])
    # stdvr = np.std(df[refsc])
    #
    # wn = wp/stdvw
    # cn = cp/stdvc
    # rn = rp/stdvr
    #
    # if np.max(cn) > 60:
    #     print('index = {}'.format(tdf.index[i]))
    #     print('Possible CH4 spike')
    #     plt.figure()
    #     plt.title('run {}'.format(i))
    #     plt.plot(cn)
    #     plt.plot(cn, 'o')
    #     plt.show()

    #
    # # compute some statistics for the original time series
    # c_moms = mf.mixed_moments(cn, wn)
    # r_moms = mf.mixed_moments(rn, wn)

    # res_qr = mf.quadrant_ratios(df, mysc=mysc, refsc=refsc, thresh=0.0)
    #



    res_wc = mf.wavelet_partition(df, mysc=mysc, refsc=refsc,
                          cdsc = 'CO2', nmin_obs=N,
                          minscale = minscale,
                          maxscale = maxscale,
                          mystdv=mystdv_c, nstdvs=nstdvs,
                          wavelet=wavelet, cond_x='c',
                          plot = testplot, datetime = tdf['date'].iloc[i])


    res_wd = mf.wavelet_partition(df, mysc='CO2', refsc=refsc,
                                  cdsc = 'CO2', nmin_obs=N,
                                  minscale = minscale,
                                  maxscale = maxscale,
                                  mystdv=mystdv_c, nstdvs=nstdvs,
                                  wavelet=wavelet, cond_x='c',
                                  plot = False, datetime = tdf['date'].iloc[i])

    # res_wco2 = mf.wavelet_partition(df, mysc='CO2', refsc=refsc,
    #                                 cdsc = 'CO2', nmin_obs=N,
    #                       minscale = minscale,
    #                       maxscale = maxscale,
    #                       mystdv=mystdv_c, nstdvs=nstdvs,
    #                       wavelet=wavelet, cond_x='c',
    #                       plot = False, datetime = tdf['date'].iloc[i])

    # res_wf = mf.wavelet_partition(df, mysc=mysc, refsc=refsc, nmin_obs=N,
    #                       minscale=minscale,
    #                       maxscale=maxscale,
    #                       mystdv=mystdv_flux,
    #                       nstdvs=nstdvs, wavelet=wavelet, cond_x='wc')
    #
    #
    # res_w2 = mf.wavelet_partition(df, mysc=mysc, refsc=refsc, nmin_obs=N,
    #                       minscale=minscale,
    #                       maxscale=maxscale,
    #                       mystdv=mystdv_c2, nstdvs=nstdvs,
    #                       wavelet=wavelet, cond_x='c2')


    # res_qc = mf.quadrant_partition(df, mysc=mysc, refsc=refsc, cond_x='c',
    #                        mystdv_en=mystdv_quad_en, nstdvs_en=nstdvs)
    #
    #
    # res_qf = mf.quadrant_partition(df, mysc=mysc, refsc=refsc, cond_x='wc',
    #                        mystdv_fl=mystdv_quad_fl, nstdvs_fl=nstdvs)


    # res_ts = mf.detect_eb_timescle(df, res_wc, wavelet=wavelet,
    #                                plot = testplot,
    #                                minscale = minscale,
    #                                maxscale = maxscale,
    #                                # minsize=minsize,
    #                                # maxsize = maxsize
    #                                datetime = tdf['date'].iloc[i]
    #                                )


    res_isr = mf.isr_fluxes(df, res_wc, res_wd, mysc=mysc, refsc=refsc,
                tscale=tscale, wavelet=wavelet,
                maxscale = maxscale_sr,
                useminsize=useminsize_sr,
                minscale = minscale_sr,
                minsize = minsize_sr,
                fs=md.fs, z_ref=md.z_ref)

    # # I can compute SR fluxes using the sr function for each scalar:
    # CH4sr = mf.isr_fluxes(df, res_wc, mysc='CH4', tscale=tscale,
    #     maxscale=maxscale_sr, useminsize=useminsize_sr,
    #     minscale=minscale_sr, minsize=minsize_sr, fs=md.fs, z_ref=md.z_ref)
    #
    # H2Osr = mf.isr_fluxes(df, res_wc, mysc='H2O', tscale=tscale,
    #     maxscale=maxscale_sr, useminsize=useminsize_sr,
    #     minscale=minscale_sr, minsize=minsize_sr, fs=md.fs, z_ref=md.z_ref)
    #
    # CO2sr = mf.isr_fluxes(df, res_wc, mysc='CO2', tscale=tscale,
    #     maxscale=maxscale_sr, useminsize=useminsize_sr,
    #     minscale=minscale_sr, minsize=minsize_sr, fs=md.fs, z_ref=md.z_ref)
    #
    # Tsr = mf.isr_fluxes(df, res_wc, mysc='T', tscale=tscale,
    #     maxscale=maxscale_sr, useminsize=useminsize_sr,
    #     minscale=minscale_sr, minsize=minsize_sr, fs=md.fs, z_ref=md.z_ref)
    #
    # Usr = mf.isr_fluxes(df, res_wc, mysc='u', tscale=tscale,
    #     maxscale=maxscale_sr, useminsize=useminsize_sr,
    #     minscale=minscale_sr, minsize=minsize_sr, fs=md.fs, z_ref=md.z_ref)

    # fluxes = {'me_srf':CH4sr['fme_sr'],    'wa_srf':H2Osr['fme_sr'],
    #     'T_srf':Tsr['fme_sr'],    'u_srf':Usr['fme_sr'], 'cd_srf':CO2sr['fme_sr'],
    #     'me_ecf':CH4sr['fme_ec'], 'wa_ecf':H2Osr['fme_ec'],
    #     'T_ecf':Tsr['fme_ec'], 'u_ecf':Usr['fme_ec'],  'cd_ecf':CO2sr['fme_ec']}

    # A, B = sa.emp_pdf(cn, nbins=nbins)
    # pdf.at[i, 'x_me'], pdf.at[i, 'p_me'] = sa.emp_pdf(cn, nbins=nbins)
    # pdf.at[i, 'x_wa'], pdf.at[i, 'p_wa'] = sa.emp_pdf(rn, nbins=nbins)
    # pdf.at[i, 'x_en'], pdf.at[i, 'p_en'] = sa.emp_pdf(res_wc['cn_ener'],
    #                                                   nbins=nbins)
    # pdf.at[i, 'x_ba'], pdf.at[i, 'p_ba'] = sa.emp_pdf(res_wc['cn_back'],
    #                                                   nbins=nbins)


    # save all scalar valued results in a data frame
    # res_dicts = {'wa':r_moms, 'me':c_moms, 'qr':res_qr,'w2':res_w2,'ts':res_ts,
    #          'wc':res_wc, 'wf':res_wf, 'qc':res_qc, 'qf':res_qf, 'sr':res_isr,
    #              'fluxes':fluxes}

    # res_dicts = {'qr':res_qr,'w2':res_w2,'ts':res_ts,
    #              'wc':res_wc, 'wf':res_wf, 'qc':res_qc, 'qf':res_qf,
    #              'wco2':res_wco2, 'srco2':CO2sr,
    #              'sr':res_isr, 'fluxes':fluxes}

    res_dicts = {'wc':res_wc, 'wd':res_wd, 'sr':res_isr}
    if first_run:
        colname1 = ['csv_name']
        # turb_variables = list(resquad.keys())
        # init_data0 = np.zeros((nruns, len(turb_variables))) * np.nan
        rdf = pd.DataFrame(tdf[colname1].values, index=np.arange(nruns),
                           columns=colname1)
        # rdf['csv_name'] = rdf['csv_name'].astype('str')
        first_run = False

    for key, val in res_dicts.items():
        for key2, val2 in val.items():
            # if key2.split('_')[0] in vars_to_save:
            if np.isscalar(val2):
                rdf.at[i, '{}_{}'.format(key, key2)] = val2


if save_outdata:
    rdf.to_csv( os.path.join(md.outdir_data, 'results_rdf.csv'))
    rdf.to_csv( os.path.join(md.outdir_data, 'results_rdf_{}.csv'.format(mystdv_c)))
    # rdf = pd.read_csv( os.path.join(md.outdir_data, 'results_rdf.csv'))
    # rdf = pd.read_csv( os.path.join(md.outdir_data, 'results_rdf.csv'),header=[0,1], index_col = 0)
    # pdf.to_csv( os.path.join(md.outdir_data, 'results_pdf.csv'))
    # pdf.to_pickle( os.path.join(md.outdir_data, 'results_pdf.csv'))
# pdf2 = pd.read_pickle( os.path.join(md.outdir_data, 'results_pdf.csv'))


final_time = time.time()
exec_time = (final_time - init_time)/60.0 # in minutes
print('execution time was {} minutes'.format(exec_time))

# print('average c ebullition flux fraction = ', np.mean(rdf['wc_en_frac_flux']))
# print('average c background flux fraction = ', np.mean(rdf['wc_ba_frac_flux']))
# print('average c2 ebullition flux fraction = ', np.mean(rdf['w2_en_frac_flux']))
# print('average c2 background flux fraction = ', np.mean(rdf['w2_ba_frac_flux']))
# print('average cf ebullition flux fraction = ', np.mean(rdf['wf_en_frac_flux']))
# print('average cf background flux fraction = ', np.mean(rdf['wf_ba_frac_flux']))

# plt.figure()
# # plt.plot(rdf['ts_l'], rdf['sr_ustar']**4, 'o')
# plt.plot(rdf['ts_l'], rdf['sr_ustar'], 'o')
# # plt.plot(rdf['wc_en_frac_time'], rdf['sr_ustar'], 'o')
# plt.show()

# np.mean(df['CH4'])
# np.mean(df['CO2'])

# plt.figure()
# plt.plot(df['CH4'])

# plt.figure()
# plt.plot(cn, 'g')
# plt.plot(rn, 'b')
# plt.show()
#
# np.max(cn)
# np.min(cn)
#
# A = np.log10(np.logspace(np.min(cn), np.max(cn), 10))

# plot a CH4 time series
metn = (df['CH4'] - np.mean(df['CH4']))/np.std(df['CH4'])
watn = (df['H2O'] - np.mean(df['H2O']))/np.std(df['H2O'])
cdtn = (df['CO2'] - np.mean(df['CO2']))/np.std(df['CO2'])
wnwn = (df['w'] - np.mean(df['w']))/np.std(df['w'])
tt = np.arange(np.size(metn))/10 # seconds

import matplotlib
matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize = (6, 4))
plt.plot(tt[50:2000], metn[50:2000], 'k')
plt.ylabel(r"$CH_4$ $c' // \sigma_n$")
plt.xlabel("time [s]")
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'chr_time_series.png'), dpi = 300)
plt.show()

plt.figure()
plt.plot(wnwn, metn, 'o')
plt.plot([0, 0], [-2, 18], '--k')
plt.plot([-5.5, 5.5], [0, 0], '--k')
plt.ylabel("$c'/\sigma_c$")
plt.xlabel("$w'/\sigma_w$")
plt.show()

CH4j = np.vstack([metn, wnwn])
H2Oj = np.vstack([watn, wnwn])
CO2j = np.vstack([cdtn, wnwn])
jpdf_ch4 = gaussian_kde(CH4j)(CH4j)
jpdf_h2o = gaussian_kde(H2Oj)(H2Oj)
jpdf_co2 = gaussian_kde(CO2j)(CO2j)



xmin = wnwn.min()
xmax = wnwn.max()
ymin = metn.min()
ymax = metn.max()


# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])


# Z = np.reshape(jpdf_ch4(positions).T, X.shape)

fig, axes = plt.subplots(1, 3, figsize = (16, 7))
# fig.suptitle('Run recorded {}'.format(mydatetime))
axes[0].scatter(wnwn, metn, c = jpdf_ch4)
axes[0].set_ylabel("$c'/\sigma_c  \quad c = CH_4$")
axes[0].set_xlabel("$w'/\sigma_w$")

axes[0].plot([0, 0], [-5, 20], '--k')
axes[0].plot([-5.5, 5.5], [0, 0], '--k')
axes[1].scatter(wnwn, watn, c = jpdf_h2o)
axes[1].set_ylabel("$c'/\sigma_c \quad c = H_2O$")
axes[1].set_xlabel("$w'/\sigma_w$")

axes[1].plot([0, 0], [-5, 20], '--k')
axes[1].plot([-5.5, 5.5], [0, 0], '--k')
axes[2].scatter(wnwn, cdtn, c = jpdf_co2)
axes[2].set_ylabel("$c'/\sigma_c  \quad c = CO_2$")
axes[2].set_xlabel("$w'/\sigma_w$")

axes[2].plot([0, 0], [-5, 20], '--k')
axes[2].plot([-5.5, 5.5], [0, 0], '--k')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(md.outdir_plot, 'density_scatter_plots.png'), dpi = 300)


