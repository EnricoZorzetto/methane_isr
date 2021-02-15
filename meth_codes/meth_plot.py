

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import methfun as mf
import methdata as md
from scipy.interpolate import UnivariateSpline

# to register datetimes in matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
mf.matplotlib_update_settings() # make prettier plots
# to avoid displaying plots: change backend
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('Qt5Agg')

# os.environ['PATH'] = os.environ['PATH'] + ':/'

# length of time series
N = 2**14
tdf0 = pd.read_csv(os.path.join(md.outdir_data, 'results_tdf.csv'),index_col=0)
sdf0 = pd.read_csv(os.path.join(md.outdir_data, 'scalars_sdf.csv'),
                   header=[0,1], index_col = 0)

# fwat = sdf0['flux_ec', 'H2O'].values
# q005fwat = np.quantile(fwat, 0.05)
# q01fwat = np.quantile(fwat, 0.25)
# q05fwat = np.quantile(fwat, 0.5)
# q095fwat = np.quantile(fwat,0.95)


##################################### CONDITION FOR RETAINING RUNS:: ###########
cond1 = (tdf0['ustar'] > 0.2) \
        & (tdf0['is_stationary'] == True) \
        & (tdf0['exists'] == True) \
& (tdf0['windir'] > 230) \
  & (tdf0['windir'] < 270) \
  & (tdf0['length'] >= N)\
        & (tdf0['h2o_is_local'] == True) \
        & (sdf0['flux_ec', 'H2O'] > 0)

###################################### FOR PLOTTING ONLY:: #####################
cond_noz0 = (tdf0['ustar'] > 0.2) \
        & (tdf0['is_stationary'] == True) \
        & (tdf0['exists'] == True) \
        & (tdf0['length'] >= N) \
        & (tdf0['h2o_is_local'] == True) \
        & (sdf0['flux_ec', 'H2O'] > 0)

############### OTHER CONDITIONS, JUST TO COUNT HOW MANY RUNS ARE EXCLUDED:: ###

cond_turb_intensity = (tdf0['ustar'] > 0.2)
cond_wind_sector = (tdf0['windir'] > 230) & (tdf0['windir'] < 270)
cond_stationary = (tdf0['is_stationary'] == True)
cond_water_vapor = (tdf0['h2o_is_local'] == True)  & (sdf0['flux_ec', 'H2O'] > 0)


nruns_turb_int = np.size(cond_turb_intensity[cond_turb_intensity > 0])
nruns_wind_sector = np.size(cond_wind_sector[cond_wind_sector > 0])
nruns_stationary = np.size(cond_stationary[cond_stationary > 0])
nruns_water_vapor = np.size(cond_water_vapor[cond_water_vapor > 0])




# only to plot all wind rirections::
tdfnoz0 = tdf0[cond_noz0].copy()
sdfnoz0 = sdf0[cond_noz0].copy()

# for the rest of the analysis::
tdf = tdf0[cond1].copy()
sdf = sdf0[cond1].copy()

rdf = pd.read_csv( os.path.join(md.outdir_data, 'results_rdf.csv'), index_col=0)
# rdf = pd.read_csv( os.path.join(md.outdir_data, 'results_rdf_0.5.csv'), index_col=0)
# rdf = pd.read_csv( os.path.join(md.outdir_data, 'results_rdf_0.3.csv'), index_col=0)
nruns = np.shape(rdf)[0]

rdf['datetime'] = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M')
sdf['datetime'] = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M')
tdf['datetime'] = pd.to_datetime(tdf['csv_name'], format='%Y%m%d_%H%M')
tdf0['datetime'] = pd.to_datetime(tdf0['csv_name'], format='%Y%m%d_%H%M')
sdf0['datetime'] = pd.to_datetime(tdf0['csv_name'], format='%Y%m%d_%H%M')
# datetimes = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M', errors='ignore')




print('total number of tuns (after processing the raw data) = ', nruns)
print('total number of runs in this wind sector = {}'.format(nruns_wind_sector))
print('total number of runs passing stationarity checks = {}'.format(nruns_stationary))
print('total number of runs passing water vapor checks = {}'.format(nruns_water_vapor))
print('total number of runs passing turbulence intensity checks = {}'.format(nruns_turb_int))
# plt.figure()
# plt.plot(rdf['wc_wa_M30'], rdf['wc_en_M30'], 'or')
# plt.plot(rdf['wc_wa_M30'], rdf['wc_me_M30'], 'og')
# plt.plot(rdf['wc_wa_M30'], rdf['wc_wa_M30'], 'k')
# plt.plot(rdf['wc_wa_M30'], rdf['wc_ba_M30'], 'oc')
# plt.plot(rdf['wc_wa_M30'], rdf['wc_bar_M30'], 'oy')
# plt.show()


# plt.plot(rdf['wc_wa_M30'], rdf['wc_en_M30'], 'or')
#
# plt.figure()
# plt.scatter(rdf['sr_tewa'].values, rdf['sr_teme'].values, alpha = 0.6, s = 18, color = 'g', marker = '^', label = 'CH4')
# plt.scatter(rdf['sr_tewa'].values, rdf['sr_teba'].values, alpha = 0.6, s = 18, color = 'c', marker = '^', label = 'CH4')
# plt.scatter(rdf['sr_tewa'].values, rdf['sr_teen'].values, alpha = 0.6, s = 18, color = 'r', marker = '^', label = 'CH4')
# plt.scatter(rdf['srco2_tewa'].values, -rdf['srco2_teme'].values, alpha = 0.6, s = 18, color = 'y', marker = '^', label = 'CO2')
# plt.plot(rdf['sr_tewa'].values, rdf['sr_tewa'].values, 'k')
# plt.show()


plt.figure()
plt.xlabel('Wind Direction [degrees]')
plt.ylabel(r'Roughness length $z_0$ [m]')
plt.plot(tdf0['windir'], tdf0['z0'], 'o', label = 'All runs', alpha = 0.6)
plt.plot(tdfnoz0['windir'], tdfnoz0['z0'], 'o', label = 'All wind sector', alpha = 0.6)
plt.plot(tdf['windir'], tdf['z0'], 'o', label = 'filtered', alpha = 0.6)

plt.plot([230, 230], [np.min(tdfnoz0['z0'].values), np.max(tdfnoz0['z0'].values)], '--k')
plt.plot([270, 270], [np.min(tdfnoz0['z0'].values), np.max(tdfnoz0['z0'].values)], '--k')
# plt.plot(tdfnoz0['windir'], tdfnoz0['z0'], 'o', label = 'included')
plt.legend()
# plt.yscale()
plt.savefig( os.path.join(md.outdir_plot, 'SI_z0_wind_direction.png'))
plt.close()


plt.figure()
plt.xlabel('Wind Direction [degrees]')
plt.ylabel(r'Transport efficiency $e_T$')
plt.plot(tdfnoz0['windir'], sdfnoz0['eT', 'H2O'], 'ob', label = r'$H_2O$', alpha = 0.4)
plt.plot(tdfnoz0['windir'], sdfnoz0['eT', 'CH4'], 'og', label = r'$CH_4$', alpha = 0.4)
plt.plot(tdfnoz0['windir'], sdfnoz0['eT', 'CO2'], 'oc', label = r'$CO_2$', alpha = 0.4)

plt.plot(tdf['windir'], sdf['eT', 'H2O'], 'ob', alpha = 0.6)
plt.plot(tdf['windir'], sdf['eT', 'CH4'], 'og', alpha = 0.6)
plt.plot(tdf['windir'], sdf['eT', 'CO2'], 'oc', alpha = 0.6)

plt.plot([230, 230], [np.min(sdfnoz0['eT', 'CO2'].values), np.max(sdfnoz0['eT', 'CH4'].values)], '--k')
plt.plot([270, 270], [np.min(sdfnoz0['eT', 'CO2'].values), np.max(sdfnoz0['eT', 'CH4'].values)], '--k')
# plt.plot(tdfnoz0['windir'], tdfnoz0['z0'], 's', label = 'without z0 filter')
# plt.plot(tdf['windir'], sdf['z0'], 'o', label = 'with z0 filter')
# plt.plot(tdfnoz0['windir'], tdfnoz0['z0'], 'o', label = 'included')
plt.legend()
plt.savefig( os.path.join(md.outdir_plot, 'SI_eT_wind_direction.png'))
plt.close()


plt.figure()
plt.xlabel('Wind Direction [degrees]')
plt.ylabel(r'Scalar Skewness $M_{30}$')
plt.plot(tdfnoz0['windir'].values, sdfnoz0['M30', 'H2O'].values, 'ob', alpha = 0.4)
plt.plot(tdfnoz0['windir'].values, sdfnoz0['M30', 'CH4'].values, 'og', alpha = 0.4)
plt.plot(tdf['windir'].values, np.abs(sdf['M30', 'H2O'].values), 'ob', label = r'$H_2O$', alpha = 0.6)
plt.plot(tdf['windir'].values, np.abs(sdf['M30', 'CH4'].values), 'og', label = r'$CH_4$', alpha = 0.6)
plt.plot([230, 230], [np.min(sdfnoz0['M30', 'H2O'].values), np.max(sdfnoz0['M30', 'CH4'].values)], '--k')
plt.plot([270, 270], [np.min(sdfnoz0['M30', 'H2O'].values), np.max(sdfnoz0['M30', 'CH4'].values)], '--k')
# plt.plot(tdf['windir'].values, sdf['flux_ec', 'H1O'].values, 'o')
plt.legend()
plt.yscale('log')
plt.savefig( os.path.join(md.outdir_plot, 'SI_Skew_wind_direction.png'))
plt.close()

# PRINT TRANSPORT EFFICIENCIES
print('-----------------------------------------------------------------------')
print('Transport Efficiencies eT:')

print('methane: mean = {}, stdv = {}'.format(np.mean(rdf['wc_me_eT']), np.std(rdf['wc_me_eT'])))
print('water  : mean = {}, stdv = {}'.format(np.mean(rdf['wc_wa_eT']), np.std(rdf['wc_wa_eT'])))
print('carbon : mean = {}, stdv = {}'.format(np.mean(rdf['wc_cd_eT']), np.std(rdf['wc_cd_eT'])))

print('hotspot : mean = {}, stdv = {}'.format(np.mean(rdf['wc_en_eT']), np.std(rdf['wc_en_eT'])))
print('backgr. : mean = {}, stdv = {}'.format(np.mean(rdf['wc_ba_eT']), np.std(rdf['wc_ba_eT'])))
print('-----------------------------------------------------------------------')

fig, axes = plt.subplots(1, 2, figsize = (12, 6))
axes[0].set_ylim([0, 1])
axes[1].set_ylim([0, 1])

# axes[0].set_xlim([0, 1])
# axes[1].set_xlim([0, 1])
axes[0].set_xlabel(r'Transport efficiency $e_T$ ')
axes[0].set_ylabel(r'Transport efficiency $e_T$ ')
axes[1].set_xlabel(r'$H_2O$ Transport efficiency $e_T$ ')
axes[1].set_ylabel(r'$CH_4$ Transport efficiency $e_T$ ')

# axes[0].plot(sdf['eT', 'H2O'], sdf['eT', 'H2O'], 'k')
# axes[1].plot(sdf['eT', 'H2O'], sdf['eT', 'H2O'], 'k')
# axes[0].plot(rdf['wc_wa_eT'], rdf['wc_wa_eT'], 'k')
# axes[0].plot(rdf['wc_wa_eT'], rdf['wc_me_eT'], 'og', label = r'$CH_4$', alpha = 0.7, markersize = 6)
# axes[0].plot(rdf['wc_wa_eT'], rdf['wc_cd_eT'], 'sb', label = r'$CO_2$', alpha = 0.7, markersize = 6)
#
# axes[1].plot(rdf['wc_wa_eT'], rdf['wc_wa_eT'], 'k')
# axes[1].plot(rdf['wc_wa_eT'], rdf['wc_me_eT'], 'og', label = r'$CH_4$ total', alpha = 0.8, markersize = 6)
# axes[1].plot(rdf['wc_we_eT'], rdf['wc_en_eT'], '^r', label = r'$CH_4$ hotspot', alpha = 0.7, markersize = 6)
# axes[1].plot(rdf['wc_wb_eT'], rdf['wc_ba_eT'], 'sc', label = r'$CH_4$ background', alpha = 0.7, markersize = 6)

mrkrsize = 12
axes[0].plot(rdf['wc_wa_eT'], rdf['wc_wa_eT'], color = 'k')
axes[0].scatter(rdf['wc_wa_eT'], rdf['wc_me_eT'], c = 'green', marker='o', label = r'$CH_4$', alpha = 0.7, s = mrkrsize)
axes[0].scatter(rdf['wc_wa_eT'], rdf['wc_cd_eT'], c = 'blue', marker='s', label = r'$CO_2$', alpha = 0.7, s = mrkrsize)
axes[1].plot(rdf['wc_wa_eT'], rdf['wc_wa_eT'], color = 'k')
axes[1].scatter(rdf['wc_wa_eT'], rdf['wc_me_eT'], c = 'green', marker='o', label = r'$CH_4$ total', alpha = 0.8, s = mrkrsize)
axes[1].scatter(rdf['wc_we_eT'], rdf['wc_en_eT'], c = 'red', marker='^', label = r'$CH_4$ hotspot', alpha = 0.7, s = mrkrsize)
axes[1].scatter(rdf['wc_wb_eT'], rdf['wc_ba_eT'], c = 'orange', marker='s', label = r'$CH_4$ background', alpha = 0.7, s = mrkrsize)
axes[0].legend(loc="lower right")
axes[1].legend(loc="lower left")

axes[0].annotate("a)", xy=(0.05, 0.9), xycoords="axes fraction")
axes[1].annotate("b)", xy=(0.05, 0.9), xycoords="axes fraction")
plt.tight_layout()
plt.savefig( os.path.join(md.outdir_plot, 'scatter_eT.png'))
plt.close()

# plt.figure()
# plt.plot(rdf['wc_me_eT'], rdf['wc_en_eT'], 'or', label = r'$CH_4$ total', alpha = 0.8, markersize = 5)
# plt.plot(rdf['wc_me_eT'], rdf['wc_ba_eT'], 'oc', label = r'$CH_4$ total', alpha = 0.8, markersize = 5)
# # plt.plot(rdf['wc_wa_eT'], rdf['wc_we_eT'], 'or', label = r'$CH_4$ total', alpha = 0.8, markersize = 5)
# # plt.plot(rdf['wc_wa_eT'], rdf['wc_wb_eT'], 'oc', label = r'$CH_4$ total', alpha = 0.8, markersize = 5)
# # plt.plot(rdf['wc_we_eT'], rdf['wc_en_eT'], 'or', label = r'$CH_4$ ebullition', alpha = 0.5, markersize = 5)
# # plt.plot(rdf['wc_wa_eT'], rdf['wc_me_eT'], 'ok', label = r'$CH_4$ total', alpha = 0.8, markersize = 5)
# # plt.plot(rdf['wc_wb_eT'], rdf['wc_ba_eT'], 'oy', label = r'$CH_4$ background', alpha = 0.5, markersize = 5)
# plt.plot(sdf['eT', 'H2O'], sdf['eT', 'H2O'], 'k')
# plt.savefig( os.path.join(md.outdir_plot, 'scatter_eT.png'))
# plt.close()


#
# plt.figure()
# plt.plot(rdf['wc_me_M30)'], rdf['wc_me_M30'], 'k')
# plt.plot(rdf['wc_me_M30)'], rdf['wc_en_M30'], 'or', label = r'$CH_4$ ener', alpha = 0.8, markersize = 5)
# plt.plot(rdf['wc_me_M30)'], rdf['wc_ba_M30'], 'oc', label = r'$CH_4$ back', alpha = 0.8, markersize = 5)
# plt.show()

fig, axes = plt.subplots(1, 2, figsize = (12, 6))
axes[0].set_xlabel(r'Stability parameter $\zeta$')
axes[1].set_xlabel(r'Stability parameter $\zeta$')
axes[0].set_ylabel(r'Transport efficiency $e_T$ ')
axes[0].set_ylim([0, 1])
axes[1].set_ylim([0, 1])
# axes[1].set_xlabel(r'$H_2O$ Transport efficiency $e_T$ ')
axes[1].set_ylabel(r'Transport efficiency $e_T$ ')
# axes[0].plot(sdf['eT', 'H2O'], sdf['eT', 'H2O'], 'k')
# axes[1].plot(sdf['eT', 'H2O'], sdf['eT', 'H2O'], 'k')
axes[0].plot(tdf['stab'], rdf['wc_me_eT'], 'og', label = 'CH4', alpha = 0.6, markersize = 5)
axes[0].plot(tdf['stab'], rdf['wc_wa_eT'], 'ob', label = 'H2O', alpha = 0.6, markersize = 5)
# axes[0].plot(tdf['stab'], rdf['wco2_me_eT'], 'oc', label = 'CO2', alpha = 0.6, markersize = 5)
axes[0].plot(tdf['stab'], rdf['wc_cd_eT'], 'oc', label = 'CO2', alpha = 0.6, markersize = 5)
axes[1].plot(tdf['stab'], rdf['wc_en_eT'], 'or', label = 'CH4 hotspot', alpha = 0.5, markersize = 5)
axes[1].plot(tdf['stab'], rdf['wc_me_eT'], 'ok', label = 'CH4 total', alpha = 0.8, markersize = 5)
axes[1].plot(tdf['stab'], rdf['wc_ba_eT'], 'oy', label = 'CH4 background', alpha = 0.5, markersize = 5)

axes[0].annotate("a)", xy=(0.04, 0.90), xycoords="axes fraction")
axes[1].annotate("b)", xy=(0.04, 0.90), xycoords="axes fraction")
axes[0].legend()
axes[1].legend()
plt.savefig( os.path.join(md.outdir_plot, 'stability_eT.png'))
plt.close()



# check EC vs WAVELET FILETERED FLUXES
# small difference in the pre-proessing fluxes
# bacause I am unsing entire time series there, not up to N = 2**14 points
fig, axes = plt.subplots(2, 1)
fig.suptitle('Effect of wavelet filtering on total fluxes')

# axes[0].plot(rdf['wc_wa_flux_ec'], rdf['sr_fwa_ec'], 's', label = 'H2O EC srfun')
# axes[0].plot(rdf['sr_fwa_ec'], rdf['sr_fwa_ec'], 'k', label = 'H2O EC srfun')
axes[0].plot(sdf['flux_ec', 'H2O'], rdf['sr_fwa_ec'], 's', label = 'H2O EC srfun')
axes[0].plot(sdf['flux_ec', 'H2O'], rdf['wc_wa_flux_ec'], 'o', label = 'H2O EC wcfun') # unfiltered
axes[0].plot(sdf['flux_ec', 'H2O'], rdf['wc_wa_flux_wt'], '.', label = 'H2O WA Filtered wcfun')
axes[0].plot(sdf['flux_ec', 'H2O'], sdf['flux_ec', 'H2O'], 'k')
axes[0].legend()
axes[0].set_ylabel('EC flux')
axes[0].set_xlabel('Wavelet flux')

axes[1].plot(sdf['flux_ec', 'CH4'], rdf['sr_fme_ec'], 's', label = 'CH4 EC srfun')
axes[1].plot(sdf['flux_ec', 'CH4'], rdf['wc_me_flux_ec'], 'o', label = 'CH4 EC wcfun') # unfiltered
axes[1].plot(sdf['flux_ec', 'CH4'], rdf['wc_me_flux_wt'], '.', label = 'CH4 WA Filtered wcfun') # wavelet filtered
axes[1].plot(sdf['flux_ec', 'CH4'], sdf['flux_ec', 'CH4'], 'k')
axes[1].set_ylabel('Wavelet flux')
axes[1].set_xlabel('EC flux')
axes[1].legend()

axes[0].annotate("a)", xy=(0.95, 0.05), xycoords="axes fraction")
axes[1].annotate("b)", xy=(0.95, 0.05), xycoords="axes fraction")
# plt.plot(rdf['sr_me_flux_ec'])
plt.xscale('log')
plt.yscale('log')
plt.savefig( os.path.join(md.outdir_plot, 'SI_filtering_effect_on_fluxes.png'))
plt.close()




########################################################################################################################

########################################################################################################################

############ PLOT PARTITION RESULTS:



# fig, axes = plt.subplots(1, 3, figsize = (15, 5))
# axes[0].plot( rdf['wc_en_frac_flux'].values, rdf['wf_en_frac_flux'].values, 'o', label = 'wf')
# axes[0].plot( rdf['wc_en_frac_flux'].values, rdf['w2_en_frac_flux'].values, 'o', label = 'w2')
# axes[0].plot( rdf['wc_en_frac_flux'].values, rdf['wc_en_frac_flux'].values, '--k')
# axes[0].set_xlabel('energetic flux fraction [wc]')
# axes[0].set_ylabel('energetic flux fraction')
# axes[1].plot( rdf['wc_en_frac_var'].values, rdf['wf_en_frac_var'].values, 'o', label = 'wf')
# axes[1].plot( rdf['wc_en_frac_var'].values, rdf['w2_en_frac_var'].values, 'o', label = 'w2')
# axes[1].plot( rdf['wc_en_frac_var'].values, rdf['wc_en_frac_var'].values, '--k')
# axes[1].set_xlabel('energetic variance fraction [wc]')
# axes[1].set_ylabel('energetic variance fraction')
# axes[2].plot( rdf['wc_en_frac_time'].values, rdf['wf_en_frac_time'].values, 'o', label = 'wf')
# axes[2].plot( rdf['wc_en_frac_time'].values, rdf['w2_en_frac_time'].values, 'o', label = 'w2')
# axes[2].plot( rdf['wc_en_frac_time'].values, rdf['wc_en_frac_time'].values, '--k')
# axes[2].set_xlabel('energetic time fraction [wc]')
# axes[2].set_ylabel('energetic time fraction')
#
# axes[0].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")
# axes[1].annotate("b)", xy=(0.1, 0.9), xycoords="axes fraction")
# axes[2].annotate("c)", xy=(0.1, 0.9), xycoords="axes fraction")
# axes[0].legend(loc='lower right')
# axes[1].legend(loc='lower right')
# axes[2].legend(loc='lower right')
# plt.tight_layout()
# plt.savefig( os.path.join(md.outdir_plot, 'SI_fractions_of_ener_flux_time.png'))
# plt.close()

# plt.figure()
# plt.plot( rdf['wc_wa_eT'].values, rdf['wc_me_eT'].values, 'o', label='CH4 wc')
# plt.plot( rdf['wc_wb_eT'].values, rdf['wc_ba_eT'].values, 'o', label='CH4 qr')
# plt.plot( rdf['wc_wa_eT'].values, rdf['wc_wa_eT'].values, '--k')
# plt.xlabel(r'$eT_{H_2O}$')
# plt.ylabel(r'$eT_{CH_4}$')
# plt.legend()
# plt.savefig(os.path.join(md.outdir_plot,
#          'partition_dir_vs_indir_flux_quadrant.png'))
# plt.close()


# plt.figure()
# plt.plot(np.abs(tdf['stab'].values), rdf['wc_wa_eT'].values,  'ob', label='H2O wc')
# plt.plot(np.abs(tdf['stab'].values), rdf['wc_me_eT'].values,  'og', label='CH4 wc')
# # plt.plot(np.abs(tdf['stab'].values), rdf['wc_ba_eT'].values/rdf['wc_wb_eT'].values,  'or', label='CH4 wc')
# # plt.plot(np.abs(tdf['stab'].values), 1*np.ones(np.shape(tdf)[0]),  'k', label='CH4 wc')
# # plt.plot(tdf['Re_star'].values, rdf['wc_me_eT'].values,  'og', label='CH4 wc')
# # plt.plot(tdf['Re_star'].values, rdf['wc_wb_eT'].values, 'o', label='CH4 qr')
# # plt.plot( rdf['wc_wa_eT'].values, tdf['stab'].values, '--k')
# # plt.xlabel(r'$eT_{H_2O}$')
# plt.xlabel(r'$|\zeta|$')
# plt.ylabel(r'$eT_{CH_4}$')
# plt.xscale('log')
# plt.legend()
# plt.savefig(os.path.join(md.outdir_plot,
#                          'partition_dir_vs_indir_flux_quadrant.png'))
# plt.close()

#
# plt.figure()
# plt.plot(rdf['wc_wa_flux_ec'].values, rdf['wc_wb_eT'].values,  'ob', label='H2O wc')
# plt.plot(rdf['wc_wa_flux_ec'].values, rdf['wc_ba_eT'].values,  'og', label='CH4 wc')
# # plt.plot(np.abs(tdf['stab'].values), rdf['wc_en_eT'].values,  'or', label='CH4 wc')
# # plt.plot(np.abs(tdf['stab'].values), rdf['wc_we_eT'].values,  'oy', label='CH4 wc')
# # plt.plot(np.abs(tdf['stab'].values), rdf['wc_ba_eT'].values/rdf['wc_wb_eT'].values,  'or', label='CH4 wc')
# # plt.plot(np.abs(tdf['stab'].values), 1*np.ones(np.shape(tdf)[0]),  'k', label='CH4 wc')
# # plt.plot(tdf['Re_star'].values, rdf['wc_me_eT'].values,  'og', label='CH4 wc')
# # plt.plot(tdf['Re_star'].values, rdf['wc_wb_eT'].values, 'o', label='CH4 qr')
# # plt.plot( rdf['wc_wa_eT'].values, tdf['stab'].values, '--k')
# # plt.xlabel(r'$eT_{H_2O}$')
# plt.xlabel(r'$F_{H2O}$')
# plt.ylabel(r'$eT_{CH_4}$')
# # plt.xscale('log')
# plt.legend()
# plt.savefig(os.path.join(md.outdir_plot,
#                          'partition_dir_vs_indir_flux_quadrant.png'))

# check they are all above the limit
print(np.min(rdf['wc_wa_eT'].values))
print(np.min(rdf['wc_me_eT'].values))
print(np.min(rdf['wc_ba_eT'].values))
print(np.min(rdf['wc_en_eT'].values))
fig, axes = plt.subplots(2, 3, figsize = (15, 10))
axes[0,0].plot(rdf['wc_wa_eT'].values, rdf['wc_me_eT'].values,  'ob', alpha = 0.6, markersize = 5)
axes[0,0].plot(rdf['wc_wa_eT'].values, rdf['wc_wa_eT'].values,  'k')
axes[0,0].set_ylim([0.4, 1])
axes[0,0].set_xlabel(r'$|\zeta|$')
axes[0,0].set_ylabel(r'$e_T$')
axes[0,0].set_title('Total flux')
axes[0,0].set_xlabel(r'$e_T$ $H_2O$')
axes[0,0].set_ylabel(r'$e_T$ $CH_4$')
axes[0, 0].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")


axes[0,1].plot(rdf['wc_wb_eT'].values, rdf['wc_ba_eT'].values,  'ob', alpha = 0.6, markersize = 5)
axes[0,1].plot(rdf['wc_wb_eT'].values, rdf['wc_wb_eT'].values, 'k')
axes[0,1].set_ylim([0.4, 1])
axes[0,1].set_title('Background')
axes[0,1].set_xlabel(r'$e_T$ $H_2O$')
axes[0,1].set_ylabel(r'$e_T$ $CH_4$')
axes[0, 1].annotate("b)", xy=(0.1, 0.9), xycoords="axes fraction")

axes[0, 2].plot(rdf['wc_we_eT'].values, rdf['wc_en_eT'].values,  'ob', alpha = 0.6, markersize = 5)
axes[0, 2].plot(rdf['wc_we_eT'].values, rdf['wc_we_eT'].values,  'k')
axes[0, 2].set_ylim([0.4, 1])
axes[0, 2].set_xlabel(r'$e_T$ $H_2O$')
axes[0, 2].set_ylabel(r'$e_T$ $CH_4$')
axes[0, 2].set_title('Hotspot')
axes[0, 2].annotate("c)", xy=(0.1, 0.9), xycoords="axes fraction")



axes[1, 0].plot(np.abs(tdf['stab'].values), rdf['wc_wa_eT'].values,  'ob', label=r'$H_2O$', alpha = 0.6, markersize = 5)
axes[1, 0].plot(np.abs(tdf['stab'].values), rdf['wc_me_eT'].values,  'sg', label=r'$CH_4$', alpha = 0.6, markersize = 5)
axes[1, 0].set_xscale('log')
axes[1, 0].set_ylim([0.4, 1])
axes[1, 0].set_xlabel(r'$|\zeta|$')
axes[1, 0].set_ylabel(r'$e_T$')
axes[1, 0].annotate("d)", xy=(0.1, 0.9), xycoords="axes fraction")

axes[1, 1].plot(np.abs(tdf['stab'].values), rdf['wc_wb_eT'].values,  'ob', label=r'$H_2O$', alpha = 0.6, markersize = 5)
axes[1, 1].plot(np.abs(tdf['stab'].values), rdf['wc_ba_eT'].values,  'sg', label=r'$CH_4$', alpha = 0.6, markersize = 5)
axes[1, 1].set_xscale('log')
axes[1, 1].set_ylim([0.4, 1])
axes[1, 1].set_xlabel(r'$|\zeta|$')
axes[1, 1].annotate("e)", xy=(0.1, 0.9), xycoords="axes fraction")

axes[1, 2].plot(np.abs(tdf['stab'].values), rdf['wc_we_eT'].values,  'ob', label=r'$H_2O$', alpha = 0.6, markersize = 5)
axes[1, 2].plot(np.abs(tdf['stab'].values), rdf['wc_en_eT'].values,  'sg', label=r'$CH_4$', alpha = 0.6, markersize = 5)
axes[1, 2].set_xscale('log')
axes[1, 2].set_ylim([0.4, 1])
axes[1, 2].set_xlabel(r'$|\zeta|$')
axes[1, 2].annotate("f)", xy=(0.1, 0.9), xycoords="axes fraction")
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot,
                         'partition_dir_vs_indir_flux_quadrant.png'), dpi = 300)
plt.close()


# fig, axes = plt.subplots(1, 3, figsize = (15, 5))
# axes[0].plot(np.abs(tdf['Re_star'].values), rdf['wc_wa_eT'].values,  'ob', label='H2O wc')
# axes[0].plot(np.abs(tdf['Re_star'].values), rdf['wc_me_eT'].values,  'og', label='CH4 wc')
# axes[0].set_xscale('log')
# axes[0].set_ylim([0.5, 1])
# axes[0].set_xlabel(r'$Re_*$')
# axes[0].set_ylabel(r'$e_T$')
# axes[0].set_title('Total flux')
#
#
# axes[1].plot(np.abs(tdf['Re_star'].values), rdf['wc_wb_eT'].values,  'ob', label='H2O wc')
# axes[1].plot(np.abs(tdf['Re_star'].values), rdf['wc_ba_eT'].values,  'og', label='CH4 wc')
# axes[1].set_xscale('log')
# axes[1].set_ylim([0.5, 1])
# axes[1].set_xlabel(r'$Re_*$')
# axes[1].set_title('Background')
#
# axes[2].plot(np.abs(tdf['Re_star'].values), rdf['wc_we_eT'].values,  'ob', label='H2O wc')
# axes[2].plot(np.abs(tdf['Re_star'].values), rdf['wc_en_eT'].values,  'og', label='CH4 wc')
# axes[2].set_xscale('log')
# axes[2].set_ylim([0.5, 1])
# axes[2].set_xlabel(r'$Re_*$')
# axes[2].set_title('Ebullition')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(md.outdir_plot,
#                          'partition_dir_vs_indir_flux_quadrant_Restar.png'))
# plt.close()
#
# def plot_moments(partition = 'wc'):
#     fig, axes = plt.subplots(ncols, nrows, figsize = (10, 15))
#
#     for j in range(ncols):
#         for i in range(nrows):
#             mymoment = mymoments[ (j-1)*nrows + i ]
#             print(mymoment)
#             print(mymoment in ['M40', 'M30'])
#
#             axes[j,i].scatter(rdf['wa_{}'.format(mymoment)],
#                   rdf['me_{}'.format(mymoment)], label = 'CH4')
#             if mymoment == 'M40':
#                     # if mymoment in ['M40', 'M30']:
#                 axes[j, i].scatter(np.abs(rdf['wa_{}'.format(mymoment)]),
#                        np.abs(rdf['{}_ba_{}'.format(partition, mymoment)]),
#                        label=partition)
#                 axes[j,i].set_xscale('log')
#                 axes[j,i].set_yscale('log')
#                 axes[j, i].set_title(r'$|{}|$'.format(mymoment))
#             else:
#                 axes[j,i].scatter(rdf['wa_{}'.format(mymoment)],
#                       rdf['{}_ba_{}'.format(partition, mymoment)],
#                       label = partition)
#                 axes[j,i].set_title(r'$ {} $'.format(mymoment))
#             axes[j,i].plot(rdf['wa_{}'.format(mymoment)],
#                   rdf['wa_{}'.format(mymoment)], 'k')
#             axes[j,i].set_xlabel('H2O')
#             axes[j,i].set_ylabel('CH4')
#     axes[0,0].legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(md.outdir_plot, 'partition_moments_{}.png'.format(partition)))
#     # plt.show()
#     plt.close()



def plot_moments_2(partition = 'wc'):
    mymoments = ['M30', 'M40', 'M21', 'M12', 'M13', 'Rcw']
    letters = np.array([["a)", "b)"],["c)", "d)"], ["e)", "f)"]])
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(ncols, nrows, figsize = (10, 15))

    for j in range(ncols):
        for i in range(nrows):
            mymoment = mymoments[ (j-1)*nrows + i ]
            print(mymoment)
            print(mymoment in ['M40', 'M30'])

            axes[j,i].plot(rdf['{}_wa_{}'.format(partition, mymoment)],
                              rdf['{}_me_{}'.format(partition, mymoment)],
            # '^b', label = partition, alpha = 0.6, markersize = 5, label = 'CH4')
            '^b', label = partition, alpha = 0.6, markersize = 5)
            if mymoment in ['M40', 'M30']:
                axes[j, i].plot(np.abs(rdf['{}_bar_{}'.format(partition, mymoment)]),
                                   np.abs(rdf['{}_ba_{}'.format(partition, mymoment)]),
                                   'or' , label=partition, alpha = 0.6, markersize = 5)
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('log')
                axes[j, i].set_title(r'$|{}|$'.format(mymoment))
            else:
                axes[j,i].plot(rdf['{}_bar_{}'.format(partition, mymoment)],
                                  rdf['{}_ba_{}'.format(partition, mymoment)],
                                  'or', label = partition,  alpha = 0.6, markersize = 5)
                axes[j,i].set_title(r'$ {} $'.format(mymoment))

            axes[j, i].annotate(letters[j, i], xy=(0.05, 0.91), xycoords="axes fraction")
            axes[j,i].plot(rdf['{}_wa_{}'.format(partition, mymoment)],
                           rdf['{}_wa_{}'.format(partition, mymoment)], 'k')
            axes[j,i].set_xlabel('H2O')
            axes[j,i].set_ylabel('CH4')
    # axes[0,0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(md.outdir_plot, 'partition_moments_2_{}.png'.format(partition)), dpi = 300)
    # plt.show()
    # plt.close()

# plot moments for the three partition methods
plot_moments_2(partition='wc')
# plot_moments_2(partition='wf')
# plot_moments_2(partition='w2')



def plot_moments_3(partition = 'wc'):
    mymoments = ['M30', 'M21', 'M12', 'M13']
    letters = np.array([["a)", "b)"],["c)", "d)"]])
    # mylabels = np.array([['30', '$M_{21}$'],['$M_{12}$', '$M_{13}$']])
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(ncols, nrows, figsize = (8, 8))

    for j in range(ncols):
        for i in range(nrows):
            mymoment = mymoments[ (j-1)*nrows + i ]
            # mylabel = mylabels[ (j-1)*nrows + i ]
            print(mymoment)
            mylabel = r'M_{{{}}}'.format(mymoment[1:])
            print(mylabel)
            print(mymoment in ['M40', 'M30'])

            axes[j,i].plot(rdf['{}_wa_{}'.format(partition, mymoment)],
                           rdf['{}_me_{}'.format(partition, mymoment)],
                           # '^b', label = partition, alpha = 0.6, markersize = 5, label = 'CH4')
                           '^b', label = partition, alpha = 0.6, markersize = 4)
            if mymoment in ['M40', 'M30']:
                axes[j, i].plot(np.abs(rdf['{}_bar_{}'.format(partition, mymoment)]),
                                np.abs(rdf['{}_ba_{}'.format(partition, mymoment)]),
                                'or' , label=partition, alpha = 0.6, markersize = 4)
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('log')
                axes[j, i].set_title(r'$|{}|$'.format(mylabel))
            else:
                axes[j,i].plot(rdf['{}_bar_{}'.format(partition, mymoment)],
                               rdf['{}_ba_{}'.format(partition, mymoment)],
                               'or', label = partition,  alpha = 0.6, markersize = 4)
                axes[j,i].set_title(r'$ {} $'.format(mylabel))

            axes[j, i].annotate(letters[j, i], xy=(0.05, 0.86), xycoords="axes fraction")
            axes[j,i].plot(rdf['{}_wa_{}'.format(partition, mymoment)],
                           rdf['{}_wa_{}'.format(partition, mymoment)], 'k')
            axes[j,i].set_xlabel(r'$H_2O$')
            axes[j,i].set_ylabel(r'$CH_4$')
    # axes[0,0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(md.outdir_plot, 'partition_moments_3_{}.png'.format(partition)), dpi = 300)
    # plt.show()
    # plt.close()


plot_moments_3(partition='wc')

# plot_moments_3(partition='wd')


plot_densities = False
if plot_densities:
    pdf = pd.read_pickle( os.path.join(md.outdir_data, 'results_pdf.csv'))
    plt.figure()
    mindensity = 1e-5
    ndecades1 = 3
    ndecades2 = 6
    for i in range(nruns):
        print(i)
        x_me = pdf['x_me'].iloc[i]
        x_wa = pdf['x_wa'].iloc[i]
        x_en = pdf['x_en'].iloc[i]
        x_ba = pdf['x_ba'].iloc[i]
        p_me = pdf['p_me'].iloc[i]
        p_wa = pdf['p_wa'].iloc[i]
        p_en = pdf['p_en'].iloc[i]
        p_ba = pdf['p_ba'].iloc[i]

        x_me[p_me < mindensity] = np.nan
        x_wa[p_wa < mindensity] = np.nan
        x_en[p_en < mindensity] = np.nan
        x_ba[p_ba < mindensity] = np.nan
        p_me[p_me < mindensity] = np.nan
        p_wa[p_wa < mindensity] = np.nan
        p_en[p_en < mindensity] = np.nan
        p_ba[p_ba < mindensity] = np.nan

        if i == 0:
            plt.plot(x_en, 10**( ndecades2 + np.log10(p_en)), '.r',
                     label='CH4 ener')
            plt.plot(x_me, 10**( ndecades1 + np.log10(p_me)), '.g',
                     label='CH4 ')
            plt.plot(x_wa, p_wa, '.b',
                     label='H2O')
            plt.plot(x_ba, 10**( -ndecades1 + np.log10(p_ba)), '.c',
                     label='CH4 back')
        else:
            plt.plot(x_en, 10 ** (ndecades2 + np.log10(p_en)), '.r')
            plt.plot(x_me, 10 ** (ndecades1 + np.log10(p_me)), '.g')
            plt.plot(x_wa, p_wa, '.b')
            plt.plot(x_ba, 10 ** (-ndecades1 + np.log10(p_ba)), '.c')
    plt.plot([0,0], [1e-8, 1e8], '--k')
    # plt.plot(XXXME[i,:], PDFME[i,:], '-og')
    plt.yscale('log')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(md.outdir_plot, 'partition_pdfs.png'))
    plt.close()


# qc_min = np.min(rdf['qc_stdv_c2'].values)
# wc_min = np.min(rdf['wc_stdv_c'].values)
# # qf_min = np.min(rdf['qf_stdv_cw'].values)
# wf_min = np.min(rdf['wf_stdv_cw'].values)
# w2_min = np.min(rdf['w2_stdv_c2'].values)
# # print(qc_min)
# print(wc_min)
# # print(qf_min)
# print(wf_min)
# print(w2_min)
#
# list(rdf.keys())


# plt.figure(figsize = (10, 10))
# plt.plot( rdf['qr_R_me_wa'].values, rdf['wc_stdv_c'].values, 'ob',  label = 'c [wavelet scalar]')
# plt.plot( rdf['qr_R_me_wa'].values, rdf['wf_stdv_cw'].values, 'or', label = 'f [wavelet flux]')
# plt.plot( rdf['qr_R_me_wa'].values, rdf['w2_stdv_c2'].values, 'oc', label = 'c2 [wavelet energy]')
# lim_wc = 0.2
# lim_wf = 0.8
# lim_w2 = 3.0
# plt.plot( rdf['qr_R_me_wa'].values, lim_wc*np.ones(nruns), '--k')
# plt.plot( rdf['qr_R_me_wa'].values, lim_wf*np.ones(nruns), '--k')
# plt.plot( rdf['qr_R_me_wa'].values, lim_w2*np.ones(nruns), '--k')
# plt.xlabel(r'correlation $R_{cr}$ between scalars $CH_4$ and $H_2O$')
# plt.ylabel(r'Standard deviation of wavelet coeffient differences')
# plt.text(0.6, lim_wc * 1.05, r'{}'.format(lim_wc), fontsize = 15)
# plt.text(0.6, lim_wf * 1.05, r'{}'.format(lim_wf), fontsize = 15)
# plt.text(0.6, lim_w2 * 1.05, r'{}'.format(lim_w2),  fontsize = 15)
# corrval = 0.98
# plt.plot([corrval, corrval], [0, 35], '--k')
# plt.text(0.93, 40, 'correlation \n {}'.format(corrval), fontsize=15)
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(title='Partition scheme')
# plt.savefig(os.path.join(md.outdir_plot, 'partition_stdvs_98.png'))
# plt.close()




fig, axes = plt.subplots(1, 2, figsize = (13, 6))
axes[0].plot( rdf['wc_Rcr'].values, rdf['wc_stdv_c'].values, 'o',
              label = 'c [wavelet scalar]', alpha = 0.7, markersize = 8)
lim_wc = 0.3
lim_wf = 0.8
lim_w2 = 3.0
# axes[0].plot( rdf['qr_R_me_wa'].values, lim_wc*np.ones(nruns), '--k')
# axes[0].plot( rdf['qr_R_me_wa'].values, rdf['wc_my_stdv']*np.ones(nruns), '--k')
# xmin, xmax, ymin, ymax = plt.axis()
axes[0].set_xlabel(r'correlation $R_{cr}$ between $CH_4$ and $H_2O$')
axes[0].set_ylabel(r'Standard deviation of $| \Delta WT^{(m)}[i] |$')
# axes[0].set_ylabel(r'Standard deviation of $\lvert \Delta WT^{(m)}[i]\rvert$')
axes[0].set_ylim([0, 0.2 + np.max(rdf['wc_stdv_c'].values)])

xmin, xmax = axes[0].get_xlim()
# axes[1].plot( x, rdf['wc_my_stdv']*np.ones(nruns), '--k')
axes[0].plot( [xmin, xmax], [rdf['wc_my_stdv'], rdf['wc_my_stdv']], '--k')
axes[0].set_xlim([xmin , xmax])
axes[0].annotate("a)", xy=(0.90, 0.90), xycoords="axes fraction")

y = rdf['wc_stdv_c'].values
# x = rdf['me_M30'].values-rdf['wa_M30'].values
x = (rdf['wc_me_M30'].values-rdf['wc_wa_M30'].values)
z = (rdf['wc_ba_M30'].values-rdf['wc_bar_M30'].values)
# z = np.polyfit(x, y, 1)
# xv = np.linspace(np.min(x), np.max(x), 100)
# yv = z[0]*xv + z[1]
axes[1].plot(x, y, 'o',  label = 'c [wavelet scalar]', alpha = 0.7, markersize = 8)
# axes[1].plot(z, y, 'or',  label = 'c [wavelet scalar]', alpha = 0.6, markersize = 8)
# axes[1].plot(-x, y, 'or',  label = 'c [wavelet scalar]', alpha = 0.8, markersize = 8)
# axes[1].plot( x, lim_wc*np.ones(nruns), '--k')
xmin, xmax = axes[1].get_xlim()
axes[1].plot( [xmin, xmax + 1], [rdf['wc_my_stdv'], rdf['wc_my_stdv']], '--k')
axes[1].set_xlim([xmin , xmax + 1])
# plt.plot(xv, yv, '--k')
# plt.plot(  [0., 0.], [ymin, ymax], '--k')
axes[1].plot(  [0., 0.], [0, np.max(rdf['wc_stdv_c'].values)+0.2], '--k')
axes[1].set_ylim([0, np.max(rdf['wc_stdv_c'].values)+0.2])
axes[1].set_xlabel(r'$M_{30, CH_4} - M_{30, H_2O}$')
axes[1].set_xscale('symlog')
# axes[1].set_ylabel(r'Standard deviation of wavelet coefficient differences')
axes[1].set_ylabel(r'Standard deviation of $ | \Delta WT^{(m)}[i] | $')
axes[1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")

plt.savefig(os.path.join(md.outdir_plot, 'partition_stdvs.png'), dpi = 300)
plt.close()


# fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# # plt.plot(rdf['ts_b'], rdf['ts_s'], 'o')
# # plt.plot(rdf['ts_b'], rdf['sr_fme_ec']*rdf['wc_en_frac_flux'], 'o')
# # plt.plot(rdf['ts_b'], rdf['wc_en_frac_flux'], 'o')
# axes[0].plot(1/rdf['ts_l'], rdf['wc_en_frac_flux'], 'o', alpha = 0.8, markersize = 8)
# # plt.plot(rdf['ts_b'], 1/20*rdf['ts_b'], 'k')
# # plt.xscale('log')
# # plt.yscale('log')
# axes[0].set_ylabel('Fraction of Ebullition Flux')
# axes[0].set_xlabel('Mean time between ebullition events [s]')
# axes[0].annotate("a)", xy=(0.9, 0.90), xycoords="axes fraction")
#
# axes[1].plot(rdf['wc_en_frac_time'], rdf['wc_en_frac_flux'], 'o', alpha = 0.8, markersize = 8)
# axes[1].set_ylabel('Fraction of Ebullition Flux')
# axes[1].set_xlabel('Fraction of active area')
# axes[1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
#
# plt.savefig(os.path.join(md.outdir_plot, 'ebull_time_scale.png'), dpi = 300)
# plt.close()


# def phim(stabv):
#     ''' stability correction function for momentum
#     From Katul et al, PRL, 2011'''
#     # if stab > 1:
#     #    print('turb_quant WARNING:: very stable - using constant phim')
#     #    return 5.47 # check
#     n = np.size(stabv)
#     myphim = np.zeros(n)
#     for i in range(n):
#         if stabv[i] > 0: # stable
#             myphim[i] =  1 + 4.7*stabv[i]
#         else:
#             myphim[i] =  (1-15*stabv[i])**(-1/4)
#     return myphim

# shear_ts = rdf['sr_ustar']/0.4/md.z_ref*phim(rdf['sr_stab'].values)
# ener_ts = rdf['sr_ustar']/0.4/md.z_ref*phim(rdf['sr_stab'].values)

# fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# xxd, yyd = mf.mylogslope(rdf['ts_shear_ts'], 1/rdf['ts_l'].values, slope=-2)
# axes[0].plot(rdf['ts_shear_ts'], 1/rdf['ts_l'], 'o', label = 'runs', alpha = 0.8, markersize = 8)
# # axes[0].plot(xxd, yyd, 'k', label = r'slope $-1$')
# axes[0].set_xlabel(r'Shear time scale [$s$]')
# axes[0].set_ylabel(r'Frequency of ebullition events [$s^{-1}$]')
#
# axes[1].plot(rdf['ts_diss_ts'], 1/rdf['ts_l'], 'o', label = 'runs', alpha = 0.8, markersize = 8)
# axes[1].set_xlabel(r'Dissipation time scale [$s$]')
# axes[1].set_ylabel(r'Frequency of ebullition events [$s^{-1}$]')
# # axes[1].set_xscale('log')
# # axes[1].set_yscale('log')
# # axes[1].legend()
# plt.savefig(os.path.join(md.outdir_plot, 'shear_ts_vs_ebull_time_scale.png'))
# plt.close()




# fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# # xxd, yyd = mf.mylogslope(rdf['ts_shear_ts'], rdf['ts_l'].values, slope=1)
# axes[0].plot(rdf['ts_shear_ts'], rdf['ts_l'], 'o', label = 'runs', alpha = 0.8, markersize = 8)
# # axes[0].plot(xxd, yyd, 'k', label = r'slope $1$')
# axes[0].set_xlabel(r'Shear time scale [$s$]')
# # axes[0].set_ylabel(r'Frequency of ebullition events [$s^{-1}$]')
# axes[0].set_ylabel(r'Frequency of ebullition events [$s^{-1}$]')
# axes[0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
# # axes[0].set_xscale('log')
# # axes[0].set_yscale('log')
#
# # xxd, yyd = mf.mylogslope(rdf['ts_diss_ts'], rdf['ts_l'].values, slope=-1)
# axes[1].plot(rdf['ts_diss_ts'], rdf['ts_l'], 'o', label = 'runs', alpha = 0.8, markersize = 8)
# # axes[1].plot(xxd, yyd, 'k', label = r'slope $1$')
# axes[1].set_xlabel(r'Dissipation time scale [$s$]')
# axes[1].set_ylabel(r'Frequency of ebullition events [$s^{-1}$]')
# axes[1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
# # axes[1].set_xscale('log')
# # axes[1].set_yscale('log')
# # axes[1].legend()
#
# # axes[2].plot(rdf['ts_Tw'], rdf['ts_l'], 'o', label = 'runs', alpha = 0.8, markersize = 8)
# # axes[2].set_xlabel(r'Dissipation time scale [$s$]')
# # axes[2].set_ylabel(r'$T_w$ [$s$]')
# # axes[1].set_xscale('log')
# # axes[1].set_yscale('log')
# # axes[1].legend()
# plt.savefig(os.path.join(md.outdir_plot, 'shear_ts_vs_ebull_time_scale.png'), dpi = 300)
# plt.close()


# plt.figure(figsize=(8, 8))
# xxd, yyd = mf.mylogslope(rdf['ts_diss_ts'], rdf['sr_alpha'].values, slope=-1)
# plt.plot(rdf['ts_diss_ts'], rdf['sr_alpha'], 'o', label = 'runs')
# plt.plot(xxd, yyd, 'k', label = r'slope $-1$')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Shear time scale [s]')
# plt.ylabel('Mean time between ebullition events [s]')
# plt.legend()
# plt.show()
# plt.savefig(os.path.join(md.outdir_plot, 'shear_ts_vs_ebull_time_scale.png'))
# plt.close()

# plt.figure()
# plt.plot(1/rdf['ts_l'], rdf['sr_fme_ec']*rdf['wc_en_frac_flux'], 'o')
# # plt.plot()
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# plt.figure()
# plt.plot(rdf['sr_fme_ec'], rdf['sr_ustar'], 'o')
# plt.show()

# plt.figure()
# # plt.plot(rdf['ts_l'], rdf['wc_en_frac_time'], 'o')
# plt.plot(1/rdf['ts_l'], rdf['wc_en_frac_time'], 'o')
# # plt.plot()
# # plt.xscale('log')
# # plt.yscale('log')
# plt.show()






# plot
# myvars = []

fig, axes = plt.subplots(1, 2, figsize=(10,6))
print(np.max(rdf['sr_fme_sr']))
axes[0].plot(rdf['sr_fme_ec'], rdf['sr_fme_sr'], 'ok', alpha = 0.3, markersize = 6)
axes[0].plot(rdf['sr_fme_ec']*rdf['wc_en_frac_flux'], rdf['sr_fen_sr'], 'or', alpha = 0.6, markersize = 6)
axes[0].plot(rdf['sr_fen_sr'], rdf['sr_fen_sr'], 'k')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title(r'$CH_4$ hotspot')
axes[0].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")
axes[0].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0].set_ylim([8E-5*40, 12E-2*50])
axes[0].set_xlim([8E-5*40, 12E-2*50])
axes[0].plot([1E-4*40, 8E-2*50], [1E-4*40, 8E-2*50], 'k')
axes[1].plot(rdf['sr_fme_ec'], rdf['sr_fme_sr'], 'ok', alpha = 0.3, markersize = 6)
axes[1].plot(rdf['sr_fme_ec']*rdf['wc_ba_frac_flux'], rdf['sr_fba_sr'], 'oc', alpha = 0.6, markersize = 6)
axes[1].plot(rdf['sr_fba_sr'], rdf['sr_fba_sr'], 'k')
# axes[1].plot(rdf['sr_fen'], rdf['sr_fen'], 'k')
axes[1].annotate("b)", xy=(0.1, 0.9), xycoords="axes fraction")
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title(r'$CH_4$ background')

axes[1].set_ylim([8E-5*40, 12E-2*50])
axes[1].set_xlim([8E-5*40, 12E-2*50])
axes[1].plot([1E-4*40, 8E-2*50], [1E-4*40, 8E-2*50], 'k')
# axes[1].set_ylim([8E-5, 12E-2])
# axes[1].set_xlim([8E-5, 12E-2])
axes[1].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
# axes[1].axes.get_yaxis().set_visible(False)
# axes[1].set_ylabel('SR flux')
# axes[1].plot([1E-4, 8E-2], [1E-4, 8E-2], 'k')
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'sr_fluxes.png'), dpi = 300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes[0,0].plot(rdf['sr_fme_ec'], rdf['sr_fme_sr'], 'og', label = 'SR')
axes[0,0].plot(rdf['sr_fme_ec'], rdf['sr_fen_sr'] + rdf['sr_fba_sr'] , '.k', label = 'ISR')
# axes[0,0].plot(rdf['sr_fme_ec'], rdf['sr_fme_ec'], 'k')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
# axes[0,0].set_xlabel('Eddy covariance flux')
# axes[0,0].set_ylabel('Surface renewal flux')
axes[0, 0].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0, 0].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0,0].legend(loc='lower right')
axes[0,0].set_title(r'$CH_4$')
axes[0, 0].set_ylim([1E-4*40, 12E-2*50])
axes[0, 0].set_xlim([1E-4*40, 12E-2*50])
axes[0, 0].plot(    [1E-4*40,  8E-2*50], [1E-4*40, 8E-2*50], 'k')
axes[0,1].plot(rdf['sr_fwa_ec'], rdf['sr_fwa_sr'], 'ob')
axes[0,1].plot(rdf['sr_fwa_ec'], rdf['sr_fwa_ec'], 'k')
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
axes[0,1].set_title(r'$H_2O$')
axes[1,0].plot(rdf['sr_fme_ec']*rdf['wc_en_frac_flux'], rdf['sr_fen_sr'], 'or')

# axes[0,1].set_xlabel('Eddy covariance flux')
# axes[0,1].set_ylabel('Surface renewal flux')

axes[0, 1].set_xlabel(r'EC flux [$m mol\, m^{-2}\, s^{-1}$]')
axes[0, 1].set_ylabel(r'SR flux [$m mol\, m^{-2}\, s^{-1}$]')
# axes[1,0].plot(rdf['sr_fen'], rdf['sr_fen'], 'k')
axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')
axes[1,0].set_title(r'$CH_4$ hotspot')
# axes[1,0].set_ylim([1E-4, 12E-2])
# axes[1,0].set_xlim([1E-4, 12E-2])
# axes[1, 0].plot([1E-4, 8E-2], [1E-4, 8E-2], 'k')

axes[1, 0].set_ylim([1E-4*40, 12E-2*50])
axes[1, 0].set_xlim([1E-4*40, 12E-2*50])
axes[1, 0].plot(    [1E-4*40,  8E-2*50], [1E-4*40, 8E-2*50], 'k')

axes[1, 0].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[1, 0].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')

axes[1,1].plot(rdf['sr_fme_ec']*rdf['wc_ba_frac_flux'], rdf['sr_fba_sr'], 'oc')
# axes[1,1].plot(rdf['sr_fba'], rdf['sr_fba'], 'k')
axes[1,1].set_xscale('log')
axes[1,1].set_yscale('log')
axes[1,1].set_title(r'$CH_4$ background')
# axes[1,1].set_ylim([1E-4, 12E-2])
# axes[1,1].set_xlim([1E-4, 12E-2])
# axes[1, 1].plot([1E-4, 8E-2], [1E-4, 8E-2], 'k')


axes[1, 1].set_ylim([1E-4*40, 12E-2*50])
axes[1, 1].set_xlim([1E-4*40, 12E-2*50])
axes[1, 1].plot(    [1E-4*40,  8E-2*50], [1E-4*40, 8E-2*50], 'k')

# axes[1,1].set_xlabel('Eddy covariance flux')
# axes[1,1].set_ylabel('Surface renewal flux')

axes[1, 1].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[1, 1].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')

axes[0, 0].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")
axes[0, 1].annotate("b)", xy=(0.1, 0.9), xycoords="axes fraction")
axes[1, 0].annotate("c)", xy=(0.1, 0.9), xycoords="axes fraction")
axes[1, 1].annotate("d)", xy=(0.1, 0.9), xycoords="axes fraction")

plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'SI_sr_fluxes_4.png'))
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(10,10))
xx, yy = mf.mylogslope(rdf['sr_Re_star'],rdf['sr_da_me'], slope = -1/4 )
axes[0,0].plot(rdf['sr_Re_star'], rdf['sr_da_me'], 'og')
axes[0,0].plot(xx, yy, 'k')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_xlabel(r'$Re_{*}$')
axes[0,0].set_ylabel(r'$Da$')
axes[0,0].set_title(r'$CH_4$')

xx, yy = mf.mylogslope(rdf['sr_Re_star'],rdf['sr_da_wa'], slope = -1/4 )
axes[0,1].plot(rdf['sr_Re_star'], rdf['sr_da_wa'], 'ob')
axes[0,1].plot(xx, yy, 'k')
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
axes[0,1].set_xlabel('$Re_{*}$')
axes[0,1].set_ylabel(r'$Da$')
axes[0,1].set_title(r'$H_2O$')

xx, yy = mf.mylogslope(rdf['sr_Re_star'],rdf['sr_da_en'], slope = -1/4 )
axes[1,0].plot(rdf['sr_Re_star'], rdf['sr_da_en'], 'or')
axes[1,0].plot(xx, yy, 'k')
axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')
axes[1,0].set_xlabel('$Re_{*}$')
axes[1,0].set_ylabel(r'$Da$')
axes[1,0].set_title(r'$CH_4$ ebullition')

xx, yy = mf.mylogslope(rdf['sr_Re_star'],rdf['sr_da_ba'], slope = -1/4 )
axes[1,1].plot(rdf['sr_Re_star'], rdf['sr_da_ba'], 'oc')
axes[1,1].plot(xx, yy, 'k')
axes[1,1].set_xscale('log')
axes[1,1].set_yscale('log')
axes[1,1].set_xlabel('$Re_{*}$')
axes[1,1].set_ylabel(r'$Da$')
axes[1,1].set_title(r'$CH_4$ background')

axes[0, 0].annotate("a)", xy=(0.9, 0.1), xycoords="axes fraction")
axes[0, 1].annotate("b)", xy=(0.9, 0.1), xycoords="axes fraction")
axes[1, 0].annotate("c)", xy=(0.9, 0.1), xycoords="axes fraction")
axes[1, 1].annotate("d)", xy=(0.9, 0.1), xycoords="axes fraction")
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'SI_sr_DaRe.png'))
plt.close()


# fig, axes = plt.subplots(2, 2, figsize=(10,10))
# xx, yy = mf.mylogslope(rdf['sr_Re_stab'],rdf['sr_da_me'], slope = -1/4 )
# axes[0,0].plot(rdf['sr_Re_stab'], rdf['sr_da_me'], 'og')
# axes[0,0].plot(xx, yy, 'k')
# axes[0,0].set_xscale('log')
# axes[0,0].set_yscale('log')
# axes[0,0].set_xlabel(r'$Re_{*}$')
# axes[0,0].set_ylabel('Da')
# axes[0,0].set_title(r'$CH4$')
#
# xx, yy = mf.mylogslope(rdf['sr_Re_stab'],rdf['sr_da_wa'], slope = -1/4 )
# axes[0,1].plot(rdf['sr_Re_stab'], rdf['sr_da_wa'], 'ob')
# axes[0,1].plot(xx, yy, 'k')
# axes[0,1].set_xscale('log')
# axes[0,1].set_yscale('log')
# axes[0,1].set_xlabel('$Re_{*}$')
# axes[0,1].set_ylabel('Da')
# axes[0,1].set_title(r'$H2O$')
#
# xx, yy = mf.mylogslope(rdf['sr_Re_stab'],rdf['sr_da_en'], slope = -1/4 )
# axes[1,0].plot(rdf['sr_Re_stab'], rdf['sr_da_en'], 'or')
# axes[1,0].plot(xx, yy, 'k')
# axes[1,0].set_xscale('log')
# axes[1,0].set_yscale('log')
# axes[1,0].set_xlabel('$Re_{*}$')
# axes[1,0].set_ylabel('Da')
# axes[1,0].set_title(r'$CH4$ Ebullition')
#
# xx, yy = mf.mylogslope(rdf['sr_Re_stab'],rdf['sr_da_ba'], slope = -1/4 )
# axes[1,1].plot(rdf['sr_Re_stab'], rdf['sr_da_ba'], 'oc')
# axes[1,1].plot(xx, yy, 'k')
# axes[1,1].set_xscale('log')
# axes[1,1].set_yscale('log')
# axes[1,1].set_xlabel('$Re_{*}$')
# axes[1,1].set_ylabel('Da')
# axes[1,1].set_title(r'$CH4$ Background')
# plt.tight_layout()
# plt.savefig(os.path.join(md.outdir_plot, 'sr_DaReStab.png'), dpi = 300)
# plt.close()

# write as a function of actual date
# plt.figure()
# # plt.plot(df['wc_en_frac_flux'], rdf['wc_me_flux_ec'], 'o')
# plt.title('Energetic fraction with different methods')
# plt.plot(rdf['wc_me_flux_ec'], 'g', label = 'total CH4')
# plt.plot(rdf['wc_en_frac_flux']*rdf['wc_me_flux_ec'], 'r', label = 'ener frac wc')
# plt.plot(rdf['w2_en_frac_flux']*rdf['w2_me_flux_ec'], 'orange', label = 'ener frac w2')
# plt.plot(rdf['wf_en_frac_flux']*rdf['wf_me_flux_ec'], 'blue', label = 'ener frac wf')
# plt.legend()
# plt.savefig(os.path.join(md.outdir_plot, 'ener_flux_fractions_allmethods.png'))
# plt.close()


# pd.to_datetime('13000101', format='%Y%m%d', errors='ignore')
# rdf['datetime'] = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M')
# datetimes = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M', errors='ignore')
datetimes = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M')



# plt.plot(rdf['datetime'])

fig, axes = plt.subplots(2, 1, figsize = (12, 8))
axes[0].plot(datetimes, rdf['wc_me_flux_ec'], 'sk', alpha = 0.6, markersize = 6, label = 'Total flux')
axes[0].plot(datetimes, rdf['wc_en_frac_flux']*rdf['wc_me_flux_ec'], 'or', alpha = 0.6, markersize = 6, label = 'hotspot flux')
axes[0].plot(datetimes, rdf['wc_ba_frac_flux']*rdf['wc_me_flux_ec'], '^c', alpha = 0.6, markersize = 6, label = 'background flux')
axes[0].set_yscale('log')
axes[0].legend(ncol = 3, bbox_to_anchor=(1., 1.2))
axes[0].set_ylabel(r'$CH_4$ Flux [$\mu mol\, m^{-2}\, s^{-1}$]', fontsize = 16)
axes[0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[0].axes.get_xaxis().set_visible(False)

axes[1].plot(datetimes, rdf['wc_en_frac_time'], '^r', alpha = 0.6, markersize = 6, label = 'Area')
axes[1].plot(datetimes, rdf['wc_en_frac_var'], 'sg',  alpha = 0.6, markersize = 6, label = 'Variance')
axes[1].plot(datetimes, rdf['wc_en_frac_flux'], 'ok', alpha = 0.6, markersize = 6, label = 'Flux')
axes[1].set_ylabel('Hotspot fraction', fontsize = 16)
axes[1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[1].get_xticks(rotation=70)
# axes[1].set_xticklabels(datetimes.dt.date, rotation = 0, fontsize = 16)
axes[1].legend()
# plt.xlabel('Date')
# axes[1].legend(ncol = 3)
axes[1].legend(ncol = 3, bbox_to_anchor=(1., 1.2))
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'ener_flux_frac_fractions.png'), dpi = 300)
plt.close()



# fig, axes = plt.subplots(ncols = 2, nrows = 1 ,figsize=(12, 6))
fig, axes = plt.subplots(ncols = 2, nrows = 2 ,figsize=(12, 12))
xdata = rdf['sr_ustar']
xv, yv = mf.mylogslope(xdata, rdf['sr_teba'], slope = 0 )
axes[0, 0].scatter(xdata, rdf['sr_teme'].values, alpha = 0.6, s = 18, color = 'g', marker = '^', label = r'$CH_4$')
axes[0, 0].scatter(xdata, rdf['sr_teen'].values, alpha = 0.6, s = 18, color = 'r', marker = 'o', label = r'$CH_4$ H')
axes[0, 0].scatter(xdata, rdf['sr_teba'].values, alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CH_4$ B')
axes[0, 0].scatter(xdata, rdf['sr_tewa'].values, alpha = 0.6, s = 20, color = 'b', marker = 's', label = r'$H_2O$')
axes[0, 0].annotate("a)", xy=(0.03, 0.90), xycoords="axes fraction")
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_ylabel('Transport efficiency')
axes[0, 0].set_xlabel(r'$u_* [m/s]$')
axes[0, 0].legend()
axes[0, 0].set_xticks([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[0, 0].set_xticklabels([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[0, 0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.6])
axes[0, 0].set_yticklabels([0.1, 0.2, 0.3,  0.4, 0.6])
xv, yv = mf.mylogslope(xdata, rdf['sr_gtv_ba'], slope = 3/4 )
axes[0, 1].scatter(xdata, rdf['sr_gtv_wa'].values, alpha = 0.6, s = 20, color = 'b', marker = 's', label = r'$H_2O$')
axes[0, 1].scatter(xdata, rdf['sr_gtv_me'].values, alpha = 0.6, s = 18, color = 'g', marker = '^', label = r'$CH_4$')
axes[0, 1].scatter(xdata, rdf['sr_gtv_ba'].values, alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CH_4$ B')
axes[0, 1].scatter(xdata, rdf['sr_gtv_en'].values, alpha = 0.6, s = 18, color = 'r', marker = 'o', label = r'$CH_4$ H')
axes[0, 1].annotate("b)", xy=(0.04, 0.90), xycoords="axes fraction")
axes[0, 1].plot(xv, yv, 'k', linewidth = 2)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_ylabel('Gas transfer velocity [m/s]')
axes[0, 1].set_xlabel(r'$u_* [m/s]$')
axes[0, 1].set_xticks([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[0, 1].set_xticklabels([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[0, 1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.6])
axes[0, 1].set_yticklabels([0.1, 0.2, 0.3,  0.4, 0.6])
# plt.tight_layout()
# plt.savefig(os.path.join(md.outdir_plot, 'teff_and_gtv.png'), dpi=300)
# plt.close()



# fig, axes = plt.subplots(ncols = 2, nrows = 1 ,figsize=(12, 6))
xdata = rdf['sr_ustar']
# xv, yv = mf.mylogslope(xdata, rdf['sr_teba_sr'], slope = -1/4 )
# axes[0].scatter(xdata, rdf['sr_teme_isr'].values, alpha = 0.6, s = 18, color = 'g', marker = 'o', label = 'CH4 ISR')
axes[1, 0].scatter(xdata, rdf['sr_tewa_sr'].values, alpha = 0.6, s = 18, color = 'b', marker = 's', label = r'$H_2O$')
axes[1, 0].scatter(xdata, rdf['sr_teme_isr'].values, alpha = 0.6, s = 18, color = 'g', marker = '^', label = r'$CH4$')
axes[1, 0].scatter(xdata, rdf['sr_teba_sr'].values, alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CH_4$ B')
axes[1, 0].scatter(xdata, rdf['sr_teen_sr'].values, alpha = 0.6, s = 18, color = 'r', marker = 'o', label = r'$CH_4$ H')
axes[1, 0].annotate("c)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylabel('Transport efficiency')
axes[1, 0].set_xlabel(r'$u_*\, [m/s]$')
axes[1, 0].legend()
axes[1, 0].set_xticks([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[1, 0].set_xticklabels([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[1, 0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.6])
axes[1, 0].set_yticklabels([0.1, 0.2, 0.3,  0.4, 0.6])
xv, yv = mf.mylogslope(xdata, rdf['sr_gtv_ba_sr'], slope = 3/4 )
axes[1, 1].scatter(xdata, rdf['sr_gtv_wa_sr'].values, alpha = 0.6, s = 18, color = 'b', marker = 's', label = r'$H_2O$')
axes[1, 1].scatter(xdata, rdf['sr_gtv_me_isr'].values, alpha = 0.6, s = 18, color = 'g', marker = '^', label = r'$CH4$')
axes[1, 1].scatter(xdata, rdf['sr_gtv_ba_sr'].values, alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CH_4$ B')
axes[1, 1].scatter(xdata, rdf['sr_gtv_en_sr'].values, alpha = 0.6, s = 18, color = 'r', marker = 'o', label = r'$CH_4$ H')
axes[1, 1].annotate("d)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[1, 1].plot(xv, yv, 'k', linewidth = 2)
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylabel('Gas transfer velocity [m/s]')
axes[1, 1].set_xlabel(r'$u_*\, [m/s]$')
axes[1, 1].set_xticks([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[1, 1].set_xticklabels([0.2, 0.3, 0.4, 0.6, 0.8], minor = True)
axes[1, 1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.6])
axes[1, 1].set_yticklabels([0.1, 0.2, 0.3,  0.4, 0.6])

axes[0, 0].set_ylim([0.03, 1.2])
axes[1, 0].set_ylim([0.03, 1.2])
axes[0, 1].set_ylim([0.03, 1.2])
axes[1, 1].set_ylim([0.03, 1.2])
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'SI_teff_and_gtv_sr_joined.png'))
plt.close()


# check two different computation of gas transfer velocity
# plt.figure()
# plt.scatter(rdf['sr_gtv_wa'].values, rdf['sr_gtv_me'].values,
#             alpha = 0.6, s = 18, color = 'g', marker = '^', label = 'CH4')
# plt.plot(rdf['sr_gtv_wa'].values, rdf['wc_enfa']/np.sqrt(rdf['wc_enva']), '.k')
# plt.plot(rdf['sr_gtv_wa'].values, rdf['wc_bafa']/np.sqrt(rdf['wc_bava']), 'oc')
# plt.scatter(rdf['sr_gtv_wa'].values, rdf['sr_gtv_en'].values,
#             alpha = 0.6, s = 18, color = 'r', marker = '*', label = 'CH4')
# plt.scatter(rdf['sr_gtv_wa'].values, rdf['sr_gtv_en_sr'].values,
#             alpha = 0.6, s = 18, color = 'r', marker = 's', label = 'CH4')
# plt.plot(rdf['sr_gtv_wa'].values, rdf['sr_gtv_wa'].values, 'k')
# plt.show()


plt.figure(figsize=(6,6))
plt.scatter(rdf['sr_gtv_wa'].values, rdf['sr_gtv_wa_sr'].values,
            alpha = 0.6, s = 18, color = 'b', marker = 'o', label = r'$H_2O$')
plt.scatter(rdf['sr_gtv_me'].values, rdf['sr_gtv_me_sr'].values,
            alpha = 0.6, s = 18, color = 'g', marker = '*', label = r'$CH_4$')
plt.scatter(rdf['sr_gtv_ba'].values, rdf['sr_gtv_ba_sr'].values,
            alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CH_4$ B')
plt.scatter(rdf['sr_gtv_en'].values, rdf['sr_gtv_en_sr'].values,
            alpha = 0.6, s = 18, color = 'r', marker = '^', label = r'$CH_4$ H')
plt.plot(rdf['sr_gtv_wa'].values, rdf['sr_gtv_wa'].values, 'k')
plt.xlabel('EC gas transfer velocity [m/s]')
plt.ylabel('SR gas transfer velocity [m/s]')
plt.legend()
plt.savefig( os.path.join(md.outdir_plot, 'gtv_sr_vs_ecwa.png'))
plt.close()


plt.figure(figsize=(6,6))
plt.scatter(rdf['sr_gtv_wa'].values, rdf['sr_gtv_wa_sr'].values,
            alpha = 0.6, s = 18, color = 'b', marker = 'o', label = r'$H_2O$')
plt.scatter(rdf['sr_gtv_me'].values, rdf['sr_gtv_me_sr'].values,
            alpha = 0.6, s = 18, color = 'g', marker = '*', label = r'$CH_4$')
plt.scatter(rdf['sr_gtv_cd'].values, rdf['sr_gtv_cd_isr'].values,
            alpha = 0.6, s = 18, color = 'y', marker = '^', label = r'$CO_2$')
# plt.scatter(rdf['sr_gtv_ba'].values, rdf['sr_gtv_ba_sr'].values,
#             alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CH_4$ back')
# plt.scatter(rdf['sr_gtv_en'].values, rdf['sr_gtv_en_sr'].values,
#             alpha = 0.6, s = 18, color = 'r', marker = '^', label = r'$CH_4$ eb.')

plt.scatter(rdf['sr_gtv_cd_ba'].values, rdf['sr_gtv_cd_ba_sr'].values,
            alpha = 0.6, s = 18, color = 'c', marker = 's', label = r'$CO_2$ B')
plt.scatter(rdf['sr_gtv_cd_en'].values, rdf['sr_gtv_cd_en_sr'].values,
            alpha = 0.6, s = 18, color = 'r', marker = '^', label = r'$CO_2$ H')
plt.plot(rdf['sr_gtv_wa'].values, rdf['sr_gtv_wa'].values, 'k')
plt.plot(-rdf['sr_gtv_wa'].values, -rdf['sr_gtv_wa'].values, 'k')
plt.xlabel('EC gas transfer velocity [m/s]')
plt.ylabel('SR gas transfer velocity [m/s]')
plt.legend()
plt.savefig( os.path.join(md.outdir_plot, 'gtv_CO2.png'))
plt.close()

# plt.figure()
# plt.plot(rdf['sr_teme'].values, rdf['sr_teme_sr'].values, 'o')
# plt.plot(rdf['sr_teme'].values, rdf['sr_teme'].values, 'k')
# plt.show()
#
#
# plt.figure()
# plt.plot(rdf['sr_teba'].values, rdf['sr_teba_sr'].values, 'o')
# plt.plot(rdf['sr_teba'].values, rdf['sr_teba'].values, 'k')
# plt.show()

#
# # plt.figure(figsize=(8, 6.5))
# fig, axes = plt.subplots(ncols = 2, nrows = 1 ,figsize=(14, 7))
# xdata = rdf['sr_ustar']
# # axes[0].scatter(xdata, rdf['sr_fme_ec']/rdf['sr_stdv_w']/rdf['sr_apime_std'], color='g',marker = '*',  alpha = 0.6, s = 20, label = 'CH4')
# # axes[0].scatter(xdata, rdf['sr_fwa_ec']/rdf['sr_stdv_w']/rdf['sr_apiwa_std'], color='b', marker = 's', alpha = 0.6, s = 20, label = 'H2O')
# # axes[0].scatter(xdata, rdf['sr_fme_ec']/rdf['sr_stdv_w']/rdf['sr_apien_std']*rdf['wc_en_frac_flux']/np.sqrt(rdf['sr_alpha']),
# #                 alpha = 0.6, s = 20, color = 'r',marker='^', label = 'CH4 Ebullition')
# # axes[0].scatter(xdata, rdf['sr_fme_ec']/rdf['sr_stdv_w']/rdf['sr_apiba_std']*rdf['wc_ba_frac_flux']/np.sqrt(1-rdf['sr_alpha']),
# #                 alpha = 0.6, s = 20,  color = 'c',marker = 'o', label = 'CH4 Background')
# # axes[0].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")
#
# xv, yv = mf.mylogslope(xdata,rdf['sr_fme_ec'].values/rdf['sr_std_me'].values/rdf['sr_stdv_w'].values, slope = -3/4/2 )
#
# axes[0].scatter(xdata, rdf['sr_fme_ec'].values/rdf['sr_stdv_w'].values/rdf['sr_std_me'].values, color='g',marker = '*',  alpha = 0.6, s = 20, label = 'CH4')
# axes[0].scatter(xdata, rdf['sr_fwa_ec'].values/rdf['sr_stdv_w'].values/rdf['sr_std_wa'].values, color='b', marker = 's', alpha = 0.6, s = 20, label = 'H2O')
# axes[0].scatter(xdata, 1/rdf['sr_alpha'].values*rdf['sr_fme_ec'].values/rdf['sr_stdv_w'].values/rdf['sr_std_en'].values*rdf['wc_en_frac_flux'].values,
#                 alpha = 0.6, s = 20, color = 'r',marker='^', label = 'CH4 Ebullition')
# axes[0].scatter(xdata, 1/(1-rdf['sr_alpha'].values)*rdf['sr_fme_ec'].values/rdf['sr_stdv_w'].values/rdf['sr_std_ba'].values*rdf['wc_ba_frac_flux'].values,
#                 alpha = 0.6, s = 20,  color = 'c',marker = 'o', label = 'CH4 Background')
# axes[0].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")
# axes[0].plot(xv, yv, 'k', linewidth = 2)
#
#
# # plot cubic spline interpolation
# # xdataso = np.sort(xdata)
# # xdataso_order = np.argsort(xdata)
# # fme = UnivariateSpline(xdataso, (rdf['sr_fme_ec']/rdf['sr_stdv_w']/rdf['sr_apime_std'])[xdataso_order])
# # fwa = UnivariateSpline(xdataso, (rdf['sr_fwa_ec']/rdf['sr_stdv_w']/rdf['sr_apiwa_std'])[xdataso_order])
# # fen = UnivariateSpline(xdataso, (rdf['sr_fme_ec']/rdf['sr_stdv_w']/rdf['sr_apien_std']*rdf['wc_en_frac_flux']/np.sqrt(rdf['sr_alpha']))[xdataso_order])
# # fba = UnivariateSpline(xdataso, (rdf['sr_fme_ec']/rdf['sr_stdv_w']/rdf['sr_apiba_std']*rdf['wc_ba_frac_flux']/np.sqrt(1-rdf['sr_alpha']))[xdataso_order])
# # # fme.set_smoothing_factor(0.5)
# # xdataso = np.sort(xdata)
# # axes[0].plot(xdataso, fme(xdataso), 'g', linewidth = 2.6)
# # axes[0].plot(xdataso, fwa(xdataso), 'b', linewidth = 2.6)
# # axes[0].plot(xdataso, fen(xdataso), 'r', linewidth = 2.6)
# # axes[0].plot(xdataso, fba(xdataso), 'c', linewidth = 2.6)
#
# # axes[0].plot(xdataso, fme(xdataso), 'g', linewidth = 2.6)
# # axes[0].plot(xdataso, fme(xdataso), '--k', linewidth = 1.6)
# # axes[0].plot(xdataso, fwa(xdataso), 'b', linewidth = 2.6)
# # axes[0].plot(xdataso, fwa(xdataso), '--k', linewidth = 1.6)
# # axes[0].plot(xdataso, fen(xdataso), 'r', linewidth = 2.6)
# # axes[0].plot(xdataso, fen(xdataso), '--k', linewidth = 1.6)
# # axes[0].plot(xdataso, fba(xdataso), 'c', linewidth = 2.6)
# # axes[0].plot(xdataso, fba(xdataso), '--k', linewidth = 1.6)
# axes[0].set_xscale('log')
# # plt.yscale('log')
# axes[0].set_ylabel('Transport Efficiency [-]')
# axes[0].set_xlabel(r'$u_*$')
# # plt.tight_layout()
# axes[0].legend()
# # plt.savefig(os.path.join(md.outdir_plot, 'transport_efficiency.png'), dpi=300)
# # plt.close()
# # plt.figure(figsize=(8, 6.5))
# # xdata = rdf['sr_Re_star']
# xv, yv = mf.mylogslope(xdata,rdf['sr_fme_ec'].values/rdf['sr_std_me'].values, slope = 3/4 )
# # axes[1].scatter(xdata, rdf['sr_fme_ec']/rdf['sr_apime_std'], color='g',marker = '*', s = 20, alpha = 0.6, label = 'CH4')
# # axes[1].scatter(xdata, rdf['sr_fwa_ec']/rdf['sr_apiwa_std'], color='b', marker = 's',s = 20, alpha = 0.6,  label = 'H2O')
# # axes[1].scatter(xdata, rdf['sr_fme_ec']/rdf['sr_apien_std']*rdf['wc_en_frac_flux']/np.sqrt(rdf['sr_alpha']),   s = 20, alpha = 0.6, color = 'r',marker='^', label = 'CH4 EB')
# # axes[1].scatter(xdata, rdf['sr_fme_ec']/rdf['sr_apiba_std']*rdf['wc_ba_frac_flux']/np.sqrt(1-rdf['sr_alpha']), s = 20, alpha = 0.6, color = 'c',marker = 'o', label = 'CH4 BA')
# # axes[1].plot(xv, yv, 'k', linewidth = 2)
# # axes[1].annotate("b)", xy=(0.1, 0.9), xycoords="axes fraction")
# # axes[1].annotate(r"3/4", xy=(0.9, 0.6), xycoords="axes fraction")
#
#
# axes[1].scatter(xdata, rdf['sr_fme_ec'].values/rdf['sr_std_me'].values, color='g',marker = '*',  alpha = 0.6, s = 20, label = 'CH4')
# axes[1].scatter(xdata, rdf['sr_fwa_ec'].values/rdf['sr_std_wa'].values, color='b', marker = 's', alpha = 0.6, s = 20, label = 'H2O')
# axes[1].scatter(xdata, 1/rdf['sr_alpha'].values*rdf['sr_fme_ec'].values/rdf['sr_std_en'].values*rdf['wc_en_frac_flux'].values,
#                 alpha = 0.6, s = 20, color = 'r',marker='^', label = 'CH4 Ebullition')
# axes[1].scatter(xdata, 1/(1-rdf['sr_alpha'].values)*rdf['sr_fme_ec'].values/rdf['sr_std_ba'].values*rdf['wc_ba_frac_flux'].values,
#                 alpha = 0.6, s = 20,  color = 'c',marker = 'o', label = 'CH4 Background')
# axes[1].annotate("a)", xy=(0.1, 0.9), xycoords="axes fraction")
# axes[1].plot(xv, yv, 'k', linewidth = 2)
#
# # plot cubic spline interpolation
# # xdataso = np.sort(xdata)
# # xdataso_order = np.argsort(xdata)
# # fme = UnivariateSpline(xdataso, (rdf['sr_fme_ec']/rdf['sr_apime_std'])[xdataso_order])
# # fwa = UnivariateSpline(xdataso, (rdf['sr_fwa_ec']/rdf['sr_apiwa_std'])[xdataso_order])
# # fen = UnivariateSpline(xdataso, (rdf['sr_fme_ec']/rdf['sr_apien_std']*rdf['wc_en_frac_flux']/np.sqrt(rdf['sr_alpha']))[xdataso_order])
# # fba = UnivariateSpline(xdataso, (rdf['sr_fme_ec']/rdf['sr_apiba_std']*rdf['wc_ba_frac_flux']/np.sqrt(1-rdf['sr_alpha']))[xdataso_order])
# # # fme.set_smoothing_factor(0.5)
# # xdataso = np.sort(xdata)
#
# # axes[1].plot(xdataso, fme(xdataso), 'g', linewidth = 2.6)
# # axes[1].plot(xdataso, fme(xdataso), '--k', linewidth = 1.6)
# # axes[1].plot(xdataso, fwa(xdataso), 'b', linewidth = 2.6)
# # axes[1].plot(xdataso, fwa(xdataso), '--k', linewidth = 1.6)
# # axes[1].plot(xdataso, fen(xdataso), 'r', linewidth = 2.6)
# # axes[1].plot(xdataso, fen(xdataso), '--k', linewidth = 1.6)
# # axes[1].plot(xdataso, fba(xdataso), 'c', linewidth = 2.6)
# # axes[1].plot(xdataso, fba(xdataso), '--k', linewidth = 1.6)
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
# axes[1].set_ylabel('Gas Transfer Velocity [m/s]')
# axes[1].set_xlabel(r'$u_*$')
# plt.tight_layout()
# # plt.legend()
# plt.savefig(os.path.join(md.outdir_plot, 'gas_transfer_velocity_and_treff.png'), dpi=300)
# plt.close()





# plt.figure()
# xv, yv = mf.mylogslope(tdf['Re_star'], rdf['sr_fme_ec'].values/rdf['sr_apime_std'].values*tdf['ustar'].values**(-1), slope=-1/4)
# plt.plot(tdf['Re_star'], rdf['sr_fme_ec'].values/rdf['sr_apime_std'].values*tdf['ustar'].values**(-1), 'o')
# plt.plot(xv, yv, 'k')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


# plt.figure()
# # xv, yv = mf.mylogslope(tdf['epsilon'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, slope=1/4)
# xv, yv = mf.mylogslope(tdf['ustar'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, slope=3/4)
# # plt.plot(tdf['epsilon'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, 'o')
# plt.plot(tdf['ustar'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, 'o')
# plt.plot(xv, yv, 'k', linewidth = 2)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
#
#
# plt.figure()
# # xv, yv = mf.mylogslope(tdf['epsilon'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, slope=1/4)
# xv, yv = mf.mylogslope(tdf['ustar'], rdf['sr_fwa_ec'].values/rdf['sr_apiwa_std'].values, slope=3/4)
# # plt.plot(tdf['epsilon'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, 'o')
# plt.plot(tdf['ustar'], rdf['sr_fwa_ec'].values/rdf['sr_apiwa_std'].values, 'o')
# plt.plot(xv, yv, 'k', linewidth = 2)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# #
# #
# plt.figure()
# # xv, yv = mf.mylogslope(tdf['epsilon'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, slope=1/4)
# xv, yv = mf.mylogslope(tdf['ustar']**(3/4)*rdf['sr_apiwa_std'].values, rdf['sr_fwa_ec'].values, slope=1)
# # plt.plot(tdf['epsilon'], rdf['sr_fme'].values/rdf['sr_apime_std'].values, 'o')
# plt.plot(tdf['ustar']**(3/4)*rdf['sr_apiwa_std'].values, rdf['sr_fwa_ec'].values, 'o')
# plt.plot(xv, yv, 'k', linewidth = 2)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# ksr = 1
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
# axes[0,0].set_title(r'$CH_4$')
# axes[0,0].plot(rdf['fluxes_me_ecf'], ksr*rdf['fluxes_me_srf'], 'og', alpha = 0.6, markersize = 6)
# axes[0,0].plot(rdf['fluxes_me_ecf'], rdf['fluxes_me_ecf'],     'k')
# axes[0,0].set_ylabel('Surface Renewal flux')
# axes[0,0].set_xlabel('Eddy covariance flux')
# axes[0,0].set_xscale('log')
# axes[0,0].set_yscale('log')
# axes[0,0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
#
# axes[0,1].set_title(r'$H_2O$')
# axes[0,1].plot(rdf['fluxes_wa_ecf'], ksr*rdf['fluxes_wa_srf'], 'ob', alpha = 0.6, markersize = 6)
# axes[0,1].plot(rdf['fluxes_wa_ecf'], rdf['fluxes_wa_ecf'], 'k')
# axes[0,1].set_xscale('log')
# axes[0,1].set_yscale('log')
#
# axes[0,1].set_ylabel('Surface Renewal flux')
# axes[0,1].set_xlabel('Eddy covariance flux')
# axes[0,1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
#
# axes[1, 0].set_title(r'Sensible heat ($T$)')
# axes[1,0].plot(rdf['fluxes_T_ecf'], ksr*rdf['fluxes_T_srf']*np.sign(rdf['fluxes_T_ecf']), 'or', alpha = 0.6, markersize = 6)
# # axes[1,0].plot(rdf['fluxes_T_ecf'], ksr*rdf['fluxes_T_srf'], 'or', alpha = 0.6, markersize = 6)
# axes[1,0].plot(rdf['fluxes_T_ecf'], rdf['fluxes_T_ecf'], 'k')
# # axes[1,0].set_xscale('log')
# # axes[1,0].set_yscale('log')
#
# axes[1,0].set_ylabel('Surface Renewal flux')
# axes[1,0].set_xlabel('Eddy covariance flux')
# axes[1,0].annotate("c)", xy=(0.05, 0.90), xycoords="axes fraction")
#
# axes[1, 1].set_title(r'Momentum ($u$)')
# axes[1,1].plot(rdf['fluxes_u_ecf'],
#    ksr*rdf['fluxes_u_srf']*np.sign(rdf['fluxes_u_ecf']), 'ok', alpha = 0.6, markersize = 6)
# # axes[1,1].plot(rdf['fluxes_cd_ecf'], ksr*rdf['fluxes_cd_srf']*np.sign(rdf['fluxes_cd_ecf']), 'ok', alpha = 0.6, markersize = 6)
# axes[1,1].plot(rdf['fluxes_u_ecf'], rdf['fluxes_u_ecf'], 'k')
# # axes[1,1].plot(rdf['fluxes_cd_ecf'], rdf['fluxes_cd_ecf'], 'k')
# axes[1,1].set_ylabel('Surface Renewal flux')
# axes[1,1].set_xlabel('Eddy covariance flux')
# axes[1,1].annotate("d)", xy=(0.05, 0.90), xycoords="axes fraction")
# plt.tight_layout()
# # plt.show()
# plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes.png'), dpi=300)
# plt.close()


# ksr = 1
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
# axes[0,0].set_title(r'$CH_4$')
# axes[0,0].plot(rdf['fluxes_me_ecf'], rdf['fluxes_me_srf'], 'og', alpha = 0.6, markersize = 6)
# axes[0,0].plot(rdf['fluxes_me_ecf'], rdf['fluxes_me_ecf'],     'k')
# axes[0,0].set_ylabel('Surface Renewal flux')
# axes[0,0].set_xlabel('Eddy covariance flux')
# axes[0,0].set_xscale('log')
# axes[0,0].set_yscale('log')
# axes[0,0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[0,1].set_title(r'$H_2O$')
# axes[0,1].plot(rdf['fluxes_wa_ecf'], rdf['fluxes_wa_srf'], 'ob', alpha = 0.6, markersize = 6)
# axes[0,1].plot(rdf['fluxes_wa_ecf'], rdf['fluxes_wa_ecf'], 'k')
# axes[0,1].set_xscale('log')
# axes[0,1].set_yscale('log')
# axes[0,1].set_ylabel('Surface Renewal flux')
# axes[0,1].set_xlabel('Eddy covariance flux')
# axes[0,1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[1, 0].set_title(r'Sensible heat ($T$)')
# axes[1,0].plot(rdf['fluxes_T_ecf'], rdf['fluxes_T_srf'], 'or', alpha = 0.6, markersize = 6)
# axes[1,0].plot(rdf['fluxes_T_ecf'], rdf['fluxes_T_ecf'], 'k')
# axes[1,0].set_ylabel('Surface Renewal flux')
# axes[1,0].set_xlabel('Eddy covariance flux')
# axes[1,0].annotate("c)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[1, 1].set_title(r'Momentum ($u$)')
# axes[1,1].plot(rdf['fluxes_u_ecf'], rdf['fluxes_u_srf'], 'ok', alpha = 0.6, markersize = 6)
# axes[1,1].plot(rdf['fluxes_u_ecf'], rdf['fluxes_u_ecf'], 'k')
# axes[1,1].set_ylabel('Surface Renewal flux')
# axes[1,1].set_xlabel('Eddy covariance flux')
# axes[1,1].annotate("d)", xy=(0.05, 0.90), xycoords="axes fraction")
# plt.tight_layout()
# # plt.show()
# plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes.png'), dpi=300)
# plt.close()


ksr = 1
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
axes[0,0].set_title(r'$CH_4$')
axes[0,0].plot(rdf['sr_fme_ec'], rdf['sr_fme_sr'], 'og', alpha = 0.6, markersize = 6)
axes[0,0].plot(rdf['sr_fme_ec'], rdf['sr_fme_ec'],     'k')
# axes[0,0].set_ylabel('Surface Renewal flux')
# axes[0,0].set_xlabel('Eddy covariance flux')

axes[0, 0].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0, 0].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[0,1].set_title(r'$H_2O$')
axes[0,1].plot(rdf['sr_fwa_ec'], rdf['sr_fwa_sr'], 'ob', alpha = 0.6, markersize = 6)
axes[0,1].plot(rdf['sr_fwa_ec'], rdf['sr_fwa_ec'], 'k')
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
# axes[0,1].set_ylabel('Surface Renewal flux')
# axes[0,1].set_xlabel('Eddy covariance flux')

axes[0, 1].set_xlabel(r'EC flux [$m mol\, m^{-2}\, s^{-1}$]')
axes[0, 1].set_ylabel(r'SR flux [$m mol\, m^{-2}\, s^{-1}$]')
axes[0,1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[1, 0].set_title(r'Sensible heat ($T$)')
axes[1,0].plot(rdf['sr_fTT_ec'], rdf['sr_fTT_sr'], 'or', alpha = 0.6, markersize = 6)
axes[1,0].plot(rdf['sr_fTT_ec'], rdf['sr_fTT_ec'], 'k')
# axes[1,0].set_ylabel('Surface Renewal flux')
# axes[1,0].set_xlabel('Eddy covariance flux')

axes[1, 0].set_xlabel(r'EC flux [$K mol\, m\, s^{-1}$]')
axes[1, 0].set_ylabel(r'SR flux [$K mol\, m\, s^{-1}$]')
axes[1,0].annotate("c)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[1, 1].set_title(r'Momentum ($u$)')
axes[1,1].plot(rdf['sr_fuu_ec'], rdf['sr_fuu_sr'], 'ok', alpha = 0.6, markersize = 6)
axes[1,1].plot(rdf['sr_fuu_ec'], rdf['sr_fuu_ec'], 'k')
# axes[1,1].set_ylabel('Surface Renewal flux')
# axes[1,1].set_xlabel('Eddy covariance flux')
# axes[1, 1].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
# axes[1, 1].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')

axes[1, 1].set_xlabel(r'EC flux [$ m^{2}\, s^{-2}$]')
axes[1, 1].set_ylabel(r'SR flux [$ m^{2}\, s^{-2}$]')
axes[1,1].annotate("d)", xy=(0.05, 0.90), xycoords="axes fraction")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes.png'), dpi=300)
plt.close()

# daytime = rdf['datetime'].dt.hour.values + rdf['datetime'].dt.minute.values/60
#
# rdf_spikes = rdf[rdf['fluxes_me_srf']> 0.005].copy()
# daytime_spikes = rdf_spikes['datetime'].dt.hour.values + rdf_spikes['datetime'].dt.minute.values/60
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
# axes[0,0].set_xlim(0.0, 23.50)
# axes[0,0].set_title(r'$CH_4$')
# axes[0,0].plot(daytime, rdf['fluxes_me_srf'], '^g', alpha = 0.6, markersize = 6)
# axes[0,0].plot(daytime, rdf['fluxes_me_ecf'],     '*k', alpha = 0.6, markersize = 6)
# axes[0,0].set_ylabel('Flux [$\mu mol\, m\, s^{-1}$]')
# axes[0,0].set_xlabel('Time of the day [hour]')
# axes[0,0].set_yscale('log')
# axes[0,0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[0,1].set_xlim(0, 23.30)
# axes[0,1].set_title(r'$H_2O$')
# axes[0,1].plot(daytime, rdf['fluxes_wa_srf'], '^b', alpha = 0.6, markersize = 6)
# axes[0,1].plot(daytime, rdf['fluxes_wa_ecf'], '*k', alpha = 0.6, markersize = 6)
# axes[0,1].set_ylabel('Flux [$\mu mol\, m\, s^{-1}$]')
# axes[0,1].set_xlabel('Time of the day [hour]')
# axes[0,1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[1, 0].set_title(r'Sensible heat ($T$)')
# axes[1,0].plot(daytime, rdf['fluxes_T_srf'], '^y', alpha = 0.6, markersize = 6)
# axes[1,0].plot(daytime, rdf['fluxes_T_ecf'], '*k', alpha = 0.6, markersize = 6)
# axes[1,0].set_ylabel('Flux [$K\, m\, s^{-1}$]')
# axes[1,0].set_xlabel('Time of the day [hour]')
# axes[1,0].annotate("c)", xy=(0.05, 0.90), xycoords="axes fraction")
# axes[1, 1].set_title(r'$CO_2$')
# axes[1,1].plot(daytime, rdf['fluxes_cd_srf'], '^c', alpha = 0.6, markersize = 6)
# axes[1,1].plot(daytime, rdf['fluxes_cd_ecf'], '*k', alpha = 0.6, markersize = 6)
# axes[1,1].set_ylabel('Flux [$\mu mol\, m\, s^{-1}$]')
# axes[1,1].set_xlabel('Time of the day [hour]')
# axes[1,1].annotate("d)", xy=(0.05, 0.90), xycoords="axes fraction")
# plt.tight_layout()
# fig.savefig(os.path.join(md.outdir_plot, 'ec_sr_daily_fluxes.png'), dpi=300)
# axes[1,1].plot(daytime_spikes, rdf_spikes['fluxes_cd_srf'], 'or', alpha = 0.6,  markersize = 8)
# axes[0,0].plot(daytime_spikes, rdf_spikes['fluxes_me_srf'], 'or', alpha = 0.6, markersize = 8)
# fig.savefig(os.path.join(md.outdir_plot, 'ec_sr_daily_fluxes_marked.png'), dpi=300)
# plt.close()


daytime = rdf['datetime'].dt.hour.values + rdf['datetime'].dt.minute.values/60

# rdf_spikes = rdf[rdf['sr_fme_isr']> 0.005].copy()
rdf_spikes = rdf[rdf['sr_fme_isr']> 0.2].copy()
daytime_spikes = rdf_spikes['datetime'].dt.hour.values + rdf_spikes['datetime'].dt.minute.values/60
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
axes[0,0].set_xlim(0.0, 23.50)
axes[0,0].set_title(r'$CH_4$')
axes[0,0].plot(daytime, rdf['sr_fme_isr'], '^g', alpha = 0.6, markersize = 6)
axes[0,0].plot(daytime, rdf['sr_fme_ec'],     '*k', alpha = 0.6, markersize = 6)
axes[0,0].set_ylabel('Flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0,0].set_xlabel('Time of the day [hour]')
axes[0,0].set_yscale('log')
axes[0,0].annotate("a)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[0,1].set_xlim(0, 23.30)
axes[0,1].set_title(r'$H_2O$')
axes[0,1].plot(daytime, rdf['sr_fwa_sr'], '^b', alpha = 0.6, markersize = 6)
axes[0,1].plot(daytime, rdf['sr_fwa_ec'], '*k', alpha = 0.6, markersize = 6)
axes[0,1].set_ylabel('Flux [$m mol\, m^{-2}\, s^{-1}$]')
axes[0,1].set_xlabel('Time of the day [hour]')
axes[0,1].annotate("b)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[1, 0].set_title(r'Sensible heat ($T$)')
axes[1,0].plot(daytime, rdf['sr_fTT_sr'], '^y', alpha = 0.6, markersize = 6)
axes[1,0].plot(daytime, rdf['sr_fTT_ec'], '*k', alpha = 0.6, markersize = 6)
axes[1,0].set_ylabel('Flux [$K\, m\, s^{-1}$]')
axes[1,0].set_xlabel('Time of the day [hour]')
axes[1,0].annotate("c)", xy=(0.05, 0.90), xycoords="axes fraction")
axes[1, 1].set_title(r'$CO_2$')
axes[1,1].plot(daytime, rdf['sr_fcd_isr'], '^c', alpha = 0.6, markersize = 6)
axes[1,1].plot(daytime, rdf['sr_fcd_ec'], '*k', alpha = 0.6, markersize = 6)
axes[1,1].set_ylabel('Flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[1,1].set_xlabel('Time of the day [hour]')
axes[1,1].annotate("d)", xy=(0.05, 0.90), xycoords="axes fraction")
plt.tight_layout()
fig.savefig(os.path.join(md.outdir_plot, 'ec_sr_daily_fluxes.png'), dpi=300)
axes[1,1].plot(daytime_spikes, rdf_spikes['sr_fcd_isr'], 'or', alpha = 0.6,  markersize = 8)
axes[0,0].plot(daytime_spikes, rdf_spikes['sr_fme_isr'], 'or', alpha = 0.6, markersize = 8)
fig.savefig(os.path.join(md.outdir_plot, 'ec_sr_daily_fluxes_marked.png'), dpi=300)

plt.close()


# plt.figure()
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_sr'],
#          'ok', alpha = 0.6, markersize = 6)
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'k')
# # plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'og')
# # plt.plot(rdf['sr_fcd_ec'], rdf['sr_fba_cd_sr'] + rdf['sr_fen_cd_sr'] , 'og')
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fba_cd_sr']+rdf['sr_fen_cd_sr'] , 'og')
# # plt.plot(rdf['fluxes_cd_ecf'],rdf['fluxes_cd_ecf'], 'k')
# # plt.plot(rdf['fluxes_cd_ecf'], rdf['srco2_fba_sr'], 'og')
# plt.ylabel('SR')
# plt.xlabel('EC')
# plt.show()

# plt.figure(figsize = (8, 8))
# # plt.title('$Co_2$')
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_isr'],
#          'ok', alpha = 0.6, markersize = 6)
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'k', label = r'$CO_2$ total flux')
# # plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'og')
# # plt.plot(rdf['sr_fcd_ec'], rdf['sr_fba_cd_sr'] + rdf['sr_fen_cd_sr'] , 'og')
# plt.plot(rdf['wc_cd_flux_ec']*rdf['wc_bd_frac_flux'], rdf['sr_fba_cd_sr'], 'oc', label = r'$CO_2$ B flux')
# plt.plot(rdf['wc_cd_flux_ec']*rdf['wc_ed_frac_flux'], rdf['sr_fen_cd_sr'], 'or', label = r'$CO_2$ H flux')
# # plt.plot(rdf['fluxes_cd_ecf'],rdf['fluxes_cd_ecf'], 'k')
# # plt.plot(rdf['fluxes_cd_ecf'], rdf['srco2_fba_sr'], 'og')
# plt.ylabel(r'SR flux [$\mu mol\, m\, s^{-1}$]')
# plt.xlabel(r'EC flux [$\mu mol\, m\, s^{-1}$]')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes_CO2.png'), dpi=300)
# plt.close()


fig, axes = plt.subplots(1, 2, figsize = (13, 7.3))
axes[0].plot(rdf['sr_fcd_ec'], rdf['sr_fcd_isr'],'ok', label = r'$CO_2$ total flux', alpha = 0.6, markersize = 6)
axes[0].plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'k',  markersize=5, alpha = 0.7)
axes[0].plot(rdf['wd_me_flux_ec']*rdf['wd_ba_frac_flux'], rdf['sr_fba_cd_sr'], 'oc', label = r'$CO_2$ B flux',  markersize=5, alpha = 0.7)
axes[0].plot(rdf['wd_me_flux_ec']*rdf['wd_en_frac_flux'], rdf['sr_fen_cd_sr'], 'or', label = r'$CO_2$ H flux',  markersize=5, alpha = 0.7)
axes[0].axvline(x=0.0, color='k', linestyle='--')
axes[0].axhline(y=0.0, color='k', linestyle='--')
axes[0].set_ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0].set_xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
axes[0].legend()
# plt.figure()
axes[1].plot(rdf['wd_me_fa'], 'k',label = r'$CO_2$ total')
axes[1].plot(rdf['wd_ba_fa'],'c',label = r'$CO_2$ B')
axes[1].plot(rdf['wd_en_fa'], 'r', label = r'$CO_2$ H')
axes[1].axhline(y=0.0, color='k', linestyle='--')
# axes[1].legend()
axes[1].set_xlabel('Run')
axes[1].set_ylabel(r'Average flux [$\mu mol\, m^{-2}\, s^{-1}$]')
# plt.show()
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes_CO2.png'), dpi=300)
plt.close()


# fig, axes = plt.subplots(111, figsize = (6, 6))
fig = plt.figure(figsize = (7.3, 7.3))
plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_isr'],'ok', label = r'$CO_2$ total flux', alpha = 0.6, markersize = 6)
plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'k',  markersize=5, alpha = 0.7)
plt.plot(rdf['wd_me_flux_ec']*rdf['wd_ba_frac_flux'], rdf['sr_fba_cd_sr'], 'oc', label = r'$CO_2$ B flux',  markersize=5, alpha = 0.7)
plt.plot(rdf['wd_me_flux_ec']*rdf['wd_en_frac_flux'], rdf['sr_fen_cd_sr'], 'or', label = r'$CO_2$ H flux',  markersize=5, alpha = 0.7)
plt.axvline(x=0.0, color='k', linestyle='--')
plt.axhline(y=0.0, color='k', linestyle='--')
plt.ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')
plt.xlabel(r'EC flux [$\mu mol\, m^{-2}\, s^{-1}$]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes_CO2_single.png'), dpi=300)
plt.close()


# datetimes = pd.to_datetime(rdf['csv_name'], format='%Y%m%d_%H%M')
plt.figure(figsize = (8, 8))
# plt.title('$Co_2$')
plt.plot(rdf['sr_fcd_ec'],
         'ok', alpha = 0.6, markersize = 6)
plt.plot(rdf['wd_me_flux_ec']*rdf['wd_ba_frac_flux'], 'oc', label = r'$CO_2$ B flux',  markersize=5, alpha = 0.7)
plt.plot(rdf['wd_me_flux_ec']*rdf['wd_en_frac_flux'], 'or', label = r'$CO_2$ H flux',  markersize=5, alpha = 0.7)

# plt.plot(rdf['sr_fba_cd_sr'], '^b', label = r'$CO_2$ B flux',  markersize=5, alpha = 0.7)
# plt.plot(rdf['sr_fen_cd_sr'], '^r', label = r'$CO_2$ H flux',  markersize=5, alpha = 0.7)
# plt.plot(rdf['fluxes_cd_ecf'],rdf['fluxes_cd_ecf'], 'k')
# plt.plot(rdf['fluxes_cd_ecf'], rdf['srco2_fba_sr'], 'og')
plt.axvline(x=0.0, color='k', linestyle='--')
plt.axhline(y=0.0, color='k', linestyle='--')
plt.ylabel(r'SR flux [$\mu mol\, m^{-2}\, s^{-1}$]')
plt.xlabel(r'Run')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'ec_day_all_fluxes_CO2.png'), dpi=300)
plt.close()




# plt.figure()
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_sr'], 'ok', alpha = 0.6, markersize = 6)
# plt.plot(rdf['sr_fcd_ec'], rdf['sr_fcd_ec'], 'k')
# plt.ylabel(r'$CO_2$ Surface Renewal flux')
# plt.xlabel(r'$CO_2$ Eddy covariance flux')
# plt.savefig(os.path.join(md.outdir_plot, 'ec_sr_all_fluxes_CO2.png'), dpi=300)
# plt.close()

plt.figure()
plt.plot(rdf['wd_en_M30'], rdf['wc_en_M30'], 'or')
plt.plot(rdf['wd_en_M30'], rdf['wd_en_M30'], 'k')
plt.plot(rdf['wd_me_M30'], rdf['wc_me_M30'], 'ok')
plt.plot(rdf['wd_ba_M30'], rdf['wc_ba_M30'], 'og')
plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel(r'$M_{30}$ $CH_4$')
plt.ylabel(r'$M_{30}$ $CO_2$')
plt.savefig(os.path.join(md.outdir_plot, 'M30_CH4_CO2.png'), dpi=300)
plt.close()

plt.figure()
plt.plot(rdf['wc_bar_M30'], rdf['wc_bar_M30'], 'k')
plt.plot(rdf['wc_bar_M30'], rdf['wc_ba_M30'], 'or')
plt.plot(rdf['wc_wa_M30'], rdf['wc_me_M30'], 'ob')
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

# rdf['time'] = rdf['datetime'].dt.hour
#
#
# plt.figure()
# # plt.plot(rdf['time'].values, rdf['sr_fme_ec'].values, 'o')
# # plt.plot(tdf['ustar'].values, rdf['wc_en_frac_time'].values, 'o')
# plt.plot(tdf['Tbar'].values, rdf['sr_fme_ec'].values*rdf['wc_en_frac_flux'].values, 'o')
# plt.plot(tdf['Tbar'].values, rdf['sr_fme_ec'].values, 'o')
# plt.yscale('log')
# # plt.plot(tdf['Tbar'].values, rdf['wc_en_frac_flux'], 'o')
# # plt.plot(rdf['time'].values, tdf['Tbar'], 'o')
# plt.show()

# plt.figure()
# plt.plot(rdf['wco2_me_M30'])
# plt.show()

# plt.figure()
# # plt.plot(rdf['time'].values, rdf['sr_fme_ec'].values, 'o')
# # plt.plot(tdf['ustar'].values, rdf['wc_en_frac_time'].values, 'o')
# plt.plot(tdf['stab'].values, rdf['sr_fme_ec'].values*rdf['wc_en_frac_flux'].values, 'o')
# plt.plot(tdf['stab'].values, rdf['sr_fme_ec'].values, 'o')
# plt.yscale('log')
# # plt.plot(tdf['Tbar'].values, rdf['wc_en_frac_flux'], 'o')
# # plt.plot(rdf['time'].values, tdf['Tbar'], 'o')
# plt.show()

# import pickle

# plt.figure()
# plt.plot(rdf['sr_ustar'], rdf['wc_en_frac_flux'], 'o')
# # plt.plot(rdf['sr_ustar'], rdf['wc_ba_frac_time'], 'o')
# # plt.plot(tdf['Re0'], rdf['wc_ba_frac_flux'], 'o')
# plt.xscale('log')
# plt.yscale('log')
# plt.plot()



# read environmental data: Pressure and Water table:#
#___________________________________________________#
dfpr = pd.read_csv(os.path.join('..', 'methane_data', 'SMEARII_p.csv'))
dfpr['datetime'] = pd.to_datetime(dfpr.iloc[:,:6])
dfpr.rename(columns = {'HYY_META.Pamb0':'patm'}, inplace = True)
# dfpr.set_index(dfpr['datetime'], inplace=True)

def pslope(y):
    # compute slope of the time series
    y = np.array(y)
    # x = 60*np.arange(np.size(y)) # time in seconds
    # slope= np.polyfit(x, y, 1)[0]
    slope = (y[-1] - y[0])/30/60
    return slope

# y = [1, 2, 3, 4, -5]
# print(pslope(y))
dfpr30 = dfpr.resample('30min', on = 'datetime').agg({'patm':[np.mean, np.std, pslope]})
dfpr30.columns = dfpr30.columns.droplevel(level=0)
dfpr30.rename(columns = {'mean':'patm_mean','std':'patm_std', 'pslope':'patm_slope'}, inplace = True)
# dfpr30['datetime'] = dfpr30.index
# .resample('1H', how={'radiation': np.sum, 'tamb': np.mean})
dfwt = pd.read_csv(os.path.join('..', 'methane_data', 'Siikaneva_wt_level.csv'))
dfwt['datetime'] = pd.to_datetime(dfwt.iloc[:,:6])
dfwt.rename(columns = {'SII1_META.WTD':'wtd'}, inplace = True)
dfwt.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], inplace = True)

rdfenv = rdf.merge(dfwt, on='datetime').merge(dfpr30, on='datetime')

plotwatertable = False
if plotwatertable:
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(dfwt['datetime'], dfwt['wtd'], '*')
    axes[0].plot(rdfenv['datetime'], rdfenv['wtd'], 'o')
    axes[0].set_ylabel(r'water table level [cm]')
    axes[1].plot(dfpr30.index, dfpr30['patm_mean'], '*')
    axes[1].plot(rdfenv['datetime'], rdfenv['patm_mean'], 'o')
    axes[1].set_xlabel('date')
    axes[1].set_ylabel(r'$p_{atm}$ hPa')
    plt.show()


# plt.plot(rdf['datetime'])

rdfenv['Ubar'] = tdf['Ubar'].values
rdfenv['Tbar'] = tdf['Tbar'].values
# envvar = 'patm_mean'
# envvar = 'patm_slope'
# envvar = 'sr_ustar'
# envvar = 'ts_shear_ts'
# envvar = 'ts_diss_ts'
envvar1 = 'wtd'
envvar2 = 'patm_slope'
# envvar = 'Ubar'
fig, axes = plt.subplots(1, 2, figsize = (12, 6))
ebflux = rdfenv['wc_en_frac_flux']*rdf['wc_me_flux_ec']
# axes[0].set_xlabel(r'$\frac{\Delta p_{atm}}{\Delta t}$ [$hPa \quad s^{-1}$]', fontsize = 16)
axes[0].plot(rdfenv[envvar1], rdfenv['wc_me_flux_ec'], 'sk', alpha = 0.6, markersize = 6, label = 'Total flux')
axes[0].plot(rdfenv[envvar1],ebflux , 'or', alpha = 0.6, markersize = 6, label = 'Hotspot flux')
axes[0].plot( [np.min(rdfenv[envvar1]), np.max(rdfenv[envvar1])],
      [np.median(rdfenv['wc_me_flux_ec']), np.median(rdfenv['wc_me_flux_ec'])], 'k')
axes[0].plot( [np.min(rdfenv[envvar1]), np.max(rdfenv[envvar1])],
              [np.median(ebflux), np.median(ebflux)], 'r')

# axes[0].plot(rdfenv[envvar], rdfenv['wc_ba_frac_flux']*rdf['wc_me_flux_ec'], '^c', alpha = 0.6, markersize = 6, label = 'Background flux')
axes[0].set_xlabel(r'Relative water table depth [cm]', fontsize = 16)
axes[0].set_yscale('log')
# axes[0].legend(ncol = 3, bbox_to_anchor=(1., 1.2))
axes[0].set_ylabel(r'CH4 Flux [$\mu mol\, m^{-2}\, s^{-1}$]', fontsize = 16)
axes[0].annotate("a)", xy=(0.04, 0.90), xycoords="axes fraction")

axes[1].plot( [0, 0], [np.min(ebflux), np.max(rdfenv['wc_me_flux_ec'])], '--k')
axes[1].plot(rdfenv[envvar2], rdfenv['wc_me_flux_ec'], 'sk', alpha = 0.6, markersize = 6, label = 'Total flux')
axes[1].plot(rdfenv[envvar2],ebflux , 'or', alpha = 0.6, markersize = 6, label = 'Hotspot flux')
axes[1].plot( [np.min(rdfenv[envvar2]), np.max(rdfenv[envvar2])],
              [np.median(rdfenv['wc_me_flux_ec']), np.median(rdfenv['wc_me_flux_ec'])], 'k')
axes[1].plot( [np.min(rdfenv[envvar2]), np.max(rdfenv[envvar2])],
              [np.median(ebflux), np.median(ebflux)], 'r')
axes[1].set_ylabel(r'CH4 Flux [$\mu mol\, m^{-2}\, s^{-1}$]', fontsize = 16)
axes[1].set_yscale('log')
# axes[1].set_ylabel('Ebullition flux fraction', fontsize = 16)
# axes[1].set_xlabel(r'${}$'.format(envvar), fontsize = 16)
axes[1].set_xlabel(r'$\frac{\Delta p_{atm}}{\Delta t}$ [$hPa \quad s^{-1}$]', fontsize = 16)
axes[1].annotate("b)", xy=(0.04, 0.90), xycoords="axes fraction")
# axes[1].legend()
# axes[1].legend(ncol = 3, bbox_to_anchor=(1., 1.2))
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(md.outdir_plot, 'ener_flux_frac_fractions_envvar_{}_{}.png'.format(envvar1, envvar2)), dpi = 300)
plt.close()
#
# plt.figure()
# plt.plot(rdf['sr_ustar'], tdf['ustar'], 'o')
# plt.show()

# check CO2 renewal



matplotlib.use('Qt5Agg')
#
# plt.figure()
# plt.plot(rdf['wc_me_fa'], rdf['wc_me_fa'], 'k')
# plt.plot(rdf['wc_me_fa'], rdf['wc_ba_fa'], 'oc')
# plt.plot(rdf['wd_me_fa'], rdf['wd_me_fa'], 'sc')
# # plt.plot(rdf['wc_da_fa'], rdf['wc_defa'], 'sr')
# plt.plot(rdf['wc_me_fa'], rdf['wc_en_fa'], 'or')
# plt.show()
#
# plt.figure()
# plt.plot(rdf['wd_me_fa'])
# plt.plot(rdf['wd_ba_fa'])
# plt.plot(rdf['wd_en_fa'], 'o')
# plt.show()
#
#
# plt.figure()
# plt.plot(rdf['sr_stab'], rdf['wd_me_eT'], 'oy')
# plt.plot(rdf['sr_stab'], rdf['wd_ba_eT'], 'oc')
# plt.plot(rdf['sr_stab'], rdf['wd_en_eT'], 'or')
# plt.plot(rdf['sr_stab'], rdf['wc_wa_eT'], 'ob')
# # plt.plot(rdf['wd_bafa'])
# # plt.plot(rdf['wd_enfa'], 'o')
# plt.show()
#
#
# plt.figure()
# plt.plot(rdf['wc_wa_eT'], rdf['wd_me_eT'], 'oy')
# plt.plot(rdf['wc_wa_eT'], rdf['wc_me_eT'], 'og')
# plt.plot(rdf['wc_wa_eT'], rdf['wd_ba_eT'], 'oc')
# plt.plot(rdf['wc_wa_eT'], rdf['wd_en_eT'], 'or')
# plt.plot(rdf['wc_wa_eT'], rdf['wc_wa_eT'], 'k')
# # plt.plot(rdf['wd_bafa'])
# # plt.plot(rdf['wd_enfa'], 'o')
# plt.show()
#
#
# plt.figure()
# # plt.plot(rdf['wc_en_frac_flux']*rdf['wc_me_flux_ec'], '-og')
# plt.plot(rdf['wd_en_frac_flux']*rdf['wd_me_flux_ec'], '-oc')
# plt.plot(rdf['wd_ba_frac_flux']*rdf['wd_me_flux_ec'], '-or')
# plt.show()
#
# plt.figure()
# plt.plot(rdf['wc_en_frac_time'], '-og')
# plt.plot(rdf['wd_en_frac_time'], '-oc')
# plt.show()

plt.figure()
plt.plot(rdf['wc_Rcr'], rdf['wc_stdv_c'], 'o')
plt.plot(np.abs(rdf['wd_Rcr']), rdf['wd_stdv_c'], 'or')
plt.show()
