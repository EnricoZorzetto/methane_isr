'''------------------------------------------------------------------------

#########################################################
Module with functions for the Methane ebullition project
#########################################################

Includes the following functions:
    -----------------------------
    load_data
    WindDir
    coordrot
    correct_lags
    simple_despiking
    remove_missing_data
    flux_stationarity
    phim
    phis
    integral_scale_fft
    turb_quant
    scalar_quant
    mixed_moments  - From Gaby
    Delta_So       - From Gaby
    Transport_Eff  - From Gaby
    REA_Beta       - From Gaby
    REA_Beta_Milne - From Gaby
    quadrant_fluxes
    compute_energetic_fraction
    coeffs_2_array
    array_2_coeffs
    detect_ebullition
    intermittent_surface_renewal
    Iwata_partitioning



------------------------------------------------------------------------'''
import os
import shutil
# import datetime
# import math
from scipy.signal import correlate
import zipfile
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import methdata as md
import pywt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.special import gamma
from scipy.stats import gaussian_kde
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.datasets import make_regression
from sklearn.neighbors import KernelDensity
import spectral_analysis as sa
from scipy import linalg
#import numba
from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice



def matplotlib_update_settings():
    # http://wiki.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    # this is a latex constant, don't change it.
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.8 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 12
    tick_size = inverse_latex_scale * 8

    # learn how to configure:
    # http://matplotlib.sourceforge.net/users/customizing.html
    params = {
        'axes.labelsize': text_size,
        'legend.fontsize': tick_size,
        'legend.handlelength': 2.5,
        'legend.borderaxespad': 0,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'font.size': text_size,
        'text.usetex': False,
        'figure.figsize': fig_size,
        # include here any neede package for latex
        # 'text.latex.preamble': [r'\usepackage{amsmath}',
        #                         ],
    }
    plt.rcParams.update(params)
    return


def dbs(int_or_str):
    ''' Return a string, and add a zero in front of it
    its length was shorter than 2
    (e.g., adds tenths to a single digit string)'''
    mystring = str(int_or_str)
    if len(mystring) < 2:
        return '0' + mystring
    return mystring


def mylogslope(xdata, ydata, slope = -5/3, npoints = 1000, vert_offset = 0):
    """ compute xx and yy to plot with given log slope """

    # xx1 = np.arange(50)
    # yy1 = xx1**(3/5)
    # xxv, yyv = mylogslope(xx1, yy1, slope = 3/5)
    #
    # plt.figure()
    # plt.plot(xx1, yy1, 'o')
    # plt.plot(xxv, yyv, 'k')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    def func(x, intercept):
        return 10**( (slope)*np.log10(x) + intercept)
    parhat = sc.optimize.curve_fit(func, xdata, ydata)[0]
    xhat = np.linspace(np.min(xdata), np.max(xdata), npoints)
    yhat = func(xhat, parhat+vert_offset)
    return xhat, yhat


def load_data_Siikaneva(date_time, datapath, myvars):
    ''' Read data from the Siikaneva dataset:
        Load a df with the data from a single run for the given time stamp
        with velocity components in [m/s] and Temperature in [K]
        gas units are unchanged [micromol / mol] or [mmol/ mol]
        datapath: folder with the data. must contain subdirectories
        namesd: Siikaneva_YYYY / files.zip
        INPUT:
            datetime
            datapath [must include subfolder for each year of data]
            site = 'Siikaneva_1' -> only this is supported for now
        RETURNS:
        df -> dataframe with the data [all np.nan if data not available]
        exists -> True if the data was available
        '''

    # extract time  and date components
    year = str(date_time.year)
    doy = str(date_time.dayofyear)
    # print(doy)
    mmin = str(date_time.minute)
    if len(mmin)==1:
        mmin = '0{}'.format(mmin)
    hour = str(date_time.hour)
    if len(hour)==1:
        hour = '0{}'.format(hour)
    hourmin = hour + mmin

    # if site == 'Siikaneva1': # if not add other datasets

    # first create temporary folder if it does not exists
    # first check if file is already unzipped in the temporary folder
    # if not unzip the entire day
    # read the run of interest
    # if last run of the day, clean temp folder afterwards

    path_raw  = os.path.join(datapath, 'Siikaneva_{}'.format(year))
    path_temp = os.path.join(datapath, 'Siikaneva_temp')
    if not os.path.exists(path_temp):
        os.makedirs(path_temp)
    zip_filename = 'Siikaneva_{}{}.zip'.format(doy, year)
    zip_file = os.path.join(path_raw, zip_filename)
    filename = '{}{}.RAW'.format(doy, hourmin)
    ext_file = os.path.join(path_temp, filename)

    # check if target file exists or if needs to be unzipped
    if not os.path.isfile(ext_file):
        if os.path.isfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(path_temp)
            if not os.path.isfile(ext_file):
                myfile_exists = False
            else:
                myfile_exists = True
        else:
            myfile_exists = False
    else:
        myfile_exists = True

    if myfile_exists:
        # print(ext_file)
        df = pd.read_csv( os.path.join(ext_file),
                    sep ='\s+', header=None, names = myvars,
                    index_col = 0, dtype = np.float64)

        # divide by 100 velocity components [cm/s] -> [m/s]
        # multiply T by 100 [celsius*100] and add 273.15 to T
        # unit of gases are micromol/mol of wet air

        df['u'] = df['u']*0.01            # [m/s]
        df['v'] = df['v']*0.01            # [m/s]
        df['w'] = df['w']*0.01            # [m/s]
        df['T'] = df['T']*0.01 + 273.15   # [K]

        # set gas concentrations equal to np.nan when negative (missing data)
        # df['CH4'][df['CH4']<0] = np.nan
        # df['CO2'][df['CO2']<0] = np.nan
        # df['H2O'][df['H2O']<0] = np.nan

        df[df['CH4']<0] = np.nan
        df[df['CO2']<0] = np.nan
        df[df['H2O']<0] = np.nan


    else: # return an array of not a numbers with length 18000:
        # m = 18000 # expected length of a run
        # x = np.ones(m)*np.nan
        # mydict = {var:x for var in myvars}
        # df = pd.DataFrame(mydict)
        df = None

    if hourmin == '2330':
        # remove all temporary unzipped files if last file of the day
        shutil.rmtree(path_temp)

    # else:
    #     print('WARNING: specify valid dataset')
    return df, myfile_exists


def WindDir(ubar, vbar):
    ''' compute wind direction
    based on Olli Peltola routine
    INPUT:
        Mu -> x-component of mean wind
        Mv -> y-component of mean wind
    RETURN:
        windir -> wind direction in degrees (0; 360)'''
    windir = np.arctan2(vbar, ubar)*180.0/np.pi + 180.0
    if windir < 0:   windir = windir + 360.0
    if windir > 360: windir = windir - 360.0
    return windir


def coordrot(df, D = 3):
    ''' Adapted from Olli Peltola codes -
        given a dataframe including wind components
        with labels u, v, w (x, y, z = vertical dir),
        rootate u along mean horizontal wind direction
        D = number of velocity components (default D = 3)
        RETURN:
            - new dataframe df with rotated components u, v, w
            - angles: array with the three rotation angles'''
    theta = 0.0
    phi = 0.0
    psi = 0.0
    u0 = df['u'].values
    v0 = df['v'].values
    w0 = df['w'].values
    u0bar = np.mean(u0)
    v0bar = np.mean(v0)
    w0bar = np.mean(w0)
    if D >= 1:
        ws = np.sqrt(u0bar**2 + v0bar**2)
        ce = u0bar/ws # cos eta
        se = v0bar/ws # sin eta
        theta = np.arcsin(se)
        # u1 = u0
        u1 = u0*ce + v0*se
        v2 = -u0*se + v0*ce
        # theta = np.arctan(v0bar/u0bar)
        # u1 = u0*np.cos(theta) + v0*np.sin(theta)
        # v2 = -u0*np.sin(theta) + v0*np.cos(theta)
    if D >= 2:
        u1bar = np.mean(u1)
        w0bar = np.mean(w0)
        phi = np.arctan(w0bar/u1bar)
        u2 = u1*np.cos(phi) + w0*np.sin(phi)
        w2 = -u1*np.sin(phi) + w0*np.cos(phi)
        # save the three rotation angles
        angles = np.array([theta, phi, psi])*180.0/np.pi
        df['u'] = u2
        df['v'] = v2 # just first rotation
        df['w'] = w2
        return df, angles


    # compute lag of scalars vs vertical velocity w
def correct_lags(df, laglim, fs):
    '''---------------------------------------------------
    compute lags for the gas scalar time series
    maximizing cross correlation with vertical velocity
    INPUT::
        df -> dataframe with time series u, v, w, T, CH4, H2O, CO2
        laglim -> limits for possible lag
    OUTPUT:
        array of lags
        new dataframe with lag-corrected time series
    ----------------------------------------------------'''
    # gasvars = ['CH4', 'H2O', 'CO2']
    myvars = list(df.keys())
    lag_dict = {}
    m = np.size(df['w'].values)
    # print(m)
    dfcorr = pd.DataFrame( {var:np.ones(m)*np.nan for var in myvars} ) # basevec
    for var in myvars:
        # print(myvars)
        if var in ['u', 'v', 'w', 'T']:
            # print(var)
            lag_dict[var] = 0.0 # no lag
            dfcorr[var] = df[var].values
        else:
            # print(var)
            cp = df[var].values - np.mean(df[var].values)
            wp = df['w'].values - np.mean(df['w'].values)
            # they must have the same size, and be synchronized
            # xcorr = np.correlate(dfp[var].values, dfp['w'].values)
            xcorr = correlate(cp, wp)/np.size(cp)
            lagpoints = np.arange(-m + 1, m)
            lagssec = lagpoints/fs
            inbounds = np.logical_and(lagssec >= laglim[var][0],
                                      lagssec <= laglim[var][1] )
            # np.size(inbounds[inbounds])
            mylags = lagssec[inbounds]
            mycorr = xcorr[inbounds]
            maxindex = np.argmax( np.abs(mycorr))
            maxlag_sec  = mylags[maxindex] # lag in seconds
            maxlag_points = int(maxlag_sec*fs) # in sampling points
            # print(maxlag_points)
            lag_dict[var] = maxlag_sec # save lag in seconds

            # cpcorr = np.ones(m)*np.nan
            # cpcorr[:m-maxlag_points] =
            dfcorr[var].iloc[:m - maxlag_points] = df[var].values[maxlag_points:]

#            ##############################################################
#            # code for testing this function
#            dfcorr, lag_dict = correct_lags(df, laglim, fs)
#            mysc = 'H2O'
#            m = 18000
#            #c = dfcorr['w'].values[ :m - int(lag_dict[mysc]*fs) ]
#            #w = dfcorr[mysc].values[:m - int(lag_dict[mysc]*fs) ]
#            c = df['w'].values[ :m - int(lag_dict[mysc]*fs) ]
#            w = df[mysc].values[:m - int(lag_dict[mysc]*fs) ]
#            cp = c - np.mean(c)
#            wp = w - np.mean(w)
#            m1 = np.size(cp)
#            xcorr = correlate(cp, wp)/np.size(cp) # must divide by length
#            xlags = np.arange(-m1 + 1, m1)
#            mylags = np.logical_and(xlags > -10000,  xlags < 10000)
#            mycorr = xcorr[mylags]
#            plt.plot(xlags[mylags], mycorr, 'o')
#            ##############################################################
    return dfcorr, lag_dict



def RunningMedian(seq, M):
    """
     Purpose: Find the median for the points in a sliding window (odd number in size)
              as it is moved from left to right by one point at a time.
      Inputs:
            seq -- list containing items for which a running median (in a sliding window)
                   is to be calculated
              M -- number of items in window (window size) -- must be an integer > 1
      Otputs:
         medians -- list of medians with size N - M + 1
       Note:
         1. The median of a finite list of numbers is the "center" value when this list
            is sorted in ascending order.
         2. If M is an even number the two elements in the window that
            are close to the center are averaged to give the median (this
            is not by definition)
    """
    seq = iter(seq)
    s = []
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq,M)]
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes
    median = lambda : s[m] if bool(M&1) else (s[m-1]+s[m])*0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()
    medians = [median()]

    # Now slide the window by one point to the right for each new position (each pass through
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()          # pop oldest from left
        d.append(item)             # push newest in from right
        del s[bisect_left(s, old)] # locate insertion point and then remove old
        insort(s, item)            # insert newest such that new sort is not required
        medians.append(median())
    return medians


def despike(df, vars, plot = False):

    # Nspikes = np.zeros(len(vars))
    dfout = df.copy()
    Nspikes = {var:0 for var in vars}
    for iv, var in enumerate(vars):
        c = df[var].values
        cout = c.copy()
        mc = c.copy()
        n = np.size(mc)
        # n = np.size(c)
        mc[3:(n-3)] = RunningMedian(c, 7)
        # mc = RunningMedian(c, 7)
        # c = c[3:(n-3)]
        mc[:3] = mc[3]
        mc[(n-3):] = mc[-4]
        dc = c - mc
        bins = 25

        apu = np.abs(np.diff(dc))
        maxbins = np.floor( (np.max(dc)-np.min(dc))/np.min(apu[apu>1e-10]))

        if maxbins % 2 == 0:
            maxbins = maxbins - 1

        if maxbins > 300:
            maxbins = 299

        xmax = np.nan
        xmin = np.nan

        while (( np.isnan(xmax) or np.isnan(xmin) ) and bins < maxbins):

            nelements, edges = np.histogram(dc, bins = bins)
            centers = edges[1:] - np.abs(edges[1] - edges[0])/2

            cond1 = np.logical_and(nelements == 0, centers < 0)
            if np.any(cond1):
                xmin = np.max(centers[cond1])

            cond2 = np.logical_and(nelements == 0, centers > 0)
            if np.any(cond2):
                ind = np.min(np.where(cond2))
                xmax = centers[ind]

            if np.logical_or(np.isnan(xmin), np.isnan(xmax)):
                xmax = np.nan
                bins = bins*2 + 1

        if (~np.isnan(xmin) & ~np.isnan(xmax)):
            fout = np.where(np.logical_or(dc >= xmax, dc <= xmin))

            fin = np.setdiff1d(np.arange(np.size(c)), fout)
            cout[fout] = np.interp(fout, fin, c[fin])
            Nspikes[var] = np.size(fout)
            # dfout[var] = cout
        elif (~np.isnan(xmin) & np.isnan(xmax)):
            fout = np.where(dc <= xmin)

            fin = np.setdiff1d(np.arange(np.size(c)), fout)
            cout[fout] = np.interp(fout, fin, c[fin])
            Nspikes[var] = np.size(fout)
            # dfout[var] = cout
        elif (np.isnan(xmin) & ~np.isnan(xmax)):
            fout = np.where(dc >= xmax)

            fin = np.setdiff1d(np.arange(np.size(c)), fout)
            cout[fout] = np.interp(fout, fin, c[fin])
            Nspikes[var] = np.size(fout)
            # dfout[var] = cout


        # if strcmp(vars_all{iv}, 'CH4')
        # Nspks = Nspks + sum(data.(vars_all
        # {iv}) < 1.8);
        # Nspks = Nspks + sum(data.(vars_all
        # {iv}) > 10);
        # data.(vars_all
        # {iv})(data.(vars_all{iv}) < 1.8) = nanmedian(data.(vars_all
        # {iv}));
        # data.(vars_all
        # {iv})(data.(vars_all{iv}) > 10) = nanmedian(data.(vars_all
        # {iv}));
        # end

        if var == 'CH4':
        #
            Nspikes[var] = Nspikes[var] + np.sum( cout < 1.8)
            Nspikes[var] = Nspikes[var] + np.sum( cout > 10)

            cout[np.logical_or(cout < 1.8, cout > 10)] = np.nanmedian(cout)
            # df[var][(df[var > 10]) | (df[var < 1.8]) ] = np.nanmedian(
            #                                                df[var].values)

            # for CH4, check if there are values of cn' > 60
            # and in case there are set them equal to the median to the run
            cn = (cout - np.mean(cout))/np.std(cout)
            cout[cn > 60] = np.nanmedian(cout)

        dfout[var] = cout

        if plot:
            plt.figure()
            plt.plot(df[var].values)
            plt.plot(dfout[var].values)
            plt.title(var)
            plt.show()

        # A = df['CH4'][np.logical_or(df['CH4'].values > 1, df['CH4'].values< -1)] = np.nanmedian(
        #      df['CH4'].values
        # )

        # fin = np.setdiff1d(np.arange(np.size(c)), fout)
        # c[fout] = np.interp(fout, fin, c[fin])
        # Nspikes[var] = np.size(fout)
        # dfout[var] = c

        # x = df['u'].values
        # y = x.copy()
        # n = len(y)
        # y[3:(n - 3)] = RunningMedian(x, 7)
        #
        # plt.figure()
        # plt.plot(x)
        # plt.plot(y, 'o')
        #
        # nelements, edges = np.histogram(x, bins=8)

    return dfout, Nspikes








def simple_despiking(df, lim = 12):
    '''--------------------------------------
    Simple function to despike a time series from eddy covariance measurements
    or gas analyzer. Remove all spikes above/below lim * standard deviation
    of each variable in the dataframe df, with respect to the mean.
    set spike values to np.nan to be removed afterwards.
    *** before applying this function, make sure to remove known missing data
    such as (-9999) values to compute reliable mean ***
    RETURN:
        df -> new df with spikes converted to NaN
        nspikes -> dict with number of spikes for each variable
        totspikes -> total number of spikes removed from the df
    ----------------------------------------'''
    myvars = list(df.keys())
    nspikes = {}
    for var in myvars:
        sample = df[var].values
        sample = sample[~np.isnan(sample)]
        mad = np.abs(sample - np.mean(sample))
        m = np.size(mad)
        # print('number of nans = ', np.size(mad[np.isnan(mad)]))
        stdv = np.std(sample)
        is_spike = mad > lim*stdv
        spikes = mad[is_spike]
        nspikes[var] = np.size(spikes)
        if np.size(spikes) == 0:
            pass
        else:
            index_spikes = np.arange(m)[is_spike]
            for index in index_spikes:
                df[var].iloc[index] = np.nan
        totspikes = sum(nspikes.values())
    return df, nspikes, totspikes


def remove_missing_data(df, nobs_min = 2**14):
    '''--------------------------------------------------------------------
    given a df with time series of observational data,
    keeps only the longest stretch of the time series with continuous observations
    of all variables, without missing data
    missing data must be saved as np.nan
    ### NO ### if the number of continuous obs is less than nobs_min (default is 2**14)
    ### NO ### return only the first row of the data frame.
    Added -> remove spikes with unphysical values (e.g., > 12 stdv)
    --------------------------------------------------------------------'''
    # First create a flag True if there is at least one np.nan in the row
    variables = list(df.keys())
    sample =  df[variables].isnull().any(axis = 1)
    non_miss = np.size(sample[sample == False]) # number of rows without NaNs
    enough_data = True
    if non_miss < nobs_min:
        df_new = df.iloc[0:1]
    #     # df_new = df
        enough_data = False
    else:
        # sample = df['nan_flag'].values
        m = np.concatenate(( [True], sample, [True] ))  # Mask
        # Start-stop limits
        ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)
        # Get max interval, interval limits
        start, stop = ss[(ss[:,1] - ss[:,0]).argmax()]
        df_new = df.iloc[start:stop]
        if df.shape[0] < nobs_min: enough_data = False
    return df_new, enough_data


# check stationarity
# def flux_stationarity(df, fs, ws = 300, lim_diff = 0.3):
#     '''--------------------------------------------------------------------
#     check if run flux is stationary
#     according to the method by Foken & Wichura 1996
#     INPUT:
#     df = dataframe with observations
#     ws = windows size in seconds for computing local fluxes (def 300)
#     lim_diff = maximum relative difference between local and global covariance.
#                if at least for one of the variable it is larger than
#               this limit, then the run is not stationary
#               (default 30% as in Foken & Wichura 1996)
#     OUTPUT:
#     fst -> dict with absolute value of rthe relative difference between
#     flux and average of fluxes computed for shorter intervals of ws seconds
#     is_stat -> True if run is stationary
#     --------------------------------------------------------------------'''
#     myvars = list(df.keys())
#     if 'w' in myvars: myvars.remove('w')
#     if 'v' in myvars: myvars.remove('v')
#
#     N = np.shape(df)[0]
#     n = np.int(ws*fs) # np.int -> to lowest integer
#     nwindows = np.int(N/n)
#     # print(nwindows)
#     # up = df['u'].values - df['u'].mean()
#     wp = df['w'].values - df['w'].mean()
#     # fluxes = {var:np.nan for var in myvars}
#     fluxes = {}
#     for var in myvars:
#         cp = df[var].values - df[var].mean()
#         fluxes[var] = np.mean(cp*wp)
#
#     count = 0
#     loc_fluxes = {var: np.zeros(nwindows)*np.nan for var in myvars}
#     for iw in range(nwindows):
#         dfiw = df.iloc[count:count+n]
#         # print(dfiw.shape[0])
#         count = count + n
#         wpii = dfiw['w'].values - dfiw['w'].mean()
#         for var in myvars:
#             cpii = dfiw[var].values - dfiw[var].mean()
#             loc_fluxes[var][iw] = np.mean(cpii*wpii)
#     # print(loc_fluxes)
#
#     fst = {}
#     for var in myvars:
#         ave_flux = np.mean(loc_fluxes[var])
#         # print(ave_flux)
#         # fst[var] = np.abs( (ave_flux - fluxes[var])/fluxes[var])
#         fst[var] = np.abs( (ave_flux - fluxes[var]))
#
#     is_stat = True
#     max_diff = max(fst.values())
#     if max_diff > lim_diff:
#         is_stat = False
#     return fst, is_stat


def flux_stationarity(df, fs, ws=300, lim_diff=0.3):
    '''--------------------------------------------------------------------
    check if run flux is stationary
    according to the method by Foken & Wichura 1996
    INPUT:
    df = dataframe with observations
    ws = windows size in seconds for computing local fluxes (def 300)
    lim_diff = maximum relative difference between local and global covariance.
               if at least for one of the variable it is larger than
              this limit, then the run is not stationary
              (default 30% as in Foken & Wichura 1996)
    OUTPUT:
    fst -> dict with absolute value of rthe relative difference between
    flux and average of fluxes computed for shorter intervals of ws seconds
    is_stat -> True if run is stationary
    --------------------------------------------------------------------'''

    # check flux stationarity only for momentum, sensible and latent heat
    myvars = list(df.keys())
    if 'w' in myvars: myvars.remove('w')
    if 'v' in myvars: myvars.remove('v')
    if 'CH4' in myvars: myvars.remove('CH4')
    if 'CO2' in myvars: myvars.remove('CO2')

    N = np.shape(df)[0]
    n = np.int(ws * fs)  # np.int -> to lowest integer
    nwindows = np.int(np.round(N / n))
    # print(myvars)
    # print(nwindows)
    # up = df['u'].values - df['u'].mean()
    wp = df['w'].values - df['w'].mean()
    # fluxes = {var:np.nan for var in myvars}
    fluxes = {}
    for var in myvars:
        cp = df[var].values - df[var].mean()
        fluxes[var] = np.mean(cp * wp)

    count = 0
    loc_fluxes = {var: np.zeros(nwindows) * np.nan for var in myvars}
    for iw in range(nwindows):
        dfiw = df.iloc[count:min(count + n, N-1)]
        # print(count)
        # print(count+n)
        # print(dfiw.shape[0])
        count = count + n
        wpii = dfiw['w'].values - dfiw['w'].mean()
        for var in myvars:
            cpii = dfiw[var].values - dfiw[var].mean()
            # print(np.sum(np.isnan(cpii)))
            # print(np.sum(np.isnan(wpii)))
            loc_fluxes[var][iw] = np.mean(cpii * wpii)
    # print(loc_fluxes)

    fst = {}
    for var in myvars:
        ave_flux = np.mean(loc_fluxes[var])
        # print(ave_flux)
        fst[var] = np.abs( (ave_flux - fluxes[var])/fluxes[var])
        # fst[var] = np.abs((ave_flux - fluxes[var]))

    is_stat = True
    max_diff = max(fst.values())
    # use the most stringent values
    if max_diff > lim_diff:
        is_stat = False
    return fst, is_stat


def test_non_locality(tdf, sdf, plot=False):
    '''
    remove runs for which H2O has nonlocal effects
    :return:
    '''

    fvrwa = sdf['stdv', 'H2O'].values / np.abs(sdf['flux_ec', 'H2O'].values / tdf['ustar'].values)
    fvrme = sdf['stdv', 'CH4'].values / np.abs(sdf['flux_ec', 'CH4'].values / tdf['ustar'].values)
    fvrT = sdf['stdv', 'T'].values / np.abs(sdf['flux_ec', 'T'].values / tdf['ustar'].values)

    stab = tdf['stab'].values
    stabp = stab[stab > 0]
    stabn = stab[stab < 0]
    C1 = 0.95
    C2 = 0.05
    stabxx = -np.logspace(-3, 2)
    fvr_kat = C1 * (C2 - stabxx) ** (-1 / 3)
    fvr_kat2 = 0.95 * (-stabxx) ** (-1 / 3)
    fvr_vdb = np.sqrt(8.4 * (1 - 28.4 * stabxx) ** (-2 / 3))

    # exclude the points outside the flux-variance relationship
    fvupbound = 1.5
    fvupbound = 0.5

    abstab = -np.abs(stab)
    fvarbound = 1.5 * C1 * (C2 - abstab) ** (-1 / 3)
    is_local = fvrwa < fvarbound
    # tdf['local_H2O'] = localwa

    if plot:
        plt.figure()
        plt.plot(-stabn, fvrwa[stab < 0], 'or', label='Unstable')
        plt.plot(stabp, fvrwa[stab > 0], 'sg', label='Stable')
        plt.plot(-stabxx, fvr_kat2, 'k', label='katul_1995')
        plt.plot(-stabxx, 2 * fvr_kat2, '--k')
        plt.plot(-stabxx, 0.5 * fvr_kat2, '--k')
        plt.plot(-stabxx, fvr_vdb, 'r', label='VanDeBoer_2014')
        plt.plot(-stabxx, 2 * fvr_vdb, '--r')
        plt.plot(-stabxx, 0.5 * fvr_vdb, '--r')
        plt.yscale('log')
        plt.xscale('log')
        plt.title(r'$H_2O$')
        plt.ylabel(r'$\sigma_c / | c*|$')
        plt.xlabel(r'$|\zeta|$')
        plt.legend()
        plt.xlim(1e-3, 1e1)
        plt.ylim(1e-1, 1e2)
        plt.show()

        # when
        plt.figure()
        plt.plot(-stabn, fvrwa[stab < 0], 'og', label='Unstable H2O')
        plt.plot(stabp, fvrwa[stab > 0], 'sg', label='Stable H2O')
        plt.plot(-stabn, fvrme[stab < 0], 'ob', label='Unstable CH4')
        plt.plot(stabp, fvrme[stab > 0], 'sb', label='Stable CH4')

        plt.plot(-stabn, fvrT[stab < 0], 'or', label='Unstable T')
        plt.plot(stabp, fvrT[stab > 0], 'sr', label='Stable T')
        plt.plot(-stabxx, fvr_kat2, 'k', label='katul_1995')
        plt.plot(-stabxx, 2 * fvr_kat2, '--k')
        plt.plot(-stabxx, 0.5 * fvr_kat2, '--k')
        plt.plot(-stabxx, fvr_vdb, 'r', label='VanDeBoer_2014')
        plt.plot(-stabxx, 2 * fvr_vdb, '--r')
        plt.plot(-stabxx, 0.5 * fvr_vdb, '--r')
        plt.axvline(x=0.04, ymin=1e-1, ymax=1e2, color='k')
        plt.yscale('log')
        plt.xscale('log')
        plt.title(r'$H_2O$')
        plt.ylabel(r'$\sigma_c / | c*|$')
        plt.xlabel(r'$|\zeta|$')
        plt.legend()
        plt.xlim(1e-3, 1e1)
        plt.ylim(1e-1, 1e2)
        plt.show()

    return is_local


def phim(stab):
    ''' stability correction function for momentum
    From Katul et al, PRL, 2011'''
    # if stab > 1:
    #    print('turb_quant WARNING:: very stable - using constant phim')
    #    return 5.47 # check
    if stab > 0: # stable
        return 1 + 4.7*stab
    else:
        return (1-15*stab)**(-1/4)

def phis(stab):
    ''' stability correction function for momentum
    From Brutsaert, Evaporation into the atmosphere '''
    # if stab > 1:
    #     return 6
    if stab > 0:
        return 1 + 4.7*stab
    else:
        return (1-15*stab)**(-1/2)


def integral_scale_fft(x1):
    '''
    From Gaby
    compute the integral scale of a 1D data vector
    integrating the autocorrelation function estimated from FFT
    x1 = data vector (its size must be an integer power of 2)
    Is = Integral scales (in units of x1 lags)
    usage::
    Is = integral_scale(x1)
    '''
    # check it is a 1d vector or return warning:
    sh = np.shape(x1)
    if np.size(sh) > 1:
        print('''Integral scale warning:
              the input array is not a 1D array''')
        return None
    N = np.size(x1)
    x1n = (x1 - np.mean(x1))/np.std(x1)
    # autocorrelation function estimated from FFT methods
    y1 = np.fft.fft(x1n)
    y2 = y1*np.conj(y1)
    CR=np.real(np.fft.ifft(y2)/N)
    # find the zero first zero crossing
    mm = np.argmax(CR < 0)
    if mm<2:
        Is = 0.5
    else:
        # trapezoidal approximation of the area under the ACF
        Is = (CR[0] + CR[mm])/2 + np.sum(CR[1:mm])
    return Is


def ave_pos_incr(x):
    """
    compute the average positive increments of a time series
    between points space at its integral time series
    :return:  api = average positive increment
    """
    T = integral_scale_fft(x)  # in sampling points
    Tint = np.int(np.ceil(T))
    inc = (x[Tint:] - x[:-Tint])  # with reduced stdv bc I removed coeffs!!
    api = np.mean(inc[(inc > 0)])
    return api


def turb_quant(df, fs, z_ref, nobs_min = 6000):
    '''--------------------------------------------------------------------
    compute turbulent parameters
    from dataframe df wich must have fields u, v, w, T
    data must already be rotated and quality controlled
    --------------------------------------------------------------------'''
    #######################################################################
    # constant used here:
    #######################################################################
    rho = 1.225 # air density at 15degrees [Kg/m^3]
    Cp = 1005 # at 300K, [J/Kg/K]
    kv = 0.4 # Von Karman Constant
    g = 9.806 # grav [m/s^2]
    nu = 1.516e-5 # [m^2/s] # air kinematic viscosity
    #######################################################################
    n_obs = df.shape[0]
    if n_obs < nobs_min:
        print('turb quant WARNING:: too many missing data -> skip this run')
        print('only length = ', n_obs, ' sampling points')

    u = df['u'].values
    v = df['v'].values
    w = df['w'].values
    T = df['T'].values
    up = u - np.mean(u)
    vp = v - np.mean(v)
    wp = w - np.mean(w)
    Tp = T - np.mean(T)

    Ubar  = np.mean(u) # [m/s]
    ustar = ( np.mean(up*wp)**2 + np.mean(vp*wp)**2 )**(1/4) # [m/s]
    # ustar = ( -np.mean(up*wp))**(1/2) # [m/s]
    Cov_wT     = np.mean(wp*Tp) # kinematic heat flux [m/s * c ]
    T_mean = np.mean(T)
    T_skew = sc.stats.skew(T)
    H      = rho*Cp*Cov_wT # dynamic sensible heat flux (Energy flux/ unit area)
    Lmo    = -ustar**3/(kv*g*( Cov_wT/T_mean)) # [m]
    stab   = z_ref/Lmo # [adim.]
    # multiply by Ubar to obtain the relative length scales:
    Tw     = integral_scale_fft(w)/fs # in seconds
    Tu     = integral_scale_fft(u)/fs # in seconds
    Tt     = integral_scale_fft(T)/fs # in seconds
    Iu     = np.std(u)/Ubar # turbulent intensity
    Re_star = z_ref*ustar/nu
    T_stdv = np.std(T)
    U_stdv = np.std(u)
    W_stdv = np.std(w)
    T_star = Cov_wT/ustar
    myphim = phim(stab)
    myphis = phis(stab)
    # estimate epsilon and kolmogorov scales
    mech_prod = ustar**3/(kv*z_ref)*myphim # check phim
    buoy_prod = g/T_mean*Cov_wT
    epsilon = mech_prod + buoy_prod # neglect transport and pressure redistribution
    tau_eta = (nu/epsilon)**(1/2)
    eta = (nu**3/epsilon)**(1/4)
    tke = 1/2*( np.mean(up**2) + np.mean(vp**2) + np.mean(wp**2))

    # timescales of shear (=dU/dz) and dissiplation
    shear_ts = ustar/kv/z_ref*myphim
    diss_ts = tke/epsilon

    def psim(stab, stab0):
        ''' integrate stability correction function, as in Panofski 1963'''
        myfun = lambda x: (1 - phim(x))/x
        return sc.integrate.quad(myfun, stab0, stab)[0]

    # local dissipation rate at z0:
    # including the stability correction function:
    # myfunz0 = lambda x: kv*Ubar/ustar - np.log(z_ref/x) + psim(stab, x/Lmo)
    # z0 = sc.optimize.fsolve(myfunz0, 0.05)[0]

    z0 = z_ref*np.exp( -(kv*Ubar)/ustar )
    z0corr = z_ref*np.exp( -(kv*Ubar)/ustar  - psim(z_ref/Lmo, 0))
    # z0 = z0corr

    stab0 = z0/Lmo
    myphim0 = phim(stab0) # should be very very close to 1
    # epsilon0 = ustar**3/(kv*z0) + buoy_prod # do not need divide by myphim0
    # epsilon0 = ustar**3/(kv*z0) # do not need divide by myphim0
    # tau_eta0 = (nu/epsilon0)**(1/2)
    # eta0 = (nu**3/epsilon0)**(1/4)

    # if z0 > 2:
    #     print('turb quant ERROR: z0 unphysical, set it to np.nan')
    #     z0 = np.nan

    # a = 5 # coefficient for eta / order of magnitude
    # h0 = z0 + a*eta
    # to avoid problems with unphysical values of z0, to be removed later::
    # epsilon0 = max(ustar**3/(kv*h0) + buoy_prod, 1E-9) # do not need divide by myphim0
    h0 = 7.5*z0 # see Brutsaert 1975
    buoy_prod0 = g/T_mean*Cov_wT
    mech_prod0 = ustar**3/(kv*h0)*phim(h0/Lmo)
    epsilon0 = buoy_prod0 + mech_prod0
    # epsilon0 = ustar**3/(kv*h0) # do not need divide by myphim0
    t0 = epsilon0**(-1/3)*h0**(2/3)
    tau_eta0 = (nu/epsilon0)**(1/2)
    eta0 = (nu**3/epsilon0)**(1/4)
    # Roughness Reynolds number
    Re0 = ustar*z0/nu

    stats = {'Ubar':Ubar, 'Tbar':T_mean, 'ustar':ustar,
             'H':H, 'Cov_wT':Cov_wT, 'Lmo':Lmo,
             'stab':stab, 'Tu':Tu, 'Tw':Tw, 'Tt':Tt,
             'phim':myphim, 'phis':myphis,
            'T_stdv':T_stdv, 'U_stdv':U_stdv, 'W_stdv':W_stdv,
             'T_star':T_star, 'T_skew':T_skew ,
            'Re_star':Re_star, 'n_obs':n_obs,
            'Iu':Iu, 'z0corr':z0corr,
            'epsilon':epsilon, 'tau_eta':tau_eta, 'eta':eta,
             'z0':z0, 'stab0':stab0, 'myphim0':myphim0, 'epsilon0':epsilon0,
             'tau_eta0':tau_eta0, 'eta0':eta0,
             'Re0':Re0, 'tke':tke, 'shear_ts':shear_ts, 'diss_ts':diss_ts,
             'h0':h0, 't0':t0}
    return stats


def scalar_quant(scalar, df, turb_stats, fs, z_ref, nobs_min = 6000):
    '''--------------------------------------------------------------------
    compute flux and other statistics for a scalar quantity c
    INPUT::
        -> c = my scalar time series
        -> w = my vartical velocity time series
        -> turb_stats: dictionary output of 'turb_quant' function
        -> fs: sampling frequency
        -> z_ref: obs height
        -> nobs_min = 6000: minimum length of tine series, in sampling points
    OUTPUT::
        -> sdf: dictionary with scalar fluxes and other statistics
        ----------------------------------------------------------------'''
    mydict = {}
    n_obs = df.shape[0]
    if n_obs < nobs_min:
        print('turb quant WARNING:: too many missing data -> skip this run')
        print('only length = ', n_obs, ' sampling points')
    wp = df['w'].values - df['w'].mean()
    cp = df[scalar].values - df[scalar].mean()
    mydict['stdv'] = np.std(cp)
    mydict['mean'] = df[scalar].mean()
    mydict['flux_ec'] = np.mean(wp*cp)
    mydict['cstar'] = mydict['flux_ec']/turb_stats['ustar']
    mydict['Tc'] = integral_scale_fft(cp)/fs # in time units [s]
    mydict['Lc'] = integral_scale_fft(cp)/fs*turb_stats['Ubar'] # in meters (Taylor Hypothesis)
    # surface renewal analysis / fluxes::
    if scalar == 'u':
        phix = turb_stats['phim']
    else:
        phix = turb_stats['phis']

    # constants for the surface renewal estimates
    ksrE = 6
    ksr0 = 6
    ksrI = 6
    # minmaxdiff = np.quantile(df[scalar].values, 0.90) - np.quantile(df[scalar].values, 0.1)

    # timesc = np.int(np.ceil(turb_stats['Tw']*md.fs))
    timesc = np.int(np.ceil(mydict['Tc']*md.fs))
    # slopes = (df[scalar].values[timesc:] - df[scalar].values[:-timesc])/timesc
    slopes = (df[scalar].values[timesc:] - df[scalar].values[:-timesc])
    # wvals  = wp[timesc:]
    # aps = np.mean(slopes[(slopes > 0) & (wvals > 0 )])
    aps = np.mean(slopes[(slopes > 0)])

    mydict['flux_srE'] = ksrE*aps*np.sqrt(md.DM[scalar]/turb_stats['tau_eta' ])*np.sign(mydict['flux_ec'])
    mydict['flux_sr0'] = ksr0*aps*np.sqrt(md.DM[scalar]/turb_stats['tau_eta0'])*np.sign(mydict['flux_ec'])
    mydict['flux_srI'] = ksrI*aps*np.sqrt(md.DM[scalar]/turb_stats['t0'      ])*np.sign(mydict['flux_ec'])

    # mydict['flux_srE'] = ksrE*mydict['stdv']*np.sqrt(md.DM[scalar]/turb_stats['tau_eta' ])*np.sign(mydict['flux_ec'])
    # mydict['flux_sr0'] = ksr0*mydict['stdv']*np.sqrt(md.DM[scalar]/turb_stats['tau_eta0'])*np.sign(mydict['flux_ec'])
    # mydict['flux_srI'] = ksrI*mydict['stdv']*np.sqrt(md.DM[scalar]/turb_stats['t0'      ])*np.sign(mydict['flux_ec'])
    mydict['DC_E'] = md.kv*turb_stats['ustar']/phix*mydict['stdv']*np.sqrt(turb_stats['tau_eta']/md.DM[scalar])
    mydict['DC_0'] = md.kv*turb_stats['ustar']/phix*mydict['stdv']*np.sqrt(turb_stats['tau_eta0']/md.DM[scalar])
    mydict['DC_I'] = md.kv*turb_stats['ustar']/phix*mydict['stdv']*np.sqrt(turb_stats['t0']/md.DM[scalar])
    mydict['flux_fv'] = mydict['stdv']*(turb_stats['T_stdv']/0.95**3*md.kv*md.g*z_ref/turb_stats['Tbar'])**0.5*np.sign(mydict['flux_ec'])

    # SR transport efficiency (= correlation coeff btw c and w) obtained from surface renewal:
    mydict['tr_eff_sr0'] = mydict['flux_sr0']/turb_stats['W_stdv']/mydict['stdv']
    mydict['tr_eff_srI'] = mydict['flux_srI']/turb_stats['W_stdv']/mydict['stdv']
    mydict['tr_eff_srE'] = mydict['flux_srE']/turb_stats['W_stdv']/mydict['stdv']
    mydict['tr_eff_ec' ]  = mydict['flux_ec']/turb_stats['W_stdv']/mydict['stdv'] # this is simply Rcw

    # large eddy model flux::
    alphac = 1
    mydict['Kt'        ] = md.kv*turb_stats['ustar']*md.z_ref/phix
    mydict['tau_relax' ] =  md.z_ref**2/mydict['Kt']
    mydict['flux_lem'  ] = alphac*mydict['stdv']*np.sqrt(mydict['Kt']/mydict['tau_relax' ])*np.sign(mydict['flux_ec'])
    mydict['tr_eff_lem'] = mydict['flux_lem']/turb_stats['W_stdv']/mydict['stdv']
    return mydict


def mixed_moments(c, w):
    '''------------------------------------------------------------------------
    compute mixed moments and cumulant expansion quantities
    From Olli Peltola codes
    for a scalar time series c.
    Moments Mij = i-th order in the scalar c, j-th order in the vert. velocity w
    ------------------------------------------------------------------------'''
    # if normalize: # if added by EZ to apply to partial time series - filtered
    cn = (c - np.mean(c))/np.std(c)
    wn = (w - np.mean(w))/np.std(w)
    # else:
    #     cn = c
    #     wn = w
    # M11 = Rcw correlation coefficient i.e. transport efficiency
    Rcw = np.mean(cn*wn)
    M03 = np.mean(wn**3)      # vert vel skewness
    M04 = np.mean(wn**4)      # vert vel skewness
    M40 = np.mean(cn**4)      # scalar   skewness
    M30 = np.mean(cn**3)      # scalar   skewness
    M21 = np.mean(cn**2*wn)
    M12 = np.mean(cn*wn**2)
    M13 = np.mean(cn*wn**3)
    # M04 = np.mean(cn**4)      # scalar kurtosis / flattness
    # CEM and ICEM - From Olli Peltola code
    ICEM = (1/(2*(np.sqrt(2*np.pi)*Rcw)))*(M21-M12)
    # CEM approximation to Delta So
    C1=(1+Rcw)*((1/6)*(M03-M30)+0.5*(M21-M12))
    C2=-((1/6)*(2-Rcw)*(M03-M30)+0.5*(M21-M12))
    C3=(1+Rcw)/(Rcw*(2*np.pi)**0.5)
    XD1=(1+Rcw)
    CEM=C3*((2*C1/(XD1**2))+(C2/XD1))
    d_Time_CEM=-M30/(3*np.sqrt(2*np.pi))

    # Repeat for the case of upward fluxes
    if Rcw > 0:
        cn=-cn
        Rwcp=np.mean(wn*cn)
        M03p=np.mean(wn**3)
        M30p=np.mean(cn**3)
        M21p=np.mean((cn**2)*wn)
        M12p=np.mean((wn**2)*cn)
        ICEM = (1/(2*(np.sqrt(2*np.pi)*Rwcp)))*(M21p-M12p)
        C1   = (1+Rwcp)*((1/6)*(M03p-M30p)+0.5*(M21p-M12p))
        C2   = -((1/6)*(2-Rwcp)*(M03p-M30p)+0.5*(M21p-M12p))
        C3   = (1+Rwcp)/(Rwcp*(2*np.pi)**0.5)
        XD1  = (1+Rwcp)
        CEM  = C3*((2*C1/(XD1**2))+(C2/XD1))
        d_Time_CEM = -M30p/(3*np.sqrt(2*np.pi))

    res =     {'Rcw':Rcw,
               'M03':M03,
               'M04':M04,
               'M40':M40,
               'M30':M30,
               'M21':M21,
               'M12':M12,
               'M13':M13,
               'dSo_ICEM':ICEM,
               'dSo_CEM':CEM,
               'd_Time_CEM':d_Time_CEM,
                }
    return res



def partial_mixed_moments(c, w, N = 2**14, normalize = False):
    '''------------------------------------------------------------------------
    Moments Mij = i-th order in the scalar c, j-th order in the vert. velocity w
    If normalize = False, assume original variables are already normalized
    ------------------------------------------------------------------------'''
    if normalize: # if added by EZ to apply to partial time series - filtered
        cn = (c - np.mean(c))/np.std(c)
        wn = (w - np.mean(w))/np.std(w)
        # vertical velocity was already normalized
        # wn = w
    else:
        cn = c
        wn = w

    # sc = np.std(cn)
    # sw = np.std(wn)
    # M11 = Rcw correlation coefficient i.e. transport efficiency
    # Rcw = np.mean(cn*wn)
    # M03 = np.mean(wn**3)      # vert vel skewness
    # M04 = np.mean(wn**4)      # vert vel skewness
    # M40 = np.mean(cn**4)      # scalar   skewness
    # M30 = np.mean(cn**3)      # scalar   skewness
    # M21 = np.mean(cn**2*wn)
    # M12 = np.mean(cn*wn**2)
    # M13 = np.mean(cn*wn**3)


    Rcw = 1/N*np.sum(cn*wn)
    M03 = 1/N*np.sum(wn**3)      # vert vel skewness
    M04 = 1/N*np.sum(wn**4)      # vert vel skewness
    M40 = 1/N*np.sum(cn**4)      # scalar   skewness
    M30 = 1/N*np.sum(cn**3)      # scalar   skewness
    M21 = 1/N*np.sum(cn**2*wn)
    M12 = 1/N*np.sum(cn*wn**2)
    M13 = 1/N*np.sum(cn*wn**3)

    # Rcw = 1/N*np.sum(cn*wn)/sc/sw
    # M03 = 1/N*np.sum(wn**3)/sw**3
    # M04 = 1/N*np.sum(wn**4)/sw**4
    # M40 = 1/N*np.sum(cn**4)/sc**4
    # M30 = 1/N*np.sum(cn**3)/sc**3
    # M21 = 1/N*np.sum(cn**2*wn)/sc**2/sw
    # M12 = 1/N*np.sum(cn*wn**2)/sc/sw**2
    # M13 = 1/N*np.sum(cn*wn**3)/sc/sw**3

    res =     {'Rcw':Rcw,
               'M03':M03,
               'M04':M04,
               'M40':M40,
               'M30':M30,
               'M21':M21,
               'M12':M12,
               'M13':M13,
               }
    return res


def Delta_So(c, w):
    ''' Compute relative weights of quadrants -
    based on code by Gaby and Olli '''
    eps = np.finfo(float).eps
    cn = (c - np.mean(c))/np.std(c)
    wn = (w - np.mean(w))/np.std(w)
    Rwc = np.mean(wn*cn)
    F = np.sum(wn*cn) + eps
    Q1 = np.logical_and(wn > 0, cn > 0)
    Q2 = np.logical_and(wn < 0, cn > 0)
    Q3 = np.logical_and(wn < 0, cn < 0)
    Q4 = np.logical_and(wn > 0, cn < 0)
    Size_Q1 = np.size(Q1[Q1])
    Size_Q2 = np.size(Q2[Q2])
    Size_Q3 = np.size(Q3[Q3])
    Size_Q4 = np.size(Q4[Q4])
    Total_size = Size_Q1 + Size_Q2 + Size_Q3 + Size_Q4
    FQ2 = np.sum(wn[Q2]*cn[Q2])
    FQ3 = np.sum(wn[Q3]*cn[Q3])
    FQ1 = np.sum(wn[Q1]*cn[Q1])
    FQ4 = np.sum(wn[Q4]*cn[Q4])
    if Rwc > eps:
        dSo = (FQ3-FQ1)/(F+eps)
        Frac_Sweep = Size_Q3/Total_size
        Frac_Eject = Size_Q1/Total_size
        dQ_Time = Frac_Sweep-Frac_Eject
    else:
        dSo = (FQ2-FQ4)/(F+eps)
        Frac_Sweep = Size_Q2/Total_size
        Frac_Eject = Size_Q4/Total_size
        dQ_Time = Frac_Sweep-Frac_Eject

    res = {'dSo':dSo,
           'dQ_Time':dQ_Time
           }
    return res


def Transport_Eff(c, w):
    '''compute transport efficiency
    Based on Olli and Gaby code'''
    eps = np.finfo(float).eps
    cn = (c - np.mean(c))/np.std(c)
    wn = (w - np.mean(w))/np.std(w)
    Rwc = np.mean(wn*cn)
    F = wn*cn + eps
    Qup = F>0
    Qdn = F<0
    F_up = np.sum(F[Qup])
    F_dn = np.sum(F[Qdn])
    if Rwc>0:
        eT=(F_up+F_dn)/(F_up+eps)
    else:
        eT=(F_dn+F_up)/(F_dn+eps)
    # Transport efficiency for a joint-Gaussian process
    NUM_e = 2*np.pi*np.abs(Rwc)
    DEN_e =  ( 2*np.sqrt(1-Rwc**2)+np.pi*np.abs(Rwc)+
               2*np.abs(Rwc)*np.arcsin(np.abs(Rwc)))
    eT_GAUSS = NUM_e/(DEN_e + eps)
    res = {'eT':eT,
           'eT_GAUSS':eT_GAUSS
           }
    return res


def REA_Beta(c, w):
    '''compute transport efficiency
    Based on Olli and Gaby code'''
    # eps = np.finfo(float).eps
    cn = (c - np.mean(c))/np.std(c)
    wn = (w - np.mean(w))/np.std(w)
    Rwc = np.mean(wn*cn)
    # F = wn*cn + eps
    Qup = wn > 0
    Qdn = wn < 0
    Cup = np.mean(cn[Qup])
    Cdn = np.mean(cn[Qdn])
    Wup = np.mean(wn[Qup])
    Wdn = np.mean(wn[Qdn])
    Beta_p = (Wup-Wdn)**(-1)
    Slope_REA = (Cup-Cdn)/(Wup-Wdn)
    Slope_REA_up = Cup/Wup
    Slope_REA_dn = Cdn/Wdn
    flux_REA = Beta_p*(Cup-Cdn)*np.std(c)*np.std(w)
    res = {'Beta_p':Beta_p,
           'Slope_REA':Slope_REA,
           'Slope_REA_up':Slope_REA_up,
           'Slope_REA_dn':Slope_REA_dn,
           'Rwc':Rwc,
           'flux_REA':flux_REA
           }
    return res


def REA_Beta_Milne(c, w):
    '''compute transport efficiency
    Based on Olli and Gaby code'''
    cn  = (c - np.mean(c))/np.std(c)
    wn  = (w - np.mean(w))/np.std(w)
    Rwc = np.mean(wn*cn)
    M04 = np.mean(wn**4)
    M13 = np.mean(cn*wn**3)
    Qup = wn > 0
    Qdn = wn < 0
    Cup = np.mean(cn[Qup])
    Cdn = np.mean(cn[Qdn])
    Beta_REA = Rwc/(Cup-Cdn)
    Beta_Milne = 0.557/(1+(4/27)*(3*M04/4-M13/Rwc))
    res = {
           'Beta_REA':Beta_REA,
           'Beta_Milne':Beta_Milne,
           'M04':M04,
           'M13':M13,
           'Rwc':Rwc
           }
    return res


def compute_eT(c, w, normalize = True):
    """compute transport efficiency
    of scalar c given vertical velocity w"""
    if normalize:
        cn = (c - np.mean(c))/np.std(c)
        wn = (w - np.mean(w))/np.std(w)
    else:
        cn = c
        wn = w
    fii = cn*wn
    flux = np.mean(fii)
    if flux > 0:
        eT = 1 - np.abs(np.sum(fii[fii < 0])/np.sum(fii[fii > 0]))
    else:
        eT = 1 - np.abs(np.sum(fii[fii > 0]) / np.sum(fii[fii < 0]))
    return eT


# def quadrant_fluxes(c, w, thresholding = False, threshold = 0.5):
#     '''------------------------------------------------------------------------
#     computes the fractional contribution of the 4 quandrants to
#     turbulent fluxes (F), energy fluctuations, and total time (T).
#     ------------------------------------------------------------------------'''
#     cp0  = c - np.mean(c)
#     wp0  = w - np.mean(w)
#     nfluxii = wp0*cp0/np.std(wp0)/np.std(cp0)
#     if thresholding:
#         # see Raupach 1981:
#         mymask = np.abs(nfluxii) > threshold*np.mean(np.abs(nfluxii))
#         wp = wp0[mymask]
#         cp = cp0[mymask]
#     else:
#         wp = wp0
#         cp = cp0
#
#     m = np.size(wp)
#     ej = np.zeros(m , dtype = bool)
#     sw = np.zeros(m , dtype = bool)
#     inw = np.zeros(m , dtype = bool)
#     outw = np.zeros(m , dtype = bool)
#     flux_sign = np.sign (np.mean(wp*cp) )
#     for ii in range(m):
#         if wp[ii] > 0 and cp[ii]*flux_sign > 0: # ejection
#             ej[ii] = True
#         elif wp[ii] < 0 and cp[ii]*flux_sign < 0: # sweep
#             sw[ii] = True
#         elif wp[ii] < 0 and cp[ii]*flux_sign > 0: # inward interaction
#             inw[ii] = True
#         elif wp[ii] > 0 and cp[ii]*flux_sign < 0: # outward interaction
#             outw[ii] = True
#
#     fluxii = wp*cp
#     energies = cp*cp
#     total_variance = np.mean(energies)
#     # net_flux = np.mean(fluxii)
#
#     # energy fractions:
#     E_eje   = 1/m*np.sum(energies[ej])/total_variance
#     E_swe   = 1/m*np.sum(energies[sw])/total_variance
#     E_inwa  = 1/m*np.sum(energies[inw])/total_variance
#     E_outwa = 1/m*np.sum(energies[outw])/total_variance
#
#     # residence time fractions:
#     T_eje   = np.size(fluxii[ej])/np.size(fluxii)
#     T_swe   = np.size(fluxii[sw])/np.size(fluxii)
#     T_inwa  = np.size(fluxii[inw])/np.size(fluxii)
#     T_outwa = np.size(fluxii[outw])/np.size(fluxii)
#
#     # absolute value components of the flux []
#     FR_eje = 1/m*np.abs(np.sum(fluxii[ej]))
#     FR_swe = 1/m*np.abs(np.sum(fluxii[sw]))
#     FR_inwa = 1/m*np.abs(np.sum(fluxii[inw]))
#     FR_outwa = 1/m*np.abs(np.sum(fluxii[outw]))
#
#     quad_fluxes = {
#                    'FR_sw':FR_swe,
#                    'FR_ej':FR_eje,
#                    'FR_in':FR_inwa,
#                    'FR_out':FR_outwa,
#                    'T_sw': T_swe,
#                    'T_ej': T_eje,
#                    'T_in': T_inwa,
#                    'T_out':T_outwa,
#                    'E_sw' :E_swe,
#                    'E_ej' :E_eje,
#                    'E_in' :E_inwa,
#                    'E_out':E_outwa
#                    }
#     return quad_fluxes


def quadrant_ratios(df, mysc='CH4', refsc='H2O', thresh=0):
    '''compute fraction of energetic flux, variance and time
       for a scalar time series mysc, with respect to a reference
       scalar quantity refsc, and w being vertical velocity

      Return (eps_) the ratio of direct vs indirects contributions to
      energy, flux and time for reference and my scalar
      Also:
      Returns the enegrgetic fractions on flux, energy and time
      and the dimensional methane positive flux contribution only (me_dir_flux)

       '''
    me = df[mysc].values
    wa = df[refsc].values
    w = df['w'].values

    mep = me - np.mean(me)
    wap = wa - np.mean(wa)
    wp = w - np.mean(w)

    R_me_wa = np.mean(mep*wap)/np.std(mep)/np.std(wap)

    mefluxii = mep*wp
    wafluxii = wap*wp
    meflux = np.mean(mefluxii)
    medirflux = 1/np.size(wp)*np.sum(mefluxii[mefluxii > 0]) # only positive contrib

    # select threshold = 1stdv of abs fluxes
    mefluxii_stdv = np.std(np.abs(mefluxii))
    wafluxii_stdv = np.std(np.abs(wafluxii))

    mevarii = mep**2
    wavarii = wap**2
    # mevar = np.mean(mevarii)

    # boolean indicating direct vs indirect contributions to the flux
    # thresh = 0
    # medirect = mefluxii > 0
    # wadirect = wafluxii > 0
    medirect = np.logical_and(mefluxii > 0, np.abs(mefluxii) > thresh*mefluxii_stdv)
    wadirect = np.logical_and(wafluxii > 0, np.abs(wafluxii) > thresh*wafluxii_stdv)

    meindir = np.logical_and(mefluxii < 0, np.abs(mefluxii) > thresh*mefluxii_stdv)
    waindir = np.logical_and(wafluxii < 0, np.abs(wafluxii) > thresh*wafluxii_stdv)

    # energies
    varwap = wavarii[wadirect]
    varwan = wavarii[waindir]
    varmep = mevarii[medirect]
    varmen = mevarii[meindir]
    eps_var_wa = np.sum(varwap)/np.sum(varwan)
    eps_var_me = np.sum(varmep)/np.sum(varmen)
    # frac_ener_var = (eps_var_me-eps_var_wa)*np.mean(varmen)/mevar
    frac_ener_var = (eps_var_me-eps_var_wa)*np.sum(varmen)/np.sum(mevarii)
    if (eps_var_me-eps_var_wa) < 0:
        frac_ener_var = 0 # water more imbalanced tham CH4, no ebullition

    # fluxes
    fiwap = wafluxii[wadirect]
    fiwan = wafluxii[waindir]
    eps_flux_wa = np.sum(fiwap)/np.sum(fiwan)
    fimep = mefluxii[medirect]
    fimen = mefluxii[meindir]
    eps_flux_me = np.sum(fimep)/np.sum(fimen)
    me_flux_eT = 1 - np.abs(np.sum(fimen)/np.sum(fimep))
    wa_flux_eT = 1 - np.abs(np.sum(fiwan)/np.sum(fiwap))
    frac_ener_flux = (1 - eps_flux_wa/eps_flux_me)*medirflux/meflux
    if eps_flux_me - eps_flux_wa > 0:
        frac_ener_flux = 0 # water more imbalanced tham CH4, no ebullition

    # times
    dirt_me = np.size(fimep)
    indt_me = np.size(fimen)
    dirt_wa = np.size(fiwap)
    indt_wa = np.size(fiwan)
    eps_time_wa = dirt_wa/indt_wa
    eps_time_me = dirt_me/indt_me
    frac_ener_time = (eps_time_me - eps_time_wa)*indt_me/(indt_me + dirt_me)
    if (dirt_me/indt_me - dirt_wa/indt_wa) < 0:
        frac_ener_time = 0


    res = {'en_frac_flux': frac_ener_flux,
           'en_frac_var': frac_ener_var,
           'en_frac_time': frac_ener_time,
           'wa_di_time': eps_time_wa,
           'wa_di_var':  eps_var_wa,
           'wa_di_flux': eps_flux_wa,
           'me_di_time': eps_time_me,
           'me_di_var':  eps_var_me,
           'me_di_flux': eps_flux_me,
           'me_eT_flux': me_flux_eT,
           'wa_eT_flux': wa_flux_eT,
           'me_dir_flux': medirflux,
           'R_me_wa': R_me_wa
           }
    return res


# new quadrant partitioning based on Lorenz curve of the flux:
def quadrant_partition(df, mysc='CH4', refsc='H2O', cond_x='c',
                       mystdv_en=0.4, nstdvs_en=5, mystdv_fl=0.2, nstdvs_fl=5):
    """
    quadrant analysis to compute ebullition fraction
    :return:
    """

    c = df[mysc].values
    r = df[refsc].values
    w = df['w'].values

    cn = (c - np.mean(c)) / np.std(c)
    rn = (r - np.mean(r)) / np.std(r)
    wn = (w - np.mean(w)) / np.std(w)

    # compute contributions to fluxes and energies
    wafii = wn * rn
    mefii = wn * cn
    wa2 = rn * rn
    me2 = cn * cn
    ccdiff = np.abs(me2 - wa2)
    wcdiff = np.abs(mefii - wafii)

    if cond_x == 'wc':
        wc_is_ener = np.abs(wcdiff) > nstdvs_fl * mystdv_fl

    elif cond_x == 'wcp':
        wc_is_ener = np.logical_and(np.abs(wcdiff) > nstdvs_fl * mystdv_fl,
                                    mefii > 0)
    elif cond_x == 'c':
        wc_is_ener = np.abs(ccdiff) > nstdvs_en * mystdv_en

    elif cond_x == 'cp':
        wc_is_ener = np.logical_and(np.abs(ccdiff) > nstdvs_en * mystdv_en,
                                    mefii > 0)
    else:
        print("quadrant fluxes error: insert a valid cond_x \
        (= 'c' or 'wc' for fluxes or variances respectively")

    enerfii = mefii[wc_is_ener]
    backfii = mefii[np.logical_not(wc_is_ener)]
    enervii = me2[wc_is_ener]
    backvii = me2[np.logical_not(wc_is_ener)]

    cn_ener = cn[wc_is_ener]  # background part of the time series
    cn_back = cn[np.logical_not(wc_is_ener)]  # background part of the cn time series
    wn_ener = wn[wc_is_ener]  # background part of the cn wn time series
    wn_back = wn[np.logical_not(wc_is_ener)]  # background part of the wn time series

    en_mom = mixed_moments(cn_ener, wn_ener)
    ba_mom = mixed_moments(cn_back, wn_back)

    res = {}

    res['en_eT_flux'] = (1 - np.abs( np.sum(enerfii[enerfii < 0])
                                     / np.sum(enerfii[enerfii > 0])))
    res['ba_eT_flux'] = (1 - np.abs( np.sum(backfii[backfii < 0])
                                     / np.sum(backfii[backfii > 0])))
    res['en_eT_var'] = 1 - np.sum(enervii[enerfii < 0])/np.sum(enervii[enerfii > 0])
    res['ba_eT_var'] =  1 - np.sum(backvii[backfii < 0])/np.sum(backvii[backfii > 0])
    res['en_eT_time'] = 1 - np.size(enervii[enerfii < 0]) / np.size(enervii[ enerfii > 0])
    res['ba_eT_time'] = 1 - np.size(backvii[backfii < 0]) / np.size(backvii[backfii > 0])
    res['en_frac_flux'] = np.sum(enerfii) / np.sum(mefii)
    res['en_frac_var'] = np.sum(enervii) / np.sum(me2)
    res['en_frac_time'] = np.size(enerfii) / np.size(mefii)
    res['ba_frac_time'] = 1 - np.size(enerfii) / np.size(mefii)

    res['cn_ener'] = cn_ener
    res['cn_back'] = cn_back
    res['wn_ener'] = wn_ener
    res['wn_back'] = wn_back


    res['stdv_c2'] = np.std(ccdiff)
    res['stdv_cw'] = np.std(wcdiff)

    res['en_M40'] = en_mom['M40']
    res['en_M30'] = en_mom['M30']
    res['en_M21'] = en_mom['M21']
    res['en_M12'] = en_mom['M12']
    res['en_M13'] = en_mom['M13']
    res['en_Rcw'] = en_mom['Rcw']

    res['ba_M40'] = ba_mom['M40']
    res['ba_M30'] = ba_mom['M30']
    res['ba_M12'] = ba_mom['M12']
    res['ba_M21'] = ba_mom['M21']
    res['ba_M13'] = ba_mom['M13']
    res['ba_Rcw'] = ba_mom['Rcw']

    return res

def wavelet_partition_old(df, mysc='CH4', refsc='H2O', nmin_obs=2**14,
                      # minsize = 16,
                      # maxsize = 2**14,
                      toplotsc='CO2',
                      minscale = 0.5,
                      maxscale = 500,
                      mystdv = 0.2, nstdvs = 3, wavelet = 'haar',
                      cond_x = 'c', plot = False,
                      datetime = None):
    """
    Partition a time series of concentrations based on Wavelet reansform
    """
    # remove excess observations if time series is longer than 2*14
    N = nmin_obs
    if df.shape[0] < N:
        print('wavelet partition error: time series not long enough')
    else:
        df = df[:N]


    turb_quants = turb_quant(df, md.fs, md.z_ref, nobs_min=nmin_obs)

    wn = (df['w'] - np.mean(df['w'])) / np.std(df['w'])
    cn = (df[mysc] - np.mean(df[mysc])) / np.std(df[mysc])
    rn = (df[refsc] - np.mean(df[refsc])) / np.std(df[refsc])


    Rcr = np.mean(cn*rn)

    WTcn = pywt.wavedec(cn, wavelet=wavelet)
    WTrn = pywt.wavedec(rn, wavelet=wavelet)
    WTwn = pywt.wavedec(wn, wavelet=wavelet)



    # [len(x) for x in WTc] 1-8192

    # all scales - skipping coarse grained signal
    # sizes = np.arange(len(WTc)) # skip coarse grained signal
    # sizes = np.arange(len(WTc)) # skip coarse grained signal
    # sizes = np.array([len(x) for x in WTc])
    # M = np.real(np.size(sizes))
    # scales = np.array([2.0**(M-x) for x in sizes])

    # Following Scanlon and Albertson notation
    # mm = np.arange(1, len(WTc)) # skip coarse grained signal
    mm0 = np.arange(len(WTcn)-1, 0, -1) # skip coarse grained signal
    M = np.log2(N)
    mm = np.hstack((M, mm0)) # add scale of coarse grained signal
    scales = 2**mm
    sizes = 2**(M-mm)

    # sizes2 = np.array([len(x) for x in WTc])[1:] # remove coarse grained signal
    # scales2 = N/sizes2

    ti = scales/md.fs
    tin = ti/turb_quants["Tw"]

    # onscales = tin[tin > minscale]
    # onsizes = sizes[tin > minscale]
    # minsize = onsizes[-5]

    # condition to keep the coefficients:

    # keep only scales of interest
    # cond_keep = np.logical_and(tin >= minscale, sizes >= minsize)
    cond_keep = np.logical_and(tin >= minscale, tin <= maxscale)
    onsizes = sizes[cond_keep]
    ontins = tin[cond_keep]
    maxsize = np.max(onsizes)
    minsize = np.min(onsizes)
    maxtin = np.max(ontins)


    WTcn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTcn)]
    WTrn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTrn)]
    WTwn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTwn)]

    # cn_filtered = pywt.waverec(WTcn0, wavelet=wavelet)
    # rn_filtered = pywt.waverec(WTrn0, wavelet=wavelet)
    # wn_filtered = pywt.waverec(WTwn0, wavelet=wavelet)

    # cn = cn_filtered
    # rn = rn_filtered
    # wn = wn_filtered
    #
    #
    # eT_me_nofilter = compute_eT(cn, wn)
    # eT_wa_nofilter = compute_eT(rn, wn)
    # # eT_me = compute_eT(cn_filtered, wn_filtered)
    # # eT_wa = compute_eT(rn_filtered, wn_filtered)
    #
    # # eT_me = compute_eT(cn_filtered, wn)
    # # eT_wa = compute_eT(rn_filtered, wn)
    #
    # eT_me = compute_eT(cn, wn)
    # eT_wa = compute_eT(rn, wn)

    # transform to arrays
    WTcn_non0 = [x for i, x in enumerate(WTcn) if cond_keep[i]]
    WTrn_non0 = [x for i, x in enumerate(WTrn) if cond_keep[i]]
    WTwn_non0 = [x for i, x in enumerate(WTwn) if cond_keep[i]]
    WAc0 = sa.coeffs_2_array(WTcn_non0)[0]
    WAr0 = sa.coeffs_2_array(WTrn_non0)[0]
    WAw0 = sa.coeffs_2_array(WTwn_non0)[0]

    # compute abs differences between CH4 and H2O
    # I am not using them now only outside to determine nstdv
    diff_c    = np.abs(WAc0 - WAr0)
    # diff_cn    = np.abs( (WAr0 - WAc0)/WAc0)
    diff_cn    = 0
    diff_c2   = np.abs(WAc0**2 - WAr0**2)
    diff_cw   = np.abs(WAc0 * WAw0 - WAr0 * WAw0)
    stdv_c    = np.std(diff_c)
    # stdv_cn    = np.std(diff_cn)
    stdv_cn    = 0
    stdv_c2   = np.std(diff_c2)
    stdv_cw   = np.std(diff_cw)


    WTen0  = [np.zeros(len(x)) for x in WTcn0]
    WTbn0  = [np.zeros(len(x)) for x in WTcn0]
    WTrbn0 = [np.zeros(len(x)) for x in WTcn0] # water background
    WTren0 = [np.zeros(len(x)) for x in WTcn0] # water background
    for i, (c, r, e, b, w ,rb, re) in enumerate(
                zip(WTcn0, WTrn0, WTen0, WTbn0, WTwn0, WTrbn0, WTren0)):
        n = np.size(c)
        for j in range(n):
            if cond_x == 'c2':
                condx = np.abs(c[j]**2 - r[j]**2) > nstdvs * mystdv
            elif cond_x == 'c':
                condx = np.abs(c[j] - r[j]) > nstdvs * mystdv
            elif cond_x == 'cn':
                # condx = np.logical_and(np.abs(c[j] - r[j]) > nstdvs * mystdv,
                # condx = np.abs((c[j] - r[j])) > 2* nstdvs * mystdv
               condx=np.logical_and(np.abs((r[j] - c[j])) > nstdvs * mystdv,
                                                            # np.abs(c[j]) > np.abs(r[j]))
                                       # c[j]**2 > np.mean(r**2))
                                       c[j]**2 > np.mean(WAr0 ** 2))
                # c[j]*w[j]> 0)
                      # w[j]*c[j] > w[j]*r[j])
                      # c[j]**2 > r[j]**2)
            elif cond_x == 'cp':
                condx = np.logical_and(np.abs(c[j] - r[j]
                              ) > nstdvs * mystdv, c[j]*w[j] > 0)
            elif cond_x == 'wc':
                condx = np.abs(c[j]*w[j] - r[j]*w[j]
                               ) > nstdvs * mystdv
            elif cond_x == 'wcp':
                condx = np.logical_and(np.abs(c[j]*w[j] - r[j]*w[j]) >
                               nstdvs * mystdv, c[j]*w[j] > 0)
            else:
                print('wavelet partition error: insert valid value for condx')

            if cond_keep[i]:
                if condx: # energetic case
                    b[j] = 0
                    e[j] = c[j]
                    rb[j] = 0
                    re[j] = r[j]
                else: # background case
                    e[j] = 0
                    b[j] = c[j]
                    rb[j] = r[j]
                    re[j] = 0

 # tests


    # COMPUTE DIRECT AND INDIRECT FLUX CONTRIBUTIONS FROM BACK / ENER FLUXES
    # include eddies of all sizes
    # now multiply each by its original stdv
    WTAc = sa.coeffs_2_array(WTcn0)[0]*np.std(df[mysc])
    # sz = sa.coeffs_2_array(WTcn0)[1]
    WTAe = sa.coeffs_2_array(WTen0)[0]*np.std(df[mysc])
    WTAb = sa.coeffs_2_array(WTbn0)[0]*np.std(df[mysc])
    WTAw = sa.coeffs_2_array(WTwn0)[0]*np.std(df['w'])
    WTAr = sa.coeffs_2_array(WTrn0)[0]*np.std(df[refsc])
    WTArb = sa.coeffs_2_array(WTrbn0)[0]*np.std(df[refsc])
    WTAre = sa.coeffs_2_array(WTren0)[0]*np.std(df[refsc])

    if plot:
        WTAc00 = sa.coeffs_2_array(WTcn0)[0]
        WTAr00 = sa.coeffs_2_array(WTrn0)[0]

        WTAr00 = sa.coeffs_2_array(WTrn0)[0]
        WTAr00 = sa.coeffs_2_array(WTrn0)[0]
        WTAe00 = sa.coeffs_2_array(WTen0)[0]
        # WTAe = sa.coeffs_2_array(WTen0)[0]

        plt.figure()
        plt.plot(WTAr00, WTAc00, 'o')
        plt.plot(WTAr00, WTAe00, 'or')
        plt.plot(WTAr00, WTAr00, 'k')
        plt.show()


    # plot Lorenz Curve for check
    # LORme = np.cumsum( np.sort(WTAc**2))/np.sum(WTAc**2)
    # LORwa = np.cumsum( np.sort(WTAr**2))/np.sum(WTAr**2)
    #
    # Xme = np.linspace(0, 1, np.size(LORme))
    #
    # plt.figure()
    # plt.plot(Xme, LORme, 'g')
    # plt.plot(Xme, LORwa, 'b')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.show()

    # [len(x) for x in WTen0]
    # [len(x) for x in WTrbn0]
    # [len(x[x>0]) for x in WTrbn0]
    # [len(x[x>0]) for x in WTbn0]
    # [len(x[x>0]) for x in WTen0]

    # compute contributions to the fluxes from each wavelet coefficient
    en_fii = WTAe * WTAw
    ba_fii = WTAb * WTAw
    me_fii = WTAc * WTAw
    wa_fii = WTAr * WTAw
    wb_fii = WTArb * WTAw
    we_fii = WTAre * WTAw

    # we_fii[np.abs(we_fii) > 0]
    # en_fii[np.abs(en_fii) > 0]

    wb_fdir = wb_fii[wb_fii > 0]
    we_fdir = we_fii[we_fii > 0]
    wa_fdir = wa_fii[wa_fii > 0]
    me_fdir = me_fii[me_fii > 0]
    en_fdir = en_fii[en_fii > 0]
    ba_fdir = ba_fii[ba_fii > 0]

    wb_find = wb_fii[wb_fii < 0]
    we_find = we_fii[we_fii < 0]
    wa_find = wa_fii[wa_fii < 0]
    me_find = me_fii[me_fii < 0]
    en_find = en_fii[en_fii < 0]
    ba_find = ba_fii[ba_fii < 0]

    wb_eT_wave = 1 - np.abs( np.sum(wb_find)/np.sum(wb_fdir))
    we_eT_wave = 1 - np.abs( np.sum(we_find)/np.sum(we_fdir))
    wa_eT_wave = 1 - np.abs( np.sum(wa_find)/np.sum(wa_fdir))
    me_eT_wave = 1 - np.abs( np.sum(me_find)/np.sum(me_fdir))
    en_eT_wave = 1 - np.abs( np.sum(en_find)/np.sum(en_fdir))
    ba_eT_wave = 1 - np.abs( np.sum(ba_find)/np.sum(ba_fdir))

   # eddy covariance fluxes (for comparison with wavelet over these scales only)
    me_flux_ec = np.mean( (df[mysc] - np.mean(df[mysc]))
                         *(df['w'] - np.mean(df['w'])))
    wa_flux_ec = np.mean( (df[refsc] - np.mean(df[refsc]))
                          *(df['w'] - np.mean(df['w'])))

    # wavelet fluxes (in the range of scales considered here only)
    me_flux_wt = 1 / N * np.sum(WTAc * WTAw)
    en_flux_wt = 1 / N * np.sum(WTAe * WTAw)
    ba_flux_wt = 1 / N * np.sum(WTAb * WTAw)
    wa_flux_wt = 1 / N * np.sum(WTAr * WTAw)
    wb_flux_wt = 1 / N * np.sum(WTArb * WTAw)
    we_flux_wt = 1 / N * np.sum(WTAre * WTAw)







    # mefluxn2 = 1 / N * np.sum(WTAc**2 * WTAw**2)
    # enfluxn2 = 1 / N * np.sum(WTAe**2 * WTAw**2)
    # bafluxn2 = 1 / N * np.sum(WTAb**2 * WTAw**2)
    # wafluxn2 = 1 / N * np.sum(WTAr**2 * WTAw**2)
    # wabfluxn2 = 1 / N * np.sum(WTArb**2 * WTAw**2)


    # same for the variances
    en_vii = WTAe ** 2
    ba_vii = WTAb ** 2
    me_vii = WTAc **2
    wa_vii = WTAr **2
    wb_vii = WTArb **2
    we_vii = WTAre **2


    # en_frac_flux = en_flux_wt / me_flux_wt
    # ba_frac_flux = ba_flux_wt / me_flux_wt
    # bw_frac_flux = np.sum(wb_fii)/np.sum(wa_fii)
    # ew_frac_flux = np.sum(we_fii)/np.sum(wa_fii)
    # en_frac_var = np.sum(en_vii)/np.sum(me_vii)
    # ba_frac_var = np.sum(ba_vii)/np.sum(me_vii)
    # bw_frac_var = np.sum(wb_vii)/np.sum(wa_vii)
    # ew_frac_var = np.sum(we_vii)/np.sum(wa_vii)


    # var_back_ttime = (bw_frac_var)*ba_frac_var

    # compute fraction of the total flux and variance in this range of scales:
    me_flufrac_scales = me_flux_wt/me_flux_ec
    wa_flufrac_scales = wa_flux_wt/wa_flux_ec
    me_varfrac_scales = 1/N*np.sum(WTAc**2)/np.var(df[mysc])
    wa_varfrac_scales = 1/N*np.sum(WTAr**2)/np.var(df[refsc])

    # fix from here
    # back to the time domain
    enn = pywt.waverec(WTen0, wavelet=wavelet)
    ban = pywt.waverec(WTbn0, wavelet=wavelet)
    # barn = pywt.waverec(WTrbn0, wavelet=wavelet)

    # print("wc MODIFIED!") ###
    # is_eb = enn**2 > np.var(ban)
    enm = enn - np.median(enn) # centered in the median
    is_eb = enm**2 > np.var(ban)
    # is_eb = np.logical_and(enm**2 > np.var(ban), enn > 0)
    # is_eb = enn > np.median(enn)
    # is_eb = np.logical_and(enn**2 > np.var(ban), enn > np.median(enn))
    # is_eb = enn > 0
    ###
    # is_eb = enn**2 > ban**2
    # is_eb = enn**2 > 0
    enn_is_eb = enn.copy()
    enn_is_eb[np.logical_not(is_eb)] = np.nan
    # is_eb = np.logical_and(enn > np.std(ban), enn > 0)
    en_frac_time = np.size(is_eb[is_eb])/np.size(ban)
    ba_frac_time = 1 - en_frac_time

    # measures based on fluxes
    cn_ebii = cn[is_eb].copy()
    rn_ebii = rn[is_eb].copy()
    wn_ebii = wn[is_eb].copy()

    en_frac_flux_T = np.sum(cn_ebii*wn_ebii)/np.sum(cn*wn)
    ba_frac_flux_T = 1 - en_frac_flux_T
    ew_frac_flux_T = np.sum(rn_ebii*wn_ebii)/np.sum(rn*wn)
    bw_frac_flux_T = 1 - ew_frac_flux_T

    en_frac_var_T = np.sum(cn_ebii**2)/np.sum(cn**2)
    ba_frac_var_T = 1 - en_frac_var_T
    ew_frac_var_T = np.sum(rn_ebii**2)/np.sum(rn**2)
    bw_frac_var_T = 1 - ew_frac_var_T



    cn_baii = cn[np.logical_not(is_eb)].copy()
    rn_baii = rn[np.logical_not(is_eb)].copy()
    wn_baii = wn[np.logical_not(is_eb)].copy()

    # average fluxes and stdv in the ebull / background times only
    enfa = np.mean(cn_ebii*wn_ebii)*np.std(df['w'])*np.std(df[mysc]) # ave flux
    enva = np.mean(cn_ebii*cn_ebii)*np.std(df[mysc])*np.std(df[mysc]) # ave flux
    wefa = np.mean(rn_ebii*wn_ebii)*np.std(df['w'])*np.std(df[refsc]) # ave flux
    weva = np.mean(rn_ebii*rn_ebii)*np.std(df[refsc])*np.std(df[refsc]) # ave flux
    bafa = np.mean(cn_baii*wn_baii)*np.std(df['w'])*np.std(df[mysc]) # ave flux
    bava = np.mean(cn_baii*cn_baii)*np.std(df[mysc])*np.std(df[mysc]) # ave flux
    wbfa = np.mean(rn_baii*wn_baii)*np.std(df['w'])*np.std(df[refsc]) # ave flux
    wbva = np.mean(rn_baii*rn_baii)*np.std(df[refsc])*np.std(df[refsc]) # ave flux
    mefa = np.mean(cn*wn)*np.std(df['w'])*np.std(df[mysc]) # ave flux
    meva = np.mean(cn*cn)*np.std(df[mysc])*np.std(df[mysc]) # ave flux
    wafa = np.mean(rn*wn)*np.std(df['w'])*np.std(df[refsc]) # ave flux
    wava = np.mean(rn*rn)*np.std(df[refsc])*np.std(df[refsc]) # ave flux

    # same for water:
    wen = pywt.waverec(WTren0, wavelet=wavelet)
    wbn = pywt.waverec(WTrbn0, wavelet=wavelet)

    # plt.figure()
    # plt.plot(enn)
    # plt.plot(ban)
    # plt.show()



    # eT_en = compute_eT(enn, wn_filtered)
    # eT_ba = compute_eT(ban, wn_filtered)
    # eT_we = compute_eT(wen, wn_filtered)
    # eT_wb = compute_eT(wbn, wn_filtered)

    # eT_en = compute_eT(enn, wn)
    # eT_ba = compute_eT(ban, wn)
    # eT_we = compute_eT(wen, wn)
    # eT_wb = compute_eT(wbn, wn)

    # ennt = cn[is_eb].copy()
    # went = rn[is_eb].copy()
    # enwt = wn[is_eb].copy()
    # bnnt = cn[np.logical_not(is_eb)].copy()
    # wbnt = rn[np.logical_not(is_eb)].copy()
    # bnwt = wn[np.logical_not(is_eb)].copy()
    #
    # eT_en = compute_eT(ennt, enwt)
    # eT_ba = compute_eT(bnnt, bnwt)
    # eT_we = compute_eT(went, enwt)
    # eT_wb = compute_eT(wbnt, bnwt)

    # print("wc MODIFIED!") ###
    # is_eb_wa = wen**2 > np.var(wbn)
    # is_eb_wa = wen > np.median(wen)
    # wem = wen - np.median(wen) # centered in the median
    wem = wen - np.median(wen) # centered in the median
    is_eb_wa = wem**2 > np.var(wbn)
    # is_eb_wa = np.logical_and(wem**2 > np.var(wbn), wen > 0)
    # is_eb_wa = np.logical_and(wen**2 > np.var(wbn), wen > np.median(wen))
    # is_eb_wa = wen > 0
    # wen_is_eb = wen.copy()
    # wen_is_eb[np.logical_not(is_eb_wa)] = np.nan

    ew_frac_time = np.size(wen[is_eb_wa])/np.size(wen)
    bw_frac_time = 1 - ew_frac_time

    # en_mom = mixed_moments(enn, wn)
    # ba_mom = mixed_moments(ban, wn)
    # bar_mom = mixed_moments(wbn, wn)


    # moments of original water and methane
    me_mom = mixed_moments(cn, wn) # no need to normalize them
    wa_mom = mixed_moments(rn, wn) # no need to normalize them
    # do in for time-filtered time series instead
    en_mom = partial_mixed_moments(cn_ebii, wn_ebii, N = N, normalize = False)
    ba_mom = partial_mixed_moments(cn_baii, wn_baii, N = N, normalize = False)
    bar_mom =partial_mixed_moments(rn_baii, wn_baii,N = N,  normalize = False)


    eT_me = compute_eT(cn, wn)
    eT_wa = compute_eT(rn, wn)
    eT_en = compute_eT(cn_ebii, wn_ebii, normalize = False)
    eT_ba = compute_eT(cn_baii, wn_baii, normalize = False)
    eT_we = compute_eT(rn_ebii, wn_ebii, normalize = False)
    eT_wb = compute_eT(rn_baii, wn_baii, normalize = False)

    # fnc_ebii = cn_ebii*wn_ebii
    # fnc_baii = cn_baii*wn_baii
    # fnr_ebii = rn_ebii*wn_ebii
    # fnr_baii = rn_baii*wn_baii
    # mefiii = cn*wn
    # wafiii = rn*wn

    # eT_en = 1 - np.abs(  np.sum(fnc_ebii[fnc_ebii < 0]) /np.sum(fnc_ebii[fnc_ebii > 0])                   )
    # eT_we = 1 - np.abs(  np.sum(fnr_ebii[fnr_ebii < 0]) /np.sum(fnr_ebii[fnr_ebii > 0])                   )
    # eT_ba = 1 - np.abs(  np.sum(fnc_baii[fnc_baii < 0]) /np.sum(fnc_baii[fnc_baii > 0])                   )
    # eT_wb = 1 - np.abs(  np.sum(fnr_baii[fnr_baii < 0]) /np.sum(fnr_baii[fnr_baii > 0])                   )
    #
    # eT_me = 1 - np.abs(  np.sum(mefiii[mefiii < 0]) /np.sum(mefiii[mefiii > 0])  )
    # eT_wa = 1 - np.abs(  np.sum(wafiii[wafiii < 0]) /np.sum(wafiii[wafiii > 0])  )
    #



    if plot:

        # print('Energetic flux fraction = ', en_frac_flux)
        # print('Energetic variance fraction = ', en_frac_var)
        # print('Energetic time fraction = ', en_frac_time)
        # print('correlation between scalars Rcr = {}'.format(Rcr))
        # matplotlib_update_settings()
        minutes = np.arange(1, np.size(rn)+1)/md.fs/60
        offst = 0.16
        offst2 = 0.12

        # cn = cn[8000:]
        # rn = rn[8000:]
        # enn = enn[8000:]
        # ban = ban[8000:]
        # minutes = minutes[8000:] - minutes[8000] # reset time
        # is_eb = is_eb[8000:]

        # mybins = np.log([ -0.27, -0.])
        # xc, yc = np.hist(cn)
        # A = (np.logspace( np.log10(np.min(cn+1)), np.log10(np.max(cn+1)), 50))
        # yc, xce = np.histogram(cn, bins = A, density = True)
        # xc = xce[1:] -(xce[1]-xce[0])/2

        cdn = (df[toplotsc] - np.mean(df[toplotsc]))/np.std(df[toplotsc])
        xd, yd = sa.emp_pdf(cdn, 20)
        xc, yc = sa.emp_pdf(cn, 20)
        xr, yr = sa.emp_pdf(rn, 20)
        # xb, yb = sa.emp_pdf(ban, 20)
        xb, yb = sa.emp_pdf(cn_baii, 20)
        xrb, yrb = sa.emp_pdf(wbn, 20)
        # xe, ye = sa.emp_pdf(enn, 20)
        xe, ye = sa.emp_pdf(cn_ebii, 20)
        maxfreq = max(np.max(yc), np.max(yr), max(yb) )

        # kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(xc.reshape(-1, 1))
        # yc = np.exp(kde.score_samples(xc.reshape(-1, 1)))
        # plt.figure()
        # plt.plot(xc, yc, 'g')
        # plt.plot(xr, yr, 'b')
        # plt.yscale('log')
        # plt.show()
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize = (10, 14))
        if datetime:
            fig.suptitle('Run measured {}'.format(datetime), fontsize = 24)
        axes[0,0].set_ylabel(r"$c'/\sigma_c$ ($CH_4$)")
        axes[0,0].annotate("a)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[1,0].annotate("b)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[2,0].annotate("c)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[3,0].annotate("d)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        # axes[0,0].set_xlabel('time [minutes]')
        # axes[0,0].plot(minutes, cn, 'k', alpha = 0.6, label = 'CH4')
        axes[0,0].plot(minutes, cn, 'g', alpha = 1, label = r'$CH_4$')

        axes[0, 0].plot([minutes[0], minutes[-1]], [np.std(cn), np.std(cn)], '--k')
        axes[0, 0].plot([minutes[0], minutes[-1]], [-np.std(cn), -np.std(cn)], '--k')
        # axes[0,0].plot(minutes, ban, 'c', alpha = 1, label = 'CH4 back')
        # axes[0,0].plot(minutes, enn, 'r', alpha = 1, label = 'CH4 en')
        # axes[0,0].plot(minutes, cn_filtered, 'k', label = 'CH4 filtered')
        # axes[0,0].plot(rn, '--b', label = 'H2O')
        # axes[0,0].legend()

        box = axes[0, 0].get_position()
        axes[0, 0].set_position([box.x0, box.y0, box.width + offst, box.height])
        # axes[0, 0].axes.get_xaxis().set_visible(False)
        axes[0, 0].set_yscale('symlog')
        axes[0, 1].set_yscale('symlog')
        # axes[1,0].plot(rn)
        axes[1,0].plot(minutes, rn, 'b', label = r'$H_2O$')

        axes[1, 0].plot([minutes[0], minutes[-1]], [np.std(rn), np.std(rn)], '--k')
        axes[1, 0].plot([minutes[0], minutes[-1]], [-np.std(rn), -np.std(rn)], '--k')
        # axes[1,0].plot(minutes, rn_filtered, 'k', label = 'H2O filtered')
        axes[1,0].set_ylabel(r"$r'/\sigma_r$ ($H_2O$)")
        # axes[1,0].set_xlabel('time [minutes]')
        # axes[1, 0].axes.get_xaxis().set_visible(False)
        # axes[1, 0].legend()

        box = axes[1, 0].get_position()
        axes[1, 0].set_position([box.x0, box.y0, box.width + offst, box.height])

        # do commom y axis:
        ymin0, _ = axes[0,0].get_ylim()
        ymin1, _ = axes[1,0].get_ylim()
        ymin = min(ymin0, ymin1)
        axes[0, 0].set_ylim(bottom=ymin)
        axes[1, 0].set_ylim(bottom=ymin)
        axes[0, 1].set_ylim(bottom=ymin)
        axes[1, 1].set_ylim(bottom=ymin)



        # second row
        axes[0,1].plot(yc, xc, 'g', linewidth = 2)
        # axes[0,1].plot(yb, xb, 'c', linewidth = 2)
        # axes[0,1].plot(ye, xe, 'r', linewidth = 2)
        # axes[0,1].plot(yr, xr, '--b', linewidth = 2)
        box = axes[0, 1].get_position()
        axes[0, 1].set_position([box.x0+offst2, box.y0, box.width - offst + offst-offst2, box.height])
        # axes[0, 1].axes.get_yaxis().set_visible(False)
        # axes[0, 1].axes.get_xaxis().set_visible(False)
        axes[0, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[0, 1].set_xscale('log')
        axes[0, 1].plot([0.002, 2.1], [0, 0], '--k')
        # axes[0, 1].set_xlabel('Frequency')
        axes[1,1].plot(yr, xr, 'b', linewidth = 2)
        box = axes[1, 1].get_position()
        axes[1, 1].set_position([box.x0+offst2, box.y0, box.width -offst + offst-offst2, box.height])
        # axes[1, 1].axes.get_yaxis().set_visible(False)
        # axes[1, 1].axes.get_xaxis().set_visible(False)
        axes[1, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[1, 1].set_xscale('log')
        # axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].plot([0.002, 2.1], [0, 0], '--k')

        axes[0, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[0, 1].set_xticklabels([0.002, 0.02, 0.2, 2])
        axes[1, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[1, 1].set_xticklabels([0.002, 0.02, 0.2, 2])


        axes[1, 0].set_yscale('symlog')
        axes[1, 1].set_yscale('symlog')

        # third row
        # axes[2, 0].plot(minutes, ban, 'c', linewidth = 1)
        axes[2, 0].plot(minutes, cn, 'c', linewidth = 1)
        # axes[2, 0].plot(minutes, enn, 'r', alpha = 1, linewidth = 1)
        axes[2, 0].plot(minutes[is_eb], cn[is_eb], '.r', linewidth = 0.9, markersize = 4)
        axes[2, 0].plot([minutes[0], minutes[-1]], [np.std(ban), np.std(ban)], '--k')
        axes[2, 0].plot([minutes[0], minutes[-1]], [-np.std(ban), -np.std(ban)], '--k')
        box = axes[2, 0].get_position()
        axes[2, 0].set_position([box.x0, box.y0, box.width + offst, box.height])
        # axes[0, 2].axes.get_yaxis().set_visible(False)
        # axes[0, 2].axes.get_xaxis().set_visible(False)
        # axes[2, 0].set_xlim([0.4E-2, maxfreq + 0.1])
        # axes[2, 0].set_xscale('log')
        # axes[2,0].set_xlabel('time [minutes]')
        axes[2,0].set_ylabel(r"$c'/\sigma_c$ ($CH_4$)")
        # axes[0, 1].set_xlabel('Frequency')
        # axes[2,1].plot(ye, xe, 'r', linewidth = 2)
        # axes[2,1].plot(yrb, xrb, 'k', linewidth = 2)
        # axes[2,1].plot(yr, xr, '--b', linewidth = 2)
        axes[2,1].plot(yb, xb, 'c', linewidth = 2)
        axes[2,1].plot(ye, xe, 'r', linewidth = 2)
        axes[2, 1].plot( [0.002, 2.1], [0, 0],'--k')
        box = axes[2, 1].get_position()
        axes[2, 1].set_position([box.x0+offst2, box.y0, box.width -offst + offst-offst2, box.height])
        # axes[2, 1].axes.get_yaxis().set_visible(False)
        # axes[1, 2].axes.get_yaxis().set_visible(False)
        axes[2, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[2, 1].set_xscale('log')
        # axes[2, 1].set_xlabel('Frequency')
        axes[2, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[2, 1].set_xticklabels([0.002, 0.02, 0.2, 2])


        axes[2, 0].set_yscale('symlog')
        axes[2, 1].set_yscale('symlog')


        axes[3, 0].plot(minutes, cdn, 'c', alpha = 1, linewidth = 1)
        axes[3, 0].plot([minutes[0], minutes[-1]], [np.std(cdn), np.std(cdn)], '--k')
        axes[3, 0].plot([minutes[0], minutes[-1]], [-np.std(cdn), -np.std(cdn)], '--k')
        box = axes[3, 0].get_position()
        axes[3, 0].set_position([box.x0, box.y0, box.width + offst, box.height])
        axes[3,0].set_xlabel('time [minutes]')
        axes[3,0].set_ylabel(r"$c'/\sigma_c$ ($CO_2$)")

        axes[3,1].plot(yd, xd, 'c', linewidth = 2)
        axes[3, 1].plot( [0.002, 2.1], [0, 0],'--k')
        box = axes[3, 1].get_position()
        axes[3, 1].set_position([box.x0+offst2, box.y0, box.width -offst + offst-offst2, box.height])
        # axes[3, 1].axes.get_yaxis().set_visible(False)
        # axes[1, 2].axes.get_yaxis().set_visible(False)
        axes[3, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[3, 1].set_xscale('log')
        axes[3, 1].set_xlabel('Frequency density')
        axes[3, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[3, 1].set_xticklabels([0.002, 0.02, 0.2, 2])

        axes[2,0].set_xlabel('time [minutes]')
        axes[2,1].set_xlabel('Frequency density')
        axes[1,0].set_xlabel('time [minutes]')
        axes[1,1].set_xlabel('Frequency density')
        axes[0,0].set_xlabel('time [minutes]')
        axes[0,1].set_xlabel('Frequency density')

        axes[3, 1].set_yscale('symlog')
        axes[3, 0].set_yscale('symlog')


        ylims = axes[0, 0].get_ylim()
        axes[0, 1].set_ylim(ylims)
        ylims = axes[1, 0].get_ylim()
        axes[1, 1].set_ylim(ylims)
        ylims = axes[2, 0].get_ylim()
        axes[2, 1].set_ylim(ylims)
        ylims = axes[3, 0].get_ylim()
        axes[3, 1].set_ylim(ylims)

        # move histograms y-labels on the right
        axes[0, 1].yaxis.set_label_position("right")
        axes[0, 1].yaxis.tick_right()
        axes[1, 1].yaxis.set_label_position("right")
        axes[1, 1].yaxis.tick_right()
        axes[2, 1].yaxis.set_label_position("right")
        axes[2, 1].yaxis.tick_right()
        axes[3, 1].yaxis.set_label_position("right")
        axes[3, 1].yaxis.tick_right()

        # plt.tight_layout()
        plt.savefig(os.path.join(md.outdir_plot, 'traces_{}.png'.format(datetime)))
        # plt.tight_layout()
        #
        # axes[0,1].plot(enn)
        # axes[0,1].plot(enn_is_eb)
        # axes[0,1].plot([0, np.size(enn)], [np.std(ban), np.std(ban)], '--r')
        # axes[0,1].plot([0, np.size(enn)], [-np.std(ban), -np.std(ban)], '--r')
        # axes[0,1].plot(enn)
        # axes[0,1].set_ylabel('enn')
        # axes[1,1].plot(ban)
        # axes[1,1].set_ylabel('ban')
        plt.show()

        plt.figure()
        # plt.plot(minutes, cn_filtered)
        plt.plot(minutes, cn, 'k')
        # plt.plot(minutes, rn, 'b')
        plt.plot(minutes, ban, 'c')
        plt.plot(minutes[is_eb], enn[is_eb], '.r', linewidth = 0.9)
        plt.xlim([18,24])
        plt.xlabel('time [minutes]')
        plt.ylabel(r"$c'/\sigma_c$")
        # plt.plot(minutes, ban, 'c', linewidth = 1)
        plt.plot(minutes, enn, 'r', alpha = 0.8, linewidth = 1)
        plt.savefig(os.path.join(md.outdir_plot, 'traces_2_{}.png'.format(datetime)))
        plt.show()


        print('Plot saved')

    #############  END PLOTTING ##############

    res = {}
    res['en_frac_flux'] =  en_frac_flux_T  # energetic flux fraction
    res['ba_frac_flux'] =  ba_frac_flux_T  # background flux fraction
    res['en_frac_var'] =  en_frac_var_T  # energetic  variance fraction
    res['ba_frac_var'] =  ba_frac_var_T  # background variance fraction
    res['en_frac_time'] =  en_frac_time  # energetic  time fraction
    res['ba_frac_time'] =  ba_frac_time  # background time fraction
    res['en_eT_wave'] = en_eT_wave
    res['ba_eT_wave'] = ba_eT_wave
    res['wb_eT_wave'] = wb_eT_wave
    res['we_eT_wave'] = we_eT_wave
    res['wa_eT_wave'] = wa_eT_wave
    res['me_eT_wave'] = me_eT_wave
    res['minsize'] = minsize
    res['maxsize'] = maxsize
    res['maxtin'] = maxtin

    # computed in time, not wavelet domain
    # res['me_nofilter_eT'] = eT_me_nofilter
    # res['wa_nofilter_eT'] = eT_wa_nofilter
    res['me_eT'] = eT_me
    res['wa_eT'] = eT_wa
    res['en_eT'] = eT_en
    res['ba_eT'] = eT_ba
    res['we_eT'] = eT_we
    res['wb_eT'] = eT_wb

    # ratio of total to (in this range of scales) fluxes and variances
    res['me_flufrac_scales'] = me_flufrac_scales
    res['wa_flufrac_scales'] = wa_flufrac_scales
    res['me_varfrac_scales'] = me_varfrac_scales
    res['wa_varfrac_scales'] = wa_varfrac_scales
    res['bw_frac_var'] = bw_frac_var_T
    res['ew_frac_var'] = ew_frac_var_T
    res['bw_frac_flux'] = bw_frac_flux_T
    res['ew_frac_flux'] = ew_frac_flux_T
    # res['bw_frac_time'] = bw_frac_time
    # res['ew_frac_time'] = ew_frac_time

    # normalized squared fluxes
    # res['en_flux2'] = enfluxn2
    # res['ba_flux2'] = bafluxn2
    # res['wab_flux2']= wabfluxn2
    # res['wa_flux2'] = wafluxn2
    # res['me_flux2'] = mefluxn2


    res['en_flux_wt'] = en_flux_wt
    res['ba_flux_wt'] = ba_flux_wt
    res['wb_flux_wt'] = wb_flux_wt
    res['we_flux_wt'] = we_flux_wt
    res['wa_flux_wt'] = wa_flux_wt
    res['me_flux_wt'] = me_flux_wt

    res['wa_flux_ec'] = wa_flux_ec
    res['me_flux_ec'] = me_flux_ec

    # res['en_eT_var'] = en_var_eT # ratio of direct / indirect variance for ener component
    # res['ba_eT_var'] = ba_var_eT # ratio of direct / indirect variance for back component
    res['cn_ener'] = enn
    res['cn_back'] = ban
    res['stdv_c2'] = stdv_c2
    res['stdv_cw'] = stdv_cw
    res['stdv_c'] = stdv_c
    res['stdv_cn'] = stdv_cn
    res['my_stdv'] = mystdv
    res['Rcr'] = Rcr
    res['WTe'] = WTen0 # energetic wavelet coefficents
    res['WTb'] = WTbn0 # energetic wavelet coefficents


    res['enfa'] = enfa
    res['enva'] = enva
    res['wefa'] = wefa
    res['weva'] = weva
    res['bafa'] = bafa
    res['bava'] = bava
    res['wbfa'] = wbfa
    res['wbva'] = wbva
    res['mefa'] = mefa
    res['meva'] = meva
    res['wafa'] = wafa
    res['wava'] = wava


    # res['var_back_ttime'] = var_back_ttime


    res['me_M40'] = me_mom['M40']
    res['me_M30'] = me_mom['M30']
    res['me_M21'] = me_mom['M21']
    res['me_M12'] = me_mom['M12']
    res['me_M13'] = me_mom['M13']
    res['me_Rcw'] = me_mom['Rcw']

    res['wa_M40'] = wa_mom['M40']
    res['wa_M30'] = wa_mom['M30']
    res['wa_M21'] = wa_mom['M21']
    res['wa_M12'] = wa_mom['M12']
    res['wa_M13'] = wa_mom['M13']
    res['wa_Rcw'] = wa_mom['Rcw']

    res['en_M40'] = en_mom['M40']
    res['en_M30'] = en_mom['M30']
    res['en_M21'] = en_mom['M21']
    res['en_M12'] = en_mom['M12']
    res['en_M13'] = en_mom['M13']
    res['en_Rcw'] = en_mom['Rcw']

    res['ba_M40'] = ba_mom['M40']
    res['ba_M30'] = ba_mom['M30']
    res['ba_M21'] = ba_mom['M21']
    res['ba_M12'] = ba_mom['M12']
    res['ba_M13'] = ba_mom['M13']
    res['ba_Rcw'] = ba_mom['Rcw']


    res['bar_M40'] = bar_mom['M40']
    res['bar_M30'] = bar_mom['M30']
    res['bar_M21'] = bar_mom['M21']
    res['bar_M12'] = bar_mom['M12']
    res['bar_M13'] = bar_mom['M13']
    res['bar_Rcw'] = bar_mom['Rcw']

    return res


def isr_fluxes(df, res_wc, res_wd, mysc='CH4', refsc='H2O', cdsc = 'CO2',
               tscale='tau_eta',
               wavelet = 'haar',
               maxscale = 50, useminsize=True,
               minsize = 4, minscale = 0.5,
               fs = 10, z_ref = 2.8):

    # tscale = 'tau_eta' or 'tau_eta0'
    if tscale == 'tau_eta':
        ksr = 8
    elif tscale == 'tau_eta0':
        ksr = 4 # for scales -> 0.5 to 5 Tw
        # ksr = 4.0 # for scales -> 0.5 to 5 Tw

    wp = (df['w'] - np.mean(df['w']))
    cp = (df[mysc] - np.mean(df[mysc]))
    rp = (df[refsc] - np.mean(df[refsc]))
    dp = (df[cdsc] - np.mean(df[cdsc]))

    Tp = (df['T'] - np.mean(df['T']))
    up = (df['u'] - np.mean(df['u']))


    mytdf = turb_quant(df, fs, z_ref, nobs_min=6000)

    stdw = np.std(df['w'])
    stdc = np.std(df[mysc])
    stdr = np.std(df[refsc])
    stdd = np.std(df[cdsc])

    # stdT = np.std(df['T'])
    # stdu = np.std(df['u'])

    # compute eddy-covariance fluxes
    fme_ec = np.mean(cp*wp)
    fwa_ec = np.mean(rp*wp)
    fcd_ec = np.mean(dp*wp)
    fTT_ec = np.mean(Tp*wp)
    fuu_ec = np.mean(up*wp)

    # compute integral time scales
    Tw = integral_scale_fft(wp) / md.fs  # in time units [s]
    Tc = integral_scale_fft(cp) / md.fs  # in time units [s]
    Tr = integral_scale_fft(rp) / md.fs  # in time units [s]
    Td = integral_scale_fft(dp) / md.fs  # in time units [s]

    # Tuu = integral_scale_fft(up) / md.fs  # in time units [s]
    # TTT = integral_scale_fft(Tp) / md.fs  # in time units [s]

    # COMPUTE FRACTION OF ACTIVE AREA
    alpha = res_wc['en_frac_time'] # FROM WAVELET PARTITION - Ch4
    alphad = res_wc['en_frac_time'] # FROM WAVELET PARTITION - CO2


    apime_std = np.std(sa.filter_large_scales(df[mysc].values,    Tw,
                      Ubar=mytdf['Ubar'], z_ref = md.z_ref, fs = md.fs,
                      minscale=minscale,
                      maxscale = maxscale,
                      minsize=minsize,
                      useminsize=useminsize))

    apiwa_std = np.std(sa.filter_large_scales(df[refsc].values,    Tw,
                      Ubar=mytdf['Ubar'], z_ref = md.z_ref, fs = md.fs,
                      minscale=minscale,
                      maxscale = maxscale,
                      minsize=minsize,
                      useminsize=useminsize))

    apicd_std = np.std(sa.filter_large_scales(df[cdsc].values,    Tw,
                      Ubar=mytdf['Ubar'], z_ref = md.z_ref, fs = md.fs,
                      minscale=minscale,
                      maxscale = maxscale,
                      minsize=minsize,
                      useminsize=useminsize))

    apiuu_std = np.std(sa.filter_large_scales(df['u'].values,    Tw,
                      Ubar=mytdf['Ubar'], z_ref = md.z_ref, fs = md.fs,
                      minscale=minscale,
                      maxscale = maxscale,
                      minsize=minsize,
                      useminsize=useminsize))

    apiTT_std = np.std(sa.filter_large_scales(df['T'].values,    Tw,
                      Ubar=mytdf['Ubar'], z_ref = md.z_ref, fs = md.fs,
                      minscale=minscale,
                      maxscale = maxscale,
                      minsize=minsize,
                      useminsize=useminsize))

    apien_std = apime_std*res_wc['en_frac_var']**(1/2)/np.sqrt(alpha)
    apiba_std = apime_std*res_wc['ba_frac_var']**(1/2)/np.sqrt(1-alpha)


    # # apien_cd_std = apicd_std*res_wc['ed_frac_var']**(1/2)/np.sqrt(alpha)
    # apiba_cd_std = apicd_std*res_wc['bd_frac_var']**(1/2)/np.sqrt(1-alpha)
    apien_cd_std = apicd_std*res_wd['en_frac_var']**(1/2)/np.sqrt(alphad)
    apiba_cd_std = apicd_std*res_wd['ba_frac_var']**(1/2)/np.sqrt(1-alphad)

    # en_cd_M30 = res_wc['end_M30']
    # ba_cd_M30 = res_wc['bad_M30']
    en_cd_M30 = res_wd['en_M30']
    ba_cd_M30 = res_wd['ba_M30']

    me_M30 = np.mean(cp**3)/np.std(cp)**3
    wa_M30 = np.mean(rp**3)/np.std(rp)**3
    cd_M30 = np.mean(dp**3)/np.std(dp)**3
    uu_M30 = np.mean(up**3)/np.std(up)**3
    TT_M30 = np.mean(Tp**3)/np.std(Tp)**3
    en_M30 = res_wc['en_M30']
    ba_M30 = res_wc['ba_M30']




    # DC_me = np.sign(fme_ec)*ksr*apime_std*      ( 1 + (np.abs(me_M30))   **(1/3) )
    DC_me = ksr*apime_std*      ( 1 + (np.abs(me_M30))   **(1/3) )
    DC_wa = np.sign(fwa_ec)*ksr*apiwa_std*      ( 1 + (np.abs(wa_M30))   **(1/3) )
    DC_cd = np.sign(fcd_ec)*ksr*apicd_std*      ( 1 + (np.abs(cd_M30))   **(1/3) )
    DC_uu = np.sign(fuu_ec)*ksr*apiuu_std*      ( 1 + (np.abs(uu_M30))   **(1/3) )
    DC_TT = np.sign(fTT_ec)*ksr*apiTT_std*      ( 1 + (np.abs(TT_M30))   **(1/3) )
    # DC_en = np.sign(res_wc['en_fa'])*ksr*apien_std*      ( 1 + (np.abs(en_M30))   **(1/3) )
    # DC_ba = np.sign(res_wc['ba_fa'])*ksr*apiba_std*      ( 1 + (np.abs(ba_M30))   **(1/3) )
    DC_en = ksr*apien_std*      ( 1 + (np.abs(en_M30))   **(1/3) )
    DC_ba = ksr*apiba_std*      ( 1 + (np.abs(ba_M30))   **(1/3) )
    # DC_cd_en = np.sign(res_wd['en_fa'])*ksr*apien_cd_std*( 1 + np.sign(res_wd['en_fa'])*np.sign(en_cd_M30)*(np.abs(en_cd_M30))**(1/3) )
    # DC_cd_ba = np.sign(res_wd['ba_fa'])*ksr*apiba_cd_std*( 1 + np.sign(res_wd['ba_fa'])*np.sign(ba_cd_M30)*(np.abs(ba_cd_M30))**(1/3) )
    DC_cd_en = np.sign(res_wd['en_fa'])*ksr*apien_cd_std*( 1 + (np.abs(en_cd_M30)   **(1/3)) )
    DC_cd_ba = np.sign(res_wd['ba_fa'])*ksr*apiba_cd_std*( 1 + (np.abs(ba_cd_M30)   **(1/3)) )



    # DC_me = np.sign(fme_ec)*ksr*apime_std*               ( 1 + np.log( 1 + (np.abs(me_M30))    ))
    # DC_wa = np.sign(fwa_ec)*ksr*apiwa_std*               ( 1 + np.log( 1 + (np.abs(wa_M30))    ))
    # DC_cd = np.sign(fcd_ec)*ksr*apicd_std*               ( 1 + np.log( 1 + (np.abs(cd_M30))    ))
    # DC_uu = np.sign(fuu_ec)*ksr*apiuu_std*               ( 1 + np.log( 1 + (np.abs(uu_M30))    ))
    # DC_TT = np.sign(fTT_ec)*ksr*apiTT_std*               ( 1 + np.log( 1 + (np.abs(TT_M30))    ))
    # DC_en = np.sign(res_wc['en_fa'])*ksr*apien_std*      ( 1 + np.log( 1 + (np.abs(en_M30))    ))
    # DC_ba = np.sign(res_wc['ba_fa'])*ksr*apiba_std*      ( 1 + np.log( 1 + (np.abs(ba_M30))    ))
    # DC_cd_en = np.sign(res_wd['en_fa'])*ksr*apien_cd_std*( 1 + np.log( 1 + (np.abs(en_cd_M30)) ))
    # DC_cd_ba = np.sign(res_wd['ba_fa'])*ksr*apiba_cd_std*( 1 + np.log( 1 + (np.abs(ba_cd_M30)) ))



    # DC_me = np.sign(fme_ec)*ksr*apime_std*      ( 1 + 1/ksr*(np.abs(me_M30))   **(1/3) )
    # DC_wa = np.sign(fwa_ec)*ksr*apiwa_std*      ( 1 + 1/ksr*(np.abs(wa_M30))   **(1/3) )
    # DC_cd = np.sign(fcd_ec)*ksr*apicd_std*      ( 1 + 1/ksr*(np.abs(cd_M30))   **(1/3) )
    # DC_uu = np.sign(fuu_ec)*ksr*apiuu_std*      ( 1 + 1/ksr*(np.abs(uu_M30))   **(1/3) )
    # DC_TT = np.sign(fTT_ec)*ksr*apiTT_std*      ( 1 + 1/ksr*(np.abs(TT_M30))   **(1/3) )
    # DC_en = np.sign(res_wc['enfa'])*ksr*apien_std*      ( 1 + 1/ksr*(np.abs(en_M30))   **(1/3) )
    # DC_ba = np.sign(res_wc['bafa'])*ksr*apiba_std*      ( 1 + 1/ksr*(np.abs(ba_M30))   **(1/3) )
    # DC_cd_en = np.sign(res_wd['enfa'])*ksr*apien_cd_std*( 1 + 1/ksr*(np.abs(en_cd_M30))**(1/3) )
    # DC_cd_ba = np.sign(res_wd['bafa'])*ksr*apiba_cd_std*( 1 + 1/ksr*(np.abs(ba_cd_M30))**(1/3) )


    # DC_me = np.sign(fme_ec)*apime_std*              (ksr + (np.abs(me_M30))   **(1/1) )
    # DC_wa = np.sign(fwa_ec)*apiwa_std*              (ksr + (np.abs(wa_M30))   **(1/1) )
    # DC_cd = np.sign(fcd_ec)*apicd_std*              (ksr + (np.abs(cd_M30))   **(1/1) )
    # DC_uu = np.sign(fuu_ec)*apiuu_std*              (ksr + (np.abs(uu_M30))   **(1/1) )
    # DC_TT = np.sign(fTT_ec)*apiTT_std*              (ksr + (np.abs(TT_M30))   **(1/1) )
    # DC_en = np.sign(res_wc['enfa'])*apien_std*      (ksr + (np.abs(en_M30))   **(1/1) )
    # DC_ba = np.sign(res_wc['bafa'])*apiba_std*      (ksr + (np.abs(ba_M30))   **(1/1) )
    # DC_cd_en = np.sign(res_wd['enfa'])*apien_cd_std*(ksr + (np.abs(en_cd_M30))**(1/1) )
    # DC_cd_ba = np.sign(res_wd['bafa'])*apiba_cd_std*(ksr + (np.abs(ba_cd_M30))**(1/1) )



    fme_sr = DC_me*np.sqrt(md.DM[mysc]/mytdf[tscale])
    fwa_sr = DC_wa*np.sqrt(md.DM[refsc]/mytdf[tscale])
    fcd_sr = DC_cd*np.sqrt(md.DM[cdsc]/mytdf[tscale])

    fuu_sr = DC_uu*np.sqrt(md.DM['u']/mytdf[tscale])
    fTT_sr = DC_TT*np.sqrt(md.DM['T']/mytdf[tscale])
    # if needed add Daltons, transport efficiencies and GTV for u and T

    da_me = fme_ec/mytdf['ustar']/DC_me
    da_wa = fwa_ec/mytdf['ustar']/DC_wa
    da_cd = fcd_ec/mytdf['ustar']/DC_cd
    da_en = (fme_ec*res_wc['en_frac_flux']
             /mytdf['ustar']/DC_en/np.sqrt(alpha))
    da_ba = (fme_ec*res_wc['ba_frac_flux']
             /mytdf['ustar']/DC_ba/np.sqrt(1-alpha))

    # da_cd_en = (fcd_ec*res_wc['ed_frac_flux']
    #          /mytdf['ustar']/DC_cd_en/np.sqrt(alpha))
    # da_cd_ba = (fcd_ec*res_wc['bd_frac_flux']
    #          /mytdf['ustar']/DC_cd_ba/np.sqrt(1-alpha))

    fba_sr = (1-alpha)*DC_ba*np.sqrt(md.DM[mysc]/mytdf[tscale])
    fen_sr =   (alpha)*DC_en*np.sqrt(md.DM[mysc]/mytdf[tscale])

    # fba_cd_sr = (1-alpha)*DC_cd_ba*np.sqrt(md.DM[mysc]/mytdf[tscale])
    # fen_cd_sr =   (alpha)*DC_cd_en*np.sqrt(md.DM[mysc]/mytdf[tscale])
    fba_cd_sr = (1-alphad)*DC_cd_ba*np.sqrt(md.DM[mysc]/mytdf[tscale])
    fen_cd_sr =   (alphad)*DC_cd_en*np.sqrt(md.DM[mysc]/mytdf[tscale])


    gtv_wa = fwa_ec/stdr
    gtv_me = fme_ec/stdc
    gtv_cd = fcd_ec/stdd

    # accounting for the active area
    gtv_ba = res_wc['ba_frac_flux']*fme_ec/stdc/np.sqrt(res_wc['ba_frac_var']*(1-alpha))
    gtv_en = res_wc['en_frac_flux']*fme_ec/stdc/np.sqrt(res_wc['en_frac_var']*(alpha))

    # gtv_cd_ba = res_wc['bd_frac_flux']*fcd_ec/stdd/np.sqrt(res_wc['bd_frac_var']*(1-alpha))
    # gtv_cd_en = res_wc['ed_frac_flux']*fcd_ec/stdd/np.sqrt(res_wc['ed_frac_var']*(alpha))

    gtv_cd_ba = res_wd['ba_frac_flux']*fcd_ec/stdd/np.sqrt(res_wd['ba_frac_var']*(1-alphad))
    gtv_cd_en = res_wd['en_frac_flux']*fcd_ec/stdd/np.sqrt(res_wd['en_frac_var']*(alphad))


    teba = gtv_ba/stdw
    teen = gtv_en/stdw

    teba_cd = gtv_cd_ba/stdw
    teen_cd = gtv_cd_en/stdw

    teme = gtv_me/stdw
    tewa = gtv_wa/stdw
    tecd = gtv_cd/stdw

    gtv_wa_sr = fwa_sr/stdr
    gtv_me_sr = fme_sr/stdc
    gtv_cd_sr = fcd_sr/stdd

    # gtv_me_isr = ((1-alpha)*fba_sr + alpha*fen_sr)/stdc
    gtv_me_isr = (fba_sr + fen_sr)/stdc
    gtv_cd_isr = (fba_cd_sr + fen_cd_sr)/stdd

    # gtv_cd_ba_sr = fba_cd_sr/stdd/np.sqrt(res_wc['bd_frac_var']*(1-alpha))
    # gtv_cd_en_sr = fen_cd_sr/stdd/np.sqrt(res_wc['ed_frac_var']*(alpha))
    gtv_cd_ba_sr = fba_cd_sr/stdd/np.sqrt(res_wd['ba_frac_var']*(1-alphad))
    gtv_cd_en_sr = fen_cd_sr/stdd/np.sqrt(res_wd['en_frac_var']*(alphad))

    gtv_ba_sr = fba_sr/stdc/np.sqrt(res_wc['ba_frac_var']*(1-alpha))
    gtv_en_sr = fen_sr/stdc/np.sqrt(res_wc['en_frac_var']*(alpha))

    # gtv_wa_sr = fwa_sr/apiwa_std/ksr
    # gtv_me_sr = fme_sr/apime_std/ksr
    # gtv_ba_sr = fba_sr/apime_std/ksr/np.sqrt(res_wc['ba_frac_var']*(1-alpha))
    # gtv_en_sr = fen_sr/apime_std/ksr/np.sqrt(res_wc['en_frac_var']*(alpha))

    teba_sr = gtv_ba_sr/stdw
    teen_sr = gtv_en_sr/stdw
    teme_isr = gtv_me_isr/stdw
    teme_sr = gtv_me_sr/stdw
    tewa_sr = gtv_wa_sr/stdw

    tecd_isr = gtv_cd_isr/stdw
    tecd_sr = gtv_cd_sr/stdw

    teba_cd_sr = gtv_cd_ba_sr/stdw
    teen_cd_sr = gtv_cd_en_sr/stdw

    # fen_sr2 =   (alpha)*apien_std*np.sqrt(md.DM[mysc]/ts_eff)
    # gtv_en_sr2 = fen_sr2/stdc/np.sqrt(res_wc['en_frac_var']*(alpha))
    # teen_sr2 = gtv_en_sr2/stdw

    res = {}
    res['fme_sr'] = fme_sr
    res['fwa_sr'] = fwa_sr
    res['fcd_sr'] = fcd_sr

    res['fme_isr'] = fba_sr + fen_sr
    res['fcd_isr'] = fba_cd_sr + fen_cd_sr

    res['fuu_sr'] = fuu_sr
    res['fTT_sr'] = fTT_sr

    res['fba_sr'] = fba_sr
    res['fen_sr'] = fen_sr
    res['fba_cd_sr'] = fba_cd_sr
    res['fen_cd_sr'] = fen_cd_sr

    res['fme_ec'] = fme_ec
    res['fwa_ec'] = fwa_ec
    res['fcd_ec'] = fcd_ec

    res['fuu_ec'] = fuu_ec
    res['fTT_ec'] = fTT_ec

    res['Re_star'] = mytdf['Re_star']
    res['stab'] = mytdf['stab']
    res['ustar'] = mytdf['ustar']
    res['stdv_me'] = stdc
    res['stdv_wa'] = stdr
    res['stdv_cd'] = stdd
    res['stdv_w'] = stdw

    res['Tc'] = Tc
    res['Tr'] = Tr
    res['Td'] = Td
    res['Tw'] = Tw
    res['alpha'] = alpha # frac of Active TIME / AREA

    # Dalton Numbers, Gas transfer Velocities and Transport Efficiencies
    res['da_me'] = da_me
    res['da_wa'] = da_wa
    res['da_cd'] = da_cd
    res['da_en'] = da_en
    res['da_ba'] = da_ba
    # res['da_cd_en'] = da_cd_en
    # res['da_cd_ba'] = da_cd_ba

    res['gtv_me'] = gtv_me
    res['gtv_wa'] = gtv_wa
    res['gtv_cd'] = gtv_cd
    res['gtv_en'] = gtv_en
    res['gtv_ba'] = gtv_ba

    res['gtv_cd_en'] = gtv_cd_en
    res['gtv_cd_ba'] = gtv_cd_ba
    res['teme'] = teme
    res['tewa'] = tewa
    res['tecd'] = tecd
    res['teba'] = teba
    res['teen'] = teen
    res['teba_cd'] = teba_cd
    res['teen_cd'] = teen_cd

    res['gtv_me_isr'] = gtv_me_isr
    res['gtv_cd_isr'] = gtv_cd_isr
    res['gtv_me_sr'] = gtv_me_sr
    res['gtv_wa_sr'] = gtv_wa_sr
    res['gtv_cd_sr'] = gtv_cd_sr
    res['gtv_en_sr'] = gtv_en_sr
    res['gtv_ba_sr'] = gtv_ba_sr
    res['gtv_cd_en_sr'] = gtv_cd_en_sr
    res['gtv_cd_ba_sr'] = gtv_cd_ba_sr

    res['teme_isr'] = teme_isr
    res['tecd_isr'] = tecd_isr
    res['teme_sr'] = teme_sr
    res['tewa_sr'] = tewa_sr
    res['tecd_sr'] = tecd_sr
    res['teba_sr'] = teba_sr
    res['teen_sr'] = teen_sr
    res['teba_cd_sr'] = teba_cd_sr
    res['teen_cd_sr'] = teen_cd_sr

    res['apime_std'] = apime_std
    res['apiwa_std'] = apiwa_std
    res['apicd_std'] = apicd_std
    res['apiba_std'] = apiba_std
    res['apien_std'] = apien_std
    res['apiba_cd_std'] = apiba_cd_std
    res['apien_cd_std'] = apien_cd_std

    res['DC_me'] = DC_me
    res['DC_wa'] = DC_wa
    res['DC_cd'] = DC_cd
    res['DC_ba'] = DC_ba
    res['DC_en'] = DC_en
    res['DC_cd_ba'] = DC_cd_ba
    res['DC_cd_en'] = DC_cd_en

    # res['fen_sr2'] = fen_sr2
    # res['gtv_en_sr2'] = gtv_en_sr2
    # res['teen_sr2'] = teen_sr2
    return res


def detect_eb_timescle(df, res_wc, mysc='CH4', wavelet = 'haar', plot=False,
                       minscale = 1, maxscale = 500, filter = 'nscales',
                       minsize = 2**2, maxsize = 2**11,
                       datetime = None):
    # ebullition timescale detection
    # based on alternating surface renewals
    # get WTe from some partition scheme
    # partition must be the result of a partition scheme, such as res_wc
    # (must have item 'WTe' with the energetic ebullition wavelet components )


    turb_quants = turb_quant(df, md.fs, md.z_ref, nobs_min=6000)

    cn = (df[mysc] - np.mean(df[mysc])) / np.std(df[mysc])
    N = np.size(cn)
    # wn = (df['w'] - np.mean(df['w'])) / np.std(df['w'])

    # Tw = integral_scale_fft(wn)/md.fs
    Tc = integral_scale_fft(cn)/md.fs

    # WTe = res_wc['WTe']
    WTe0 = res_wc['WTe'] # remove coarse grained signal
    WTe0 = WTe0[1:]
    WTe0.reverse()

    #
    # print('WTe')
    # print([len(x) for x in WTe0])


    # mm = np.arange(len(WTe0)-1, 0, -1) # skip coarse grained signal
    mm = np.arange(len(WTe0)) # skip coarse grained signal
    M = np.log2(N)
    scales0 = 2**mm
    sizes0 = 2**(M-mm)

    # print(sizes)

    sizes = np.array([len(x) for x in WTe0])
    scales = N/sizes

    ti = scales/md.fs
    tin = ti/turb_quants["Tw"]

    # if filter == 'nscales':
    # cond_keep = np.logical_and(tin > minscale, sizes >= minsize)
    cond_keep = np.logical_and(tin >= minscale, tin <= maxscale)
    # print(cond_keep)
    # else:
    #     cond_keep = np.logical_and(sizes <= maxsize, sizes >= minsize)


    # WTe = [x for i, x in enumerate(WTe0[1:]) if cond_keep[i]]
    # WTe = [x for i, x in enumerate(WTe0[1:]) if cond_keep[i]]
    WTe = [x for i, x in enumerate(WTe0) if cond_keep[i]]
    myti = ti[cond_keep]
    mytin = tin[cond_keep]
    mysizes = sizes[cond_keep]

    mynscales = len(myti)

    # print("mynscales = ", mynscales)

    # fracs = np.array([x/y for x, y in zip()])

    accumulate_prob = True
    if not accumulate_prob:
        pno = np.array([ np.size(x[x < 1e-6])/np.size(x) for x in WTe])
    else:
        pno = np.zeros(mynscales)
        for j in range(mynscales):
            if j == 0:
                wtej = np.abs(WTe[j]) > 1E-6  # IF Pebull positive
            elif j > 0:
                wtejm1 = np.logical_or(wtej[::2], wtej[1::2])
                wtej0 = np.abs(WTe[j]) > 1E-6  # p not ebullition
                wtej = np.logical_or(wtejm1, wtej0)
            nebj = np.sum(wtej)
            nj = np.size(wtej)
            pno[j] = 1 - nebj / nj

    # print('pno')
    # print(pno)
    # weighted average to get the fraction of ON time
    # alpha = 1 - np.sum(pno*mytin)/np.sum(mytin)
    # alpha = np.mean(pno)

    def feb(t, b, s):
        peb = b/(b+s)*np.exp(-t/b)
        return peb

    # alternatively, POISSON PROCESS A LA COX & ISHAM
    def febp(t, l, d):
        pebp = np.exp( -l*d - l*t)
        return pebp

    # parameters with time in seconds
    (b, s), pcov_ren = curve_fit(feb, myti, pno,
            bounds=([0,0],[np.inf,np.inf]))

    (l, d), pcov_poi = curve_fit(febp, myti, pno,
            bounds=([0,0],[np.inf, np.inf]))

    if plot:

        matplotlib_update_settings()
        plt.figure()
        # plt.plot(scales, np.log(p_no), 'o')
        # plt.plot(scales, p_no, 'o')

        # plt.plot([Tw, Tw], [np.min(pno), np.max(pno)], '--k')
        # plt.plot(mytin, feb(myti, b, s), 'r')
        mytin2 = np.linspace(np.min(mytin), np.max(mytin), 200)
        myti2 = np.linspace(np.min(myti), np.max(myti), 200)
        plt.title('Run measured {}'.format(datetime), fontsize = 24)
        plt.plot(mytin2, febp(myti2, l, d), 'b', linewidth = 2)
        # plt.plot(mytin2, febp(myti2, l, 0), '--b', linewidth = 2)
        plt.scatter(mytin, pno, marker = 'o', facecolors='none',
                    s = 50, edgecolors = 'k')
        # plt.plot(mytscales, m * mytscales + q, 'r')
        # plt.plot(scales[condx], np.log(p_no[condx]), 'o')
        # plt.plot(sizes, np.log(p_no), 'o')
        # plt.plot(sizes[condx], np.log(p_no[condx]), 'o')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel(r'Normalized time scale $t/T_w$')
        plt.ylabel('Probability of no ebullition [-]')
        plt.savefig(os.path.join(md.outdir_plot, 'ebull_prob_{}.png'.format(datetime)))
        plt.show()

    poff = np.exp(-d*l)
    pon = 1-poff
    # alpha = 1-np.exp(-l*turb_quants['Tw'])
    # alpha = 1-np.exp(-l*Tc)
    alpha_tau_eta0 = 1-np.exp(-l*turb_quants['tau_eta0'])
    alpha_Tw = 1-np.exp(-l*turb_quants['Tw'])
    # alpha = 1-np.exp(-l*turb_quants['Tw'])
    # alpha = 1-l*turb_quants['tau_eta']
    alpha = 1/l*turb_quants['Tw']

    res = {}
    res['s'] = s  # ebullition average duration
    res['b'] = b
    res['l'] = l  # Poisson arrival rate (lambda)
    res['d'] = d  # Poisson average duration
    res['var_l'] = pcov_poi[0,0]
    res['var_d'] = pcov_poi[1,1]
    res['var_b'] = pcov_ren[0,0]
    res['var_s'] = pcov_ren[1,1]
    res['alpha'] = alpha
    res['alpha_Tw'] = alpha_Tw
    res['alpha_tau_eta0'] = alpha_tau_eta0
    res['Tw'] = turb_quants['Tw']
    res['shear_ts'] = turb_quants['shear_ts']
    res['diss_ts'] = turb_quants['diss_ts']
    res['poff'] = poff # prob of being off (in space or time)
    res['pon']  = pon # prob of being on (in space or time)

    return res



def wavelet_partition(df, mysc='CH4', refsc='H2O', nmin_obs=2**14,
                      cdsc='CO2',
                      minscale = 0.5,
                      maxscale = 200,
                      mystdv = 0.3, nstdvs = 3, wavelet = 'haar',
                      cond_x = 'c',
                      plot = False,
                      datetime = None):
    """-----------------------------------------------------------------
    Partition a time series of concentrations based on Wavelet transform
    INPUT:
        mysc: scalar of interest
        refsc: reference scalar
        nmin_obs:
    OUTPUT:
    -----------------------------------------------------------------"""
    # remove excess observations if time series is longer than 2*14
    N = nmin_obs
    if df.shape[0] < N:
        print('wavelet partition error: time series not long enough')
    else:
        df = df[:N]

    # compute turbulent quantities
    turb_quants = turb_quant(df, md.fs, md.z_ref, nobs_min=nmin_obs)

    # compute normalize fluctuations of w, c, r
    wn = (df['w'] - np.mean(df['w'])) / np.std(df['w'])
    cn = (df[mysc] - np.mean(df[mysc])) / np.std(df[mysc])
    rn = (df[refsc] - np.mean(df[refsc])) / np.std(df[refsc])
    dn = (df[cdsc] - np.mean(df[cdsc])) / np.std(df[cdsc])


    Rcr = np.mean(cn*rn)

    # wavelet decomposition
    WTcn = pywt.wavedec(cn, wavelet=wavelet)
    WTrn = pywt.wavedec(rn, wavelet=wavelet)
    WTdn = pywt.wavedec(dn, wavelet=wavelet)
    WTwn = pywt.wavedec(wn, wavelet=wavelet)

    # Following Scanlon and Albertson (2001) notation
    mm0 = np.arange(len(WTcn)-1, 0, -1) # skip coarse grained signal
    M = np.log2(N)
    mm = np.hstack((M, mm0)) # add scale of coarse grained signal
    scales = 2**mm
    sizes = 2**(M-mm)

    # compute normalize scales
    ti = scales/md.fs
    tin = ti/turb_quants["Tw"]

    cond_keep = np.logical_and(tin >= minscale, tin <= maxscale)
    onsizes = sizes[cond_keep]
    ontins = tin[cond_keep]
    maxsize = np.max(onsizes)
    minsize = np.min(onsizes)
    maxtin = np.max(ontins)

    # filter wavelet coefficent base on scales
    WTcn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTcn)]
    WTrn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTrn)]
    # WTdn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTdn)]
    WTwn0 = [x if cond_keep[i] else np.zeros(np.int(sizes[i])) for i, x in enumerate(WTwn)]

    # transform to arrays, without the zeros:
    WTcn_non0 = [x for i, x in enumerate(WTcn) if cond_keep[i]]
    WTrn_non0 = [x for i, x in enumerate(WTrn) if cond_keep[i]]
    # WTdn_non0 = [x for i, x in enumerate(WTdn) if cond_keep[i]]
    # WTwn_non0 = [x for i, x in enumerate(WTwn) if cond_keep[i]]
    WAc0 = sa.coeffs_2_array(WTcn_non0)[0]
    WAr0 = sa.coeffs_2_array(WTrn_non0)[0]
    # WAd0 = sa.coeffs_2_array(WTdn_non0)[0]
    # WAw0 = sa.coeffs_2_array(WTwn_non0)[0]

    # FIT A LINE TO THE NON_ZERO WT ONLY
    XX = WAr0
    YY = WAc0
    huber = HuberRegressor(fit_intercept=True).fit(XX.reshape(-1, 1), YY)
    ahat = huber.coef_
    bhat = huber.intercept_

    diff_c    = np.abs(WAc0 - (ahat*WAr0+bhat))
    stdv_c    = np.std(diff_c)


    WTen0  = [np.zeros(len(x)) for x in WTcn0] # CH4 ener
    WTbn0  = [np.zeros(len(x)) for x in WTcn0] # CH4 back
    WTrbn0 = [np.zeros(len(x)) for x in WTcn0] # water background
    WTren0 = [np.zeros(len(x)) for x in WTcn0] # water energetic

    for i, (c, r, e, b, w, rb, re) in enumerate(
                zip(WTcn0, WTrn0, WTen0, WTbn0, WTwn0, WTrbn0, WTren0)):
        n = np.size(c)
        for j in range(n):
            if cond_x == 'c': # methane
                # condx = np.abs(c[j] - r[j]) > nstdvs * mystdv
                condx = np.abs(c[j] - (ahat*r[j] + bhat)) > nstdvs * mystdv
                # condd = np.abs(d[j] - (ahatcd*r[j] + bhatcd)) > nstdvs * mystdv
            else:
                print('wavelet partition error: insert valid value for condx')

            if cond_keep[i]:
                if condx: # energetic case
                    b[j] = 0
                    e[j] = c[j]
                    rb[j] = 0
                    re[j] = r[j]
                else: # background case
                    e[j] = 0
                    b[j] = c[j]
                    rb[j] = r[j]
                    re[j] = 0

    # COMPUTE DIRECT AND INDIRECT FLUX CONTRIBUTIONS FROM BACK / ENER FLUXES
    # now multiply each by its original stdv
    WTAc = sa.coeffs_2_array(WTcn0)[0]*np.std(df[mysc])
    WTAe = sa.coeffs_2_array(WTen0)[0]*np.std(df[mysc])
    WTAb = sa.coeffs_2_array(WTbn0)[0]*np.std(df[mysc])
    WTAw = sa.coeffs_2_array(WTwn0)[0]*np.std(df['w'])
    WTAr = sa.coeffs_2_array(WTrn0)[0]*np.std(df[refsc])
    WTArb = sa.coeffs_2_array(WTrbn0)[0]*np.std(df[refsc])
    WTAre = sa.coeffs_2_array(WTren0)[0]*np.std(df[refsc])

    if plot:
        WTAc00 = sa.coeffs_2_array(WTcn0)[0]
        WTAr00 = sa.coeffs_2_array(WTrn0)[0]
        WTAe00 = sa.coeffs_2_array(WTen0)[0]

        WTAc00p = WTAc00[np.abs(WTAc00) > 1E-6]
        WTAr00p = WTAr00[np.abs(WTAc00) > 1E-6]
        WTAe00p = WTAe00[np.abs(WTAc00) > 1E-6]

        plt.figure()
        plt.plot(WTAr00p, WTAc00p, 'o')
        plt.plot(WTAr00p[np.abs(WTAe00p)>1E-6], WTAe00p[np.abs(WTAe00p)>1E-6], 'or')
        plt.plot(WTAr00p, WTAr00p, 'k')
        plt.plot(WTAr00p, ahat*WTAr00p + bhat, 'g')
        plt.savefig(os.path.join(md.outdir_plot, 'wt_coeffs_{}.png'.format(datetime)))
        plt.show()

    # compute contributions to the fluxes from each wavelet coefficient
    # en_fii = WTAe * WTAw
    # ba_fii = WTAb * WTAw
    # me_fii = WTAc * WTAw
    # wa_fii = WTAr * WTAw
    # wb_fii = WTArb * WTAw
    # we_fii = WTAre * WTAw

    # wb_fdir = wb_fii[wb_fii > 0]
    # we_fdir = we_fii[we_fii > 0]
    # wa_fdir = wa_fii[wa_fii > 0]
    # me_fdir = me_fii[me_fii > 0]
    # en_fdir = en_fii[en_fii > 0]
    # ba_fdir = ba_fii[ba_fii > 0]
    #
    # wb_find = wb_fii[wb_fii < 0]
    # we_find = we_fii[we_fii < 0]
    # wa_find = wa_fii[wa_fii < 0]
    # me_find = me_fii[me_fii < 0]
    # en_find = en_fii[en_fii < 0]
    # ba_find = ba_fii[ba_fii < 0]

    # wb_eT_wave = 1 - np.abs( np.sum(wb_find)/np.sum(wb_fdir))
    # we_eT_wave = 1 - np.abs( np.sum(we_find)/np.sum(we_fdir))
    # wa_eT_wave = 1 - np.abs( np.sum(wa_find)/np.sum(wa_fdir))
    # me_eT_wave = 1 - np.abs( np.sum(me_find)/np.sum(me_fdir))
    # en_eT_wave = 1 - np.abs( np.sum(en_find)/np.sum(en_fdir))
    # ba_eT_wave = 1 - np.abs( np.sum(ba_find)/np.sum(ba_fdir))

   # eddy covariance fluxes (for comparison with wavelet over these scales only)
    me_flux_ec = np.mean( (df[mysc] - np.mean(df[mysc]))
                         *(df['w'] - np.mean(df['w'])))
    wa_flux_ec = np.mean( (df[refsc] - np.mean(df[refsc]))
                          *(df['w'] - np.mean(df['w'])))
    cd_flux_ec = np.mean( (df[cdsc] - np.mean(df[cdsc]))
                          *(df['w'] - np.mean(df['w'])))

    # wavelet fluxes (in the range of scales considered here only)
    me_flux_wt = 1 / N * np.sum(WTAc * WTAw)
    en_flux_wt = 1 / N * np.sum(WTAe * WTAw)
    ba_flux_wt = 1 / N * np.sum(WTAb * WTAw)
    wa_flux_wt = 1 / N * np.sum(WTAr * WTAw)
    wb_flux_wt = 1 / N * np.sum(WTArb * WTAw)
    we_flux_wt = 1 / N * np.sum(WTAre * WTAw)

    # compute fraction of the total flux and variance in this range of scales:
    me_flufrac_scales = me_flux_wt/me_flux_ec
    wa_flufrac_scales = wa_flux_wt/wa_flux_ec
    me_varfrac_scales = 1/N*np.sum(WTAc**2)/np.var(df[mysc])
    wa_varfrac_scales = 1/N*np.sum(WTAr**2)/np.var(df[refsc])

    # back to the time domain: compute energetic time fraction
    enn = pywt.waverec(WTen0, wavelet=wavelet)
    ban = pywt.waverec(WTbn0, wavelet=wavelet)

    enm = enn - np.median(enn) # centered in the median
    is_eb = enm**2 > np.var(ban)

    enn_is_eb = enn.copy()
    enn_is_eb[np.logical_not(is_eb)] = np.nan
    en_frac_time = np.size(is_eb[is_eb])/np.size(ban)
    ba_frac_time = 1 - en_frac_time

    # measures based on fluxes
    cn_ebii = cn[is_eb].copy()
    rn_ebii = rn[is_eb].copy()
    wn_ebii = wn[is_eb].copy()
    # dn_ebii = dn[is_eb].copy() # carbon dioxide

    cn_baii = cn[np.logical_not(is_eb)].copy()
    rn_baii = rn[np.logical_not(is_eb)].copy()
    wn_baii = wn[np.logical_not(is_eb)].copy()
    # dn_baii = dn[np.logical_not(is_eb)].copy() # carbon dioxide

    # active fractions of fluxes and variances
    en_frac_flux_T = np.sum(cn_ebii*wn_ebii)/np.sum(cn*wn)
    ba_frac_flux_T = 1 - en_frac_flux_T
    ew_frac_flux_T = np.sum(rn_ebii*wn_ebii)/np.sum(rn*wn)
    bw_frac_flux_T = 1 - ew_frac_flux_T
    # do not consider this because they may have different signs
    # ed_frac_flux_T = np.sum(dn_ebii*wn_ebii)/np.sum(dn*wn)
    # bd_frac_flux_T = 1 - ed_frac_flux_T

    en_frac_var_T = np.sum(cn_ebii**2)/np.sum(cn**2)
    ba_frac_var_T = 1 - en_frac_var_T
    ew_frac_var_T = np.sum(rn_ebii**2)/np.sum(rn**2)
    bw_frac_var_T = 1 - ew_frac_var_T
    # ed_frac_var_T = np.sum(dn_ebii**2)/np.sum(dn**2)
    # bd_frac_var_T = 1 - ed_frac_var_T

    # average fluxes and variances in the ebullition / background times only
    # CH4
    en_fa = np.mean(cn_ebii*wn_ebii)*np.std(df['w'])*np.std(df[mysc])
    en_va = np.mean(cn_ebii*cn_ebii)*np.std(df[mysc])*np.std(df[mysc])
    ba_fa = np.mean(cn_baii*wn_baii)*np.std(df['w'])*np.std(df[mysc])
    ba_va = np.mean(cn_baii*cn_baii)*np.std(df[mysc])*np.std(df[mysc])
    me_fa = np.mean(cn*wn)*np.std(df['w'])*np.std(df[mysc])
    me_va = np.mean(cn*cn)*np.std(df[mysc])*np.std(df[mysc])

    # H2O
    we_fa = np.mean(rn_ebii*wn_ebii)*np.std(df['w'])*np.std(df[refsc])
    we_va = np.mean(rn_ebii*rn_ebii)*np.std(df[refsc])*np.std(df[refsc])
    wb_fa = np.mean(rn_baii*wn_baii)*np.std(df['w'])*np.std(df[refsc])
    wb_va = np.mean(rn_baii*rn_baii)*np.std(df[refsc])*np.std(df[refsc])
    wa_fa = np.mean(rn*wn)*np.std(df['w'])*np.std(df[refsc])
    wa_va = np.mean(rn*rn)*np.std(df[refsc])*np.std(df[refsc])

    # CO2
    # defa = np.mean(dn_ebii*wn_ebii)*np.std(df['w'])*np.std(df[cdsc])
    # deva = np.mean(dn_ebii*dn_ebii)*np.std(df[cdsc])*np.std(df[cdsc])
    # dbfa = np.mean(dn_baii*wn_baii)*np.std(df['w'])*np.std(df[cdsc])
    # dbva = np.mean(dn_baii*dn_baii)*np.std(df[cdsc])*np.std(df[cdsc])
    # dafa = np.mean(dn*wn)*np.std(df['w'])*np.std(df[cdsc])
    # dava = np.mean(dn*dn)*np.std(df[cdsc])*np.std(df[cdsc])

    # moments of original methane, water and carbon dioxide time series
    me_mom = mixed_moments(cn, wn) # no need to normalize them
    wa_mom = mixed_moments(rn, wn) # no need to normalize them
    # cd_mom = mixed_moments(dn, wn) # no need to normalize them

    # do in for time-filtered time series instead
    en_mom = partial_mixed_moments(cn_ebii, wn_ebii, N = N, normalize = False)
    ba_mom = partial_mixed_moments(cn_baii, wn_baii, N = N, normalize = False)
    enr_mom =partial_mixed_moments(rn_ebii, wn_ebii, N = N,  normalize = False)
    bar_mom =partial_mixed_moments(rn_baii, wn_baii, N = N,  normalize = False)
    # end_mom =partial_mixed_moments(dn_ebii, wn_ebii, N = N,  normalize = False)
    # bad_mom =partial_mixed_moments(dn_baii, wn_baii, N = N,  normalize = False)


    eT_me = compute_eT(cn, wn)
    eT_wa = compute_eT(rn, wn)
    eT_cd = compute_eT(dn, wn)
    eT_en = compute_eT(cn_ebii, wn_ebii, normalize = False)
    eT_ba = compute_eT(cn_baii, wn_baii, normalize = False)
    eT_we = compute_eT(rn_ebii, wn_ebii, normalize = False)
    eT_wb = compute_eT(rn_baii, wn_baii, normalize = False)
    # eT_de = compute_eT(dn_ebii, wn_ebii, normalize = False)
    # eT_db = compute_eT(dn_baii, wn_baii, normalize = False)

    if plot:

        minutes = np.arange(1, np.size(rn)+1)/md.fs/60
        offst = 0.16
        offst2 = 0.12

        # xd, yd = sa.emp_pdf(dn, 20)
        # xc, yc = sa.emp_pdf(cn, 20)
        # xr, yr = sa.emp_pdf(rn, 20)
        # xb, yb = sa.emp_pdf(cn_baii, 20)
        # xe, ye = sa.emp_pdf(cn_ebii, 20)

        def symlogspace(x, npoints=100):
            xmin = np.min(x)
            if xmin < 0:
                offset = 0.1 - xmin
            else:
                offset = 0.0
            xpr = x + offset
            pmin = np.log10(np.min(xpr))
            pmax = np.log10(np.max(xpr) + 1) # higher offset to plot high values
            sls_pr = np.logspace(pmin, pmax, npoints)
            sls = sls_pr - offset - 0.2 # lower offset to plot low values
            return sls

        npoints_kde = 40
        edges_xd = symlogspace(dn, npoints=npoints_kde)
        edges_xc = symlogspace(cn, npoints=npoints_kde)
        edges_xr = symlogspace(rn, npoints=npoints_kde)
        edges_xb = symlogspace(cn_baii, npoints=npoints_kde)
        edges_xe = symlogspace(cn_ebii, npoints=npoints_kde)

        yd, _ = np.histogram(dn, bins = edges_xd, density = True)
        yc, _ = np.histogram(cn, bins = edges_xc, density = True)
        yr, _ = np.histogram(rn, bins = edges_xr, density = True)
        yb, _ = np.histogram(cn_baii, bins = edges_xb, density = True)
        ye, _ = np.histogram(cn_ebii, bins = edges_xe, density = True)

        xd = edges_xd[:-1] + 0.5*(edges_xd[1:]-edges_xd[:-1])
        xc = edges_xc[:-1] + 0.5*(edges_xc[1:]-edges_xc[:-1])
        xr = edges_xr[:-1] + 0.5*(edges_xr[1:]-edges_xr[:-1])
        xb = edges_xb[:-1] + 0.5*(edges_xb[1:]-edges_xb[:-1])
        xe = edges_xe[:-1] + 0.5*(edges_xe[1:]-edges_xe[:-1])


        maxfreq = max(np.max(yc), np.max(yr), max(yb) )

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize = (10, 14))
        if datetime:
            fig.suptitle('Run measured {}'.format(datetime), fontsize = 24)
        axes[0,0].set_ylabel(r"$c'/\sigma_c$ ($CH_4$)")
        axes[0,0].annotate("a)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[1,0].annotate("b)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[2,0].annotate("c)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[3,0].annotate("d)", xy=(0.03, 0.87), fontsize = 14, xycoords="axes fraction")
        axes[0,0].plot(minutes, cn, 'g', alpha = 1, label = r'$CH_4$')

        axes[0, 0].plot([minutes[0], minutes[-1]], [np.std(cn), np.std(cn)], '--k')
        axes[0, 0].plot([minutes[0], minutes[-1]], [-np.std(cn), -np.std(cn)], '--k')
        box = axes[0, 0].get_position()
        axes[0, 0].set_position([box.x0, box.y0, box.width + offst, box.height])
        axes[0, 0].set_yscale('symlog')
        axes[0, 1].set_yscale('symlog')

        axes[1,0].plot(minutes, rn, 'b', label = r'$H_2O$')
        axes[1, 0].plot([minutes[0], minutes[-1]], [np.std(rn), np.std(rn)], '--k')
        axes[1, 0].plot([minutes[0], minutes[-1]], [-np.std(rn), -np.std(rn)], '--k')
        axes[1,0].set_ylabel(r"$r'/\sigma_r$ ($H_2O$)")
        box = axes[1, 0].get_position()
        axes[1, 0].set_position([box.x0, box.y0, box.width + offst, box.height])

        # do commom y axis:
        ymin0, _ = axes[0,0].get_ylim()
        ymin1, _ = axes[1,0].get_ylim()
        ymin = min(ymin0, ymin1)
        axes[0, 0].set_ylim(bottom=ymin)
        axes[1, 0].set_ylim(bottom=ymin)
        axes[0, 1].set_ylim(bottom=ymin)
        axes[1, 1].set_ylim(bottom=ymin)

        # second row
        # xd, edges_d = np.histogram(dn, bins = xd, density = 'True')
        # axes[0, 1].hist(dn, bins = xd, density='True')
        axes[0,1].plot(yc, xc, 'g', linewidth = 2)
        box = axes[0, 1].get_position()
        axes[0, 1].set_position([box.x0+offst2, box.y0, box.width - offst + offst-offst2, box.height])
        axes[0, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[0, 1].set_xscale('log')
        axes[0, 1].plot([0.002, 2.1], [0, 0], '--k')
        axes[1,1].plot(yr, xr, 'b', linewidth = 2)
        box = axes[1, 1].get_position()
        axes[1, 1].set_position([box.x0+offst2, box.y0, box.width -offst + offst-offst2, box.height])
        axes[1, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[1, 1].set_xscale('log')
        axes[1, 1].plot([0.002, 2.1], [0, 0], '--k')
        axes[0, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[0, 1].set_xticklabels([0.002, 0.02, 0.2, 2])
        axes[1, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[1, 1].set_xticklabels([0.002, 0.02, 0.2, 2])
        axes[1, 0].set_yscale('symlog')
        axes[1, 1].set_yscale('symlog')

        # third row
        axes[2, 0].plot(minutes, cn, 'grey', linewidth = 1)
        axes[2, 0].plot(minutes[is_eb], cn[is_eb], 'or', linewidth = 0.9, markersize = 2)
        axes[2, 0].plot([minutes[0], minutes[-1]], [np.std(ban), np.std(ban)], '--k')
        axes[2, 0].plot([minutes[0], minutes[-1]], [-np.std(ban), -np.std(ban)], '--k')
        box = axes[2, 0].get_position()
        axes[2, 0].set_position([box.x0, box.y0, box.width + offst, box.height])
        axes[2,0].set_ylabel(r"$c'/\sigma_c$ ($CH_4$)")
        axes[2,1].plot(yb, xb, 'grey', linewidth = 2)
        axes[2,1].plot(ye, xe, '--r', linewidth = 2)
        axes[2, 1].plot( [0.002, 2.1], [0, 0],'--k')
        box = axes[2, 1].get_position()
        axes[2, 1].set_position([box.x0+offst2, box.y0, box.width -offst + offst-offst2, box.height])
        axes[2, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[2, 1].set_xscale('log')
        axes[2, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[2, 1].set_xticklabels([0.002, 0.02, 0.2, 2])
        axes[2, 0].set_yscale('symlog')
        axes[2, 1].set_yscale('symlog')

        meth_spikes = cn > 3
        axes[3, 0].plot(minutes, dn, 'grey', alpha = 1, linewidth = 1)
        axes[3, 0].plot(minutes[meth_spikes], dn[meth_spikes], 'or', linewidth = 0.9, markersize = 2)
        axes[3, 0].plot([minutes[0], minutes[-1]], [np.std(dn), np.std(dn)], '--k')
        axes[3, 0].plot([minutes[0], minutes[-1]], [-np.std(dn), -np.std(dn)], '--k')
        box = axes[3, 0].get_position()
        axes[3, 0].set_position([box.x0, box.y0, box.width + offst, box.height])
        axes[3,0].set_xlabel('time [minutes]')
        axes[3,0].set_ylabel(r"$c'/\sigma_c$ ($CO_2$)")
        axes[3,1].plot(yd, xd, 'grey', linewidth = 2)
        axes[3, 1].plot( [0.002, 2.1], [0, 0],'--k')
        box = axes[3, 1].get_position()
        axes[3, 1].set_position([box.x0+offst2, box.y0, box.width -offst + offst-offst2, box.height])
        axes[3, 1].set_xlim([0.4E-2, maxfreq + 0.1])
        axes[3, 1].set_xscale('log')
        axes[3, 1].set_xlabel('Frequency density')
        axes[3, 1].set_xticks([0.002, 0.02, 0.2, 2])
        axes[3, 1].set_xticklabels([0.002, 0.02, 0.2, 2])
        axes[2,0].set_xlabel('time [minutes]')
        axes[2,1].set_xlabel('Frequency density')
        axes[1,0].set_xlabel('time [minutes]')
        axes[1,1].set_xlabel('Frequency density')
        axes[0,0].set_xlabel('time [minutes]')
        axes[0,1].set_xlabel('Frequency density')

        axes[3, 1].set_yscale('symlog')
        axes[3, 0].set_yscale('symlog')


        ylims = axes[0, 0].get_ylim()
        axes[0, 1].set_ylim(ylims)
        ylims = axes[1, 0].get_ylim()
        axes[1, 1].set_ylim(ylims)
        ylims = axes[2, 0].get_ylim()
        axes[2, 1].set_ylim(ylims)
        ylims = axes[3, 0].get_ylim()
        axes[3, 1].set_ylim(ylims)

        # move histograms y-labels on the right
        axes[0, 1].yaxis.set_label_position("right")
        axes[0, 1].yaxis.tick_right()
        axes[1, 1].yaxis.set_label_position("right")
        axes[1, 1].yaxis.tick_right()
        axes[2, 1].yaxis.set_label_position("right")
        axes[2, 1].yaxis.tick_right()
        axes[3, 1].yaxis.set_label_position("right")
        axes[3, 1].yaxis.tick_right()

        # plt.tight_layout()
        plt.savefig(os.path.join(md.outdir_plot, 'traces_{}.png'.format(datetime)))
        plt.show()

    #############  END PLOTTING ##############

    res = {}
    res['en_frac_flux'] =  en_frac_flux_T  # energetic flux fraction
    res['ba_frac_flux'] =  ba_frac_flux_T  # background flux fraction
    res['en_frac_var'] =  en_frac_var_T  # energetic  variance fraction
    res['ba_frac_var'] =  ba_frac_var_T  # background variance fraction
    res['en_frac_time'] =  en_frac_time  # energetic  time fraction
    res['ba_frac_time'] =  ba_frac_time  # background time fraction
    # res['en_eT_wave'] = en_eT_wave
    # res['ba_eT_wave'] = ba_eT_wave
    # res['wb_eT_wave'] = wb_eT_wave
    # res['we_eT_wave'] = we_eT_wave
    # res['wa_eT_wave'] = wa_eT_wave
    # res['me_eT_wave'] = me_eT_wave
    res['minsize'] = minsize
    res['maxsize'] = maxsize
    res['maxtin'] = maxtin

    # computed in time, not wavelet domain
    # res['me_nofilter_eT'] = eT_me_nofilter
    # res['wa_nofilter_eT'] = eT_wa_nofilter
    res['me_eT'] = eT_me
    res['wa_eT'] = eT_wa
    res['cd_eT'] = eT_cd
    res['en_eT'] = eT_en
    res['ba_eT'] = eT_ba
    res['we_eT'] = eT_we
    res['wb_eT'] = eT_wb
    # res['de_eT'] = eT_de
    # res['db_eT'] = eT_db

    # ratio of total to (in this range of scales) fluxes and variances
    res['me_flufrac_scales'] = me_flufrac_scales
    res['wa_flufrac_scales'] = wa_flufrac_scales
    res['me_varfrac_scales'] = me_varfrac_scales
    res['wa_varfrac_scales'] = wa_varfrac_scales
    res['bw_frac_var'] = bw_frac_var_T
    res['ew_frac_var'] = ew_frac_var_T
    res['bw_frac_flux'] = bw_frac_flux_T
    res['ew_frac_flux'] = ew_frac_flux_T
    # res['bd_frac_var'] = bd_frac_var_T
    # res['ed_frac_var'] = ed_frac_var_T
    # res['bd_frac_flux'] = bd_frac_flux_T
    # res['ed_frac_flux'] = ed_frac_flux_T

    res['en_flux_wt'] = en_flux_wt
    res['ba_flux_wt'] = ba_flux_wt
    res['wb_flux_wt'] = wb_flux_wt
    res['we_flux_wt'] = we_flux_wt
    res['wa_flux_wt'] = wa_flux_wt
    res['me_flux_wt'] = me_flux_wt

    res['wa_flux_ec'] = wa_flux_ec
    res['me_flux_ec'] = me_flux_ec
    res['cd_flux_ec'] = cd_flux_ec

    res['cn_ener'] = enn
    res['cn_back'] = ban
    # res['stdv_c2'] = stdv_c2
    # res['stdv_cw'] = stdv_cw
    res['stdv_c'] = stdv_c
    # res['stdv_d'] = stdv_d
    # res['stdv_cn'] = stdv_cn
    res['my_stdv'] = mystdv
    res['Rcr'] = Rcr
    res['WTe'] = WTen0 # energetic wavelet coefficents
    res['WTb'] = WTbn0 # energetic wavelet coefficents

    # average fluxes and variances in each section
    res['en_fa'] = en_fa
    res['en_va'] = en_va
    res['ba_fa'] = ba_fa
    res['ba_va'] = ba_va
    res['me_fa'] = me_fa
    res['me_va'] = me_va

    res['we_fa'] = we_fa
    res['we_va'] = we_va
    res['wb_fa'] = wb_fa
    res['wb_va'] = wb_va
    res['wa_fa'] = wa_fa
    res['wa_va'] = wa_va

    # res['defa'] = defa
    # res['deva'] = deva
    # res['dbfa'] = dbfa
    # res['dbva'] = dbva
    # res['dafa'] = dafa
    # res['dava'] = dava

    res['me_M40'] = me_mom['M40']
    res['me_M30'] = me_mom['M30']
    res['me_M21'] = me_mom['M21']
    res['me_M12'] = me_mom['M12']
    res['me_M13'] = me_mom['M13']
    res['me_Rcw'] = me_mom['Rcw']

    res['wa_M40'] = wa_mom['M40']
    res['wa_M30'] = wa_mom['M30']
    res['wa_M21'] = wa_mom['M21']
    res['wa_M12'] = wa_mom['M12']
    res['wa_M13'] = wa_mom['M13']
    res['wa_Rcw'] = wa_mom['Rcw']

    # res['cd_M40'] = cd_mom['M40']
    # res['cd_M30'] = cd_mom['M30']
    # res['cd_M21'] = cd_mom['M21']
    # res['cd_M12'] = cd_mom['M12']
    # res['cd_M13'] = cd_mom['M13']
    # res['cd_Rcw'] = cd_mom['Rcw']

    res['en_M40'] = en_mom['M40']
    res['en_M30'] = en_mom['M30']
    res['en_M21'] = en_mom['M21']
    res['en_M12'] = en_mom['M12']
    res['en_M13'] = en_mom['M13']
    res['en_Rcw'] = en_mom['Rcw']

    res['ba_M40'] = ba_mom['M40']
    res['ba_M30'] = ba_mom['M30']
    res['ba_M21'] = ba_mom['M21']
    res['ba_M12'] = ba_mom['M12']
    res['ba_M13'] = ba_mom['M13']
    res['ba_Rcw'] = ba_mom['Rcw']

    res['enr_M40'] = enr_mom['M40']
    res['enr_M30'] = enr_mom['M30']
    res['enr_M21'] = enr_mom['M21']
    res['enr_M12'] = enr_mom['M12']
    res['enr_M13'] = enr_mom['M13']
    res['enr_Rcw'] = enr_mom['Rcw']


    res['bar_M40'] = bar_mom['M40']
    res['bar_M30'] = bar_mom['M30']
    res['bar_M21'] = bar_mom['M21']
    res['bar_M12'] = bar_mom['M12']
    res['bar_M13'] = bar_mom['M13']
    res['bar_Rcw'] = bar_mom['Rcw']

    # res['end_M40'] = end_mom['M40']
    # res['end_M30'] = end_mom['M30']
    # res['end_M21'] = end_mom['M21']
    # res['end_M12'] = end_mom['M12']
    # res['end_M13'] = end_mom['M13']
    # res['end_Rcw'] = end_mom['Rcw']
    #
    #
    # res['bad_M40'] = bad_mom['M40']
    # res['bad_M30'] = bad_mom['M30']
    # res['bad_M21'] = bad_mom['M21']
    # res['bad_M12'] = bad_mom['M12']
    # res['bad_M13'] = bad_mom['M13']
    # res['bad_Rcw'] = bad_mom['Rcw']

    return res