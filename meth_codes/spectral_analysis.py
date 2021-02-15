# functions to perform some spectral analyses


import os
import numpy as np
import matplotlib.pyplot as plt
# import stats as st
# from scipy import signal
# import scipy as sc
import pywt


def dwt_power_spectrum(data, fs, wavelet='haar', mode='periodic', density=True, Ubar = 1):
    # give it ks instead of ks for wavenumber
    # if wavenumber =1, need to provide mean velocity U
    # keep only data up to the nearest power of 2
    ndata = np.size(data)
    Mi = np.log2(ndata)
    M = np.rint(Mi)
    N = np.int(2**M)
    x = data[:N]
    coeffs = pywt.wavedec(x, wavelet = wavelet, mode = mode)
    # skip the first output - coarse grained signal = signal average
    # print(len(coeffs))
    sizes = np.array([np.size(coeffs[i]) for i in range(1, len(coeffs))])
    print('sizes = ', sizes)
    # total energy at each scale
    TE = np.array([1/N*np.sum(coeffs[i]**2) for i in range(1, len(coeffs))])
    MEm = np.array([np.mean(coeffs[i]**2) for i in range(1, len(coeffs))]) # mean energy at scale m = TE*2**m
    # s2 = np.sum(TE) # total variance of the time series
    m = np.log2(N/sizes)  # scale index
    # print('m = ', m)
    # if wavenumber:
    #     Rm = 2**m/fs*U # scales of decompositions [T]
    #     Km = 2*np.pi/Rm
    #     # wsd = TE/(2*np.pi*np.log(2)*fs)*U*2**m
    #     wsd = TE/(2*np.pi*np.log(2)*fs)*U*2**m
    # else:
    # Rm = 2**m/fs # time or spatial scale
    # Km = 1/Rm # frequency of wavenumber
    # Following Scanlon and Albertson 2001 notation::
    fm = fs/(2**m) # frequency (cfr Fourier)
    Km = 2*np.pi*fs/(2**m)/Ubar # wavenumber
    # fm = Km*Ubar
    # Km = fm/Ubar # wavenumber
    Rm = (2*np.pi)/Km # eddy length scale (= D)
    if density:
        # wsd = TE/(np.log(2)*fs)*2**m ## Equivalent
        wsd = MEm/(np.log(2)*fs)*Ubar
    else:
        wsd = TE # (average) wavelet energy at each scale
    # returns scale, wavenumber, frequency and spectral density
    return Rm, Km, fm, wsd


def dwt_csd(x, y, fs, wavelet='haar', mode='periodic', density=True, Ubar = 1):
    # if wavenumber =1, need to provide mean velocity U
    # keep only data up to the nearest power of 2    # reduce the two sample at the common length, maximum power of 2
    len_x = np.size(x)
    len_y = np.size(y)
    min_len = min(len_x, len_y)
    Mi = np.log2(min_len)
    M = np.floor(Mi)
    N = np.int(2**M)
    x = x[:N]
    y = y[:N]
    x_coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode)
    y_coeffs = pywt.wavedec(y, wavelet=wavelet, mode=mode)
    sizes = np.array([np.size(x_coeffs[i]) for i in range(1, len(x_coeffs))])
    TE = np.array([1/N*np.sum(x_coeffs[i]*np.conj(y_coeffs[i])) for i in range(1, len(x_coeffs))])
    ME = np.array([np.mean(x_coeffs[i]*np.conj(y_coeffs[i])) for i in range(1, len(x_coeffs))])
    m = np.log2(N/sizes)  # scale index
    fm = fs/(2**m) # frequency (cfr Fourier)
    Km = 2*np.pi*fs/(2**m)/Ubar # wavenumber
    Rm = (2*np.pi)/Km # eddy length scale (= D)
    if density:
        wsd = ME/(np.log(2)*fs)
    else:
        wsd = TE
    # returns scale, wavenumber, frequency and spectral density
    return Rm, Km, fm, wsd


def dwt_cohe(x, y, fs, wavelet='haar', mode='periodic', Ubar = 1):
    # if wavenumber =1, need to provide mean velocity U
    # keep only data up to the nearest power of 2    # reduce the two sample at the common length, maximum power of 2
    len_x = np.size(x)
    len_y = np.size(y)
    min_len = min(len_x, len_y)
    Mi = np.log2(min_len)
    M = np.floor(Mi)
    N = np.int(2**M)
    x = x[:N]
    y = y[:N]
    x_coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode)
    y_coeffs = pywt.wavedec(y, wavelet=wavelet, mode=mode)
    sizes = np.array([np.size(x_coeffs[i]) for i in range(1, len(x_coeffs))])
    # TE = np.array([1/N*np.sum(x_coeffs[i]*np.conj(y_coeffs[i])) for i in range(1, len(x_coeffs))])
    MEc = np.array([np.mean(x_coeffs[i]*np.conj(y_coeffs[i])) for i in range(1, len(x_coeffs))])
    MEx = np.array([np.mean(x_coeffs[i]*np.conj(x_coeffs[i])) for i in range(1, len(x_coeffs))])
    MEy = np.array([np.mean(y_coeffs[i]*np.conj(y_coeffs[i])) for i in range(1, len(x_coeffs))])
    m = np.log2(N/sizes)  # scale index
    fm = fs/(2**m) # frequency (cfr Fourier)
    Km = 2*np.pi*fs/(2**m)/Ubar # wavenumber
    Rm = (2*np.pi)/Km # eddy length scale (= D)
    wcohe = MEc**2 / (MEx * MEy)
    # returns scale, wavenumber, frequency and spectral density
    return Rm, Km, fm, wcohe


def dwt_flux(x, y, wavelet='haar', mode='periodic'):
    '''------------------------------------------------
    compute the flux using discrete wavelet transform
    keep only data up to the nearest power of 2
    ------------------------------------------------'''
    len_x = np.size(x)
    len_y = np.size(y)
    min_len = min(len_x, len_y)
    Mi = np.log2(min_len)
    M = np.floor(Mi)
    N = np.int(2**M)
    x = x[:N]
    y = y[:N]
    WT_x = pywt.wavedec(x, wavelet=wavelet, mode=mode)
    WT_y = pywt.wavedec(y, wavelet=wavelet, mode=mode)
    # skip the first element of the lists = coarse grained signal
    flux_wave = 1/N*np.sum( np.array( [   np.sum(WT_x[j]*np.conj(WT_y[j])) for j in range(1, len(WT_x))]   ))
    res = {'flux_wa':flux_wave}
    return res


def fourier_power_spectrum(data, fs, density=True, Ubar = 1):
    """
    Compute Fourier Power spectrum / spectral density
    To compare with freq must keep Ubar = 1.
    Otherwise, integrates with respect to Km

    # compare with the following scipy functions::
    # in case scaling = 'spectrum' or 'density'
    # f, Pxx = signal.welch(data, fs, nperseg=2 ** 14, scaling='density')
    # f_den, Pxx_den = signal.periodogram(data, fs, scaling='density')

    # ARGUMENTS:
        data = array with data
        fs = sampling density
        density = True (if True compute power spectral density, if not power spectrum)
        Ubar = 1: 'velocity' to convert time in space and compute wavenumbers Km.
    # RETURN:
        ps = power spectral density or spectrum
        freq = array of frequencies
        ks = array of wavenumbers (using Taylor with Ubar)
    """
    ndata = np.size(data)
    Mi = np.log2(ndata)
    M = np.floor(Mi)
    n = np.int(2**M)
    x = data[:n]
    y = np.fft.fft(x)
    p2 = np.conj(y)*y
    # p2 = y.real**2 + y.imag**2
    freq = fs*np.arange(1, int(n/2))/n
    ks = freq/Ubar
    ps = 2/n*p2[1:int(n/2)]/fs*Ubar
    if not density:
        ps = ps*np.abs(freq[1]-freq[0])/Ubar
    return ps, freq, ks


def fourier_csd(x, y, fs, density=True, Ubar = 1):
    # reduce the two sample at the common length, maximum power of 2
    # compute Fourier cross power spectral density
    len_x = np.size(x)
    len_y = np.size(y)
    min_len = min(len_x, len_y)
    Mi = np.log2(min_len)
    M = np.floor(Mi)
    n = np.int(2**M)
    x = x[:n]
    y = y[:n]
    #########################
    xf = np.fft.fft(x)
    yf = np.fft.fft(y)
    cs = np.conj(xf)*yf # Fourier cross spectrum
    # freq = fs*np.arange(1, int(n/2))/n # frequencies corresponding to the first half of power spectrum values,
    # TE = np.sum(cs) # total Energy
    # # print('Total Energy =', TE)
    # if density:
    #     ps = 2*cs[1:int(n/2)]/TE/np.abs(freq[1]-freq[0])   # power spectrum
    # else:
    #     ps = 2/n*cs[1:int(n/2)]   # Energy spectrum
    freq = fs*np.arange(1, int(n/2))/n
    ks = freq/Ubar
    ps = 2/n*cs[1:int(n/2)]/fs*Ubar
    if not density:
        ps = ps*np.abs(freq[1]-freq[0])/Ubar
    return ps, freq, ks



def fourier_coherence(x, y, fs, Ubar = 1):
    # reduce the two sample at the common length, maximum power of 2
    # compute Fourier cross power spectral density
    len_x = np.size(x)
    len_y = np.size(y)
    min_len = min(len_x, len_y)
    Mi = np.log2(min_len)
    M = np.floor(Mi)
    n = np.int(2**M)
    x = x[:n]
    y = y[:n]
    #########################
    xf = np.fft.fft(x)
    yf = np.fft.fft(y)
    psx = np.conj(xf)*xf
    psy = np.conj(yf)*yf
    cs = np.conj(xf)*yf # Fourier cross spectrum
    freq = fs*np.arange(1, int(n/2))/n
    ks = freq/Ubar
    psc = 2/n*cs[1:int(n/2)]/fs*Ubar
    psx0 = 2/n*psx[1:int(n/2)]/fs*Ubar
    psy0 = 2/n*psy[1:int(n/2)]/fs*Ubar
    cohe = np.abs(psc)**2 / psx0 / psy0
    # cohe = cohe[1:int(n/2)]
    return cohe, freq, ks


def fourier_coherence(x, y, fs, Ubar = 1, nblocks = 64):
    # reduce the two sample at the common length, maximum power of 2
    # compute Fourier cross power spectral density
    len_x = np.size(x)
    len_y = np.size(y)
    min_len = min(len_x, len_y)
    Mi = np.log2(min_len)
    M = np.floor(Mi)
    n = np.int(2**M)
    x = x[:n]
    y = y[:n]
    #########################
    # I must divide the signal in blocks:
    # nblocks = 64
    bsize = n//nblocks
    mx = np.reshape(x, (nblocks, bsize))
    my = np.reshape(y, (nblocks, bsize))

    freq = fs * np.arange(1, int(bsize / 2)) / bsize
    ks = freq/Ubar
    nfreqs = np.size(freq)

    # psx = np.zeros((4, 2))
    psx = np.zeros((nblocks, nfreqs))
    psy = np.zeros((nblocks, nfreqs))
    psc = np.zeros((nblocks, nfreqs))

    ########################
    for i in range(nblocks):
        xf = np.fft.fft(mx[i, :])
        yf = np.fft.fft(my[i, :])
        tempx = 2*fs**2/bsize*np.conj(xf)*xf
        tempy = 2*fs**2/bsize*np.conj(yf)*yf
        tempc = 2*fs**2/bsize*np.conj(xf)*yf


        psx[i, :] = tempx[1:int(bsize/2)]
        psy[i, :] = tempy[1:int(bsize/2)]
        psc[i, :] = tempc[1:int(bsize/2)]

    apsx = np.mean(psx, 0)
    apsy = np.mean(psy, 0)
    apsc = np.mean(psc, 0)

    # psc = 2/n*cs[1:int(n/2)]/fs*Ubar
    # psx0 = 2/n*psx[1:int(n/2)]/fs*Ubar
    # psy0 = 2/n*psy[1:int(n/2)]/fs*Ubar
    # cohe = np.abs(psc)**2 / psx0 / psy0
    # cohe = np.abs(apsc) / np.sqrt( apsx * apsy)
    cohe = np.abs(apsc)**2 / ( apsx * apsy)
    # cohe = cohe[1:int(n/2)]
    # return cohe, freq, ks
    return cohe, freq, ks


# def fourier_power_spectrum(data, fs, density=True):
#     ndata = np.size(data)
#     Mi = np.log2(ndata)
#     # print(Mi)
#     M = np.floor(Mi)
#     # print(M)
#     n = np.int(2**M)
#     # print(N)
#     x = data[:n]
#     #########################
#     y = np.fft.fft(x)
#     p2 = np.conj(y)*y
#     print(np.sum(p2))
#     freq = fs*np.arange(1, int(n/2))/n # frequencies corresponding to the first half of power spectrum values,
#     TE = np.sum(p2) # total Energy
#     print('Total Energy =', TE)
#     if density:
#         # ps = 2*p2[1:int(n/2)]/TE/np.abs(freq[1]-freq[0])   # power spectrum
#         ps = 1/4*p2[1:int(n/2)]/TE/np.abs(freq[1]-freq[0])   # power spectrum
#     else:
#         # ps = 2/n*p2[1:int(n/2)]   # Energy spectrum
#         ps = 1 / n * p2[1:int(n / 2)]  # Energy spectrum
#     return freq, ps




# def fourier_smoothed_spectrum(x, fs, windsize=2048, density=True):
#     ''' compute and alternatively plot the fourier power spectrum
#     of a signal (1-D array x) sampled with frequency fs [Hz]
#     if plot -> plot it
#     if smoothed -> smooth it using a hamming window
#     windsize -> windows in which compute the spectra to be average
#     (keep it a power of 2: 2048, 4096, 8192, ...)
#     Returns:
#     non-smoothed spectrum spectrum
#     smoothed spectrum ssp
#     and their relative frequency arrays: freq,freqs '''
#     nave = np.int(np.floor(np.size(x)/windsize))
#     vec = x[:np.int(nave*windsize)]
#     mat = vec.reshape(nave, windsize)
#     slen = np.int(windsize/2-1)
#     spectra = np.zeros((slen, nave))
#     for ii in range(nave):
#         myvec = mat[ii, :]
#         hann = np.hanning(windsize)*myvec # apply hanning window to current sample
#         freq, pi = fourier_power_spectrum(hann, fs, density=density)
#         spectra[:,ii] = pi
#     ps = (np.mean(spectra, axis=1))
#     # if plot:
#     #     spectrum, freqg = fourier_power_spectrum(x, fs, density=density)
#     #     x53 = freq
#     #     y53 = freq**(-5/3)
#     #     plt.figure()
#     #     plt.plot(freqg, spectrum,'o', label='observed')
#     #     plt.plot(freq, ps,'r', label='smoothed')
#     #     plt.plot(x53,y53,'k', label='-5/3')
#     #     plt.xlabel('f [Hz]')
#     #     plt.ylabel('fft power spectrum')
#     #     plt.xscale('log')
#     #     plt.yscale('log')
#     #     plt.show()
#     return freq, ps
#
#
# def fourier_smoothed_cross_spectrum(x, y, fs, windsize=2048, density=True):
#     nx = np.size(x)
#     ny = np.size(y)
#     n = min(nx, ny)
#     x = x[:n]
#     y = y[:n]
#     nave = np.int(np.floor(n/windsize))
#     vx = x[:np.int(nave*windsize)]
#     vy = y[:np.int(nave*windsize)]
#     matx = vx.reshape(nave, windsize)
#     maty = vy.reshape(nave, windsize)
#     slen = np.int(windsize/2-1)
#     spectra = np.zeros((slen, nave))
#     for ii in range(nave):
#         myvecx = matx[ii, :]
#         myvecy = maty[ii, :]
#         hannx = np.hanning(windsize)*myvecx
#         hanny = np.hanning(windsize)*myvecy
#         freq, cs = fourier_cross_spectrum(hannx, hanny, fs, density=density)
#         spectra[:, ii] = cs
#     ps = (np.mean(spectra, axis=1))
#     return freq, ps


def coeffs_2_array(coeffs):
    ''' transform wavelet coefficients
    list of arrays in a single array'''
    nscales = len(coeffs)
    coeff_array = np.array([])
    sizes = np.zeros(nscales)
    for mm in range(nscales):
        sizes[mm] = np.size(coeffs[mm])
        coeff_array = np.concatenate((coeff_array, coeffs[mm]), axis = 0)
    np.size(coeff_array)
    return coeff_array, sizes


def array_2_coeffs(array, sizes):
    ''' thansform wavelet coefficients
    from single array to list of arrays at each scale'''
    N = np.size(array) # must be a power of 2
    nscales = np.size(sizes)
    coeffs = []
    start = int(0)
    end = int(sizes[0])
    for mm in range(nscales):
        coeffs.append(array[start:end])
        start = end
        if mm+1 < nscales:
            end = np.int(end + sizes[mm+1])
    return coeffs


def dft_statistics(data, fs, wavelet='haar', mode='periodic'):
    # compute dft statistics following Katul et al, 1994
    ndata = np.size(data)
    Mi = np.log2(ndata)
    M = np.rint(Mi)
    N = np.int(2**M)
    x = data[:N]
    coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode)
    sizes = np.array([np.size(coeffs[i]) for i in range(1, len(coeffs))])
    ME2 = np.array([np.mean(coeffs[i]**2) for i in range(1, len(coeffs))])
    ME3 = np.array([np.mean(coeffs[i]**3) for i in range(1, len(coeffs))])
    ME4 = np.array([np.mean(coeffs[i]**4) for i in range(1, len(coeffs))]) # contributions to Kurtosis
    m = np.log2(N/sizes)  # scale index
    Rm = 2**m/fs # time or spatial scale
    Km = 1/Rm # frequency of wavenumber
    EK = ME2/(np.log(2)*fs) # power spectral density
    SD = 1/(np.log(2)*fs)*(ME4 - ME2**2)**0.5 # standard deviation
    CV = SD/EK # coeff of variation of the power spectral density
    SF = ME3/ME2**(3/2) # wavelet skewness factor
    FF = ME4/ME2**(4/2) # wavelet flatness factor
    wavestats = {'Scale': Rm, 'Frequency':Km, 'Spectrum':EK, 'Stdv':SD, 'CV': CV, 'Skewness':SF, 'Kurtosis':FF}
    return wavestats


# def dft_cross_statistics(x, y, fs, wavelet='haar', mode='periodic'):
#     # compute dft statistics following Katul et al, 1994
#     len_x = np.size(x)
#     len_y = np.size(y)
#     min_len = min(len_x, len_y)
#     Mi = np.log2(min_len)
#     M = np.floor(Mi)
#     N = np.int(2**M)
#     x = x[:N]
#     y = y[:N]
#     x_coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode)
#     y_coeffs = pywt.wavedec(y, wavelet=wavelet, mode=mode)
#     ncoeffs = len(x_coeffs)
#     sizes = np.array([np.size(x_coeffs[i]) for i in range(1, ncoeffs)])
#     ME2 = np.array([np.mean(x_coeffs[i]*y_coeffs[i]) for i in range(1, ncoeffs)])
#     ME4 = np.array([np.mean((x_coeffs[i]*y_coeffs[i])**2) for i in range(1, ncoeffs)])
#     ME3 = np.array([np.mean(coeffs[i]**3) for i in range(1, ncoeffs)])
#     ME4 = np.array([np.mean(coeffs[i]**4) for i in range(1, ncoeffs)]) # contributions to Kurtosis
#     m = np.log2(N/sizes)  # scale index
#     Rm = 2**m/fs # time or spatial scale
#     Km = 1/Rm # frequency of wavenumber
#     EK = ME2/(np.log(2)*fs) # cross power spectral density
#
#     SD = 1/(np.log(2)*fs)*(ME4 - ME2**2)**0.5 # standard deviation
#     CV = SD/EK # coeff of variation of the power spectral density
#     SF = ME3/ME2**(3/2) # wavelet skewness factor
#     FF = ME4/ME2**(4/2) # wavelet flatness factor
#     wavestats = {'Scale': Rm, 'Frequency':Km, 'Spectrum':EK, 'Stdv':SD, 'CV': CV, 'Skewness':SF, 'Kurtosis':FF}
#     return wavestats


def emp_pdf(sample, nbins=10, log = False):
    '''
    -------------------------------------------------
    compute the empirical frequency density of a sample
    INPUT:
           sample
           nbins (default = 10)
    OUTPUT:
            binc, dens
    -------------------------------------------------
    '''
    # add: make sure it is a 1-D numpy array
    if log:
        print('warning - check')
        # # sbagliato mi sa!!!
        # # print('using log bins')
        # min = np.min(sample)
        # max = np.max(sample)
        # bins = np.logspace(np.log10(min), np.log10(max), nbins+1)
        # (counts, edges) = np.histogram(sample, bins)
    else:
        (counts, edges) = np.histogram(sample, nbins)
    dx = (edges[2]-edges[1])
    dens = counts/dx/np.sum(counts)
    binc = edges[:-1]+dx/2
    return binc, dens


def plot_cdf(df):
    men = (df['CH4']-df['CH4'].mean())/np.std(df['CH4'])
    wan = (df['H2O']-df['H2O'].mean())/np.std(df['H2O'])

    me_binc, me_pdf = emp_pdf(men, nbins = 30)
    wa_binc, wa_pdf = emp_pdf(wan, nbins = 30)

    plt.figure()
    plt.plot(me_binc, me_pdf, 'g')
    plt.plot(wa_binc, wa_pdf, 'b')
    plt.show()



def wave_filter(x, wavelet = 'haar', maxsize = 2**14, minsize = 2**1):
    """
    Filter a time series by applying a orthogonal wavelet transform
    and removing all scales with more than maxsize coeffs, or less than minsize coeffs
    """
    WT = pywt.wavedec(x, wavelet=wavelet)
    sizes = [len(x) for x in WT]
    WTf = []

    for ii, si in enumerate(sizes):
        if (si <= minsize or si >= maxsize) and ii > 0 :
            WTf.append(np.zeros(si))
        else:
            WTf.append(WT[ii])
    # keep the coarse grained signal = average
    xf = pywt.waverec(WTf, wavelet=wavelet)
    return xf


def filter_large_scales(rp, Tc, Ubar=1, z_ref=2.8, fs = 10, wavelet='haar',
                        plot=False, minscale = 0.5,
                        maxscale = 50, minsize = 4, useminsize = True):
    # filter time series around its integral time scale Tc
    # using discrete wavelets
    WTwa = pywt.wavedec(rp,  wavelet=wavelet)

    ############################################################################
    M = np.rint(np.log2(np.size(rp)))
    mm0 = np.arange(len(WTwa)-1, 0, -1) # skip coarse grained signal
    # M = np.log2(N)
    mm = np.hstack((M, mm0)) # add scale of coarse grained signal
    scales = 2**mm
    sizes = 2**(M-mm)

    # sizes2 = np.array([len(x) for x in WTc])[1:] # remove coarse grained signal
    # scales2 = N/sizes2

    ti = scales/fs
    tin = ti/Tc


    # keep only scales of interest
    # cond_keep = np.logical_and(tin >= minscale, sizes >= minsize)
    if useminsize:
        cond_keep = np.logical_and(tin >= minscale, sizes >= minsize)
    else:
        cond_keep = np.logical_and(tin >= minscale, tin <= maxscale)

    onsizes = sizes[cond_keep]
    # ontins = tin[cond_keep]
    maxsize2 = np.max(onsizes) + 1E-4
    # maxsize2 = np.max(onsizes)
    minsize2 = np.min(onsizes) - 1E-4
    # minsize2 = np.min(onsizes)

    # alternatively::
    # cond_keep = tin >= minscale
    # onsizes = sizes[cond_keep]
    #
    # # keep 3 total scales like this
    # maxsize2 = onsizes[-1] + 1E-4
    # minsize2 = onsizes[-5] - 1E-4
    # print('Hello World!')
    ############################################################################

    # ii = np.arange(len(WTwa)-1) # skip coarse grained signal
    # # ii = np.arange(len(WTwa))
    # si = np.array([len(x) for x in WTwa[1:]]) # = 2**ii
    # # li = 2**(14-ii)*tdf['Ubar'].iloc[i]/md.fs
    # li = 2**(14-ii)*Ubar/fs
    # ti = 2**(14-ii)/fs
    # is_large_scale = li > z_ref

    # Tc = mf.integral_scale_fft(rp) / fs  # in time units [s]
    # is_large_scale_2 = ti > 0.5*Tc # from that order of magnitude
    # on_sizes = si[is_large_scale_2]
    # maxsize = on_sizes[-1]
    # minsize = on_sizes[-5] # to 3 order of magn larger
    # minsize = 4
    # print('maxsize = ', maxsize2)
    # print('minsize = ', minsize2)
    # print(maxsize)
    # print(minsize)
    # print(on_sizes)
    # maxsize = 2048
    # maxsize = 2**11
    # maxsize = 1024
    # maxsize = 512
    # minsize = 124
    # minsize = 64
    wa_fil = wave_filter(rp, wavelet=wavelet,
                         maxsize=maxsize2, minsize=minsize2)
    # wa_fil = wave_filter(rp, wavelet=wavelet, maxsize=128, minsize=4)

    if plot:
        plt.figure()
        plt.plot(rp)
        plt.plot(wa_fil)
        plt.show()
    return wa_fil
