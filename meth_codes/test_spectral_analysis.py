# test the functions in the spectral_analysis module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spectral_analysis as sa
from scipy import signal
# from fbm import FBM
from stochastic.continuous import FractionalBrownianMotion
from stochastic.noise import FractionalGaussianNoise


# import methdata as md


# 1) test with synthetic data


leng = 33000
fbm = FractionalBrownianMotion(hurst=0.55, t=leng)
s = fbm.sample(leng)
times = fbm.times(leng)

plt.plot(times, s)
# plt.show()
plt.close()

fs = 100
x = s
Ubar = 1


# Compare Fourier and Wavelet Spectra:

ps, freq, ks = sa.fourier_power_spectrum(x, fs, density=True, Ubar = 1)

Rm, Km, fm, wsd = sa.dwt_power_spectrum(x, fs, wavelet = 'haar', density = True, Ubar = 1)

f, Pxx = signal.welch(x, fs, nperseg=2**14/16, scaling = 'density') # scaling density or spectrum
f_den, Pxx_den = signal.periodogram(x, fs, scaling='density')
# f_den, Pxx_den = signal.periodogram(x, fs, scaling='spectrum')

plt.figure()
plt.plot(freq, ps, label = 'fou')
plt.plot(f_den, Pxx_den, '--r', label = 'period')
# plt.plot(f, (2*np.pi)*Pxx, label = 'welch')
plt.plot(f, Pxx, label = 'welch')
# plt.plot(fm, wsd*(2*np.pi), label = 'wave')
plt.plot(fm, wsd, label = 'wave')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
# plt.close()



# load real data
df = pd.read_csv('../methane_data/Siikaneva_2013_cleaned/20130601_0000.csv')

N = 2**14
df = df[:N]

x = df['w'] # horizontal velocity
Ubar = np.mean(df['u']) #  mean velocity [m/s]
fs = 0.1 # Hertz [Hz]
dt = 1/fs # seconds [s]

# plt.figure()
# plt.plot(x)

# ps, freq, Kmf = sa.fourier_power_spectrum(x, fs, density=True, Ubar = Ubar)
ps, freq, ks, = sa.fourier_power_spectrum(x, fs, density=True, Ubar = Ubar)
Rm, Km, fm, wsd = sa.dwt_power_spectrum(x, fs, wavelet = 'haar', density = True, Ubar = Ubar)
f_den, Pxx_den = signal.periodogram(x, fs, scaling='density')
f, Pxx = signal.welch(x, fs, nperseg=2**14/16, scaling = 'density') # scaling density or spectrum
plt.figure()
# plt.plot(2*np.pi*freq/Ubar, ps*Ubar, label = 'Fourier')
plt.plot(freq/Ubar, ps, label = 'Fourier')
# plt.plot(f, Pxx*Ubar, label = 'Welch')
# plt.plot(f*2*np.pi/Ubar, Pxx*Ubar, label = 'Welch')
# plt.plot(f*2*np.pi/Ubar, Pxx*Ubar/2/np.pi, label = 'Welch')
plt.plot(f_den/Ubar, Pxx_den*Ubar, label = 'Period')
plt.plot(f/Ubar, Pxx*Ubar, label = 'Welch')
plt.plot(fm/Ubar, wsd, label = 'Wavelet')
plt.ylabel('Power spectral density [unit of x]^2 * [m]')
plt.xlabel('Wavenumber [m^-1]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.plot()
# plt.close()
#

x = df['T'].values
y = df['w'].values
# test cross spectra densities e.g. Heatflux
Rm, Km, fm, wsd = sa.dwt_csd( x, y, fs, wavelet='haar', mode='periodic', density=True, Ubar = Ubar)
ps, freq, ks = sa.fourier_csd(x, y, fs, density=True, Ubar = Ubar)

fsc, csdsc =  signal.csd(x, y, fs=fs, nperseg=2**14/1, scaling='density')

plt.figure()
plt.plot(freq, np.abs(ps), label = 'Fourier')
plt.plot(fsc, np.abs(csdsc)*Ubar, '--',label = 'Scipy csd')
plt.plot(fm, np.abs(wsd), label = 'Wavelet')
plt.ylabel('Cross power spectral density [unit of x]^2 * [m]')
plt.xlabel('Wavenumber [m^-1]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# test Fourier spectral coherence::
# Note: use the scipy version is better than mine.
# Probably we need to smooth the signal with hanning window
# averaging among different blocks is not enough
cohe, freq, ks = sa.fourier_coherence(x, y, fs, Ubar = Ubar, nblocks = 4)
# fsc, cohesc =  signal.coherence(x, y, fs=fs, nperseg=2**14/1)
fsc, cohesc =  signal.coherence(x, y, fs=fs)
Rm, Km, fm, wcohe = sa.dwt_cohe( x, y, fs, wavelet='haar', mode='periodic', Ubar = Ubar)

plt.figure()
plt.plot(freq, np.abs(cohe), label = 'Fourier')
plt.plot(fm, np.abs(wcohe), label = 'Wavelet')
plt.plot(fsc, np.abs(cohesc), '--',label = 'Scipy csd')
plt.ylabel('Signal coherence')
plt.xlabel('Wavenumber [m^-1]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()



# plt.figure()
# # plt.plot(np.sqrt(psx0*psy0), 'o')
# # plt.plot(cs[1:int(n/2)], 'o')
# # plt.plot(freq, cohe)
# plt.plot(freq, np.abs(apsc))
# # plt.plot(np.abs(cohe))
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# remains to be checked only dwt_statistics
