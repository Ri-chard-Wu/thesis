import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

 
import h5py 
from os import listdir
from os.path import isfile, join
import scipy as sp




def cycles2us(x):
    return x*2.325e-3



def traverseHDF(f):

    for i in f.keys(): # DatasetType
        if type(f[i]) == DatasetType:
            #print('Dataset ',f[i].name,': ',f[i])
            print('Dataset ',f[i].name,': ',f[i][:],'\n')
        else: # GroupType
            for j in f[i].keys():
                if type(f[i][j]) == DatasetType:
                    print('Dataset ',f[i][j].name,': ',f[i][j][:],'\n')
                else:
                    traverseHDF(f[i][j])


plt.figure()



# ################## Fit Onetone #######################
 


def hangerfunc(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

def hangerS21func(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return a0 + np.abs(hangerfunc(x, *p)) - scale*(1-Q0/Qe)

def hangerS21func_sloped(x, *p):
    f0, Qi, Qe, phi, scale, a0, slope = p
    return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope*(x-f0)

def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))

def fithanger(xdata, ydata):
    
    fitparams = [None]*7

    fitparams[0]=np.average(xdata)
    fitparams[1]=5000
    fitparams[2]=1000
    fitparams[3]=0
    fitparams[4]=max(ydata)-min(ydata)
    fitparams[5]=np.average(ydata)
    fitparams[6]=(ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])


    bounds = (
        [np.min(xdata), -np.inf, -np.inf, -np.inf, 0, min(ydata), -np.inf],
        [np.max(xdata), np.inf, np.inf, np.inf, np.inf, max(ydata), np.inf],
        )
    
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=fitparams)
    except RuntimeError: 
        pOpt, pCov = None, None

    return pOpt, pCov
 

file_name = './q1/r1_1.hdf5' 
file = h5py.File(file_name,'r')


# DatasetType = type(file['Channels'])
# GroupType = type(file['Data'])
# traverseHDF(file) 

fpts = file['/Data/Data'][:,0,0]/1e6
I = file['/Data/Data'][:,1,0]
Q = file['/Data/Data'][:,2,0]

# plt.plot(fpts, abs(I + 1j*Q))
# plt.show()
 

l, u = 0, 0 # for 012

amps = abs(I + 1j*Q)#[l:-u] 

para, cov = fithanger(fpts, amps)
paraName = ['f0', 'Qi', 'Qe', 'phi', 'scale', 'a0', 'slope']

Qtot = 1/(1/para[1] + 1/para[2])
f0 = para[0]
df = f0 / Qtot


if (para is None): print('fit failed.')

mse = np.mean(np.sqrt(np.abs(np.diag(cov) / para)))

plt.subplot(231, xlabel='Frequency (MHz)', ylabel='ADC level (a.u.)', title='OneTone')
plt.plot(fpts, amps, marker='o', label='data')
plt.plot(fpts, hangerS21func_sloped(fpts, *para), label='fit')

plt.legend()
# plt.show()
# plt.title('OneTone')
# plt.xlabel('Hz')
# plt.ylabel('ADC level (a.u.)')
# plt.savefig('./fitting_results/OneTone.png')




# # ################## Fit Twotone #######################



file_name = './q1/q1_twotone_5.hdf5' 
file = h5py.File(file_name,'r')
 

u = 10
l = 80

fpts = file['/Data/Data'][:,0,0][l:-u]
I = file['/Data/Data'][:,1,0][l:-u]
Q = file['/Data/Data'][:,2,0][l:-u]


# plt.plot(fpts, abs(I + 1j*Q))
# plt.show()
  

def func(x, a, x0, gamma, offset):
    return  - a * (gamma/2*np.pi) / ( (x - x0)**2 + (0.5 * gamma)**2 ) + offset
 


x = fpts/1e6
amp = abs(I + 1j*Q)
 

# plt.show()


guess = ([amp.max() - amp.min(), x[amp.argmin()], 2, amp[-1]])

opt, cov = curve_fit(func, x, amp, p0=guess, maxfev=1000000)





plt.subplot(234, xlabel='Frequency (MHz)', ylabel='ADC level (a.u.)', title='TwoTone') 
plt.plot(x, amp, marker='o', label='data')

plt.plot(x, func(x, *opt), label='fit')
plt.legend()
 
# plt.show()
# plt.savefig('./fitting_results/TwoTone.png')



################## Fit T2 #######################

def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0 = p
    return yscale * np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-x/decay) + y0

def fitdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*5
    else: fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=(max(ydata)-min(ydata))/2
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    bounds = (
        [0.75*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.3*(max(xdata)-min(xdata)), np.min(ydata)],
        [1.25*fitparams[0], 30/(max(xdata)-min(xdata)), 360, np.inf, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov


file_name = './q1/q1_T2Ramsey_3.hdf5' 
file = h5py.File(file_name,'r')
 

u = 1
l = 1

fpts = file['/Data/Data'][:,0,0][l:-u]
I = file['/Data/Data'][:,1,0][l:-u]
Q = file['/Data/Data'][:,2,0][l:-u]


# plt.plot(fpts, abs(I + 1j*Q))
# plt.show()


 
x = fpts 
amp = abs(I + 1j*Q)

 
 

pOpt,_= fitdecaysin(x, amp)


plt.subplot(235, xlabel='Time (us)', ylabel='ADC level (a.u.)', title='T2')
plt.plot(x, amp, marker='o', label='data')
plt.plot(x, decaysin(x, *pOpt), label='fit')

plt.legend()
# plt.show()
# plt.savefig('./fitting_results/T2.png')


# ################# Fit T1 #######################





file_name = './q1/q1_T1_2.hdf5' 
file = h5py.File(file_name,'r')
 

u = 1
l = 1

x_pts = file['/Data/Data'][:,0,0][l:-u]
I = file['/Data/Data'][:,1,0][l:-u]
Q = file['/Data/Data'][:,2,0][l:-u]

amp = abs(I + 1j*Q)


# plt.plot(x_pts, abs(I + 1j*Q))
# plt.show()

def expfunc(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale*np.exp(-(x-x0)/decay)

def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    else: fitparams = np.copy(fitparams)
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0]-ydata[-1]
    if fitparams[2] is None: fitparams[2] = xdata[0]
    if fitparams[3] is None: fitparams[3] = (xdata[-1]-xdata[0])/5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov


pOpt, _= fitexp(x_pts, amp)
 
 

plt.subplot(232, xlabel='Time (us)', ylabel='ADC level (a.u.)', title='T1') 
plt.plot(x_pts, amp, marker='o', label='data')
plt.plot(x_pts, expfunc(x_pts, *pOpt), label='fit')
  
plt.legend() 
# plt.show()
# plt.savefig('./fitting_results/T1.png')

 
# ################## Fit Rabi #######################



def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) + y0

def fitsin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    else: fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    max_ind = np.argmax(np.abs(fourier[1:])) + 1
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=np.mean(ydata)
    bounds = (
        [0.5*fitparams[0], 0.2/(max(xdata)-min(xdata)), -360, np.min(ydata)],
        [2*fitparams[0], 5/(max(xdata)-min(xdata)), 360, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov



file_name = './q1/q1_rabi.hdf5' 
file = h5py.File(file_name,'r')
 

u = 1
l = 1

x_pts = file['/Data/Data'][:,0,0][l:-u]
I = file['/Data/Data'][:,1,0][l:-u]
Q = file['/Data/Data'][:,2,0][l:-u]

x_pts = cycles2us(x_pts)
amp = abs(I + 1j*Q)


# plt.plot(x_pts, amp)
# plt.show() 


pOpt, _= fitsin(x_pts, amp)



plt.subplot(233, xlabel='Time (us)', ylabel='ADC level (a.u.)', title='Rabi') 

plt.plot(x_pts, amp, marker='o', label='data')
plt.plot(x_pts, sinfunc(x_pts, *pOpt), label=f'fit')
plt.legend()
 
# plt.savefig('./fitting_results/Rabi.png')
plt.tight_layout()

plt.show()

 