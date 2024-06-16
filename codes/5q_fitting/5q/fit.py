import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


 
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


 



 

# figsize = (11,3)

# # # # ################## Fit Onetone - david shuster method #######################
 


# def hangerfunc(x, *p):
#     f0, Qi, Qe, phi, scale, a0 = p
#     Q0 = 1 / (1/Qi + np.real(1/Qe))
#     return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

# def hangerS21func(x, *p):
#     f0, Qi, Qe, phi, scale, a0 = p
#     Q0 = 1 / (1/Qi + np.real(1/Qe))
#     return a0 + np.abs(hangerfunc(x, *p)) - scale*(1-Q0/Qe)

# def hangerS21func_sloped(x, *p):
#     f0, Qi, Qe, phi, scale, a0, slope = p
#     return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope*(x-f0)

# def hangerphasefunc(x, *p):
#     return np.angle(hangerfunc(x, *p))

# def fithanger(xdata, ydata):
    
#     fitparams = [None]*7

#     # paraName = ['f0', 'Qi', 'Qe', 'phi', 'scale', 'a0', 'slope']
#     fitparams[0]=np.average(xdata)
#     fitparams[1]=30000 # TODO: tune this if fit failed.
#     fitparams[2]=5000
#     fitparams[3]=0
#     fitparams[4]=max(ydata)-min(ydata)
#     fitparams[5]=np.average(ydata)
#     fitparams[6]=(ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])


#     bounds = (
#         [np.min(xdata), -np.inf, -np.inf, -np.inf, 0, min(ydata), -np.inf],
#         [np.max(xdata), np.inf, np.inf, np.inf, np.inf, max(ydata), np.inf],
#         )
    
#     for i, param in enumerate(fitparams):
#         if not (bounds[0][i] < param < bounds[1][i]):
#             fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))

#     pOpt = fitparams
#     pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
#     try:
#         pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=fitparams)
#     except RuntimeError: 
#         pOpt, pCov = None, None

#     return pOpt, pCov
 


# amps3 = []
# fpts3 = []

# for i in range(3):
#     file_name = f'./5q_muxed_6_15/muxed_onetone2-q{i}.hdf5' 
#     file = h5py.File(file_name,'r')
 
    
#     fpts = file['/Data/Data'][:,0,0]/1e6
#     I = file['/Data/Data'][:,1,0]
#     Q = file['/Data/Data'][:,2,0]

#     amps = abs(I + 1j*Q)

#     fpts3.append(fpts)
#     amps3.append(amps)

#     # plt.plot(fpts, amps, label=f'q{i}')

# # plt.legend()
# # plt.show() 
# # exit()

 
# def fitOneTone(ax, fpts, amps):
 

#     para, cov = fithanger(fpts, amps)
#     paraName = ['f0', 'Qi', 'Qe', 'phi', 'scale', 'a0', 'slope']
 
#     if (para is None): 
#         print('fit failed.')
#         return

#     Qtot = 1/(1/para[1] + 1/para[2])
#     f0 = para[0]
#     df = f0 / Qtot
#     mse = np.mean(np.sqrt(np.abs(np.diag(cov) / para)))
    
#     ax.plot(fpts, amps, marker='o', label=f'data')
#     # ax.plot(fpts, hangerS21func_sloped(fpts, *para), label=f'fit, fr: {f0:.2f}, Qi: {para[1]:.1f}, Qc: {para[2]:.1f}')
#     ax.plot(fpts, hangerS21func_sloped(fpts, *para), label=f'fit, fr: {f0:.2f} MHz')

 

# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
# for i in range(3): 
#     fitOneTone(axs[i], fpts3[i], amps3[i])
#     axs[i].set_xlabel('Frequency (MHz)')
#     axs[i].set_ylabel('ADC amplitude (a.u.)') 
#     axs[i].set_title(f'q{i+1}') 
#     axs[i].legend()   
# plt.tight_layout()    
 
 
# # plt.show()
# plt.savefig('./fitting_results/characterization/OneTone.png')
# # exit()


# # # # ################## Fit Twotone #######################



# amps3 = []
# fpts3 = []

# for i in range(3):
#     file_name = f'./5q_muxed_6_15/muxed_twotone2-q{i}.hdf5' 
#     file = h5py.File(file_name,'r')
 
    
#     fpts = file['/Data/Data'][:,0,0]/1e6
#     I = file['/Data/Data'][:,1,0]
#     Q = file['/Data/Data'][:,2,0]

#     amps = abs(I + 1j*Q)

#     fpts3.append(fpts)
#     amps3.append(amps)

#     # plt.plot(fpts, amps, label=f'q{i}')

# # plt.legend()
# # plt.show() 
# # exit()


 
# def fitTwoTone(ax, fpts, amps):

#     def func(x, a, x0, gamma, offset):
#         return  - a * (gamma/2*np.pi) / ( (x - x0)**2 + (0.5 * gamma)**2 ) + offset
     
#     guess = ([amps.max() - amps.min(), fpts[amps.argmin()], 2, amps[-1]])

#     opt, cov = curve_fit(func, fpts, amps, p0=guess, maxfev=1000000)
   
#     ax.plot(fpts, amps, marker='o', label=f'data')
#     ax.plot(fpts, func(fpts, *opt), label=f'fit, f01: {opt[1]:.2f} MHz')
    
 

# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
# for i in range(3): 
#     fitTwoTone(axs[i], fpts3[i], amps3[i])
#     axs[i].set_xlabel('Frequency (MHz)')
#     axs[i].set_ylabel('ADC amplitude (a.u.)') 
#     axs[i].set_title(f'q{i+1}') 
#     axs[i].legend()   
# plt.tight_layout() 
 
# # plt.show()
# plt.savefig('./fitting_results/characterization/TwoTone.png')
# # exit()

 
# # ################## Fit T2 #######################
 


# amps3 = []
# dpts3 = []

# for i in range(3):
#     file_name = f'./5q_muxed_6_15/muxed_t2r_2.hdf5' 
#     file = h5py.File(file_name,'r')
    
#     # print(file['/Data/Data'].shape)
#     # exit()
    
#     dpts = file['/Data/Data'][:,0,i] * 1e-3 # us.
#     I = file['/Data/Data'][:,2,i]
#     Q = file['/Data/Data'][:,3,i]

#     amps = abs(I + 1j*Q)

#     dpts3.append(dpts)
#     amps3.append(amps)

#     # plt.plot(dpts, amps, label=f'q{i+1}')

# # plt.legend()
# # plt.show() 
# # exit()
 


# def fit_t2(ax, dpts, amps, qidx):
    
#     import scipy as sp
#     def decaysin(x, *p):
#         yscale, freq, phase_deg, decay, y0 = p
#         return yscale * np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-x/decay) + y0

#     def fitdecaysin(xdata, ydata, fitparams=None):
#         if fitparams is None: fitparams = [None]*5
#         else: fitparams = np.copy(fitparams)
#         fourier = np.fft.fft(ydata)
#         fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
#         fft_phases = np.angle(fourier)
#         sorted_fourier = np.sort(fourier)
#         max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
#         if max_ind == 0:
#             max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
#         max_freq = np.abs(fft_freqs[max_ind])
#         max_phase = fft_phases[max_ind]
#         if fitparams[0] is None: fitparams[0]=(max(ydata)-min(ydata))/2
#         if fitparams[1] is None: fitparams[1]=max_freq
#         # if fitparams[2] is None: fitparams[2]=0
#         if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
#         if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
#         if fitparams[4] is None: fitparams[4]=np.mean(ydata)
#         bounds = (
#             [0.75*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.3*(max(xdata)-min(xdata)), np.min(ydata)],
#             [1.25*fitparams[0], 10/(max(xdata)-min(xdata)), 360, np.inf, np.max(ydata)]
#             )
#         for i, param in enumerate(fitparams):
#             if not (bounds[0][i] < param < bounds[1][i]):
#                 fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
#                 print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
#         pOpt = fitparams
#         pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
#         try:
#             pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
#             # return pOpt, pCov
#         except RuntimeError: 
#             print('Warning: fit failed!')
#             return None, None
#         return pOpt, pCov

     

    
#     ax.plot(dpts, amps, marker='o', label='data')

    
#     if(qidx==1):return
#     pOpt, _= fitdecaysin(dpts, amps)
#     amps_fit = decaysin(dpts, *pOpt)    
#     ax.plot(dpts, amps_fit, label=f'fit, T2R: {pOpt[3]:.2f} us')
     
#     # detune = pOpt[1] # MHz. 


# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
# for i in range(3): 
    
#     fit_t2(axs[i], dpts3[i], amps3[i], i)
    
#     axs[i].set_xlabel('delay (us)')
#     axs[i].set_ylabel('ADC amplitude (a.u.)') 
#     axs[i].set_title(f'q{i+1}')   
#     axs[i].legend()        

# plt.tight_layout() 
# # plt.show()
# plt.savefig('./fitting_results/characterization/T2R.png')
# # exit()
 
# # # ################# Fit T1 #######################

 

# amps3 = []
# dpts3 = []

# for i in range(3):
#     file_name = f'./5q_muxed_6_15/muxed_T1_2.hdf5' 
#     file = h5py.File(file_name,'r')
 
    
#     dpts = file['/Data/Data'][:,0,i] * 1e-3 # us.
#     I = file['/Data/Data'][:,2,i]
#     Q = file['/Data/Data'][:,3,i]

#     amps = abs(I + 1j*Q)

#     dpts3.append(dpts)
#     amps3.append(amps)

# #     plt.plot(dpts, amps, label=f'q{i+1}')

# # plt.legend()
# # plt.show() 
# # exit()

 

# def fitT1(ax, dpts, amps):

#     def expfunc(x, *p):
#         y0, yscale, x0, decay = p
#         return y0 + yscale*np.exp(-(x-x0)/decay)

#     def fitexp(xdata, ydata, fitparams=None):
#         if fitparams is None: fitparams = [None]*4
#         else: fitparams = np.copy(fitparams)
#         if fitparams[0] is None: fitparams[0] = ydata[-1]
#         if fitparams[1] is None: fitparams[1] = ydata[0]-ydata[-1]
#         if fitparams[2] is None: fitparams[2] = xdata[0]
#         if fitparams[3] is None: fitparams[3] = (xdata[-1]-xdata[0])/5
#         pOpt = fitparams
#         pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
#         try:
#             pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
#             # return pOpt, pCov
#         except RuntimeError: 
#             print('Warning: fit failed!')
#             # return 0, 0
#         return pOpt, pCov


#     ax.plot(dpts, amps, marker='o', label='data')
 
#     pOpt, _= fitexp(dpts, amps)
#     amps_fit = expfunc(dpts, *pOpt)    
#     ax.plot(dpts, amps_fit, label=f'fit, T1: {pOpt[3]:.2f} us')
 

# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
# for i in range(3): 
    
#     fitT1(axs[i], dpts3[i], amps3[i])
    
#     axs[i].set_xlabel('delay (us)')
#     axs[i].set_ylabel('ADC amplitude (a.u.)') 
#     axs[i].set_title(f'q{i+1}')   
#     axs[i].legend()        

# plt.tight_layout() 
# # plt.show()
# plt.savefig('./fitting_results/characterization/T1.png')
# # exit() 

 
 
# # # ################## Fit Rabi #######################

 
 
# amps3 = []
# lpts3 = []

# for i in range(3):
#     file_name = f'./5q_muxed_6_15/muxed_Length_Rabi_3.hdf5' 
#     file = h5py.File(file_name,'r')
  
#     lpts = file['/Data/Data'][:,0,i] * 1e-3 # us.
#     I = file['/Data/Data'][:,2,i]
#     Q = file['/Data/Data'][:,3,i]

#     amps = abs(I + 1j*Q)

#     lpts3.append(lpts)
#     amps3.append(amps)

# #     plt.plot(lpts, amps, label=f'q{i+1}')
# # plt.legend()
# # plt.show() 
# # exit()


  


 

# def fit_length_rabi(ax, lpts, amps):
    
 
#     def decaysin(x, *p):
#         yscale, freq, phase_deg, decay, y0 = p
#         return yscale * np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-x/decay) + y0

#     def fitdecaysin(xdata, ydata, fitparams=None):
#         if fitparams is None: fitparams = [None]*5
#         else: fitparams = np.copy(fitparams)
#         fourier = np.fft.fft(ydata)
#         fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
#         fft_phases = np.angle(fourier)
#         sorted_fourier = np.sort(fourier)
#         max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
#         if max_ind == 0:
#             max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
#         max_freq = np.abs(fft_freqs[max_ind])
#         max_phase = fft_phases[max_ind]
#         if fitparams[0] is None: fitparams[0]=(max(ydata)-min(ydata))/2
#         if fitparams[1] is None: fitparams[1]=max_freq
#         # if fitparams[2] is None: fitparams[2]=0
#         if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
#         if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
#         if fitparams[4] is None: fitparams[4]=np.mean(ydata)
#         bounds = (
#             [0.75*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.3*(max(xdata)-min(xdata)), np.min(ydata)],
#             [1.25*fitparams[0], 30/(max(xdata)-min(xdata)), 360, np.inf, np.max(ydata)]
#             )
#         for i, param in enumerate(fitparams):
#             if not (bounds[0][i] < param < bounds[1][i]):
#                 fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
#                 print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
#         pOpt = fitparams
#         pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
#         try:
# #             pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
#             pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams)
            
#             # return pOpt, pCov
#         except RuntimeError: 
#             print('Warning: fit failed!')
#             # return 0, 0
#         return pOpt, pCov
    
#     def sinfunc(x, *p):
#         yscale, freq, phase_deg, y0 = p
#         return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) + y0

#     def fitsin(xdata, ydata, fitparams=None):
#         if fitparams is None: fitparams = [None]*4
#         else: fitparams = np.copy(fitparams)
#         fourier = np.fft.fft(ydata)
#         fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
#         fft_phases = np.angle(fourier)
#         max_ind = np.argmax(np.abs(fourier[1:])) + 1
#         max_freq = np.abs(fft_freqs[max_ind])
#         max_phase = fft_phases[max_ind]
#         if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
#         if fitparams[1] is None: fitparams[1]=max_freq
#         if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
#         if fitparams[3] is None: fitparams[3]=np.mean(ydata)
#         bounds = (
#             [0.5*fitparams[0], 0.2/(max(xdata)-min(xdata)), -360, np.min(ydata)],
#             [2*fitparams[0], 5/(max(xdata)-min(xdata)), 360, np.max(ydata)]
#             )
#         for i, param in enumerate(fitparams):
#             if not (bounds[0][i] < param < bounds[1][i]):
#                 fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
#                 print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
#         pOpt = fitparams
#         pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
#         try:
            
# #             pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
#             pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams)
#             # return pOpt, pCov
#         except RuntimeError: 
#             print('Warning: fit failed!')
#             # return 0, 0
#         return pOpt, pCov

 
#     pOpt, _= fitdecaysin(lpts, amps) 
    
#     amps_fit = decaysin(lpts, *pOpt)
    

#     fit_skip_start = 10
    
#     pi_len = lpts[fit_skip_start:][np.argmin(amps_fit[fit_skip_start:])] # us.
    
    
#     ax.plot(lpts, amps, marker='o', label='data')
    
#     rabi_freq = pOpt[1] # MHz.
#     rabi_period = 1 / rabi_freq # us.
#     pi2_len = pi_len - rabi_period/4 # us.
    
#     ax.plot(lpts, amps_fit, label=f'fit, Rabi freq: {rabi_freq :.2f} MHz')
  
#     ax.axvline(x=pi_len,  color='b', label=f'pi len: {pi_len*1e3:.0f} ns')
#     ax.axvline(x=pi2_len, color='r', label=f'pi/2 len: {pi2_len*1e3:.0f} ns')
    
  

# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
# for i in range(3): 
    
#     fit_length_rabi(axs[i], lpts3[i], amps3[i])
    
#     axs[i].set_xlabel('Pulse length (us)')
#     axs[i].set_ylabel('ADC amplitude (a.u.)') 
#     axs[i].set_title(f'q{i+1}')   
#     axs[i].legend()        

# plt.tight_layout()    
# # plt.show()
# plt.savefig('./fitting_results/characterization/LenRabi.png')
 

################## Crosstalk #######################




 
for q in range(3):
    for f in range(3):

        amps3 = []
        lpts3 = []

        for i in range(3):

            file_name = f'./5q_muxed_6_15/muxed_crosstalk_lenRabi_q{q}-f{f}.hdf5' 
            file = h5py.File(file_name,'r')
        
            lpts = file['/Data/Data'][:,0,i] * 1e-3 # us.
            I = file['/Data/Data'][:,2,i]
            Q = file['/Data/Data'][:,3,i]

            amps = abs(I + 1j*Q)

            lpts3.append(lpts)
            amps3.append(amps)

        #     plt.plot(lpts, amps, label=f'q{q+1}-f{f+1}-meas{i+1}')
            
        # plt.legend()
        # plt.show() 

 



        def fit_length_rabi(ax, lpts, amps):
            
        
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
         
                    pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams)
                     
                except RuntimeError: 
                    print('Warning: fit failed!')
                    return None, None
                  
                return pOpt, pCov
             

            ax.plot(lpts, amps, marker='o', label='data')


            pOpt, pCov= fitdecaysin(lpts, amps) 

            

            if(pOpt is None): return

            mse = np.mean(np.sqrt(np.abs(np.diag(pCov) / pOpt)))
            
            if(mse > 0.4): return

            amps_fit = decaysin(lpts, *pOpt)  
            fit_skip_start = 10 
            pi_len = lpts[fit_skip_start:][np.argmin(amps_fit[fit_skip_start:])] # us.
             
            rabi_freq = pOpt[1] # MHz.
            rabi_period = 1 / rabi_freq # us.
            pi2_len = pi_len - rabi_period/4 # us.
            
            ax.plot(lpts, amps_fit, label=f'fit, Rabi freq: {rabi_freq :.2f} MHz')
        
            # ax.axvline(x=pi_len,  color='b', label=f'pi len: {pi_len*1e3:.0f} ns')
            # ax.axvline(x=pi2_len, color='r', label=f'pi/2 len: {pi2_len*1e3:.0f} ns')
            
        


        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
            
        for i in range(3): 
            
            fit_length_rabi(axs[i], lpts3[i], amps3[i])
            
            axs[i].set_xlabel('Pulse length (us)')
            axs[i].set_ylabel('ADC amplitude (a.u.)') 
            axs[i].set_title(f'drv q{q+1}, para q{f+1}, meas q{i+1}')   
            axs[i].legend()        

        plt.tight_layout()    
        # plt.show()
        plt.savefig(f'./fitting_results/crosstalk/crosstalk_q{q+1}-f{f+1}.png')
        