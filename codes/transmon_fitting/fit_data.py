import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd



# ################## Fit Onetone #######################

plt.figure()

def lorenztian(f, f0, kappa, A, B):
    t = A / ((f - f0) ** 2 + (kappa / 2) ** 2) + B
    return t


file_data = './transmon002_data/OneTone_SingleTrace_data.csv'
file_axisinfo = './transmon002_data/OneTone_SingleTrace_axisInfo.csv'


df_data = pd.read_csv(file_data, header=None)

a = df_data.values
b = a.reshape(a.shape[0])
c = [complex(i[1:-1]) for i in b] if type(b[0]) is type('') else b
data_i = np.array([i.real for i in c])
data_q = np.array([i.imag for i in c])
n_points = len(data_i)



df_axisinfo = pd.read_csv(file_axisinfo)
unit = df_axisinfo.values[0][1]
start = float(df_axisinfo.values[1][1])
stop = float(df_axisinfo.values[2][1])
x = np.linspace(start, stop, n_points)
ComplexTrunc = np.array([i + 1j*q for i, q in zip(data_i, data_q) ])
amp = np.array([np.abs(i + 1j*q) for i, q in zip(data_i, data_q) ])

divisor = 1
x = x[:int(x.shape[0]/divisor)]
amp = amp[:int(amp.shape[0]/divisor)]
ComplexTrunc = ComplexTrunc[:int(ComplexTrunc.shape[0]/divisor)]

drop = 20
x = x[drop:int(x.shape[0]/divisor)-drop]
amp = amp[drop:int(amp.shape[0]/divisor)-drop]
ComplexTrunc = ComplexTrunc[drop:int(ComplexTrunc.shape[0]/divisor)-drop]
FreqTrunc = x
plt.plot(x, amp, marker='o', label='data')
 

AbsComplex = amp
MaxAbs = np.max(AbsComplex)
ComplexTrunc /= MaxAbs 
MaxAbs = np.max(AbsComplex)
MinAbs = np.min(AbsComplex)
MaxInd = AbsComplex.argmax()
f0_guess = FreqTrunc[MaxInd]
kappa_guess = (FreqTrunc[-1] - FreqTrunc[0]) / 4
B_guess = MinAbs
A_guess = (MaxAbs - MinAbs) * (kappa_guess / 2) ** 2


guess = ([f0_guess, kappa_guess, A_guess, B_guess]) 
qopt, qcov = curve_fit(lorenztian, FreqTrunc, AbsComplex, guess)
f0_fit, kappa_fit, A_fit, B_fit = qopt
kappa_fit = np.abs(kappa_fit)

plt.plot(x, lorenztian(x, *qopt), label='fit')
plt.legend()
plt.title('OneTone')
plt.xlabel('Hz')
plt.ylabel('ADC level (a.u.)')
# plt.show() 
plt.savefig('./fitting results/OneTone.png')




# ################## Fit Twotone #######################

plt.figure()
def func(x, a, x0, gamma, offset):
    return  - a * (gamma/2*np.pi) / ( (x - x0)**2 + (0.5 * gamma)**2 ) + offset

file_data = './transmon002_data/TwoTone_SingleTrace_data.csv'
file_axisinfo = './transmon002_data/TwoTone_SingleTrace_axisInfo.csv'


df_data = pd.read_csv(file_data, header=None)

a = df_data.values
b = a.reshape(a.shape[0])
c = [complex(i[1:-1]) for i in b] if type(b[0]) is type('') else b
data_i = np.array([i.real for i in c])
data_q = np.array([i.imag for i in c])
n_points = len(data_i)



df_axisinfo = pd.read_csv(file_axisinfo)
unit = df_axisinfo.values[0][1]
start = float(df_axisinfo.values[1][1])
stop = float(df_axisinfo.values[2][1])
x = np.linspace(start, stop, n_points)
amp = np.array([np.abs(i + 1j*q) for i, q in zip(data_i, data_q) ])

divisor = 1
x = x[:int(x.shape[0]/divisor)]
amp = amp[:int(amp.shape[0]/divisor)]

drop = 20
x = x[drop:int(x.shape[0]/divisor)-drop]
amp = amp[drop:int(amp.shape[0]/divisor)-drop]

plt.plot(x, amp, marker='o', label='data')

# plt.show()


guess = ([amp.max() - amp.min(), x[amp.argmin()], 2, amp[-1]])

opt, cov = curve_fit(func, x, amp, p0=guess, maxfev=1000000)

plt.plot(x, func(x, *opt), label='fit')
plt.legend()
plt.title('TwoTone')
plt.xlabel('MHz')
plt.ylabel('ADC level (a.u.)')
# plt.show()
plt.savefig('./fitting results/TwoTone.png')



################## Fit T2 #######################

plt.figure()
def func(x, a, f, c, d, t2):
    return a * np.cos(2 * np.pi * f * (x-c)) * np.exp(-(x-c) / t2) + d

file_data = './transmon002_data/T2_SingleTrace_data.csv'
file_axisinfo = './transmon002_data/T2_SingleTrace_axisInfo.csv'


df_data = pd.read_csv(file_data, header=None)

a = df_data.values
b = a.reshape(a.shape[0])
c = [complex(i[1:-1]) for i in b] if type(b[0]) is type('') else b
data_i = np.array([i.real for i in c])
data_q = np.array([i.imag for i in c])
n_points = len(data_i)



df_axisinfo = pd.read_csv(file_axisinfo)
unit = df_axisinfo.values[0][1]
start = float(df_axisinfo.values[1][1])
stop = float(df_axisinfo.values[2][1])
x = np.linspace(start, stop, n_points)
amp = np.array([np.abs(i + 1j*q) for i, q in zip(data_i, data_q) ])

divisor = 1.5
x = x[:int(x.shape[0]/divisor)]
amp = amp[:int(amp.shape[0]/divisor)]

plt.plot(x, amp, marker='o', label='data')

# plt.show()


rabi_guess = 1
decay_guess = 1

t2_guess = 1
guess = ([amp.max() - amp.min(),
         0.5/np.abs(x[amp.argmax()] - x[amp.argmin()]),
          x[0], 
          amp[-1], 
          t2_guess])

opt, cov = curve_fit(func, x, amp, p0=guess, maxfev=1000000)

plt.plot(x, func(x, *opt), label='fit')
plt.legend()
plt.title('T2')
plt.xlabel('us')
plt.ylabel('ADC level (a.u.)')
# plt.show()
plt.savefig('./fitting results/T2.png')


################# Fit T1 #######################
plt.figure()
def func(x, a, b, c, d):
    return a*np.exp(-(x-c)/b) + d

file_data = './transmon002_data/T1_SingleTrace_data.csv'
file_axisinfo = './transmon002_data/T1_SingleTrace_axisInfo.csv'


df_data = pd.read_csv(file_data, header=None)

a = df_data.values
b = a.reshape(a.shape[0])
c = [complex(i[1:-1]) for i in b] if type(b[0]) is type('') else b
data_i = np.array([i.real for i in c])
data_q = np.array([i.imag for i in c])
n_points = len(data_i)



df_axisinfo = pd.read_csv(file_axisinfo)
unit = df_axisinfo.values[0][1]
start = float(df_axisinfo.values[1][1])
stop = float(df_axisinfo.values[2][1])
x = np.linspace(start, stop, n_points)
amp = np.array([np.abs(i + 1j*q) for i, q in zip(data_i, data_q) ])
plt.plot(x, amp, marker='o', label='data')
# plt.show()

rabi_guess = 1
decay_guess = 1
guess = ([amp[0]-amp[-1], x[-1]/5, 0.0, amp[-1]])
opt, cov = curve_fit(func, x, amp, p0=guess, maxfev=1000000)


plt.plot(x, func(x, *opt), label='fit')
plt.legend()
plt.title('T1')
plt.xlabel('us')
plt.ylabel('ADC level (a.u.)')
# plt.show()
plt.savefig('./fitting results/T1.png')

 
################## Fit Rabi #######################


plt.figure()
def Rabi_func(x,a,b,c,d,t1):
    return a * np.cos(b*(x-c)) * np.exp(-(x-c)/t1) + d

file_data = './transmon002_data/Rabi_SingleTrace_data.csv'
file_axisinfo = './transmon002_data/Rabi_SingleTrace_axisInfo.csv'

df_data = pd.read_csv(file_data, header=None)

a = df_data.values
b = a.reshape(a.shape[0])
c = [complex(i[1:-1]) for i in b] if type(b[0]) is type('') else b
data_i = np.array([i.real for i in c])
data_q = np.array([i.imag for i in c])
n_points = len(data_i)



df_axisinfo = pd.read_csv(file_axisinfo)
unit = df_axisinfo.values[0][1]
start = float(df_axisinfo.values[1][1])
stop = float(df_axisinfo.values[2][1])
x = np.linspace(start, stop, n_points)
amp = np.array([np.abs(i + 1j*q) for i, q in zip(data_i, data_q) ])
plt.plot(x, amp, marker='o', label='data')

rabi_guess = 1
decay_guess = 1
guess = ([(amp.max() - amp.min())/2, rabi_guess, x[0]/rabi_guess, amp.mean(), decay_guess])
opt, cov = curve_fit(Rabi_func, x, amp, p0=guess, maxfev=1000000)

plt.plot(x, Rabi_func(x, *opt), label='fit')
plt.legend()
plt.title('Rabi')
plt.xlabel('us')
plt.ylabel('ADC level (a.u.)')
plt.savefig('./fitting results/Rabi.png')
# plt.show()
