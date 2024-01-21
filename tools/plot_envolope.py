
#################### gauss
from qick.helpers import cosine, gauss, triang, DRAG

si = 5 * 4
length = 5*si

a = gauss(mu=length/2-0.5, si=si, length=length, maxv=30000)
plt.plot(a, marker='o')


#################### flat_top



# flat_length = blkcfg['pulse_info']["length"]
flat_length = 10

# si = blkcfg['pulse_info']["sigma"]
si = 5 * 1
maxv = 30000
gauss_length = 5 * si
mu = gauss_length / 2

t1 = gauss_length / 2
t2 = t1 + flat_length
t3 = t2 + gauss_length / 2

# x1 = np.linspace(0, t1, 100)
x1 = np.arange(0, t1, si/20)
y1 = maxv * np.exp(- (x1 - mu)**2 / si**2)

x2 = np.linspace(t1, t2, 100)
y2 = y1[-1] * np.ones(len(x2))

# x3 = np.linspace(gauss_length / 2, gauss_length, 100)
x3 = np.arange(gauss_length / 2, gauss_length, si/20)
y3 = maxv * np.exp(- (x3 - mu)**2 / si**2)
# x3 = np.arange(t2, t3, si/20)
x3 += flat_length

# x1 += t_off_us
# x2 += t_off_us
# x3 += t_off_us

x = list(x1) + list(x2) + list(x3)
y = list(y1) + list(y2) + list(y3)

plt.plot(x, y, marker='o')
#################### const


length = 10
step=0.3
x0 = np.arange(0, length/5, step)
y0 = [0]*len(x0)

x1 = np.arange(length/5, length/5+length, step)
y1 = maxv * np.ones(len(x1))


x2 = np.arange(length/5+length, length/5+length+length/5, step)
y2 = [0]*len(x0)


epsilon = 1e-9
x = list(x0)+list(x1)+list(x2)
y = y0+list(y1)+y2

plt.plot(x, y, marker='o')



