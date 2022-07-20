import h5py
import numpy as np
import matplotlib.pyplot as plt

dat=h5py.File('resting.snirf','r')
d1=np.array(dat.get('/nirs/data1/dataTimeSeries'));
#m=dat.get('/nirs/data1/measurementList');
#print(m)
#print(d1)
#print(np.shape(d1))


plt.plot(d1[:,0])
plt.show()

plt.plot(d1[:,1])
plt.show()

plt.plot(d1[:,2])
plt.show()

plt.plot(d1[:,3])
plt.show()

plt.plot(d1[:,4])
plt.show()

plt.plot(d1[:,5])
plt.show()

plt.plot(d1[:,6])
plt.show()

plt.plot(d1[:,7])
plt.show()

#d2 = d1[:,4:8]
#d2 = d2[1550:,:]
#d2 = d2[0:12000,:]
#d2 = d2[0::4,:]
#d2 = d2[1:,:]

#d1 = d1[:,0:4]
#d1 = d1[1550:,:]
#d1 = d1[0:12000,:]
#d1 = d1[0::4,:]
#d1 = d1[1:,:]
#d = np.dstack((d1, d2))



print(np.shape(d1))

#print(np.shape(d2))

#print(np.shape(d))

