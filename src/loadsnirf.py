import h5py
import numpy as np
import matplotlib.pyplot as plt

listie1 = ['33', '34', '36', '37', '38', '39', '40', '41', '43', '44', '46', '47', '49', '51']
listie2 = ['86', '91', '92', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104']
# for e in listie:
	# dat=h5py.File('resting'+e+'.snirf','r')
	# d1=np.array(dat.get('/nirs/data1/dataTimeSeries'));
	# print(d1.shape)
    
dat1=h5py.File('resting'+'33'+'.snirf','r')
d3=np.array(dat1.get('/nirs/probe/sourcePos2D'));
d4=np.array(dat1.get('/nirs/probe/detectorPos2D'));
print('source')
print(d3)
print('detector')
print(d4)
#print(d5)

l=list(dat1.get('/nirs/data1/measurementList1' ));
#print(l)

for j in range(56):
    if j>0:
        valS = np.array(dat1.get('/nirs/data1/measurementList'+str(j)+'/'+str(l[5])))
        tyS = valS.shape
        valD = np.array(dat1.get('/nirs/data1/measurementList'+str(j)+'/'+str(l[3])))
        tyD = valD.shape
        print(j-1,l[5],valS,tyS,l[3],valD,tyD)





# dat2=h5py.File('resting'+'104'+'.snirf','r')
# d3=np.array(dat2.get('/nirs/probe/sourcePos2D'));
# d4=np.array(dat2.get('/nirs/probe/detectorPos2D'));
# print('source')
# print(d3)
# print('detector')
# print(d4)
# #print(d5)

# l=list(dat2.get('/nirs/data1/measurementList1' ));
# #print(l)

# for j in range(113):
    # if j>0:
        # valS = np.array(dat2.get('/nirs/data1/measurementList'+str(j)+'/'+str(l[5])))
        # valD = np.array(dat2.get('/nirs/data1/measurementList'+str(j)+'/'+str(l[3])))
        # print(j-1,l[5],valS,l[3],valD)



#print(m)
#print(d1)
#print(np.shape(d1))
#print(np.shape(d2))



#plt.plot(d1[:,11])
#plt.show()

#plt.plot(d1[:,16])
#plt.show()

#plt.plot(d1[:,])
#plt.show()

#plt.plot(d1[:,3])
#plt.show()

#plt.plot(d1[:,4])
#plt.show()

#plt.plot(d1[:,5])
#plt.show()

#plt.plot(d1[:,6])
#plt.show()

#plt.plot(d1[:,7])
#plt.show()

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



#print(np.shape(d1))

#print(np.shape(d2))

#print(np.shape(d))

