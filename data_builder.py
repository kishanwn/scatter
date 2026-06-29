import scipy.io as sio
import numpy as np
import os


data = np.zeros((1000,5, 262144))

inx =list(range(0, 262144, 100))
data_short = np.zeros((1000,5, len(inx)))

x = sio.loadmat("x.mat")
y = sio.loadmat("y.mat")

X,Y =  np.meshgrid( x["x"], y["y"])

c = 0

for i in range(1288, 2288):#  specify the data range 
    q = sio.loadmat("./data/q" + str(i) + ".mat")
    #print(len(q["nu_"]))
    real = sio.loadmat("./data/real" + str(i) + ".mat")
    #print(real["realu"])
    imag = sio.loadmat("./data/imag" + str(i) + ".mat")
    #print(imag["imagu"])
    #print(X[:].shape)
    
    data_short[c,0, : ] = np.reshape(X[:].squeeze() , 262144  )[inx]
    data_short[c,1, : ] = np.reshape(Y[:].squeeze() , 262144 )[inx]
    data_short[c,2, : ] = q["nu_"][inx].squeeze() 
    #print(np.reshape(real["realu"][:], 262144  )) # .squeeze())
    data_short[c,3, : ] = np.reshape(real["realu"][:], 262144  )[inx] # real["realu"][:] #.squeeze()
    data_short[c,4, : ] = np.reshape(imag["imagu"][:], 262144  )[inx] # imag["imagu"][:].squeeze()
    
    '''
    data[c,0, : ] = np.reshape(X[:].squeeze() , 262144  )
    data[c,1, : ] = np.reshape(Y[:].squeeze() , 262144  )
    data[c,2, : ] = q["nu_"][:].squeeze() 
    #print(np.reshape(real["realu"][:], 262144  )) # .squeeze())
    data[c,3, : ] = np.reshape(real["realu"][:], 262144  ) # real["realu"][:] #.squeeze()
    data[c,4, : ] = np.reshape(imag["imagu"][:], 262144  ) # imag["imagu"][:].squeeze()
    '''
    c = c + 1
    
print(data_short)


data_ = {"data": data_short}
#si("./data/data1000.mat", data_ )o.savemat
#np.save("./data/data1000.mat", data_ )
np.savez_compressed('compressed_dataset.npz', data=data_short)

loaded = np.load('compressed_dataset.npz')
my_array = loaded['data']
print(my_array)


'''
data = {"data": data_short}
#si("./data/data1000.mat", data_ )o.savemat
#np.save("./data/data1000.mat", data_ )
np.savez_compressed('compressed_dataset.npz', data=data)

loaded = np.load('compressed_dataset.npz')
my_array = loaded['data_short']
print(my_array)
'''