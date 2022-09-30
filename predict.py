from tensorflow import keras
import numpy as np
import scipy as sp
from scipy.io import loadmat


collection=[[2,4],[2,23],[2, 14],[2, 15],[2,12],[1, 2],[1, 8],[2, 21],[2, 16],[2, 20],[1, 15],[3, 14],[1, 20],[1, 21],[2, 17],[2, 5],[2, 19],[2, 7],[3, 19],[0, 15],[3, 15],[1, 7],[2, 8],[2, 13]]

model = keras.models.load_model('models')

matlabfile = loadmat("data_EEG_AI.mat")
temp=matlabfile['data']
temp=np.array([temp[m[1],801//4*m[0]:801//4*(m[0]+1),:]for m in collection])
data=np.transpose(temp,(2,0,1))
data=np.stack([np.stack([sp.signal.cwt(data[j,:,i],sp.signal.ricker,np.arange(1,32)) for i in range(24)]) for j in range(78)])
data=np.transpose(data,(0,2,3,1))




print(model.predict(data,10))