from scipy.io import loadmat
import pandas as pd
import numpy as np
import scipy as sp

matlabfile = loadmat("data_EEG_AI.mat")

print(matlabfile.keys())
labels = ['channel_labels', 'data', 'label', 'time_points']

data_dict = dict()
channel_label_list = [] 

file=open("PCorr.csv","a")

# 1. store the name of the label to the channel_label_list
#for i in matlabfile['channel_labels']:
#    channel_label_list.append(str(list(i[0])[0]))
#print(channel_label_list)
temp=matlabfile['data']
data=np.transpose(temp,(2,0,1))
data=np.array_split(data,4,axis=2)


mn=np.zeros(shape=(4,24,26,26))
best=np.zeros(shape=(96))
for i in range(1,4):

    for j in range(24):

        for m in range(7800):

            for n in range(7800):
                mm=m//300
                nn=n//300
                mn[i,j,mm,nn]+=abs(sp.stats.pearsonr(np.ndarray.flatten(data[i][mm:mm+1,j,:]),np.ndarray.flatten(data[i][nn:nn+1,j,:])).statistic)
                
        

        best[i*24+j]=np.array(np.mean(mn[i,j,:,:]))
        print(i,j,m//300,n//300,"mean=",np.mean(mn[i,j,:,:]))
        for m in range(26):

            for n in range(26):
                file.write(str(mn[i][j,m,n]/300)+',')
            file.write('\n')
        file.write('\n')
        file.close()   
        file=open("PCorr.csv","a")


mn=mn/300


        
file.close()        

stor=np.argsort(best)
best=np.take_along_axis(best,stor,axis=0)
location=np.take_along_axis(np.arange(96),stor,axis=0)
for i in range(24):
    print(location[i]//4,location[i]%24, best[i])


matlabfile = loadmat("data_EEG_AI.mat")
