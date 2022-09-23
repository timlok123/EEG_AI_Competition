import numpy as np
import matplotlib.pyplot as plt
import csv

a=np.zeros((96,26,26))
c1=0
with open("PCorr.csv") as fl:
    reader=csv.reader(fl)
    for row in reader:
        for u in row:
            if u:

                a[c1//676,(c1//26)%26,c1%26]=float(u)
                c1+=1
a=a/300

b=np.zeros((4,4,24))

print(a[0,0,0]+a[0,0,0+1:])
#avg correlation of timezone vs channel
for i in range(96):
    b[0,i//24,i%24]=np.trace(a[i])/26
    b[1,i//24,i%24]=(np.sum(a[i])-np.trace(a[i]))/(26*25)
    b[2,i//24,i%24]=np.var(np.ndarray.flatten(np.array([np.concatenate((a[i,j,0:j],a[i,j,j+1:])) for j in range(26)])))
b[3]=b[1]+b[2]

fig=plt.figure()
#ax1=fig.add_subplot(411)
ax2=fig.add_subplot(311)
ax3=fig.add_subplot(312)
ax4=fig.add_subplot(313)
#ax1.set_title("self correlation")
ax2.set_title("correlation")
ax3.set_title("variance")
ax4.set_title("correlation+variance")
#cax1=ax1.matshow(b[0])
cax2=ax2.matshow(b[1])
cax3=ax3.matshow(b[2])
cax4=ax4.matshow(b[3])
#fig.colorbar(cax1)
fig.colorbar(cax2)
fig.colorbar(cax3)
fig.colorbar(cax4)
plt.show()



#finds minimal correlation
indi=np.argsort(np.ndarray.flatten(b[3]))
for i in range(24):
    print(indi[i]//24,indi[i]%24)

#This does not do much
for i in range(96):

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_title(f"timezone {indi[i]//24}, channel {indi[i]%24}, variance {b[3,indi[i]//24,indi[i]%24]}")
    cax=ax.matshow(a[indi[i]])
    fig.colorbar(cax)

    plt.show()
