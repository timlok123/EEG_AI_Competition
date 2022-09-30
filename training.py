import tensorflow as tf

from scipy.io import loadmat
import pandas as pd
import numpy as np
import scipy as sp

matlabfile = loadmat("data_EEG_AI.mat")

print(matlabfile.keys())
labels = ['channel_labels', 'data', 'label', 'time_points']

data_dict = dict()
channel_label_list = [] 

# 1. store the name of the label to the channel_label_list
#for i in matlabfile['channel_labels']:
#    channel_label_list.append(str(list(i[0])[0]))
#print(channel_label_list)
temp=matlabfile['data']

collection=[[2,4],[2,23],[2, 14],[2, 15],[2,12],[1, 2],[1, 8],[2, 21],[2, 16],[2, 20],[1, 15],[3, 14],[1, 20],[1, 21],[2, 17],[2, 5],[2, 19],[2, 7],[3, 19],[0, 15],[3, 15],[1, 7],[2, 8],[2, 13]]

#randomizer=np.random.shuffle(np.arange(7800))
randomizer=np.arange(7800)
randomizer=np.random.permutation(randomizer)
temp=np.array([temp[m[1],801//4*m[0]:801//4*(m[0]+1),:]for m in collection])
label=np.array(([i//300 for i in randomizer]))
data=np.transpose(np.array(np.take(temp,randomizer,axis=2)),(2,0,1))
#print(data)
'''f,t,data=sp.signal.stft(data[:,:,:],250,nperseg=64)
data=np.transpose(data,(0,2,3,1))
data=np.concatenate((data.real,data.imag),axis=3)'''

data=np.stack([np.stack([sp.signal.cwt(data[j,:,i],sp.signal.ricker,np.arange(1,32)) for i in range(24)]) for j in range(7800)])
data=np.transpose(data,(0,2,3,1))
print("done")


train=data[:6240,:,:,]
test=data[6240:,:,:,]
trainlabel=label[:6240]
testlabel=label[6240:]
print(type(train),type(trainlabel))
print(train.shape,trainlabel.shape)
data[7000]


"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5,5) ,padding="same",  input_shape=(31,24,24)))
model.add(tf.keras.layers.MaxPooling2D((1, 7),padding="same"))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (7, 13),padding="same" ))
model.add(tf.keras.layers.MaxPooling2D((3, 7),padding="same"))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (7, 13),padding="same"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='softmax'))
model.add(tf.keras.layers.Dense(26))
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
history = model.fit(train, trainlabel, epochs=20, validation_data=(test, testlabel),batch_size=1)
model.save('models')"""

#Based on DenseNet
def Dense(x):
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x=tf.keras.layers.Conv2D(128,(1,1),padding="same")(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x=tf.keras.layers.Conv2D(128,(3,3),padding="same")(x)
    return x

def DenseNet2():
    input=tf.keras.layers.Input(shape=(31,24,24))
    x=tf.keras.layers.Conv2D(128,(5),padding="same",strides=2)(input)
    x=tf.keras.layers.MaxPooling2D((2,2),padding="same")(x)
  
    for i in range(3):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Conv2D(128,(1,1),padding="same")(x)
    x=tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")(x)

    for i in range(6):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Conv2D(128,(1,1),padding="same")(x)
    x=tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")(x)

    for i in range(12):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Conv2D(128,(1,1),padding="same")(x)
    x=tf.keras.layers.MaxPooling2D((2,2),padding="same")(x)

    for i in range(8):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.GlobalMaxPooling2D()(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(4096, activation='softmax')(x)
    x=tf.keras.layers.Dense(4096, activation='softmax')(x)
    output=tf.keras.layers.Dense(26)(x)

    model=tf.keras.Model(inputs=input,outputs=output)
    #print(model.summary())



    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

    history = model.fit(train, trainlabel,epochs=30, 
                    validation_data=(test, testlabel))
    model.save('models')
DenseNet2()

"""def DenseNet1():
    input=tf.keras.layers.Input(shape=(24,801,1))
    x=tf.keras.layers.Conv2D(32,(7,42),padding="same",strides=2)(input)
    x=tf.keras.layers.MaxPooling2D((1,3),padding="same")(x)
    for i in range(6):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
    x=tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")(x)
    for i in range(12):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
    x=tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")(x)
    for i in range(24):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
    x=tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")(x)
    for i in range(16):
        if i==0:
            x=Dense(x)
        else:
            y=Dense(x)
            x=tf.keras.layers.Add()([x,y])
    x=tf.keras.layers.GlobalMaxPooling2D()(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(128, activation='softmax')(x)
    output=tf.keras.layers.Dense(26)(x)

    model=tf.keras.Model(inputs=input,outputs=output)
    print(model.summary())



    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

    history = model.fit(train, trainlabel, epochs=300, 
                    validation_data=(test, testlabel))"""


