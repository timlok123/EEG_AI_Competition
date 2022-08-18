import tensorflow as tf

from scipy.io import loadmat
import pandas as pd
import numpy as np

matlabfile = loadmat("data_EEG_AI.mat")

print(matlabfile.keys())
labels = ['channel_labels', 'data', 'label', 'time_points']

data_dict = dict()
channel_label_list = [] 

# 1. store the name of the label to the channel_label_list
for i in matlabfile['channel_labels']:
    channel_label_list.append(str(list(i[0])[0]))
print(channel_label_list)
temp=matlabfile['data']

#randomizer=np.random.shuffle(np.arange(7800))
randomizer=np.arange(7800)
randomizer=np.random.permutation(randomizer)
label=np.array(([i//300 for i in randomizer]))
data=np.transpose(np.array(np.take(temp,randomizer,axis=2)),(2,0,1))
train=data[:6240,:,]
test=data[6240:,:,]
trainlabel=label[:6240]
testlabel=label[6240:]
print(type(train),type(trainlabel))
print(train.shape,trainlabel.shape)

"""model = tf.keras.layers.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5,5) ,padding="same",  input_shape=(24,801,1)))
model.add(tf.keras.layers.MaxPooling2D((1, 7),padding="same"))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (7, 13),padding="same" ))
model.add(tf.keras.layers.MaxPooling2D((3, 7),padding="same"))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (7, 13),padding="same"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='softmax'))
model.add(tf.keras.layers.Dense(26))"""

#Based on DenseNet
def Dense(x):
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x=tf.keras.layers.Conv2D(32,(3,3),padding="same")(x)
    return x

input=tf.keras.layers.Input(shape=(24,801,1))
x=tf.keras.layers.Conv2D(32,(7,42),padding="same",strides=2)(input)
x=tf.keras.layers.AveragePooling2D((1,3),padding="same")(x)
for i in range(6):
    if i==0:
        x=Dense(x)
    else:
        y=Dense(x)
        x=tf.keras.layers.Add()([x,y])
x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
x=tf.keras.layers.AveragePooling2D((2,2),strides=2,padding="same")(x)
for i in range(12):
    if i==0:
        x=Dense(x)
    else:
        y=Dense(x)
        x=tf.keras.layers.Add()([x,y])
x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
x=tf.keras.layers.AveragePooling2D((2,2),strides=2,padding="same")(x)
for i in range(24):
    if i==0:
        x=Dense(x)
    else:
        y=Dense(x)
        x=tf.keras.layers.Add()([x,y])
x=tf.keras.layers.Conv2D(32,(1,1),padding="same")(x)
x=tf.keras.layers.AveragePooling2D((2,2),strides=2,padding="same")(x)
for i in range(16):
    if i==0:
        x=Dense(x)
    else:
        y=Dense(x)
        x=tf.keras.layers.Add()([x,y])
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Dense(256, activation='softmax')(x)
output=tf.keras.layers.Dense(26)(x)

model=tf.keras.Model(inputs=input,outputs=output)
print(model.summary())



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train, trainlabel, epochs=300, 
                    validation_data=(test, testlabel))