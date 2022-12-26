#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle

import keras
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.layers import Flatten, Dropout, ZeroPadding2D, UpSampling2D, Input, MaxPool2D
from keras.layers import Reshape, Add, Multiply, Lambda, AveragePooling2D, concatenate
from keras.activations import relu
from keras.activations import linear as linear_activation
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam, SGD, Nadam, RMSprop, Adagrad
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback, ModelCheckpoint
from keras import backend as K
from keras.regularizers import l2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, f1_score, fbeta_score, accuracy_score
from sklearn.metrics import recall_score, precision_score


# In[ ]:


#Clone the dataset
#!git clone https://github.com/indrakalita/KARC


# In[ ]:


classes = 12
img_height = 256
img_width = 256
batch_size = 32


# In[ ]:


dataset = np.load('data/train/KARC.npy')

features = dataset[:, 0]
labels = dataset[:, 1]
labels_one_hot_encoded = to_categorical(labels, classes)

x_train = features
y_train = labels_one_hot_encoded

dataset = np.load('data/val/KARC.npy')

features = dataset[:, 0]
labels = dataset[:, 1]
labels_one_hot_encoded = to_categorical(labels, classes)

x_val = features
y_val = labels_one_hot_encoded

dataset = np.load('data/test/KARC.npy')

features = dataset[:, 0]
labels = dataset[:, 1]
labels_one_hot_encoded = to_categorical(labels, classes)

x_test=features
y_test=labels_one_hot_encoded


# In[ ]:


# from sklearn.model_selection import train_test_split

# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=42)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255


# In[ ]:


def mymodel(shape=(256,256,3)):
  input1 = Input(shape=shape)
  #Depth1
  #CBR11
  x1 = Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same')(input1)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)
  #MP1
  x2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x1)
  #CBR21
  x2 = Conv2D(64, (3, 3), activation='linear', padding='same')(x2)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  #Depth2
  x3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input1)
  input2=x3
  #CBR22
  x4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  #CAT1
  x5 = concatenate([x2, x4],axis=-1)
  #MP2
  x5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),padding='same')(x5)
  #CBR31
  x6 = Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same')(x5)
  x6 = BatchNormalization()(x6)
  x6 = Activation('relu')(x6)
  #Depth3
  x7 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input2)
  input3=x7
  #CBR32
  x8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x7)
  x8 = BatchNormalization()(x8)
  x8 = Activation('relu')(x8)
  #CAT2
  x9 = concatenate([x6, x8],axis=-1)
  #MP3
  x10 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),padding='same')(x9)
  #CBR41
  x11 = Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same')(x10)
  x11= BatchNormalization()(x11)
  x11 = Activation('relu')(x11)
  #Depth4
  x12 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input3)
  #CBR42
  x13 = Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same')(x12)
  x13= BatchNormalization()(x13)
  x13 = Activation('relu')(x13)
  #CAT3
  x14 = concatenate([x11, x13],axis=-1)
  #MP4
  x15 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),padding='same')(x14)
  #CBR5
  x16 = Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same')(x15)
  x16= BatchNormalization()(x15)
  x16= Activation('relu')(x15)
  #MP5
  x17 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),padding='same')(x16)
  #CBR6
  x18 = Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same')(x17)
  x18 = BatchNormalization()(x18)
  x18 = Activation('relu')(x18)
  #AP
  x19=AveragePooling2D(pool_size=(2,2), strides=(1, 1),padding='same')(x18)
  x20 = Flatten()(x19)
  #x20 = Dense(2056, activation = 'relu')(x20)
  x20= Dense(1024, activation = 'relu')(x20)
  x20= Dropout(0.2)(x20)
  output = Dense(12, activation='softmax')(x20)
  model = Model(input1, output)
  #model.summary()
  return model


# In[ ]:


#for layer in model.layers:
#    layer.trainable = True
model1=mymodel()
op = Adam(lr=0.0001)
model1.compile(optimizer = op,loss = 'categorical_crossentropy',metrics = ['accuracy'])
model1.summary()


# In[ ]:


history=model1.fit(x_train, y_train,epochs =25, validation_data =(x_val, y_val), batch_size=batch_size)


# In[ ]:



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(' Parallel multi-scaled model accuracy for own dataset')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training_accuracy', 'validation_accuracy'], loc='lower center')
plt.show()


# In[ ]:


y_pred=model1.predict(x_test)
y_pred1 = y_pred.argmax(axis=-1)
y_test1 = y_test.argmax(axis=-1)
print(y_pred)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
confmt=confusion_matrix(y_test1,y_pred1)
print("confusion_matrix without normalization\t:")
print(confmt)
A = confmt
A = A.astype('float') / A.sum(axis=1)[:,np.newaxis]
print("confusion_matrix with normalization\t:")
print(A)
for i in range(len(A)):
  print('Class'+str(i+1)+' ; '+str(A[i,i]*100))


# In[ ]:


arrmax_row=np.amax(confmt,axis=1)
print(arrmax_row)


# In[ ]:


corr_clsf=arrmax_row.sum()
print(corr_clsf)


# In[ ]:


test_size=x_test.shape[0]
print(test_size)


# In[ ]:


final_acc=(corr_clsf/test_size)*100
print(final_acc)
np.savetxt(fname="pllcnn_finaldataset.csv", X=y_pred, delimiter=",")

