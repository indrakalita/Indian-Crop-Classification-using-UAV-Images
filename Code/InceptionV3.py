#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


print(keras.__version__)


# In[3]:


#Clone the dataset
#!git clone https://github.com/indrakalita/KARC


# In[ ]:


classes = 12
img_height = 256
img_width = 256


# In[6]:


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


# In[19]:


base_model = InceptionV3(include_top = False,weights = 'imagenet',input_shape=(img_height,img_width,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.6)(x)
predictions= Dense(classes, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

model1=model
# op = Adam(lr=0.0001)
model1.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model1.summary()


# In[21]:


history=model1.fit(x_train, y_train,epochs =25, validation_data =(x_val, y_val), batch_size=32)


# In[22]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(' Parallel multi-scaled model accuracy for own dataset')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training_accuracy', 'validation_accuracy'], loc='lower center')
plt.show()


# In[23]:


y_pred=model1.predict(x_test)
y_pred1 = y_pred.argmax(axis=-1)
y_test1 = y_test.argmax(axis=-1)
print(y_pred)


# In[24]:


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


# In[25]:


arrmax_row=np.amax(confmt,axis=1)
print(arrmax_row)


# In[26]:


corr_clsf=arrmax_row.sum()
print(corr_clsf)


# In[27]:


test_size=x_test.shape[0]
print(test_size)


# In[28]:


final_acc=(corr_clsf/test_size)*100
print(final_acc)
np.savetxt(fname="InceptionV3d6_finaldataset.csv", X=y_pred, delimiter=",")

