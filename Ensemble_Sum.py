#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import confusion_matrix


# In[2]:


incp_probmat = np.loadtxt(fname="InceptionV3d6_finaldataset.csv", delimiter=",") # prediction from pre-trained model
pllcnn_probmat = np.loadtxt(fname="pllcnn_finaldataset.csv", delimiter=",") # prediction from MFMS model
y_test = np.loadtxt(fname="finaldataset_test_probmat.csv", delimiter=",") # Actual output


# In[5]:


#Ensemble SUM
ypred_addens = pllcnn_probmat+incp_probmat
y_test1 = np.argmax(y_test,axis=-1)
y_sumensb = np.argmax(ypred_addens,axis=-1)


# In[6]:


confmt=confusion_matrix(y_test1,y_sumensb)
print("confusion_matrix without normalization\t:")
print(confmt)
A = confmt
A = A.astype('float') / A.sum(axis=1)[:,np.newaxis]
print("confusion_matrix with normalization\t:")
print(A)
for i in range(len(A)):
  print('Class'+str(i+1)+' ; '+str(A[i,i]*100))


# In[7]:


arrmax_row=np.amax(confmt,axis=1)
print(arrmax_row)


# In[8]:


corr_clsf=arrmax_row.sum()
print(corr_clsf)


# In[9]:


test_size=y_test.shape[0]
print(test_size)


# In[10]:


final_acc=(corr_clsf/test_size)*100
print(final_acc)

