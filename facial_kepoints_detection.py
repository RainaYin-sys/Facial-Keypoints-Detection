#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 23:01:47 2021

@author: yinxiaoru
"""

import os,sys

input_dir = os.path.abspath(sys.argv[1])
result_dir = os.path.abspath(sys.argv[2])

for dirname, _, filenames in os.walk(input_dir):
    for file in filenames:
        print(os.path.join(os.path.abspath(dirname),file))

# Start python import 
import math, time, random, datetime
 
# Data Manipulation       
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
#import missingno
import seaborn as sns
#plt.style.use('seaborn-whitergrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

#Machine learning
#import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection,tree,preprocessing,metrics,linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#from catboost import CatBoostClassifier,Pool,cv
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, MaxPool2D,ZeroPadding2D

#Let's be rebels and warnings for now
import warnings
warnings.filterwarnings('ignore')

# data append
train_data = pd.read_csv(os.path.join(input_dir,'training.zip'),compression='zip',header=0, sep=',',quotechar='"')
test_data = pd.read_csv(os.path.join(input_dir,'test.zip'),compression='zip', header = 0,sep = ',', quotechar='"')
IdLookupTable = pd.read_csv(os.path.join(input_dir,'IdLookupTable.csv'), header=0,sep = ',',quotechar='"')
SampleSubmission = pd.read_csv(os.path.join(input_dir,'SampleSubmission.csv'),header = 0,sep=',',quotechar='"')
train_data.head()

train_data.head().T.tail()
len(train_data['Image'][2])

train_data.info()

test_data.head()
test_data.info()

IdLookupTable.head()
IdLookupTable.info()

SampleSubmission.head()

# check missing data
## train

null_sum=train_data.isnull().sum()
null_sum.plot(kind='bar',color='pink')

train_data.fillna(method='ffill',inplace=True)
train_data.isnull().sum().plot(kind='bar',color='pink')

test_data.isnull().sum().plot(kind='bar',color='pink',title='Missing Data')

# IDLookupTable
display(IdLookupTable.isnull().sum())


# Visualize Data

vis = []
for i in range(len(train_data)):
    vis.append(train_data['Image'][i].split(' '))

## prepare data x train
array_float = np.array(vis, dtype='float')
X_train = array_float.reshape(-1,96,96,1)

## show photo
photo_visualize = array_float[1].reshape(96,96)

plt.imshow(photo_visualize,cmap='pink')
plt.title('viasualize Image')

plt.show()

## Facial Keypoints
facial_pnts_float = train_data.drop(['Image'],axis=1).values

## prepare data y train
training_data = train_data.drop('Image',axis=1)

y_train = training_data.values


## show photo image eith facial points
photo_visualize_pnts = facial_pnts_float[0]

plt.imshow(photo_visualize,cmap = 'gray')
plt.scatter(photo_visualize_pnts[0::2],photo_visualize_pnts[1::2],color = 'Pink', marker= '*')
plt.title("Image eith Facial Keypoints")  
plt.show()

# prepare and split data
train_data.shape

# Build model

## keras CNN
model = Sequential()

### layer 1
model.add(Convolution2D(32, (3,3), activation = 'relu', padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

### layer 2
model.add(Convolution2D(32,(3,3),activation = 'relu', padding = 'same', use_bias = False))
model.add(MaxPool2D(pool_size=(2,2)))

### layer 3

model.add(Convolution2D(64,(3,3),activation= 'relu',padding = 'same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

### layer4
model.add(Convolution2D(128, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))

model.summary()

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

# train data
model.fit(X_train,y_train,epochs = 5,batch_size = 32,validation_split = 0.2)

# prepare data test

test_images = []
for i in range(len(test_data)):
    item = test_data['Image'][i].split(' ')
    test_images.append(item)

array_float_test = np.array(test_images,dtype = 'float')
X_test = array_float_test.reshape(-1,96,96,1)

# predict
predict = model.predict(X_test)

# submission

#from IdLookupTable
FeatureName = list(IdLookupTable['FeatureName'])
ImageId = list(IdLookupTable['ImageId']-1)
RowId = list(IdLookupTable['RowId'])

# predict results
predict = list(predict)

Data = []
for i in list(FeatureName):
    Data.append(FeatureName.index(i))

Data_pre = []
for x,y in zip(ImageId,Data):
    Data_pre.append(predict[x][y])

RowId = pd.Series(RowId,name = 'RowId')
Location = pd.Series(Data_pre, name = 'Location')

submission = pd.concat([RowId,Location],axis = 1)
submission.to_csv(os.path.join(result_dir,'Submission.csv'),index=False)
