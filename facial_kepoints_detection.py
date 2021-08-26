#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 23:01:47 2021

@author: yinxiaoru
"""

import os

for dirname, _, filenames in os.walk('./'):
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
train_data = pd.read_csv('./training.zip',compression='zip',header=0, sep=',',quotechar='"')
test_data = pd.read_csv('./test.zip',compression='zip', header = 0,sep = ',', quotechar='"')
IdLookupTable = pd.read_csv('./IdLookupTable.csv', header=0,sep = ',',quotechar='"')
SampleSubmission = pd.read_csv('./SampleSubmission.csv',header = 0,sep=',',quotechar='"')
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

test_data.isnull().sum().plot(kind='bar',color='pink')















