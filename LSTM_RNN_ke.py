#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 04:32:23 2021

@author: krajna4ever
"""

#import all libraries 

import pandas as pd 
import numpy as np 
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras import models, layers, optimizers, callbacks,regularizers 
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM
from keras import models, layers, optimizers, callbacks,regularizers 
from tensorflow.keras.layers import Dropout

from sklearn.datasets import make_classification
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(threshold=sys.maxsize)

#import and describe dataset
dataset = pd.read_csv('beacons.csv')
grubo = dataset.describe()
df = dataset.drop(['steps'],axis=1)


#allocate the training data
X=df.iloc[:,2:]

scaler = MinMaxScaler(feature_range=(0,1))
X_scaler = scaler.fit_transform(X)
y=df.iloc[:,0:2]
X=np.array(X_scaler)

#train test and train val split
X_train, X_test, Y_train, Y_test  = train_test_split(X, y, test_size = 0.2,random_state =42)
X_val, X_test, Y_val, Y_test  = train_test_split(X, y, test_size = 0.7,random_state =42)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
Y_val = np.reshape(Y_val, (Y_val.shape[0], Y_val.shape[1]))
# start the RNN structure



rnn = Sequential()
rnn.add(LSTM(units = 500, return_sequences = True, input_shape = (X_train.shape[1], 1)))
rnn.add(LSTM(units = 300, return_sequences = True))
rnn.add(Dropout(0.1))
rnn.add(LSTM(units = 300, return_sequences = True))
rnn.add(LSTM(units = 300))   
rnn.add(Dropout(0.1))
rnn.add(Dense(units = 2))
rnn.compile(optimizer ='RMSprop', loss = 'mse', metrics=['accuracy'])

chkpt = callbacks.ModelCheckpoint(filepath='moja_modularacnn_{val_acc:.2f}.h5', monitor='val_loss',verbose=42,save_best_only=True, save_weights_only=False)
reduce_lr=callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.95,patience=5,verbose=42,cooldown=2)
callb_l=[chkpt, reduce_lr]
# Change the training epohcs to arbitrary number, preferred 150
history = rnn.fit(X_train, Y_train,epochs = 2, batch_size = 32, validation_data=(X_val, Y_val))

#eucledian distance function to include for results to compare
def eucledian_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dists = np.sqrt(dx ** 2 + dy ** 2)
    dists =np.nan_to_num(dists)
    return np.mean(dists), dists

#import the test dataset.
test_data = pd.read_csv('novabeacons.csv')
X_new_test = test_data.iloc[:,3:]



scaler = MinMaxScaler(feature_range=(0,1))
X_scaler_new = scaler.fit_transform(X_new_test)
X_scalar = np.array(X_scaler_new)
X_scaler_new = np.reshape(X_scalar, (X_scalar.shape[0], X_scalar.shape[1], 1))


predictions_new = rnn.predict(X_scaler_new)

y=test_data.iloc[:,1:3]

PredXcoor = predictions_new[:,0]
PredYcoor = predictions_new[:,1]
Xcoor = y.iloc[:,0]
Ycoor = y.iloc[:,1]

subtraction_x = abs(np.subtract(Xcoor, PredXcoor))
subtraction_y = abs(np.subtract(Ycoor, PredYcoor))

error = np.mean(subtraction_x)  
error2 = np.mean(subtraction_y)


print('Mean Absolute Error in x-axis:', round(np.mean(error), 2), 'm')
print('Mean Absolute Error in y-axis:', round(np.mean(error2), 2), 'm')

newPredXcoor = np.nan_to_num(PredXcoor)
newPredYcoor = np.nan_to_num(PredYcoor)

newXcoor = np.nan_to_num(Xcoor)
newYcoor = np.nan_to_num(Ycoor)

l2dists_mean, l2dists = eucledian_distance((newPredXcoor, newPredYcoor), (newXcoor, newYcoor))
print("Mean distance error : {}".format(l2dists_mean))
import os
cdf_error_cnn = os.path.join(os.getcwd(),'CDF_error_cnn1.png')
sortedl2_deep = np.sort(l2dists)
prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
fig, ax = plt.subplots()
lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
plt.title('CDF of Euclidean distance error for LSTM')
plt.xlabel('Distance (m)')
plt.ylabel('Probability')
plt.grid(True)
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
            line.set_linestyle('-.')

plt.savefig(cdf_error_cnn, dpi=300)
plt.show()











