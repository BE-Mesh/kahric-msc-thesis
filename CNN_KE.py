#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 03:38:39 2021

@author: krajna4ever
"""
#import all libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten,BatchNormalization
from keras.layers import Dropout
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers, callbacks,regularizers 
from keras import backend as K
np.random.seed(7)


#input the dataset and describe

df = pd.read_csv('beacons.csv')

#print(df)
data = df.values
#print(data)


X = data[:,3:]

Y = data[:,1:3]


#Normalization through min max scaler 
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.25)



X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test,Y_val_and_test, test_size = 0.7)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_test.shape, Y_val.shape)
#model the CNN
model = Sequential ([
    Dense(1000, activation = 'tanh', activity_regularizer=regularizers.l2(0.001), input_shape=(7,)),
    BatchNormalization(),
    Dense (1000, activation = 'tanh',activity_regularizer=regularizers.l2(0.001)),
    Dense (500, activation = 'tanh',activity_regularizer=regularizers.l2(0.01)),
    Dense (62, activation ='tanh',activity_regularizer=regularizers.l2(0.01)),
    Dense (2, activation = 'relu',activity_regularizer=regularizers.l2(0.01), ),])
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
rms = optimizers.RMSprop(lr=0.0001, rho=0.9)
sgd = optimizers.SGD(lr=0.001,momentum=0)
model.summary()

model.compile(optimizer = sgd, loss='mse', metrics=['accuracy'])
chkpt = callbacks.ModelCheckpoint(filepath='moja_modularacnn_{val_acc:.2f}.h5', monitor='mse',verbose=42,save_best_only=True, save_weights_only=False)
reduce_lr=callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.95,patience=5,verbose=42,cooldown=2)
callb_l=[chkpt, reduce_lr]

hist = model.fit(X_train, Y_train, batch_size=32, epochs=150, validation_data=(X_val, Y_val), callbacks=callb_l)
model.save(filepath=r'mojcnn.h5', overwrite=True)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('LOss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper right')
plt.show()




def eucledian_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dists = np.sqrt(dx ** 2 + dy ** 2)
    return np.mean(dists), dists

Ypred=model.predict(X_test)
Xcoor = Y_test[:,0]
Ycoor = Y_test[:,1]

PredXcoor = Ypred[:, 0]
PredYcoor = Ypred[:, 1]
l2dists_mean, l2dists = eucledian_distance((PredXcoor, PredYcoor), (Xcoor, Ycoor))
print("Mean distance error : {}".format(l2dists_mean))

predictions = PredXcoor
predictions2 = PredYcoor
errors = abs(predictions - Xcoor)
errors2 = abs(predictions2-Ycoor)
mape = 100 * (errors / PredXcoor)
accuracy = 100 - np.mean(mape)


print('Mean Absolute Error:', round(np.mean(errors), 2), 'm')
print('Mean Absolute Error1:', round(np.mean(errors2), 2), 'm')

import os
cdf_error_cnn = os.path.join(os.getcwd(),'CDF_error_cnn1.png')
sortedl2_deep = np.sort(l2dists)
prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
fig, ax = plt.subplots()
lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
plt.title('CDF of Euclidean distance error for CNN')
plt.xlabel('Distance (m)')
plt.ylabel('Probability')
plt.grid(True)
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
            line.set_linestyle('-.')

plt.savefig(cdf_error_cnn, dpi=300)
plt.show()

plt.boxplot(l2dists)
plt.xlabel('CNN')
