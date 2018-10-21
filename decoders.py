# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:34:57 2018

@author: tanma
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Bidirectional
from keras.optimizers import RMSprop
from keras.layers import LSTM,CuDNNLSTM,CuDNNGRU

# Importing the training set
dataset = pd.read_csv('weatherHistory.csv')
results = []
feature_indexes = [3,4,5,6]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler,RobustScaler
sc = MinMaxScaler()

# Building the RNN
regressor = Sequential()

regressor.add(Bidirectional(CuDNNLSTM(units = 8,return_sequences = True, input_shape = (None, 1))))
regressor.add(Dropout(0.1))

regressor.add(Bidirectional(CuDNNLSTM(units = 8,return_sequences = True, input_shape = (None, 1))))
regressor.add(Dropout(0.1))

regressor.add(Bidirectional(CuDNNLSTM(units = 8,return_sequences = True, input_shape = (None, 1))))
regressor.add(Dropout(0.1))

regressor.add(Bidirectional(CuDNNLSTM(units = 6,return_sequences = True)))
regressor.add(Dropout(0.1))

regressor.add(Bidirectional(CuDNNLSTM(units = 4,return_sequences = True)))
regressor.add(Dropout(0.1))

regressor.add(Bidirectional(CuDNNLSTM(units = 3)))
regressor.add(Dropout(0.1))

regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'nadam', loss = 'mean_squared_error')

# Creating a data structure with 15 timesteps and t+1 output
for i in (feature_indexes):
    training_set = dataset.iloc[:,i:i+1].values
    obs = training_set.shape[0]
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    timestep = 15
    for i in range(obs-3650*24, obs):
        X_train.append(training_set_scaled[i-timestep:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Fitting the RNN to the Training set
    filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
    checkpoint = ModelCheckpoint(filepath,monitor = "loss", verbose = 1,save_best_only = True,mode = "min")
    callbacks_list = [checkpoint]
    
    regressor.fit(X_train, y_train, epochs = 5, batch_size = 32,callbacks = callbacks_list)
    # Making the predictions and visualising the results
    inputs = []
    for i in range(obs-48, obs):
        inputs.append(training_set_scaled[i-timestep:i, 0])
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
    preds = regressor.predict(inputs)
    preds = sc.inverse_transform(preds)
    results.append(preds)
    
    # Visualising the results
    plt.plot(training_set[obs-48:], color = 'red', label = 'Real Values')
    plt.plot(preds, color = 'blue', label = 'Predicted Values')
    plt.title('Weather Prediction')
    plt.xlabel('Time')
    plt.ylabel('Dataset Object')
    plt.legend()
    plt.show()

results = np.array(results)
results = np.reshape(results, (results.shape[0], results.shape[1]))
temp = list(results[0])
apparent_temp = list(results[1])
humidity = list(results[2])
wind_speed = list(results[3])
df = pd.DataFrame(results)
df.to_csv("file_path.csv")

#Pushing Data to Firebase
from google.cloud import firestore
path=r"C:\Users\tanma\Desktop\Hackertech\hack-7b54f-firebase-adminsdk-tu2z6-36a123d635.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=path
db=firestore.Client()
forecasts = db.collection(u'forecast')
for i in range(60):
    forecasts.add({
    u'name':str(temp[i]),
    u'status':str(apparent_temp[i]),
    u'humidity':str(humidity[i]),
    u'windspeed':str(wind_speed[i])
            })







