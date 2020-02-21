# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:21:47 2020

@author: Tanmay Thakur
"""
import pickle
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from model import get_model


X_train, y_train = pickle.load(open( "data.pickle", "rb" ))

model = get_model(X_train)

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 1e-3))

cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath = "recurrent_model_initial.h5", monitor = "val_loss", mode = 'min', save_best_only = True, verbose = 1)

model.fit(X_train, y_train, epochs = 20, batch_size = 16, validation_split = 0.25, callbacks = [cp_callbacks])

model.save("recurrent_model_initial.h5")

model = tf.keras.models.load_model("recurrent_model_initial.h5")

validation_target = y_train[-3*len(X_train)//4:]
validation_predictions = []

# index of first validation input
i = -3*len(X_train)//4

while len(validation_predictions) < len(validation_target):
  p = model.predict(X_train[i].reshape(1, X_train.shape[1], X_train.shape[2]))[0] 
  i -= 1

  print(mean_squared_error(p,y_train[i]))
  # update the predictions list
  validation_predictions.append(p)