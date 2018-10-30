# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:00:43 2018

@author: Jan-Gunther Gosselke
"""

import tensorflow as tf
import keras as ke
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(6,)),
    Dense(1)
])

training = pd.DataFrame(data = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6], 'col4': [7, 8], 'col5': [9, 10], 'col6': [11, 12]})
training_y = pd.DataFrame(data = {'result' : [4, 5]})

test = pd.DataFrame(data = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6], 'col4': [7, 8], 'col5': [9, 10], 'col6': [11, 12]})
test_y = pd.DataFrame(data = {'result' : [4, 5]})

model.compile(optimizer='rmsprop', loss='mse')
model.fit(training, training_y, epochs=10)
score = model.evaluate(test, test_y, batch_size=16)