# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:00:43 2018

@author: Jan-Gunther Gosselke
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ke
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import loading_data as ld




def Fully_Connected_OneL(x, y):
    #Hyperparameters
    for i in range(1,10):
        number_of_nodes = i*5
        print("Number of Nodes " + str(number_of_nodes))
        model = Sequential([
            Dense(number_of_nodes, input_shape=(42,)),
            Dense(1)
            ])
        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()
        model.fit(x, y, validation_split=0.33, epochs=10, batch_size=128)

def Fully_Connected_TwoL(x, y):
    #Hyperparameters
    for i in range(1,10):
        number_of_nodes_l1 = i*5
        for j in range (1,0):
            number_of_nodes_l2 = j*5            
            print("Number of Nodes " + str(number_of_nodes))
            model = Sequential([
                Dense(number_of_nodes_l1, input_shape=(42,)),
                Dense(number_of_nodes_l2),
                Dense(1)
                ])
            model.compile(optimizer='rmsprop', loss='mse')
            model.summary()
            model.fit(x, y, validation_split = 0.33, epochs = 10, batch_size = 128)
    
def RNN(x, y):
    model = Sequential()
    model.add(LSTM(42, input_shape = (42, ), return_sequences=True))
    model.add(LSTM(42, return_sequences=True))
    model.add(Dense(1))
    model.compile('rmsprop', loss = 'mse')
    model.fit(x, y, validation_split = 0.33, epochs = 10, batch_size = 128)
