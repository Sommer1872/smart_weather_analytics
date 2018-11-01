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
from keras.layers import Dense, Activation
import loading_data as ld
<<<<<<< HEAD
=======
from sklearn.model_selection import KFold, cross_val_score
from keras.callbacks import History 
>>>>>>> 46285951fc36009db5dd11b2cb328dfb7ff6d227





<<<<<<< HEAD
def Fully_Connected_OneL(x, y):
    #Hyperparameters
=======
def FullyConected(x, y):
    #Hyperparameters
    history = History()
    models = {}
>>>>>>> 46285951fc36009db5dd11b2cb328dfb7ff6d227
    for i in range(1,10):
        number_of_nodes = i*5
        print("Number of Nodes" + str(number_of_nodes))
        model = Sequential([
            Dense(number_of_nodes, input_shape=(42,)),
            Dense(1)
            ])
        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()
        model.fit(x, y, validation_split=0.33, epochs=10, batch_size=128)
<<<<<<< HEAD
    
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
            model.fit(x, y, validation_split=0.33, epochs=10, batch_size=128)
    
def RNN(x, y):
    a = 1
=======
    print(history)
    
#def RNN:
    #model = 
>>>>>>> 46285951fc36009db5dd11b2cb328dfb7ff6d227
