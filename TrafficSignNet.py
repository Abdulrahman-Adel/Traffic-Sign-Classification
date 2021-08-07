# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:49:26 2021

@author: Abdelrahman
"""
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

class TrafficSignNet(object):
    @staticmethod
    def build(width, height, depth, classes):
        
        #Defining the model
        model = Sequential()
        inputShape = (height, width, depth)
        
        #Constructing the model
        model.add(Input(shape = inputShape))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding="valid"))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding='valid'))
        model.add(Flatten())
        model.add(Dense(120, activation='relu',kernel_regularizer='l2'))
        model.add(Dropout(0.25))
        model.add(Dense(classes, activation = 'softmax'))

        
        return model