# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:44:22 2021

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from TrafficSignNet import TrafficSignNet
from keras.optimizers import Adam,  SGD


df = pd.read_csv('Train.csv',usecols=['ClassId', 'Path'])
test_df = pd.read_csv('Test.csv',usecols=['ClassId', 'Path'])

df['ClassId'] = df['ClassId'].astype(str)
test_df['ClassId'] = test_df['ClassId'].astype(str)


gen_train = ImageDataGenerator(rescale=1./255,shear_range=0.2)
gen_test = ImageDataGenerator(rescale=1./255)

generator_train = gen_train.flow_from_dataframe(dataframe=df,
                                                target_size=(32, 32),
                                                x_col='Path',
                                                y_col='ClassId',
                                                batch_size=32,
                                                class_mode='categorical'
                                                )

generator_test = gen_test.flow_from_dataframe(dataframe=test_df,
                                            x_col='Path',
                                            y_col='ClassId',
                                            target_size=(32, 32),
                                            batch_size=16,
                                            class_mode='categorical'
                                            )

n_classes = df['ClassId'].nunique()

opt = SGD(learning_rate = 0.1)

model = TrafficSignNet.build(width=32, height=32, depth=3,
	classes = n_classes)

model.compile(loss="categorical_crossentropy", optimizer = opt,
	metrics=["accuracy"]) 


print("Training....")
history = model.fit(generator_train, epochs = 10) #Epoch 10/10
                                                  #1226/1226 [==============================] - 451s 368ms/step - loss: 0.4097 - accuracy: 0.9726


#Saving the model
print("\n\nSaving Model.....")
model_json = model.to_json()
with open("TrafficSign_model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("TrafficSign_model.h5")

#Evaluating the model
print("\n\nEvaluating......")
model.evaluate(generator_test) # 790/790 [==============================] - 48s 61ms/step - loss: 0.4702 - accuracy: 0.9609