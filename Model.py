# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:14:22 2020

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

train_dir = "Train"
test_dir = "Test"

df = []
df_test = []

for filename in os.listdir(train_dir):
    imgs = os.path.join(train_dir,filename)
    for img_name in os.listdir(imgs):
        try:
            img = cv2.imread(os.path.join(imgs,img_name),cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(30,30))
            df.append([img,filename])
        except:
            continue
        
for img_name in os.listdir(test_dir):
        try:
            img = cv2.imread(os.path.join(test_dir,img_name),cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(30,30))
            df_test.append(img)
        except:
            continue      
        
random.shuffle(df)

"""cv2.imshow("radnom_pic",random.choice(df)[0])

cv2.waitKey(0)
cv2.destroyAllWindows()"""  

X = []
y = []

for img, label in df:
    X.append(img)
    y.append(label)
    
X = np.array(X).reshape(-1,30,30,1).astype(np.float32)
df_test = np.array(df_test).reshape(-1,30,30,1).astype(np.float32)
y = pd.Series(np.array(y)) 

nClasses = y.nunique()

y = pd.get_dummies(y)    

X = X/255 
df_test = df_test/255 



from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, concatenate, Input
        
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

input_img = Input(shape=(30, 30, 1))

layer = inception_module(input_img, 64, 96, 128, 16, 32, 32)

#layer = inception_module(layer, 128, 128, 192, 32, 96, 64)

flat_1 = Flatten()(layer)

dense_1 = Dense(1024, activation='relu')(flat_1)
output = Dense(nClasses, activation='softmax')(dense_1)

model = Model([input_img], output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=14, batch_size=64, validation_split=0.2)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("model.h5")

y_pred = model.predict(df_test)
predictions = np.array(np.argmax(y_pred,axis=1))
predictions = pd.Series(predictions,name="Label")

predictions.to_csv("predictions.csv",index=False)
