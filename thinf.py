# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:29:46 2020

@author: neha
"""

#import tensorflow as tf
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
#from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D 

#Initilising the CNN
classifier = Sequential()
classifier.add(Conv2D(16,(3,3),input_shape = (64,64,3),activation='relu'))
#16=filters,dimention of filter,64-64 is a dimention of data which we want,3=for coloured image 
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,
                     activation = 'relu'))
classifier.add(Dense(units=1,
                     activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
classifier.summary()
from keras .preprocessing.image import ImageDataGenerator
test_datagen= ImageDataGenerator(rescale=1./255)
train_datagen= ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory("training_set",target_size=(64,64),batch_size=32,class_mode='binary')
#classification...last layer action function
#sigmoid-binary_crossentropy-binary
#softmax-sparce categorical-categorical
test_set=test_datagen.flow_from_directory("test_set",target_size=(64,64),batch_size=32,class_mode='binary')
training_set.class_indices
#fit
classifier.fit_generator(training_set,steps_per_epoch=400,epochs=3,validation_data=test_set,validation_steps=62)

#prediction
import numpy as np
from keras.preprocessing import image
import matplotlib . image as mpimg
import matplotlib . pyplot as plt
test_image=image.load_img("B:/Python ML/ThinFat/single_prediction/Thin_or_Fat_0.jpg",target_size=(64,64))
#"1" - 1 dimention is added in row ...i.e.1 col get added in data

img = mpimg . imread('B:/Python ML/ThinFat/single_prediction/Thin_or_Fat_0.jpg')
plt.imshow(img)
type(test_image)
test_image_1=image.img_to_array(test_image)
type(test_image_1)
test_image_1.shape
test_image_2=np.expand_dims(test_image_1,axis=0)
test_image_2.shape
result=classifier.predict(test_image_2)
print(result)
training_set.class_indices
if result[0][0]==1:
    print("This person is Thin")
else:
       print("This person is Fat")