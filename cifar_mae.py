#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project 37 SMAI Spring 2018

@author: abhishek
"""


import numpy as np

from keras.datasets import cifar10
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt


# ## Train on CIFAR-10 dataset
# 
# #### Load CIFAR 10 dataset.
# 
# 


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training data:")
print( "Number of examples: ", X_train.shape[0])
print( "Number of channels:",X_train.shape[3] )
print( "Image size:", X_train.shape[1], X_train.shape[2])
print("\n")
print( "Test data:")
print( "Number of examples:", X_test.shape[0])
print( "Number of channels:", X_test.shape[3])
print( "Image size:",X_test.shape[1], X_test.shape[2]) 


# #### Normalize the data.

print("mean before normalization:", np.mean(X_train)) 
print("std before normalization:", np.std(X_train))

mean=[0,0,0]
std=[0,0,0]
newX_train = np.ones(X_train.shape)
newX_test = np.ones(X_test.shape)
for i in range(3):
    mean[i] = np.mean(X_train[:,:,:,i])
    std[i] = np.std(X_train[:,:,:,i])
    
for i in range(3):
    newX_train[:,:,:,i] = X_train[:,:,:,i] - mean[i]
    newX_train[:,:,:,i] = newX_train[:,:,:,i] / std[i]
    newX_test[:,:,:,i] = X_test[:,:,:,i] - mean[i]
    newX_test[:,:,:,i] = newX_test[:,:,:,i] / std[i]
        
    
X_train = newX_train
X_test = newX_test

print("mean after normalization:", np.mean(X_train))
print("std after normalization:", np.std(X_train))


# #### Specify Training Parameters

batchSize = 32                    #-- Training Batch Size
num_classes = 10                  #-- Number of classes in CIFAR-10 dataset
num_epochs = 150                   #-- Number of epochs for training   
learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = 0.95            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch


img_rows, img_cols = 32, 32       #-- input image dimensions

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)


# #### Build a CNN network and train on dataset.

model = Sequential()                                                #-- Sequential container.

model.add(Convolution2D(32, (3, 3), padding='same',                                   #-- 6 outputs (6 filters), 3x3 convolution kernel
                        input_shape=( img_rows, img_cols, 3)))       #-- 3 input depth (RGB)
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
#model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows
model.add(Convolution2D(32, 3, 3))                                  #-- 16 outputs (16 filters), 3x3 convolution kernel
model.add(Activation('relu'))                                       #-- ReLU non-linearity
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows

model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))                                 #-- 64 outputs (64 filters), 5x5 convolution kernel
model.add(Activation('relu'))                                       #-- ReLU n on-linearity
model.add(Convolution2D(64, 3, 3))                                  #-- 64 outputs (64 filters), 5x5 convolution kernel
model.add(Activation('relu'))                                       #-- ReL U non-linearity

model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows
model.add(Dropout(0.25))

model.add(Flatten())                                                #-- eshapes a 3D tensor of 256x5x5 into 1D tensor of 16*5*5

model.add(Dense(512))                                               #-- 512 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dropout(0.5))
model.add(Dense(num_classes))                                       #-- 10 outputs fully connected layer (one for each class)
model.add(Activation('softmax'))                                    #-- converts the output to a log-probability. Useful for classification problems

print(model.summary())


# #### Compile and then train the network

sgd = SGD(lr=learningRate, decay = lr_weight_decay)
model.compile(loss='mean_absolute_error',
              optimizer='sgd',
              metrics=['accuracy'])

#-- switch verbose=0 if you get error "I/O operation from closed file"
history = model.fit(X_train, Y_train, batch_size=batchSize, epochs=num_epochs,
          verbose=1, shuffle=True, validation_data=(X_test, Y_test))


# #### Print the scores

#-- summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#-- summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#-- test the network
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# #### save the model

#cifar10_weights = model.get_weights()
#np.savez("cifar10_weights_new", cifar10_weights = cifar10_weights)

model.save('my_cifar_model_mae.h5') 

