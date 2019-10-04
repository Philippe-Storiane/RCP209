# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 05:23:17 2019

@author: a179415
"""

import os

os.chdir("C:/Users/philippe/Documents/CNAM/RCP 209")

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# data preparation
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float64')
X_train /= 255
X_train = X_train.reshape(( 60000, 784))
X_train_conv = X_train.reshape( ( 60000, 28, 28, 1))

X_test = X_test.astype('float64')
X_test /= 255
X_test = X_test.reshape( ( 10000, 784 ))
X_test_conv = X_test.reshape( ( 10000, 28, 28, 1))

y_train = to_categorical( y_train )
y_test= to_categorical( y_test )

learning_rate = 0.5
optimizer = SGD( learning_rate )

def saveModel( model, savename ):
    model_yaml = model.to_yaml()
    model_filename = savename + ".yaml"
    model_weights_filename = savename+".h5"
    with open( model_filename, "w") as yaml_file:
        yaml_file.write( model_yaml )
        print( "YAML model ", model_filename, "saved to disk")
    model.save_weights( model_weights_filename)
    print("Weights", model_weights_filename, "saved to disk")
    
    
# logistioc regression
logistic_epochs = 10
logistic_batch_size = 300
input = Input(shape=(784,))
classifier_layer = Dense( 10, activation='softmax', name='fc1')( input )
logistic = Model(inputs= input, outputs = classifier_layer)
logistic.compile( loss='categorical_crossentropy', optimizer = optimizer, metrics=[ 'accuracy' ])

logistic_history = logistic.fit(
        X_train,
        y_train,
        batch_size = logistic_batch_size,
        epochs= logistic_epochs,
        verbose = 1,
        validation_data = ( X_test, y_test))
scores = logistic.evaluate( X_test, y_test )
print("%s: %.2f%%" % (logistic.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (logistic.metrics_names[1], scores[1]*100))




    
# perceptron
perceptron_epochs = 10
perceptron_batch_size = 300
input = Input(shape=(784,))
hidden_layer = Dense( 100, activation = 'sigmoid')(input)
classifier_layer = Dense( 10, activation='softmax', name='fc1')( hidden_layer )
perceptron = Model(inputs= input, outputs = classifier_layer)
perceptron.compile( loss='categorical_crossentropy', optimizer = optimizer, metrics=[ 'accuracy' ])

perceptron_history = perceptron.fit(
        X_train,
        y_train,
        batch_size = perceptron_batch_size,
        epochs= perceptron_epochs,
        verbose = 1,
        validation_data = ( X_test, y_test))
scores = perceptron.evaluate( X_test, y_test )
print("%s: %.2f%%" % (perceptron.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (perceptron.metrics_names[1], scores[1]*100))
saveModel( perceptron, "perceptron")
# convnet
convnet_epochs = 50
convnet_batch_size = 300

input = Input(shape=(28,28,1))
conv_layer1 = Conv2D(32, (5,5), activation='sigmoid', padding='same', name='conv1')(input)
max_layer_1 = MaxPooling2D((2,2), padding='same', name='maxPooling1')( conv_layer1 )
conv_layer2 = Conv2D(64, (5,5), activation='sigmoid', padding='same', name='conv2')( max_layer_1 )
max_layer_2 = MaxPooling2D((2,2), padding='same', name='maxPooling2')( conv_layer2 )
flatten = Flatten()( max_layer_2 )
dense_layer1 = Dense( 100, activation = 'sigmoid', name='Dense1')(flatten)
classifier_layer = Dense( 10, activation='softmax', name='classifier')( dense_layer1 )
convnet = Model( inputs= input, outputs = classifier_layer)
convnet.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=[ 'accuracy' ])

convnet_history = convnet.fit(
        X_train_conv,
        y_train,
        batch_size = convnet_batch_size,
        epochs= convnet_epochs,
        verbose = 1,
        validation_data = ( X_test_conv, y_test))
scores = convnet.evaluate( X_test_conv, y_test )
print("%s: %.2f%%" % (convnet.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (convnet.metrics_names[1], scores[1]*100))
saveModel( convnet, "convnet")




