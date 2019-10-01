# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 08:00:56 2019

@author: a179415
"""

from keras.datasets import mnist

import numpy as np
import math


(  ( X_train, Y_train ), ( X_test, Y_test ) ) = mnist.load_data()

X_train = X_train.reshape(60000, 784 )
X_test = X_test.reshape(10000, 784 )
X_train = X_train.astype( 'float64')
X_test = X_test.astype( 'float64')

X_train /= 255
X_test /= 255




# one hot encoding
from keras.utils import np_utils
Y_train = np_utils.to_categorical( Y_train )
Y_test = np_utils.to_categorical( Y_test )

# graduiant descent
K = 10
L = 100
N = X_train.shape[0]
d = X_train.shape[1]
eta = 1e-1
Wh = np.random.standard_normal( size = (d,L) )  *  eta # / math.sqrt( N )
Bh = np.random.standard_normal( size = (1, L) ) * eta # / math.sqrt( N )
Wy = np.random.standard_normal( size = ( L, K) ) * eta # / math.sqrt( N )
By = np.random.standard_normal( size = (1, K) ) *  eta # / math.sqrt( N )
eta = 1e-1
numEpoc = 250
batch_size = 100
nb_batches = int(float( N )/ batch_size)
gradWh = np.zeros( ( d, L ))
gradWh = np.zeros( ( 1, L ))
gradWy = np.zeros( ( L, K ))
gradBy = np.zeros( ( ( 1, K)))


def sigmoid( u ):
    return 1.0 / ( 1.0 + np.exp( -1.0 * u ))

def softmax( s ):
    e = np.exp( s )
    return ( e.T / ( np.sum( e, axis = 1).T)).T

def forward( batch, Wh, Bh, Wy, By):
    u = np.dot(batch, Wh) + Bh
    h = sigmoid( u )
    v = np.dot( h, Wy ) + By
    return ( h, softmax( v ))


def accuracy(Wh, Bh, Wy, By, images, labels):
    ( h, pred )= forward( images, Wh, Bh, Wy, By)
    return np.where( pred.argmax( axis = 1) != labels.argmax( axis = 1), 0., 1.).mean() * 100.0

for epoch in range( numEpoc ):
    for ex in range ( nb_batches):
         x = X_train[ ex * batch_size : (ex + 1 ) * batch_size]
         y_labels = Y_train[ ex * batch_size: ( ex + 1 ) * batch_size]
         ( h, y ) = forward( x, Wh, Bh, Wy, By)
         gradWy = np.dot( h.T, ( y - y_labels)) * ( 1.0 / batch_size )
         gradBy = np.sum( y - y_labels, axis=0) * ( 1.0 / batch_size )
         deltah = np.dot( y - y_labels, Wy.T ) * ( h * ( 1 - h )) 
         gradWh = np.dot( x.T, deltah) * ( 1.0 / batch_size)
         gradBh = np.sum( deltah, axis = 0 ) * ( 1.0 / batch_size )
         Wh = Wh -eta * gradWh
         Bh = Bh - eta * gradBh
         Wy = Wy -eta * gradWy
         By = By - eta * gradBy
    acc = accuracy( Wh, Bh, Wy, By, X_test, Y_test)
    print( str(epoch) + " Accuray: " + str( acc ))
        




