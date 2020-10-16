import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from sklearn.svm import SVC
import helpers.helper_funcs as helpers

def get_l4_model():
    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(32, input_shape=input_shape, kernel_size=(2,2), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(64, kernel_size=(2, 2), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(128, kernel_size=(2, 2), padding='valid'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100,))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10,activation=tf.nn.softmax))


    # Compile the model
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def get_l3_model():
    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(32, input_shape=input_shape, kernel_size=(5,5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(64, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(100,))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10,activation=tf.nn.softmax))


    # Compile the model
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def get_l2_model():
    input_shape = (32, 32, 3)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(32, input_shape=input_shape, kernel_size=(5,5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

'''
Simple 10 digit model
'''
def get_l1_model():
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(100, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

