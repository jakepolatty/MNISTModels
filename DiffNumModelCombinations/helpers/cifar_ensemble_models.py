import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization, Lambda, PReLU
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers, optimizers
from sklearn.svm import SVC
import helpers.helper_funcs as helpers


def get_l10_model():
    input_shape = (32, 32, 3)

    model=Sequential()
    model.add(Conv2D(64,(3,3),activation="linear"))
    model.add(PReLU())
    model.add(Conv2D(64,(3,3),activation="linear"))
    model.add(PReLU())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation="linear"))
    model.add(PReLU())
    model.add(Conv2D(128,(3,3),activation="linear"))
    model.add(PReLU())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation="linear"))
    model.add(PReLU())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="linear"))
    model.add(PReLU())
    model.add(Dense(10,activation="softmax"))
    
    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

def get_l9_model():
    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

def get_l8_model():
    input_shape = (32, 32, 3)

    baseMapNum = 32
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    #opt_rms = optimizers.RMSprop(lr=0.0005,decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model

def get_l7_model():
    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_l6_model():
    input_shape = (32, 32, 3)

    model = Sequential()
 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=max_norm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_l5_model():
    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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
                loss='categorical_crossentropy', 
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
                loss='categorical_crossentropy', 
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
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    return model

'''
Simple 10 digit model
'''
def get_l1_model():
    input_shape = (32, 32, 3)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten(input_shape=input_shape)) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(100, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    return model


