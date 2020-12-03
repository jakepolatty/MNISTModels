import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from sklearn.svm import SVC
import helpers.helper_funcs as helpers


'''
L9
'''
def get_untrained_l9_all_digit_model(input_shape, x_train):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

def get_trained_l9_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l8_all_digit_model(input_shape, inputs)
    #model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)

    learning_rate_reduction = ReduceLROnPlateau(monitor="val_acc", patience=3, verbose=1, factor=0.5, min_lr=0.0001)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
        )
    datagen.fit(inputs)

    model.fit_generator(datagen.flow(inputs, labels, batch_size=64),steps_per_epoch=inputs.shape[0] // 64,epochs=epochs,verbose=1, callbacks=[learning_rate_reduction])
    return model

'''
L8
'''
def get_untrained_l8_all_digit_model(input_shape, x_train):
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

def get_trained_l8_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l8_all_digit_model(input_shape, inputs)
    #model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
        )
    datagen.fit(inputs)

    model.fit_generator(datagen.flow(inputs, labels, batch_size=32),steps_per_epoch=inputs.shape[0] // 32,epochs=3*epochs,verbose=1)
    return model


'''
L7
'''

def get_untrained_l7_all_digit_model(input_shape):
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

def get_trained_l7_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l7_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model




'''
L6
'''
def get_untrained_l6_all_digit_model(inputs):
    model = Sequential()
 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputs.shape[1:]))
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

def get_trained_l6_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l6_all_digit_model(inputs)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model


'''
L5
'''
def get_untrained_l5_all_digit_model(inputs):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputs.shape[1:]))
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

def get_trained_l5_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l5_all_digit_model(inputs)
    model.fit(x=inputs,y=labels, batch_size=32, epochs=epochs, verbose=1)
    return model


'''
L4
'''
def get_untrained_l4_all_digit_model(input_shape):
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

def get_trained_l4_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l4_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

def get_untrained_l3_all_digit_model(input_shape):
    # input_shape = (28, 28, 1)

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

def get_trained_l3_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l3_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

'''
L2
'''
def get_untrained_l2_all_digit_model(input_shape):
    # input_shape = (28, 28, 1)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(100, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

def get_trained_l2_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l2_all_digit_model(input_shape)
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

'''
Simple 10 digit model
'''
def get_untrained_l1_all_digit_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

def get_trained_l1_all_digit_model(inputs, labels, epochs=1):
    model = get_untrained_l1_all_digit_model()
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model

def get_untrained_l0_all_digit_model():

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    # model.add(Dense(100, activation=tf.nn.relu))
    # model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

def get_trained_l0_all_digit_model(inputs, labels, epochs=1):
    model = get_untrained_l1_all_digit_model()
    model.fit(x=inputs,y=labels, epochs=epochs, verbose=1)
    return model
