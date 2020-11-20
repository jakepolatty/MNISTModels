import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from tensorflow.keras.layers import BatchNormalization, Lambda, ZeroPadding2D, AveragePooling2D, Add, Input, Convolution2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from sklearn.svm import SVC
import helpers.helper_funcs as helpers


batch_size = 128
nb_classes = 10
data_augmentation = False
n = 4  # depth = 6*n + 4
k = 4  # widen factor

# the CIFAR10 images are 32x32 RGB with 10 labels
img_rows, img_cols = 32, 32
img_channels = 3


def bottleneck(incoming, count, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    outgoing = wide_basic(incoming, nb_in_filters, nb_out_filters, dropout, subsample)
    for i in range(1, count):
        outgoing = wide_basic(outgoing, nb_out_filters, nb_out_filters, dropout, subsample=(1, 1))

    return outgoing


def wide_basic(incoming, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    nb_bottleneck_filter = nb_out_filters

    if nb_in_filters == nb_out_filters:
        # conv3x3
        y = BatchNormalization(axis=1)(incoming)
        y = Activation('relu')(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_bottleneck_filter, kernel_size=3,
                          strides=subsample, padding='valid')(y)

        # conv3x3
        y = BatchNormalization(axis=1)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_bottleneck_filter, kernel_size=3,
                          strides=(1, 1), padding='valid')(y)

        return Add()([incoming, y])

    else:  # Residual Units for increasing dimensions
        # common BN, ReLU
        shortcut = BatchNormalization(axis=1)(incoming)
        shortcut = Activation('relu')(shortcut)

        # conv3x3
        y = ZeroPadding2D((1, 1))(shortcut)
        y = Convolution2D(nb_bottleneck_filter, kernel_size=3,
                          strides=subsample, padding='valid')(y)

        # conv3x3
        y = BatchNormalization(axis=1)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Convolution2D(nb_out_filters, kernel_size=3,
                          strides=(1, 1), padding='valid')(y)

        # shortcut
        shortcut = Convolution2D(nb_out_filters, kernel_size=1,
                                 strides=subsample, padding='same')(shortcut)

        return Add()([shortcut, y])


'''
L10
'''

def get_untrained_l10_all_digit_model(input_shape):
    img_input = Input(shape=(img_channels, img_rows, img_cols))

    # one conv at the beginning (spatial size: 32x32)
    x = ZeroPadding2D((1, 1))(img_input)
    x = Convolution2D(16, kernel_size=3)(x)

    # Stage 1 (spatial size: 32x32)
    x = bottleneck(x, n, 16, 16 * k, dropout=0.3, subsample=(1, 1))
    # Stage 2 (spatial size: 16x16)
    x = bottleneck(x, n, 16 * k, 32 * k, dropout=0.3, subsample=(2, 2))
    # Stage 3 (spatial size: 8x8)
    x = bottleneck(x, n, 32 * k, 64 * k, dropout=0.3, subsample=(2, 2))

    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8), strides=(1, 1))(x)
    x = Flatten()(x)
    preds = Dense(nb_classes, activation='softmax')(x)

    model = Model(input=img_input, output=preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_trained_l10_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l10_all_digit_model(inputs)

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(inputs)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(inputs, labels,
                                     batch_size=batch_size),
                        samples_per_epoch=inputs.shape[0],
                        nb_epoch=epochs)

    return model








