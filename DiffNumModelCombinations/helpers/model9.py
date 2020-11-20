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
def augment_2d(inputs, rotation=0, horizontal_flip=False, vertical_flip=False):
    """Apply additive augmentation on 2D data.

    # Arguments
      rotation: A float, the degree range for rotation (0 <= rotation < 180),
          e.g. 3 for random image rotation between (-3.0, 3.0).
      horizontal_flip: A boolean, whether to allow random horizontal flip,
          e.g. true for 50% possibility to flip image horizontally.
      vertical_flip: A boolean, whether to allow random vertical flip,
          e.g. true for 50% possibility to flip image vertically.

    # Returns
      input data after augmentation, whose shape is the same as its original.
    """
    if inputs.dtype != tf.float32:
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

    with tf.name_scope('augmentation'):
        shp = tf.shape(inputs)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if rotation > 0:
            angle_rad = rotation * 3.141592653589793 / 180.0
            angles = tf.random.uniform([batch_size], -angle_rad, angle_rad)
            f = tfa.image.transform_ops.angles_to_projective_transforms(angles,
                                                                 height, width)
            transforms.append(f)

        if horizontal_flip:
            coin = tf.less(tf.random.uniform([batch_size], 0, 1.0), 0.5)
            shape = [-1., 0., width, 0., 1., 0., 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

        if vertical_flip:
            coin = tf.less(tf.random.uniform([batch_size], 0, 1.0), 0.5)
            shape = [1., 0., 0., 0., -1., height, 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

    if transforms:
        f = tfa.image.transform_ops.compose_transforms(*transforms)
        inputs = tfa.image.transform_ops.transform(inputs, f, interpolation='BILINEAR')
    return inputs

def get_untrained_l9_all_digit_model(x_train):
    model = Sequential()
    model.add(Lambda(augment_2d,
                     input_shape=x_train.shape[1:],
                     arguments={'rotation': 8.0, 'horizontal_flip': True}))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


def get_trained_l9_all_digit_model(inputs, labels, input_shape, epochs=1):
    model = get_untrained_l9_all_digit_model(inputs)
    model.fit(x=inputs,y=labels, batch_size=32, epochs=epochs, verbose=1)
    return model





