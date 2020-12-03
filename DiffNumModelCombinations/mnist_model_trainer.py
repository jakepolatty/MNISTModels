import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.mnist_models as models

def train_and_save_models(x_train, y_train):
    # Train each of the models 
    input_shape = (28, 28, 1)

    # No hidden layers (printistic regression)
    # l0_model = models.get_trained_l0_all_digit_model(x_train, y_train)
    # l0_model.save('models/mnist/l0_model')

    # 1 hidden layer
    # l1_model = models.get_trained_l1_all_digit_model(x_train, y_train, epochs=10)
    # l1_model.save('models/mnist/l1_model')

    # 1 conv layer, 1 hidden layer
    # l2_model = models.get_trained_l2_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l2_model.save('models/mnist/l2_model')

    # 2 conv layers, 1 hidden layer
    # l3_model = models.get_trained_l3_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l3_model.save('models/mnist/l3_model')

    # 3 conv layers, 1 hidden layer
    # l4_model = models.get_trained_l4_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l4_model.save('models/mnist/l4_model')

    # 4 conv layers, 1 hidden layer
    # l5_model = models.get_trained_l5_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l5_model.save('models/mnist/l5_model')

    # 6 small conv layers, 1 hidden layer
    # l6_model = models.get_trained_l6_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l6_model.save('models/mnist/l6_model')

    # 6 large conv layers, 1 hidden layers
    # l7_model = models.get_trained_l7_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l7_model.save('models/mnist/l7_model')

    # 6 conv layers with default data augmentation
    # l8_model = models.get_trained_l8_all_digit_model(x_train, y_train, input_shape, epochs=10)
    # l8_model.save('models/mnist/l8_model')

    # 5 conv layers with data augmentation and learning rate reduction
    # l9_model = models.get_trained_l9_all_digit_model(x_train, y_train, input_shape, epochs=20)
    # l9_model.save('models/mnist/l9_model')

    # 5 conv layers with data augmentation and PReLU layers
    l10_model = models.get_trained_l10_all_digit_model(x_train, y_train, input_shape, epochs=10)
    l10_model.save('models/mnist/l10_model')

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    num_classes = 10
    #y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    train_and_save_models(x_train, y_train)

if __name__ == '__main__':
    main()