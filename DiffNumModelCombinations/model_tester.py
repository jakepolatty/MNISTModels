import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.cifar_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    y_test2 = tf.squeeze(y_test)

    #train_and_save_models(x_train, y_train)

    print("Loading models...")
    # l0_model = tf.keras.models.load_model('models/l0_model')
    # l1_model = tf.keras.models.load_model('models/cifar/l1_model')
    # l2_model = tf.keras.models.load_model('models/cifar/l2_model')
    # l3_model = tf.keras.models.load_model('models/cifar/l3_model')
    # l4_model = tf.keras.models.load_model('models/cifar/l4_model')
    # l5_model = tf.keras.models.load_model('models/cifar/l5_model')
    # l6_model = tf.keras.models.load_model('models/cifar/l6_model')
    # l7_model = tf.keras.models.load_model('models/cifar/l7_model')
    # l8_model = tf.keras.models.load_model('models/cifar/l8_model')
    # l9_model = tf.keras.models.load_model('models/cifar/l9_model')
    # l10_model = tf.keras.models.load_model('models/cifar/l10_model')

    l1_model = tf.keras.models.load_model('models/mnist/l1_model')
    l2_model = tf.keras.models.load_model('models/mnist/l2_model')
    l3_model = tf.keras.models.load_model('models/mnist/l3_model')
    l4_model = tf.keras.models.load_model('models/mnist/l4_model')
    l5_model = tf.keras.models.load_model('models/mnist/l5_model')
    l6_model = tf.keras.models.load_model('models/mnist/l6_model')
    l7_model = tf.keras.models.load_model('models/mnist/l7_model')
    l8_model = tf.keras.models.load_model('models/mnist/l8_model')
    # l9_model = tf.keras.models.load_model('models/mnist/l9_model')
    # l10_model = tf.keras.models.load_model('models/mnist/l10_model')

    l1_time = []
    l2_time = []
    l3_time = []
    l4_time = []
    l5_time = []
    l6_time = []
    l7_time = []
    l8_time = []
    l9_time = []
    l10_time = []

    for i in range(5):
        before_time = time.time()
        #accuracy = l1_model.predict(x_test)
        print("L1 Accuracy: ", l1_model.evaluate(x_test, y_test2, verbose=0)[1])
        l1_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l2_model.predict(x_test)
        print("L2 Accuracy: ", l2_model.evaluate(x_test, y_test2, verbose=0)[1])
        l2_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l3_model.predict(x_test)
        print("L3 Accuracy: ", l3_model.evaluate(x_test, y_test2, verbose=0)[1])
        l3_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l4_model.predict(x_test)
        print("L4 Accuracy: ", l4_model.evaluate(x_test, y_test2, verbose=0)[1])
        l4_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l3_model.predict(x_test)
        print("L5 Accuracy: ", l5_model.evaluate(x_test, y_test2, verbose=0)[1])
        l5_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l4_model.predict(x_test)
        print("L6 Accuracy: ", l6_model.evaluate(x_test, y_test2, verbose=0)[1])
        l6_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l3_model.predict(x_test)
        print("L7 Accuracy: ", l7_model.evaluate(x_test, y_test2, verbose=0)[1])
        l7_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l4_model.predict(x_test)
        print("L8 Accuracy: ", l8_model.evaluate(x_test, y_test2, verbose=0)[1])
        l8_time.append(time.time() - before_time)

        # before_time = time.time()
        # #accuracy = l3_model.predict(x_test)
        # print("L9 Accuracy: ", l9_model.evaluate(x_test, y_test2, verbose=0)[1])
        # l9_time.append(time.time() - before_time)

        # before_time = time.time()
        # #accuracy = l4_model.predict(x_test)
        # print("L10 Accuracy: ", l10_model.evaluate(x_test, y_test2, verbose=0)[1])
        # l10_time.append(time.time() - before_time)

    print("L1 Time:", np.mean(l1_time[1:], axis=0), l1_time)
    print("L2 Time:", np.mean(l2_time[1:], axis=0), l2_time)
    print("L3 Time:", np.mean(l3_time[1:], axis=0), l3_time)
    print("L4 Time:", np.mean(l4_time[1:], axis=0), l4_time)
    print("L5 Time:", np.mean(l5_time[1:], axis=0), l5_time)
    print("L6 Time:", np.mean(l6_time[1:], axis=0), l6_time)
    print("L7 Time:", np.mean(l3_time[1:], axis=0), l7_time)
    print("L8 Time:", np.mean(l4_time[1:], axis=0), l8_time)
    # print("L9 Time:", np.mean(l5_time[1:], axis=0), l9_time)
    # print("L10 Time:", np.mean(l6_time[1:], axis=0), l10_time)


if __name__ == '__main__':
    main()