import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.cifar_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()
    y_test = tf.squeeze(y_test)

    #train_and_save_models(x_train, y_train)

    print("Loading models...")
    # l0_model = tf.keras.models.load_model('models/l0_model')
    l1_model = tf.keras.models.load_model('models/cifar/l1_model')
    l2_model = tf.keras.models.load_model('models/cifar/l2_model')
    l3_model = tf.keras.models.load_model('models/cifar/l3_model')
    l4_model = tf.keras.models.load_model('models/cifar/l4_model')
    l5_model = tf.keras.models.load_model('models/cifar/l5_model')
    l6_model = tf.keras.models.load_model('models/cifar/l6_model')

    l1_time = []
    l2_time = []
    l3_time = []
    l4_time = []
    l5_time = []
    l6_time = []

    for i in range(5):
        before_time = time.time()
        #accuracy = l1_model.predict(x_test)
        print("L1 Accuracy: ", l1_model.evaluate(x_test, y_test, verbose=0)[1])
        l1_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l2_model.predict(x_test)
        print("L2 Accuracy: ", l2_model.evaluate(x_test, y_test, verbose=0)[1])
        l2_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l3_model.predict(x_test)
        print("L3 Accuracy: ", l3_model.evaluate(x_test, y_test, verbose=0)[1])
        l3_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l4_model.predict(x_test)
        print("L4 Accuracy: ", l4_model.evaluate(x_test, y_test, verbose=0)[1])
        l4_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l3_model.predict(x_test)
        print("L5 Accuracy: ", l5_model.evaluate(x_test, y_test, verbose=0)[1])
        l5_time.append(time.time() - before_time)

        before_time = time.time()
        #accuracy = l4_model.predict(x_test)
        print("L6 Accuracy: ", l6_model.evaluate(x_test, y_test, verbose=0)[1])
        l6_time.append(time.time() - before_time)

    print("L1 Time:", np.mean(l1_time[1:], axis=0), l1_time)
    print("L2 Time:", np.mean(l2_time[1:], axis=0), l2_time)
    print("L3 Time:", np.mean(l3_time[1:], axis=0), l3_time)
    print("L4 Time:", np.mean(l4_time[1:], axis=0), l4_time)
    print("L5 Time:", np.mean(l5_time[1:], axis=0), l5_time)
    print("L6 Time:", np.mean(l6_time[1:], axis=0), l6_time)


if __name__ == '__main__':
    main()