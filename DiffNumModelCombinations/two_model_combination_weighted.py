import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.cifar_models as models

def run_combinations(simple_model, complex_model, x_data, y_data):
    '''
    Attempt all confidence values in 0:0.1:1
    Store accuracy and time for each confidence value
    '''

    y_test_np = y_data.numpy()
    count_dicts = []
    coeffs = np.zeros((2, 10))

    s_model_probs = simple_model.predict(x_data)
    s_model_preds = np.argmax(s_model_probs, axis=1)
    unique, counts = np.unique(s_model_preds, return_counts=True)
    count_dicts.append(dict(zip(unique, counts)))

    c_model_probs = complex_model.predict(x_data)
    c_model_preds = np.argmax(c_model_probs, axis=1)
    unique, counts = np.unique(c_model_preds, return_counts=True)
    count_dicts.append(dict(zip(unique, counts)))

    for j in range(10):
        # Compute the number of times where the prediction matches the test output for the simple model
        class_count = len(np.where((s_model_preds == j) & (y_test_np == j))[0])
        coeffs[0][j] = class_count / count_dicts[0][j]
    for j in range(10):
        # Compute the number of times where the prediction matches the test output for the complex model
        class_count = len(np.where((c_model_preds == j) & (y_test_np == j))[0])
        coeffs[1][j] = class_count / count_dicts[1][j]

    conf_values = np.arange(0, 0.9, 0.1)
    conf_values = np.append(conf_values, np.arange(0.81, 0.91, 0.01))

    accuracies = []
    times = []
    simplePercents = []
    for conf_value in conf_values:
        accuracy, time, simplePercent = run_combined(simple_model, complex_model, x_data, y_data, conf_value, coeffs)
        accuracies.append(accuracy)
        times.append(time)
        simplePercents.append(simplePercent)
    
    return accuracies, times, simplePercents

def run_combined(simple_model, complex_model, inputs, labels, conf_value, coeffs):
    '''
    Runs both models on the input data with the input confidence value
    '''
    before = time.time()
    # -----------------------------------
    # All Simple
    simple_probs = simple_model.predict(inputs)
    simple_highest_probs = np.amax(simple_probs, axis=1)
    simple_preds = np.argmax(simple_probs, axis=1)

    for i in range(inputs.shape[0]):
        pred = simple_preds[i]
        simple_highest_probs[i] = simple_highest_probs[i] * coeffs[0][pred]


    # complex_probs = simple_model.predict(inputs)
    # complex_highest_probs = np.amax(simple_probs, axis=1)
    # complex_preds = np.argmax(simple_probs, axis=1)

    # for i in range(inputs.shape[0]):
    #     pred = complex_preds[i]
    #     complex_highest_probs[i] = complex_highest_probs[i] * coeffs[1][pred]

    # combined_preds = np.arange(inputs.shape[0])
    # simple_count = 0

    # for i in range(inputs.shape[0]):
    #     if simple_highest_probs[i] > complex_highest_probs[i]:
    #         combined_preds[i] = simple_preds[i]
    #         simple_count += 1
    #     else:
    #         combined_preds[i] = complex_preds[i]


    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    indices = [i for i in range(inputs.shape[0])]
    complex_indices = np.where(simple_highest_probs < conf_value, indices, None)
    complex_indices = complex_indices[complex_indices != np.array(None)] # remove None values
    complex_indices = np.asarray(complex_indices, dtype=np.int64)

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_preds = np.argmax(complex_model.predict(complex_inputs), axis=1)
    # -----------------------------------
    # Select simple
    simple_indices = np.where(simple_highest_probs >= conf_value, indices, None)
    simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
    simple_indices = np.asarray(simple_indices, dtype=np.int64)

    # ------------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, indices, simple_preds)
    np.put(combined_preds, complex_indices, complex_preds)

    simplePercent = simple_indices.shape[0] / (simple_indices.shape[0] + complex_indices.shape[0])
    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy(), time.time() - before, simplePercent

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()
    y_test = tf.squeeze(y_test)

    #train_and_save_models(x_train, y_train)

    print("Loading models...")
    # l0_model = tf.keras.models.load_model('models/l0_model')
    # l1_model = tf.keras.models.load_model('models/cifar/l1_model')
    # l2_model = tf.keras.models.load_model('models/cifar/l2_model')
    # l3_model = tf.keras.models.load_model('models/cifar/l3_model')
    # l4_model = tf.keras.models.load_model('models/cifar/l4_model')
    l7_model = tf.keras.models.load_model('models/cifar/l7_model')
    l8_model = tf.keras.models.load_model('models/cifar/l8_model')

    l7_times = []
    l8_times = []

    # for i in range(5):
    #     before_time = time.time()
    #     accuracy = l7_model.predict(x_test)
    #     #print("L1 Accuracy: ", l1_model.evaluate(x_test, y_test, verbose=0)[1])
    #     l7_times.append(time.time() - before_time)

    #     before_time = time.time()
    #     accuracy = l8_model.predict(x_test)
    #     #print("L2 Accuracy: ", l2_model.evaluate(x_test, y_test, verbose=0)[1])
    #     l8_times.append(time.time() - before_time)

    # print(l7_times)
    # print(l8_times)
    # print("L7 Time:", np.mean(l7_times[1:], axis=0))
    # print("L8 Time:", np.mean(l8_times[1:], axis=0))

    l7_l8_accuracies = []
    l7_l8_times = []
    l7_l8_percents = []

    for i in range(3):
        print("Run l7 l8... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l7_model, l8_model, x_test, y_test)
        l7_l8_accuracies.append(accuracies)
        l7_l8_times.append(times)
        l7_l8_percents.append(simplePercents)

    print("L7 L8 Accuracies:", np.mean(l7_l8_accuracies, axis=0))
    print("L7 L8 Times:", np.mean(l7_l8_times, axis=0))
    print("L7 L8 Simple Percents", np.mean(l7_l8_percents, axis=0))

if __name__ == '__main__':
    main()