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
    thresholds = np.arange(0.2, 1.0, 0.04)
    thresholds = np.append(thresholds, np.arange(0.965, 1.0, 0.005))

    all_conf_values = []
    accuracies = []
    times = []
    simple_percents = []
    complex_percents = []
    threshold_counts = []
    for threshold in thresholds:
        all_conf_values.append(threshold)
        print("Threshold", threshold)
        accuracy, time, simple_percent, threshold_count = run_combined(simple_model, complex_model,\
             x_data, y_data, threshold)

        print("accuracy:", accuracy, "time:", time)
        accuracies.append(accuracy)
        times.append(time)
        simple_percents.append(simple_percent)
        threshold_counts.append(threshold_count)
    
    return accuracies, times, simple_percents

def run_combined(simple_model, complex_model, inputs, labels, threshold):
    '''
    Runs both models on the input data with the input confidence value
    '''
    before = time.time()
    # -----------------------------------
    # All Simple
    simple_probs = simple_model.predict(inputs)
    simple_highest_probs = np.amax(simple_probs, axis=1)

    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    indices = [i for i in range(inputs.shape[0])]
    complex_indices = []
    threshold_count = 0
    for i in range(inputs.shape[0]):
        second_highest = np.partition(simple_probs[i], -2)[-2]
        diff = simple_highest_probs[i] - second_highest
        if diff < threshold:
            complex_indices.append(i)

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_preds = np.argmax(complex_model.predict(complex_inputs), axis=1)
    # -----------------------------------
    # Select simple
    simple_preds = np.argmax(simple_probs, axis=1)

    # ------------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, indices, simple_preds)
    np.put(combined_preds, complex_indices, complex_preds)

    simplePercent = 1 - (len(complex_indices) / len(indices))

    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy(), time.time() - before, simplePercent, threshold_count

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()
    y_test = tf.squeeze(y_test)

    # train_and_save_models(x_train, y_train)

    print("Loading models...")
    # l0_model = tf.keras.models.load_model('models/l0_model')
    l1_model = tf.keras.models.load_model('models/cifar/l1_model')
    l2_model = tf.keras.models.load_model('models/cifar/l2_model')
    l3_model = tf.keras.models.load_model('models/cifar/l3_model')
    l4_model = tf.keras.models.load_model('models/cifar/l4_model')

    # before_time = time.time()
    # accuracy = l0_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l1_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l2_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l3_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    # before_time = time.time()
    # accuracy = l4_model.evaluate(x_test, y_test)
    # print("Time", time.time() - before_time)

    l1_l2_accuracies = []
    l1_l3_accuracies = []
    l2_l3_accuracies = []
    l1_l2_times = []
    l1_l3_times = []
    l2_l3_times = []
    l1_l2_percents = []
    l1_l3_percents = []
    l2_l3_percents = []

    l1_l4_accuracies = []
    l2_l4_accuracies = []
    l3_l4_accuracies = []
    l1_l4_times = []
    l2_l4_times = []
    l3_l4_times = []
    l1_l4_percents = []
    l2_l4_percents = []
    l3_l4_percents = []

    for i in range(5):
        print("Run l1 l2... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l1_model, l2_model, x_test, y_test)
        l1_l2_accuracies.append(accuracies)
        l1_l2_times.append(times)
        l1_l2_percents.append(simplePercents)

        print("Run l1 l3... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l1_model, l3_model, x_test, y_test)
        l1_l3_accuracies.append(accuracies)
        l1_l3_times.append(times)
        l1_l3_percents.append(simplePercents)

        print("Run l2 l3... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l2_model, l3_model, x_test, y_test)
        l2_l3_accuracies.append(accuracies)
        l2_l3_times.append(times)
        l2_l3_percents.append(simplePercents)

        print("Run l1 l4... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l1_model, l4_model, x_test, y_test)
        l1_l4_accuracies.append(accuracies)
        l1_l4_times.append(times)
        l1_l4_percents.append(simplePercents)

        print("Run l2 l4... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l2_model, l4_model, x_test, y_test)
        l2_l4_accuracies.append(accuracies)
        l2_l4_times.append(times)
        l2_l4_percents.append(simplePercents)

        print("Run l3 l4... #" + str(i))
        accuracies, times, simplePercents = run_combinations(l3_model, l4_model, x_test, y_test)
        l3_l4_accuracies.append(accuracies)
        l3_l4_times.append(times)
        l3_l4_percents.append(simplePercents)

    print("L1 L2 Accuracies:", np.mean(l1_l2_accuracies, axis=0))
    print("L1 L2 Times:", np.mean(l1_l2_times, axis=0))
    print("L1 L2 Simple Percents", np.mean(l1_l2_percents, axis=0))
    print("L1 L3 Accuracies:", np.mean(l1_l3_accuracies, axis=0))
    print("L1 L3 Times:", np.mean(l1_l3_times, axis=0))
    print("L1 L3 Simple Percents", np.mean(l1_l3_percents, axis=0))
    print("L2 L3 Accuracies:", np.mean(l2_l3_accuracies, axis=0))
    print("L2 L3 Times:", np.mean(l2_l3_times, axis=0))
    print("L2 L3 Simple Percents", np.mean(l2_l3_percents, axis=0))

    print("L1 L4 Accuracies:", np.mean(l1_l4_accuracies, axis=0))
    print("L1 L4 Times:", np.mean(l1_l4_times, axis=0))
    print("L1 L4 Simple Percents", np.mean(l1_l4_percents, axis=0))
    print("L2 L4 Accuracies:", np.mean(l2_l4_accuracies, axis=0))
    print("L2 L4 Times:", np.mean(l2_l4_times, axis=0))
    print("L2 L4 Simple Percents", np.mean(l2_l4_percents, axis=0))
    print("L3 L4 Accuracies:", np.mean(l3_l4_accuracies, axis=0))
    print("L3 L4 Times:", np.mean(l3_l4_times, axis=0))
    print("L3 L4 Simple Percents", np.mean(l3_l4_percents, axis=0))

if __name__ == '__main__':
    main()