import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.cifar_models as models

def run_combinations(simple_model, complex_model, most_complex_model, x_data, y_data):
    '''
    Attempt all confidence values in 0:0.1:1
    Store accuracy and time for each confidence value
    '''
    conf_values = np.arange(0, 1.1, 0.1)

    all_conf_values = []
    accuracies = []
    times = []
    simple_percents = []
    complex_percents = []
    for conf_value_1 in conf_values:
        for conf_value_2 in conf_values:
            current_conf_values = [conf_value_1, conf_value_2]
            all_conf_values.append(current_conf_values)
            print("Confidence Values", conf_value_1, conf_value_2)
            accuracy, time, simple_percent, complex_percent = run_combined(simple_model, complex_model, most_complex_model,\
                 x_data, y_data, conf_value_1, conf_value_2)

            print("accuracy:", accuracy, "time:", time)
            accuracies.append(accuracy)
            times.append(time)
            simple_percents.append(simple_percent)
            complex_percents.append(complex_percent)
    
    return all_conf_values, accuracies, times, simple_percents, complex_percents

def run_combined(simple_model, complex_model, most_complex_model, inputs, labels, conf_value_1, conf_value_2):
    '''
    Runs all three models on the input data with the input confidence value
    '''
    before = time.time()

    indices = [i for i in range(inputs.shape[0])]
    # -----------------------------------
    # All Simple
    simple_probs = simple_model.predict(inputs)
    simple_highest_probs = np.amax(simple_probs, axis=1)

    simple_indices = np.where(simple_highest_probs >= conf_value_1, indices, None)
    simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
    simple_indices = np.asarray(simple_indices, dtype=np.int64)

    print('simple indices', simple_indices.shape)

    reduced_simple_probs = np.take(simple_probs, simple_indices, axis=0)
    simple_preds = reduced_simple_probs.argmax(axis=1)
    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    complex_indices = np.zeros(0)
    complex_indices = np.where(simple_highest_probs < conf_value_1, indices, None)
    complex_indices = complex_indices[complex_indices != np.array(None)] # remove None values
    complex_indices = np.asarray(complex_indices, dtype=np.int64)

    print('complex indices', complex_indices.shape)

    complex_inputs = np.take(inputs, complex_indices, axis=0)
    complex_highest_probs = None
    if complex_inputs.shape[0] == 0:
        complex_preds = []
    else:
        complex_probs = complex_model.predict(complex_inputs)
        complex_highest_probs = np.amax(complex_probs, axis=1)
        # reduced_complex_probs = np.where(complex_highest_probs >= conf_value_2, complex_probs, None)
        # reduced_complex_probs = reduced_complex_probs[reduced_complex_probs != np.array(None)] # remove None values
        # reduced_complex_probs = np.asarray(reduced_complex_probs, dtype=np.float64)
        
        complex_preds = np.argmax(complex_probs, axis=1)

    # -----------------------------------
    # Get most complex inputs
    most_complex_indices = np.zeros(0)
    most_complex_preds = []
    if not (complex_highest_probs is None):
        most_complex_indices = np.where(complex_highest_probs < conf_value_2, complex_indices, None)
        most_complex_indices = most_complex_indices[most_complex_indices != np.array(None)] # remove None values
        most_complex_indices = np.asarray(most_complex_indices, dtype=np.int64)

        print('most complex indices', most_complex_indices.shape)

        most_complex_inputs = np.take(inputs, most_complex_indices, axis=0)
        if most_complex_inputs.shape[0] != 0:
            most_complex_probs = most_complex_model.predict(most_complex_inputs)
            most_complex_highest_probs = np.amax(most_complex_probs, axis=1)
            most_complex_preds = np.argmax(most_complex_probs, axis=1)

    # -----------------------------------
    # Reorganize preds
    combined_preds = np.arange(inputs.shape[0])

    np.put(combined_preds, simple_indices, simple_preds)
    if complex_indices.shape[0] != 0:
        np.put(combined_preds, complex_indices, complex_preds)
    if most_complex_indices.shape[0] != 0:
        np.put(combined_preds, most_complex_indices, most_complex_preds)

    simple_percent = simple_indices.shape[0] / (simple_indices.shape[0] + complex_indices.shape[0] + most_complex_indices.shape[0])
    complex_percent = complex_indices.shape[0] / (simple_indices.shape[0] + complex_indices.shape[0] + most_complex_indices.shape[0])

    return tf.reduce_mean(tf.cast(tf.equal(combined_preds, labels), tf.float32)).numpy(), time.time() - before, simple_percent, complex_percent

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
    l5_model = tf.keras.models.load_model('models/cifar/l5_model')
    l6_model = tf.keras.models.load_model('models/cifar/l6_model')
    l7_model = tf.keras.models.load_model('models/cifar/l7_model')
    l8_model = tf.keras.models.load_model('models/cifar/l8_model')

    l_accuracies = []
    l_times = []
    l_simple_percents = []
    l_complex_percents = []

    for i in range(2):
        print("Run l6 l7 l8... #" + str(i))
        all_conf_values, accuracies, times, simple_percents, complex_percents = run_combinations(l5_model, l7_model, l8_model, x_test, y_test)
        l_accuracies.append(accuracies)
        l_times.append(times)
        l_simple_percents.append(simple_percents)
        l_complex_percents.append(complex_percents)

    print("Accuracies:", np.mean(l_accuracies, axis=0))
    print("Times:", np.mean(l_times, axis=0))
    print("Simple Percents:", np.mean(l_simple_percents, axis=0))
    print("Complex Percents:", np.mean(l_complex_percents, axis=0))

if __name__ == '__main__':
    main()