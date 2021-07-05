import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
import helpers.cifar_models as models

def run_combinations(simple_model_outputs, complex_model_outputs, y_data):
    '''
    Attempt all confidence values in 0:0.1:1
    Store accuracy and time for each confidence value
    '''
    conf_values = np.arange(0.01, 1.01, 0.01)

    accuracies = []
    times = []
    simplePercents = []
    for conf_value in conf_values:
        accuracy, time, simplePercent = run_combined(simple_model_outputs, complex_model_outputs, y_data, conf_value)
        accuracies.append(accuracy)
        times.append(time)
        simplePercents.append(simplePercent)
    
    return accuracies

def run_combined(simple_model_outputs, complex_model_outputs, labels, conf_value):
    '''
    Runs both models on the input data with the input confidence value
    '''
    before = time.time()
    # -----------------------------------
    # All Simple
    simple_probs = simple_model_outputs
    simple_highest_probs = np.amax(simple_probs, axis=1)

    # -----------------------------------
    # Complex predictions: get inputs of complex predictions
    indices = [i for i in range(labels.shape[0])]
    complex_indices = np.where(simple_highest_probs < conf_value, indices, None)
    complex_indices = complex_indices[complex_indices != np.array(None)] # remove None values
    complex_indices = np.asarray(complex_indices, dtype=np.int64)

    complex_probs = np.take(complex_model_outputs, complex_indices, axis=0)
    if complex_probs.shape[0] == 0:
        complex_preds = []
    else:
        complex_preds = np.argmax(complex_probs, axis=1)
    # -----------------------------------
    # Select simple
    simple_indices = np.where(simple_highest_probs >= conf_value, indices, None)
    simple_indices = simple_indices[simple_indices != np.array(None)] # remove None values
    simple_indices = np.asarray(simple_indices, dtype=np.int64)

    simple_preds = np.argmax(simple_probs, axis=1)

    # ------------------------------------
    # Reorganize preds
    combined_preds = np.arange(labels.shape[0])

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
    l1_model = tf.keras.models.load_model('models/cifar/l1_model')
    l2_model = tf.keras.models.load_model('models/cifar/l2_model')
    l3_model = tf.keras.models.load_model('models/cifar/l3_model')
    l4_model = tf.keras.models.load_model('models/cifar/l4_model')
    l5_model = tf.keras.models.load_model('models/cifar/l5_model')
    l6_model = tf.keras.models.load_model('models/cifar/l6_model')
    l7_model = tf.keras.models.load_model('models/cifar/l7_model')
    l8_model = tf.keras.models.load_model('models/cifar/l8_model')
    l9_model = tf.keras.models.load_model('models/cifar/l9_model')
    l10_model = tf.keras.models.load_model('models/cifar/l10_model')

    models = [l1_model, l2_model, l3_model, l4_model, l5_model, l6_model, l7_model, l8_model, l9_model, l10_model]
    num_models = len(models)
    num_samples = x_test.shape[0]
    output_size = 10

    # Pre-compute model outputs
    model_outputs = np.zeros((num_models, num_samples, output_size))
    for i in range(num_models):
        print("Loading model " + str(i + 1) + " outputs...")
        model = models[i]

        model_probs = model.predict(x_test)
        model_outputs[i] = model_probs


    # Compute all combinations
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            else:
                print("Running L" + str(i + 1) + " L" + str(j + 1) + "...")
                accuracies = run_combinations(model_outputs[i], model_outputs[j], y_test)
                print("Accuracies:", accuracies)


    # Compute combinations for specific model
    # current_model = 0
    # index = 0
    # all_accuracies = np.zeros((num_models - 1, 100))
    # for i in range(num_models):
    #     if i == current_model:
    #         continue
    #     else:
    #         print("Running L" + str(current_model + 1) + " L" + str(i + 1) + "...")
    #         accuracies = run_combinations(model_outputs[current_model], model_outputs[i], y_test)
    #         #print("Accuracies:", accuracies)
    #         all_accuracies[index] = accuracies
    #         index += 1

    # print("Accuracies:", all_accuracies)

if __name__ == '__main__':
    main()