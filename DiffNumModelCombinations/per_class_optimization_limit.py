import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    y_test = tf.squeeze(y_test)

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
    class_count = 10
    accuracies = compute_class_matrix(models, class_count, y_test)

    best_models = np.argmax(accuracies, axis=0)

    # Classify the test data into buckets based off of their true class
    buckets = [[] for i in range(class_count)]
    for i in range(x_test.shape[0]):
        class_peek = y_test[i]
        buckets[class_peek].append(x_test[i])

    # Loop through each of the class buckets and run the corresponding model, before adding up the correct predictions
    total_correct = 0 
    for i in range(class_count):
        bucket_inputs = np.array(buckets[i])
        probs = model.predict(bucket_inputs)
        preds = np.argmax(probs, axis=1)
        total_correct += np.sum(preds == i)

    print("Final accuracy: ", total_correct / x_test.shape[0])




def compute_class_matrix(models, class_count, y_test):
    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    count_dicts = []

    # Set up accuracy grid
    model_count = len(models)
    accuracies = np.zeros((model_count, class_count))

    # Iterate over all models and get their predicted outputs
    for i in range(model_count):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)

        unique, counts = np.unique(model_preds, return_counts=True)
        count_dicts.append(dict(zip(unique, counts)))

        # Iterate over all 10 classes
        for j in range(class_count):
            # Compute the number of times where the prediction matches the test output for that class
            class_count = len(np.where((model_preds == j) & (y_test_np == j))[0])
            accuracies[i][j] = class_count / count_dicts[i][j]

    return accuracies


if __name__ == '__main__':
    main()