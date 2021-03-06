import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

#RUN ALL 10 MODELS, PICK HEIGHEST PROBABILITY IN ENTIRE GRID
#RUN ALL 10 MODELS, PICK HEIGHEST PROBABILITY WEIGHTED BY A OR B

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()
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
    num_classes = 10

    #unweighted_optimize(models, num_classes, x_test, y_test)
    weighted_optimize(models, num_classes, x_test, y_test, weight_type="B")


def unweighted_optimize(models, num_classes, x_test, y_test):
    prob_matrix = np.zeros((len(models), x_test.shape[0]))
    pred_matrix = np.zeros((len(models), x_test.shape[0]))

    for i in range(len(models)):
        model = models[i]
        model_probs = model.predict(x_test)
        model_highest_probs = np.amax(model_probs, axis=1)
        model_preds = np.argmax(model_probs, axis=1)

        prob_matrix[i] = model_highest_probs
        pred_matrix[i] = model_preds

    final_preds = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        col = prob_matrix[:, i]
        final_preds[i] = pred_matrix[np.argmax(col), i]

    final_accuracy = tf.reduce_mean(tf.cast(tf.equal(final_preds, y_test), tf.float32)).numpy()

    print("Final accuracy: ", final_accuracy)


def weighted_optimize(models, num_classes, x_test, y_test, weight_type):
    if weight_type == "A":
        accuracies = compute_class_matrix_A(models, num_classes, x_test, y_test)
    else:
        accuracies = compute_class_matrix_B(models, num_classes, x_test, y_test)

    prob_matrix = np.zeros((len(models), x_test.shape[0]))
    pred_matrix = np.zeros((len(models), x_test.shape[0]))

    for i in range(len(models)):
        model = models[i]
        model_probs = model.predict(x_test)

        model_accuracies = np.transpose([accuracies[i]])
        model_probs = (model_probs.T * model_accuracies).T

        model_highest_probs = np.amax(model_probs, axis=1)
        model_preds = np.argmax(model_probs, axis=1)

        prob_matrix[i] = model_highest_probs
        pred_matrix[i] = model_preds

    final_preds = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        col = prob_matrix[:, i]
        final_preds[i] = pred_matrix[np.argmax(col), i]

    final_accuracy = tf.reduce_mean(tf.cast(tf.equal(final_preds, y_test), tf.float32)).numpy()

    print("Final accuracy: ", final_accuracy)



def compute_class_matrix_A(models, num_classes, x_test, y_test):
    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    unique, counts = np.unique(y_test_np, return_counts=True)
    count_dict = dict(zip(unique, counts))

    # Set up accuracy grid
    num_models = len(models)
    accuracies = np.zeros((num_models, num_classes))

    # Iterate over all models and get their predicted outputs
    for i in range(num_models):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)

        # Iterate over all 10 classes
        for j in range(num_classes):
            # Compute the number of times where the prediction matches the test output for that class
            class_count = len(np.where((model_preds == j) & (y_test_np == j))[0])
            accuracies[i][j] = class_count / count_dict[j]

    print(accuracies)
    return accuracies


def compute_class_matrix_B(models, num_classes, x_test, y_test):
    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    count_dicts = []

    # Set up accuracy grid
    num_models = len(models)
    accuracies = np.zeros((num_models, num_classes))

    # Iterate over all models and get their predicted outputs
    for i in range(num_models):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)

        unique, counts = np.unique(model_preds, return_counts=True)
        count_dicts.append(dict(zip(unique, counts)))

        # Iterate over all 10 classes
        for j in range(num_classes):
            # Compute the number of times where the prediction matches the test output for that class
            class_count = len(np.where((model_preds == j) & (y_test_np == j))[0])
            accuracies[i][j] = class_count / count_dicts[i][j]

    print(accuracies)
    return accuracies


if __name__ == '__main__':
    main()