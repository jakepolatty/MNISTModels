import tensorflow as tf
import time
import numpy as np
import random

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

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

    average_optimization(models, num_classes, x_test, y_test, weight_type="A", threshold=0.8, iterations=10)    
    #optimize(models, num_classes, x_test, y_test, weight_type="B")
    #optimize(models, num_classes, x_test, y_test)

def average_optimization(models, num_classes, x_test, y_test, weight_type, threshold, iterations):
    if weight_type == "A":
        accuracies = compute_class_matrix_A(models, num_classes, x_test, y_test)
    elif weight_type == "B":
        accuracies = compute_class_matrix_B(models, num_classes, x_test, y_test)
    else:
        accuracies = compute_class_matrix_overall(models, num_classes, x_test, y_test)

    total_accuracy = 0
    for i in range(iterations):
        total_accuracy += optimize(models, num_classes, x_test, y_test, accuracies, threshold)

    print("Average Accuracy: ", total_accuracy / iterations)


def optimize(models, num_classes, x_test, y_test, accuracies, threshold):
    num_models = len(models)
    num_samples = x_test.shape[0]
    prob_matrix = np.zeros((num_models, num_samples))
    pred_matrix = np.zeros((num_models, num_samples))

    for i in range(num_models):
        model = models[i]
        model_probs = model.predict(x_test)

        model_accuracies = np.transpose([accuracies[i]])
        model_probs = (model_probs.T * model_accuracies).T

        model_highest_probs = np.amax(model_probs, axis=1)
        model_preds = np.argmax(model_probs, axis=1)

        prob_matrix[i] = model_highest_probs
        pred_matrix[i] = model_preds

    final_preds = np.zeros(num_samples)
    for i in range(num_samples):
        col = prob_matrix[:, i]
        best_model = -1
        model_nums = list(range(num_models))

        for j in range(num_models):
            rand_index = random.randint(0, num_models - j - 1)
            model_num = model_nums.pop(rand_index)

            prob = col[model_num]
            if prob > threshold:
                best_model = model_num
                break
            elif best_model == -1 or prob > col[best_model]:
                best_model = model_num
        
        final_preds[i] = pred_matrix[best_model, i]

    final_accuracy = tf.reduce_mean(tf.cast(tf.equal(final_preds, y_test), tf.float32)).numpy()
    print("Accuracy: ", final_accuracy)
    return final_accuracy



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

def compute_class_matrix_overall(models, num_classes, x_test, y_test):
    y_test_squeezed = tf.squeeze(y_test)

    # Set up accuracy grid
    num_models = len(models)
    accuracies = np.zeros((num_models, num_classes))

    for i in range(num_models):
        model = models[i]
        accuracy = model.evaluate(x_test, y_test_squeezed, verbose=0)[1]
        for j in range(num_classes):
            accuracies[i][j] = accuracy

    print(accuracies)
    return accuracies




if __name__ == '__main__':
    main()