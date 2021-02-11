import tensorflow as tf
import time
import numpy as np
import random

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

# Method C: weights by F1 score

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

    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    weight_types = ["A", "B", "C", "O", "U"]
    run_experiment(models, num_classes, x_test, y_test, thresholds=thresholds, weight_types=weight_types, iterations=10)    


def run_experiment(models, num_classes, x_test, y_test, thresholds, weight_types, iterations, random=True):
    final_accuracies = np.zeros((len(weight_types), len(thresholds)))
    model_counts = np.zeros((len(weight_types), len(thresholds)))

    for i in range(len(weight_types)):
        weight_type = weight_types[i]
        if weight_type == "A":
            accuracies = compute_class_matrix_A(models, num_classes, x_test, y_test)
        elif weight_type == "B":
            accuracies = compute_class_matrix_B(models, num_classes, x_test, y_test)
        elif weight_type == "C":
            accuracies = compute_class_matrix_C(models, num_classes, x_test, y_test)
        elif weight_type == "O":
            accuracies = compute_class_matrix_overall(models, num_classes, x_test, y_test)
        else:
            accuracies = compute_class_matrix_all_ones(models, num_classes)

        if random:
            for j in range(len(thresholds)):
                threshold = thresholds[j]
                acc, counts = average_optimization(models, num_classes, x_test, y_test, accuracies, threshold, iterations)
                final_accuracies[i][j] = acc
                model_counts[i][j] = np.sum(counts)
                print(weight_type, " - ", threshold, " - Average Accuracy: ", final_accuracies[i][j])
        else:
            model_accuracies = compute_class_matrix_overall(models, num_classes, x_test, y_test)[:, 0]
            model_order = np.argsort(model_accuracies)
            for j in range(len(thresholds)):
                threshold = thresholds[j]
                acc, counts = optimize_linear(models, num_classes, x_test, y_test, accuracies, threshold, model_order)
                final_accuracies[i][j] = acc
                model_counts[i][j] = np.sum(counts)
                print(weight_type, " - ", threshold, " - Average Accuracy: ", final_accuracies[i][j])

    print(final_accuracies, model_counts)


def average_optimization(models, num_classes, x_test, y_test, accuracies, threshold, iterations):
    total_accuracy = 0
    model_counts = np.zeros(len(models))
    for i in range(iterations):
        acc, counts = optimize_random(models, num_classes, x_test, y_test, accuracies, threshold)
        total_accuracy += acc
        model_counts += counts

    return total_accuracy / iterations, model_counts / iterations


def optimize_random(models, num_classes, x_test, y_test, accuracies, threshold):
    num_models = len(models)
    num_samples = x_test.shape[0]
    prob_matrix = np.zeros((num_models, num_samples))
    pred_matrix = np.zeros((num_models, num_samples))

    model_counts = np.zeros(num_models)

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
            model_counts[model_num] += 1
            if prob > threshold:
                best_model = model_num
                break
            elif best_model == -1 or prob > col[best_model]:
                best_model = model_num
        
        final_preds[i] = pred_matrix[best_model, i]

    final_accuracy = tf.reduce_mean(tf.cast(tf.equal(final_preds, y_test), tf.float32)).numpy()
    #print("Accuracy: ", final_accuracy)
    return final_accuracy, model_counts


def optimize_linear(models, num_classes, x_test, y_test, accuracies, threshold, model_order):
    num_models = len(models)
    num_samples = x_test.shape[0]
    prob_matrix = np.zeros((num_models, num_samples))
    pred_matrix = np.zeros((num_models, num_samples))

    model_counts = np.zeros(num_models)

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
            model_num = model_order[j]

            prob = col[model_num]
            model_counts[model_num] += 1
            if prob > threshold:
                best_model = model_num
                break
            elif best_model == -1 or prob > col[best_model]:
                best_model = model_num
        
        final_preds[i] = pred_matrix[best_model, i]

    final_accuracy = tf.reduce_mean(tf.cast(tf.equal(final_preds, y_test), tf.float32)).numpy()
    #print("Accuracy: ", final_accuracy)
    return final_accuracy, model_counts


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

def compute_class_matrix_C(models, num_classes, x_test, y_test):
    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    unique_r, counts_r = np.unique(y_test_np, return_counts=True)
    count_dict_recall = dict(zip(unique_r, counts_r))
    count_dicts_precision = []

    # Set up accuracy grid
    num_models = len(models)
    accuracies = np.zeros((num_models, num_classes))

    # Iterate over all models and get their predicted outputs
    for i in range(num_models):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)

        unique_p, counts_p = np.unique(model_preds, return_counts=True)
        count_dicts_precision.append(dict(zip(unique_p, counts_p)))

        # Iterate over all 10 classes
        for j in range(num_classes):
            # Compute the number of times where the prediction matches the test output for that class
            class_count = len(np.where((model_preds == j) & (y_test_np == j))[0])
            recall = class_count / count_dict_recall[j]
            precision = class_count / count_dicts_precision[i][j]
            accuracies[i][j] = 2 * (precision * recall) / (precision + recall)

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

def compute_class_matrix_all_ones(models, num_classes):
    # Set up accuracy grid
    num_models = len(models)
    accuracies = np.ones((num_models, num_classes))
    print(accuracies)
    return accuracies




if __name__ == '__main__':
    main()