import tensorflow as tf
import time
import numpy as np
import random

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_val, y_val, x_test, y_test = helpers.get_cifar10_data_val()
    y_val = tf.squeeze(y_val)
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
    model_times = [0.00002400, 0.00002876, 0.00003061, 0.00003114, 0.00003804, 0.00004302, 0.00003061, 0.00003114, 0.00003804, 0.00004302]
    experiment_type = "R"; # "L" "R" S" "D"
    run_experiment(models, num_classes, x_test, y_test, x_val, y_val,
        thresholds=thresholds, weight_types=weight_types, model_times=model_times, iterations=10, experiment_type=experiment_type)    


##################
# Runtime Controls
##################
def run_experiment(models, num_classes, x_test, y_test, x_val, y_val, thresholds, weight_types, model_times, iterations, experiment_type):
    final_accuracies = np.zeros((len(weight_types), len(thresholds)))
    times = np.zeros((len(weight_types), len(thresholds)))

    for i in range(len(weight_types)):
        weight_type = weight_types[i]
        if weight_type == "A":
            accuracies = compute_class_matrix_A(models, num_classes, x_val, y_val)
        elif weight_type == "B":
            accuracies = compute_class_matrix_B(models, num_classes, x_val, y_val)
        elif weight_type == "C":
            accuracies = compute_class_matrix_C(models, num_classes, x_val, y_val)
        elif weight_type == "O":
            accuracies = compute_class_matrix_overall(models, num_classes, x_val, y_val)
        else:
            accuracies = compute_class_matrix_all_ones(models, num_classes)

        if experiment_type == "R":
            for j in range(len(thresholds)):
                threshold = thresholds[j]
                acc, counts = average_optimization(models, num_classes, x_test, y_test, accuracies, threshold, iterations)
                final_accuracies[i][j] = acc
                times[i][j] = np.dot(model_times, counts)
                print(weight_type, " - ", threshold, " - Average Accuracy: ", final_accuracies[i][j])
        elif experiment_type == "L":
            model_accuracies = compute_class_matrix_overall(models, num_classes, x_test, y_test)[:, 0]
            model_order = np.argsort(model_accuracies)

            for j in range(len(thresholds)):
                threshold = thresholds[j]
                acc, counts = optimize_linear(models, num_classes, x_test, y_test, accuracies, threshold, model_order)

                final_accuracies[i][j] = acc
                times[i][j] = np.dot(model_times, counts)
                print(weight_type, " - ", threshold, " - Average Accuracy: ", final_accuracies[i][j])
        elif experiment_type == "S":
            model_accuracies = compute_class_matrix_B(models, num_classes, x_test, y_test)
            model_rankings = np.argsort(-1*model_accuracies, axis=0)

            for j in range(len(thresholds)):
                threshold = thresholds[j]
                acc, counts = optimize_selective(models, num_classes, x_test, y_test, accuracies, threshold, model_rankings)

                final_accuracies[i][j] = acc
                times[i][j] = np.dot(model_times, counts)
                print(weight_type, " - ", threshold, " - Average Accuracy: ", final_accuracies[i][j])
        elif experiment_type == "D":
            model_rankings = compute_double_ranking_matrix(models, num_classes, x_test, y_test)

            for j in range(len(thresholds)):
                threshold = thresholds[j]
                acc, counts = optimize_double_selective(models, num_classes, x_test, y_test, accuracies, threshold, model_rankings)
 
                final_accuracies[i][j] = acc
                times[i][j] = np.dot(model_times, counts)
                print(weight_type, " - ", threshold, " - Average Accuracy: ", final_accuracies[i][j])

    print(final_accuracies, times)


def average_optimization(models, num_classes, x_test, y_test, accuracies, threshold, iterations):
    total_accuracy = 0
    model_counts = np.zeros(len(models))
    for i in range(iterations):
        acc, counts = optimize_random(models, num_classes, x_test, y_test, accuracies, threshold)
        total_accuracy += acc
        model_counts += counts

    return total_accuracy / iterations, model_counts / iterations


######################
# Optimization Methods
######################
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

def optimize_double_selective(models, num_classes, x_test, y_test, accuracies, threshold, model_rankings):
    num_models = len(models)
    num_samples = x_test.shape[0]
    prob_matrix = np.zeros((num_models, num_samples))
    pred_matrix = np.zeros((num_models, num_samples))
    second_pred_matrix = np.zeros((num_models, num_samples))

    model_counts = np.zeros(num_models)

    for i in range(num_models):
        model = models[i]
        model_probs = model.predict(x_test)

        model_accuracies = np.transpose([accuracies[i]])
        model_probs = (model_probs.T * model_accuracies).T

        model_highest_probs = np.amax(model_probs, axis=1)
        model_preds = np.argmax(model_probs, axis=1)
        second_model_preds = np.argpartition(model_probs, -2, axis=1)[:, -2]

        prob_matrix[i] = model_highest_probs
        pred_matrix[i] = model_preds
        second_pred_matrix[i] = second_model_preds

    final_preds = np.zeros(num_samples)
    for i in range(num_samples):
        col = prob_matrix[:, i]
        best_model = -1
        model_nums = list(range(num_models))

        model_num = random.randint(0, num_models - 1)

        for j in range(num_models):
            model_nums.remove(model_num)

            prob = col[model_num]
            model_counts[model_num] += 1
            if prob > threshold:
                best_model = model_num
                break
            elif best_model == -1 or prob > col[best_model]:
                best_model = model_num

            current_pred = int(pred_matrix[model_num, i])
            current_second = int(second_pred_matrix[model_num, i])
            index = 10 * current_pred + current_second

            pred_models = model_rankings[:, current_pred]
            for k in range(num_models):
                next_model = pred_models[k]
                if next_model in model_nums:
                    model_num = next_model
                    break
        
        final_preds[i] = pred_matrix[best_model, i]

    final_accuracy = tf.reduce_mean(tf.cast(tf.equal(final_preds, y_test), tf.float32)).numpy()
    #print("Accuracy: ", final_accuracy)
    return final_accuracy, model_counts

def optimize_selective(models, num_classes, x_test, y_test, accuracies, threshold, model_rankings):
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

        model_num = random.randint(0, num_models - 1)

        for j in range(num_models):
            model_nums.remove(model_num)

            prob = col[model_num]
            model_counts[model_num] += 1
            if prob > threshold:
                best_model = model_num
                break
            elif best_model == -1 or prob > col[best_model]:
                best_model = model_num

            current_pred = int(pred_matrix[model_num, i])
            pred_models = model_rankings[:, current_pred]
            for k in range(num_models):
                next_model = pred_models[k]
                if next_model in model_nums:
                    model_num = next_model
                    break
        
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

##################
# Accuracy Formats
##################
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

###################
# Complex Utilities
###################
def compute_double_ranking_matrix(models, num_classes, x_test, y_test):
    num_models = len(models)
    accuracies = compute_class_matrix_B(models, num_classes, x_test, y_test)
    raw_rankings = np.zeros((num_models, num_classes * num_classes))

    for i in range(num_models):
        for j in range(num_classes):
            for k in range(num_classes):
                col = 10 * j + k
                if j == k:
                    raw_rankings[i][col] = 0
                else:
                    raw_rankings[i][col] = accuracies[i][j] + 0.1 * accuracies[i][k]

    model_rankings = np.argsort(-1*raw_rankings, axis=0)
    return model_rankings


if __name__ == '__main__':
    main()