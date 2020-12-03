import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    y_test = tf.squeeze(y_test)

    #train_and_save_models(x_train, y_train)

    print("Loading models...")
    l1_model = tf.keras.models.load_model('models/mnist/l1_model')
    l2_model = tf.keras.models.load_model('models/mnist/l2_model')
    l3_model = tf.keras.models.load_model('models/mnist/l3_model')
    l4_model = tf.keras.models.load_model('models/mnist/l4_model')
    l5_model = tf.keras.models.load_model('models/mnist/l5_model')
    l6_model = tf.keras.models.load_model('models/mnist/l6_model')
    l7_model = tf.keras.models.load_model('models/mnist/l7_model')
    l8_model = tf.keras.models.load_model('models/mnist/l8_model')
    l9_model = tf.keras.models.load_model('models/mnist/l9_model')
    l10_model = tf.keras.models.load_model('models/mnist/l10_model')


    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    unique, counts = np.unique(y_test.numpy(), return_counts=True)
    count_dict = dict(zip(unique, counts))

    # Set up accuracy grid
    accuracies = np.zeros((10, 10))

    # Iterate over all models and get their predicted outputs
    models = [l1_model, l2_model, l3_model, l4_model, l5_model, l6_model, l7_model, l8_model, l9_model, l10_model]
    for i in range(10):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)

        # Iterate over all 10 classes
        for j in range(10):
            # Compute the number of times where the prediction matches the test output for that class
            class_count = len(np.where((model_preds == j) & (y_test_np == j))[0])
            accuracies[i][j] = class_count / count_dict[j]


    print(accuracies)


if __name__ == '__main__':
    main()