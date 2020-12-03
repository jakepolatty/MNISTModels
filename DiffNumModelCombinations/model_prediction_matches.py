import tensorflow as tf
import time
import numpy as np

import helpers.helper_funcs as helpers
#import helpers.cifar_models as models

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


    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    unique, counts = np.unique(y_test.numpy(), return_counts=True)
    count_dict = dict(zip(unique, counts))

    # Set up accuracy grid
    accuracies = np.zeros((10, 10))

    # Iterate over all models and get their predicted outputs
    models = [l1_model, l2_model, l3_model, l4_model, l5_model, l6_model, l7_model, l8_model, l9_model, l10_model]
    models_preds = []
    for i in range(10):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)
        models_preds.append(model_preds)


    for i in range(10):
        model1 = models_preds[i]

        for j in range(10):
            model2 = models_preds[j]

            # Compute the number of times where the two models match predictions
            model_count = np.count_nonzero(model1 == model2)
            accuracies[i][j] = model_count / 10000


    print(accuracies)


if __name__ == '__main__':
    main()