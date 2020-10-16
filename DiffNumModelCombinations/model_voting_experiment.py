import tensorflow as tf
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import helpers.helper_funcs as helpers
import helpers.cifar_ensemble_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()
    y_test = tf.squeeze(y_test)

    print("Loading models...")
    l1_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l1_model, epochs=10, verbose=True)
    l1_model._estimator_type = "classifier"
    l2_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l2_model, epochs=10, verbose=True)
    l2_model._estimator_type = "classifier"
    l3_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l3_model, epochs=10, verbose=True)
    l3_model._estimator_type = "classifier"
    l4_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l4_model, epochs=10, verbose=True)
    l4_model._estimator_type = "classifier"

    ensemble1 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                voting='hard')

    l1_model.fit(x_train, y_train)
    l1_model.save('models/cifar_ensemble/l1_model')

    l2_model.fit(x_train, y_train)
    l2_model.save('models/cifar_ensemble/l2_model')

    l3_model.fit(x_train, y_train)
    l3_model.save('models/cifar_ensemble/l3_model')

    l4_model.fit(x_train, y_train)
    l4_model.save('models/cifar_ensemble/l4_model')

    ensemble.fit(x_train, y_train)
    ensemble1.save('models/cifar_ensemble/hard_voting_model')

    for clf in (l1_model, l2_model, l3_model, l4_model, ensemble):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    main()