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
    l1_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l1_model, epochs=10, verbose=False)
    l1_model._estimator_type = "classifier"
    l2_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l2_model, epochs=10, verbose=False)
    l2_model._estimator_type = "classifier"
    l3_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l3_model, epochs=10, verbose=False)
    l3_model._estimator_type = "classifier"
    l4_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l4_model, epochs=10, verbose=False)
    l4_model._estimator_type = "classifier"

    l1_model.fit(x_train, y_train)
    l2_model.fit(x_train, y_train)
    l3_model.fit(x_train, y_train)
    l4_model.fit(x_train, y_train)

    accuracy_scores = []

    for clf in (l1_model, l2_model, l3_model, l4_model):
        before_time = time.time()
        y_pred = clf.predict(x_test)
        model_time = time.time() - before_time

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

        print(clf.__class__.__name__, accuracy)
        print("Time: ", model_time)



    ensemble12 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble13 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l3', l3_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble14 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble23 = VotingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble24 = VotingClassifier(estimators=[('l2', l2_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble34 = VotingClassifier(estimators=[('l3', l3_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)

    ensemble123 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l3', l3_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble124 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble134 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)
    ensemble234 = VotingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)


    ensemble1234 = VotingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                voting='soft', verbose=False, weights=accuracy_scores)

    ensemble12.fit(x_train, y_train)
    ensemble13.fit(x_train, y_train)
    ensemble14.fit(x_train, y_train)
    ensemble23.fit(x_train, y_train)
    ensemble24.fit(x_train, y_train)
    ensemble34.fit(x_train, y_train)

    ensemble123.fit(x_train, y_train)
    ensemble124.fit(x_train, y_train)
    ensemble134.fit(x_train, y_train)
    ensemble234.fit(x_train, y_train)

    ensemble1234.fit(x_train, y_train)

    for clf in (ensemble12, ensemble13, ensemble14, ensemble23, ensemble24, ensemble34,
                ensemble123, ensemble124, ensemble134, ensemble234,
                ensemble1234):
        before_time = time.time()
        y_pred = clf.predict(x_test)
        model_time = time.time() - before_time

        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
        print("Time: ", model_time)

if __name__ == '__main__':
    main()