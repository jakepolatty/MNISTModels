import tensorflow as tf
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import helpers.helper_funcs as helpers
import helpers.cifar_ensemble_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_cifar10_data()
    y_train2 = tf.keras.utils.to_categorical(y_train, 10)
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
    l5_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l1_model, epochs=10, verbose=True)
    l5_model._estimator_type = "classifier"
    l6_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l2_model, epochs=10, verbose=True)
    l6_model._estimator_type = "classifier"
    l7_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l3_model, epochs=10, verbose=True)
    l7_model._estimator_type = "classifier"
    l8_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l4_model, epochs=10, verbose=True)
    l8_model._estimator_type = "classifier"
    l9_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l9_model, epochs=10, verbose=True)
    l9_model._estimator_type = "classifier"
    l10_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l10_model, epochs=10, verbose=True)
    l10_model._estimator_type = "classifier"


    # ensemble12 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l2', l2_model)],
    #                             final_estimator=LogisticRegression())

    # ensemble13 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l3', l3_model)],
    #                             final_estimator=LogisticRegression())

    # ensemble14 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=LogisticRegression())

    # ensemble23 = StackingClassifier(estimators=[('l2', l2_model),
    #                                         ('l3', l3_model)],
    #                             final_estimator=LogisticRegression())

    # ensemble24 = StackingClassifier(estimators=[('l2', l2_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=LogisticRegression())

    # ensemble34 = StackingClassifier(estimators=[('l3', l3_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=LogisticRegression())

    # ensemble123 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l2', l2_model),
    #                                         ('l3', l3_model)],
    #                             final_estimator=HistGradientBoostingClassifier(random_state=42))

    # ensemble124 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l2', l2_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=HistGradientBoostingClassifier(random_state=42))

    # ensemble134 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l3', l3_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=HistGradientBoostingClassifier(random_state=42))

    # ensemble234 = StackingClassifier(estimators=[('l2', l2_model),
    #                                         ('l3', l3_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=HistGradientBoostingClassifier(random_state=42))

    # ensemble1234 = StackingClassifier(estimators=[('l1', l1_model),
    #                                         ('l2', l2_model),
    #                                         ('l3', l3_model),
    #                                         ('l4', l4_model)],
    #                             final_estimator=HistGradientBoostingClassifier(random_state=42))

    # ensemble5678 = StackingClassifier(estimators=[('l5', l5_model),
    #                                         ('l6', l6_model),
    #                                         ('l7', l7_model),
    #                                         ('l8', l8_model)],
    #                             final_estimator=LogisticRegression(), verbose=True)

    ensemble = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model),
                                            ('l5', l5_model),
                                            ('l6', l6_model),
                                            ('l7', l7_model),
                                            ('l8', l8_model),
                                            ('l9',l9_model),
                                            ('l10', l10_model)],
                                final_estimator=LogisticRegression(), verbose=True)

    # l1_model.fit(x_train, y_train2)
    # l2_model.fit(x_train, y_train2)
    # l3_model.fit(x_train, y_train2)
    # l4_model.fit(x_train, y_train2)
    # ensemble.fit(x_train, y_train)

    # ensemble12.fit(x_train, y_train)
    # ensemble13.fit(x_train, y_train)
    # ensemble14.fit(x_train, y_train)
    # ensemble23.fit(x_train, y_train)
    # ensemble24.fit(x_train, y_train)
    # ensemble34.fit(x_train, y_train)

    # ensemble123.fit(x_train, y_train)
    # ensemble124.fit(x_train, y_train)
    # ensemble134.fit(x_train, y_train)
    # ensemble234.fit(x_train, y_train)

    ensemble.fit(x_train, y_train)

    for clf in (ensemble, ensemble):
        before_time = time.time()
        y_pred = clf.predict(x_test)
        model_time = time.time() - before_time

        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
        print("Time: ", model_time)

if __name__ == '__main__':
    main()