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
import helpers.mnist_ensemble_models as models

def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = helpers.get_mnist_data()
    y_train2 = tf.keras.utils.to_categorical(y_train, 10)
    #y_test = tf.squeeze(y_test)

    print("Loading models...")
    l1_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l1_model, epochs=5, verbose=True)
    l1_model._estimator_type = "classifier"
    l2_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l2_model, epochs=5, verbose=True)
    l2_model._estimator_type = "classifier"
    l3_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l3_model, epochs=5, verbose=True)
    l3_model._estimator_type = "classifier"
    l4_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.get_l4_model, epochs=5, verbose=True)
    l4_model._estimator_type = "classifier"


    ensemble1 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=RandomForestClassifier(random_state=42, n_estimators=10))

    ensemble2 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=RandomForestClassifier(random_state=42))

    ensemble3 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=RandomForestClassifier(random_state=42, n_estimators=500))

    ensemble4 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42, max_iter=10))

    ensemble5 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble1.fit(x_train, y_train)
    #print("L123 Done")
    ensemble2.fit(x_train, y_train)
    #print("L124 Done")
    ensemble3.fit(x_train, y_train)
    #print("L134 Done")
    ensemble4.fit(x_train, y_train)
    #print("L234 Done")
    ensemble5.fit(x_train, y_train)
    #print("L234 Done")

    for clf in (ensemble1, ensemble2, ensemble3, ensemble4, ensemble5):
        before_time = time.time()
        y_pred = clf.predict(x_test)
        model_time = time.time() - before_time

        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
        print("Time: ", model_time)

if __name__ == '__main__':
    main()