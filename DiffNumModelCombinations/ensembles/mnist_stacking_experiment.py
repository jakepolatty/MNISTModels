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


    ensemble12 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble13 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l3', l3_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble14 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble23 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble24 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble34 = StackingClassifier(estimators=[('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble123 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l3', l3_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble124 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble134 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble234 = StackingClassifier(estimators=[('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    ensemble1234 = StackingClassifier(estimators=[('l1', l1_model),
                                            ('l2', l2_model),
                                            ('l3', l3_model),
                                            ('l4', l4_model)],
                                final_estimator=HistGradientBoostingClassifier(random_state=42))

    l1_model.fit(x_train, y_train2)
    #print("L1 Done")
    l2_model.fit(x_train, y_train2)
    #print("L2 Done")
    l3_model.fit(x_train, y_train2)
    #print("L3 Done")
    l4_model.fit(x_train, y_train2)
    #print("L4 Done")

    ensemble12.fit(x_train, y_train)
    #print("L12 Done")
    ensemble13.fit(x_train, y_train)
    #print("L13 Done")
    ensemble14.fit(x_train, y_train)
    #print("L14 Done")
    ensemble23.fit(x_train, y_train)
    #print("L23 Done")
    ensemble24.fit(x_train, y_train)
    #print("L24 Done")
    ensemble34.fit(x_train, y_train)
    #print("L34 Done")

    ensemble123.fit(x_train, y_train)
    #print("L123 Done")
    ensemble124.fit(x_train, y_train)
    #print("L124 Done")
    ensemble134.fit(x_train, y_train)
    #print("L134 Done")
    ensemble234.fit(x_train, y_train)
    #print("L234 Done")

    ensemble1234.fit(x_train, y_train)
    #print("L1234 Done")

    for clf in (l1_model, l2_model, l3_model, l4_model,
                ensemble12, ensemble13, ensemble14, ensemble23, ensemble24, ensemble34,
                ensemble123, ensemble124, ensemble134, ensemble234,
                ensemble1234):
        before_time = time.time()
        y_pred = clf.predict(x_test)
        model_time = time.time() - before_time

        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
        print("Time: ", model_time)

if __name__ == '__main__':
    main()