import numpy as np
import pandas as pd
from numpy.linalg import pinv
from numpy import identity as eye
from src.preprocessing import PreProcess
from time import time
from src.utilities import normalised_mse
from sklearn.model_selection import KFold


def train_test_itl(data, settings):
    # Training
    tt = time()
    all_performances = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        x = data['test_tasks_tr_features'][task_idx]
        y = data['test_tasks_tr_labels'][task_idx]

        cv_splits = 3
        if len(y) < cv_splits:
            # In the case we don't enough enough data for 5-fold cross-validation for training (cold start), just use random data.
            x = np.random.randn(*np.concatenate([data['test_tasks_test_features'][task_idx] for task_idx in range(len(data['test_tasks_test_features']))]).shape)
            y = np.random.uniform(0, 1, len(x))

        kf = KFold(n_splits=cv_splits)
        kf.get_n_splits(x)
        preprocessing = PreProcess(standard_scaling=False, inside_ball_scaling=False, add_bias=True)

        best_performance = np.Inf
        best_param = None
        for regul_param in settings['regul_param_range']:
            curr_val_performances = []
            for train_index, test_index in kf.split(x):
                x_tr, x_val = x.iloc[train_index], x.iloc[test_index]
                y_tr, y_val = y.iloc[train_index], y.iloc[test_index]

                x_tr, y_tr = preprocessing.transform(x_tr, y_tr, fit=True, multiple_tasks=False)
                x_val, y_val = preprocessing.transform(x_val, y_val, fit=False, multiple_tasks=False)

                model_itl = ITL(regul_param)
                model_itl.fit(x_tr, y_tr)

                val_predictions = model_itl.predict(x_val)
                val_performance = normalised_mse(y_val, val_predictions)
                curr_val_performances.append(val_performance)
            average_val_performance = np.mean(curr_val_performances)
            if average_val_performance < best_performance:
                best_performance = average_val_performance
                best_param = regul_param

        # Retrain on full training set
        x, y = preprocessing.transform(x, y, fit=True, multiple_tasks=False)
        x_test, y_test = preprocessing.transform(data['test_tasks_test_features'][task_idx], data['test_tasks_test_labels'][task_idx], fit=False, multiple_tasks=False)

        model_itl = ITL(best_param)
        model_itl.fit(x, y)

        # Testing
        test_predictions = model_itl.predict(x_test)
        all_performances.append(normalised_mse(y_test, test_predictions))
    test_performance = np.mean(all_performances)
    print(f'{"Independent":12s} | test performance: {test_performance:12.5f} | {time() - tt:6.1f}sec')

    return test_performance


class ITL:
    def __init__(self, regularization_parameter=1e-2):
        self.regularization_parameter = regularization_parameter
        self.weight_vector = None

    def fit(self, features, labels):
        dims = features.shape[1]

        weight_vector = pinv(features.T @ features + self.regularization_parameter * eye(dims)) @ features.T @ labels
        self.weight_vector = weight_vector.values.ravel()

    def predict(self, features):
        pred = (features @ self.weight_vector).to_frame()
        return pred
