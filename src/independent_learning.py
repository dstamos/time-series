from numpy import identity as eye
import pandas as pd
from src.utilities import labels_to_raw
import numpy as np
from src.utilities import handle_data, performance_check
from numpy.linalg.linalg import pinv
from time import time


class ITL:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.lags

        self.best_weight_vectors = None
        self.all_predictions = None
        self.all_raw_predictions = None
        self.all_test_perf = None

    def fit(self, test_tasks):
        test_tasks = handle_data(test_tasks, self.lags, self.settings.use_exog)

        best_weight_vectors = [None] * len(test_tasks)
        all_val_perf = [None] * len(test_tasks)
        tt = time()
        for task_idx in range(len(test_tasks)):
            best_val_performance = np.Inf
            x_train = test_tasks[task_idx].training.features.values
            y_train = test_tasks[task_idx].training.labels.values.ravel()

            x_val = test_tasks[task_idx].validation.features.values
            y_val = test_tasks[task_idx].validation.labels

            dims = x_train.shape[1]

            xtx = x_train.T @ x_train
            for regularization_parameter in self.settings.regularization_parameter_range:
                #####################################################
                # Optimisation
                from scipy.linalg import lstsq
                curr_w = lstsq(xtx + regularization_parameter * eye(dims), x_train.T @ y_train)[0]

                #####################################################
                # Validation
                curr_predictions = pd.Series(x_val @ curr_w, index=y_val.index)
                errors = performance_check(y_val, curr_predictions)
                val_performance = errors['nmse']

                if val_performance < best_val_performance:
                    validation_criterion = True
                else:
                    validation_criterion = False

                if validation_criterion:
                    best_val_performance = val_performance
                    best_weight_vectors[task_idx] = curr_w
                    all_val_perf[task_idx] = val_performance
            print(f'LTL | {task_idx:3d} | val performance: {best_val_performance:12.5f} | {time() - tt:5.2f}sec')
        self.best_weight_vectors = best_weight_vectors

        self._predict(test_tasks)

    def _predict(self, test_tasks):
        all_predictions = []
        all_raw_predictions = []
        all_test_perf = []
        for task_idx in range(len(test_tasks)):
            x_test = test_tasks[task_idx].test.features.values
            y_test = test_tasks[task_idx].test.labels

            curr_predictions = pd.Series(x_test @ self.best_weight_vectors[task_idx], index=y_test.index)
            errors = performance_check(y_test, curr_predictions)
            test_perf = errors['nmse']
            all_test_perf.append(test_perf)
            all_predictions.append(curr_predictions)
            all_raw_predictions.append(curr_predictions)
        self.all_test_perf = all_test_perf
        self.all_predictions = all_predictions
        self.all_raw_predictions = all_raw_predictions
        print('ITL | test performance: %8.5f' % (np.nanmean(all_test_perf)))
