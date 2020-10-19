from numpy import identity as eye
import pandas as pd
import numpy as np
from src.utilities import handle_data
from numpy.linalg.linalg import pinv


class ITL:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.lags

        self.best_weight_vectors = None
        self.all_test_perf = None
        self.all_predictions = None

    def fit(self, test_tasks):
        test_tasks = handle_data(test_tasks, self.lags, self.settings.use_exog)

        best_weight_vectors = [None] * len(test_tasks)
        all_val_perf = [None] * len(test_tasks)

        for task_idx in range(len(test_tasks)):
            best_val_performance = np.Inf
            x_train = test_tasks[task_idx].training.features.values
            y_train = test_tasks[task_idx].training.labels.values.ravel()

            x_val = test_tasks[task_idx].validation.features.values
            y_val = test_tasks[task_idx].validation.labels.values.ravel()

            dims = x_train.shape[1]

            for regularization_parameter in self.settings.regularization_parameter_range:
                #####################################################
                # Optimisation
                curr_w = pinv(x_train.T @ x_train + regularization_parameter * eye(dims)) @ x_train.T @ y_train

                #####################################################
                # Validation
                val_performance = self._performance_check(y_val, x_val @ curr_w)

                if val_performance < best_val_performance:
                    validation_criterion = True
                else:
                    validation_criterion = False

                if validation_criterion:
                    best_val_performance = val_performance

                    best_weight_vectors[task_idx] = curr_w

                    best_regularization_parameter = regularization_parameter
                    best_val_perf = val_performance
                    all_val_perf[task_idx] = val_performance
            print('task: %3d | best lambda: %6e | val performance: %8.5f' % (task_idx, best_regularization_parameter, best_val_perf))
        self.best_weight_vectors = best_weight_vectors
        print(f'lambda: {np.nan:6e} | val performance: {np.nanmean(all_val_perf):20.16f}')

        self._predict(test_tasks)

    def _predict(self, test_tasks):
        all_test_perf = []
        predictions = []
        for task_idx in range(len(test_tasks)):
            x_test = test_tasks[task_idx].test.features.values
            y_test = test_tasks[task_idx].test.labels.values.ravel()

            curr_prediction = pd.Series(x_test @ self.best_weight_vectors[task_idx], index=test_tasks[task_idx].test.labels.index)
            test_performance = self._performance_check(y_test, curr_prediction.values.ravel())
            all_test_perf.append(test_performance)
            predictions.append(curr_prediction)
        self.all_predictions = predictions
        self.all_test_perf = all_test_perf

    @staticmethod
    def _performance_check(y_true, y_pred):
        # Make sure that if y_true is 0 then you return 0
        rel_error = np.abs(np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true), where=(y_true != 0)))
        mape = (100 / len(y_true)) * np.sum(rel_error)
        return mape
