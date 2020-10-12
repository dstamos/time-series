from numpy import identity as eye
from sklearn.metrics import mean_squared_error
import numpy as np
from src.utilities import lag_features, prune_data
from numpy.linalg.linalg import pinv


class ITL:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.lags

        self.prediction = None

    def fit(self, test_tasks):
        test_tasks = self._handle_data(test_tasks)

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
                val_performance = mean_squared_error(y_val, x_val @ curr_w)

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
            print('task: %3d | best lambda: %6e | val MSE: %8.5f' % (task_idx, best_regularization_parameter, best_val_perf))
        print(f'lambda: {np.nan:6e} | val MSE: {np.nanmean(all_val_perf):20.16f}')

    @staticmethod
    def _performance_check(y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)

    def _handle_data(self, list_of_tasks):
        for task_idx in range(len(list_of_tasks)):
            # The features are based just on the percentage difference of values of the time series
            raw_time_series_tr = list_of_tasks[task_idx].training.raw_time_series.pct_change()
            raw_time_series_val = list_of_tasks[task_idx].validation.raw_time_series.pct_change()
            raw_time_series_ts = list_of_tasks[task_idx].test.raw_time_series.pct_change()

            y_train = list_of_tasks[task_idx].training.labels
            y_validation = list_of_tasks[task_idx].validation.labels
            y_test = list_of_tasks[task_idx].test.labels
            if self.settings.use_exog is True:
                x_train = list_of_tasks[task_idx].training.features
                features_tr = lag_features(x_train, self.lags)

                x_validation = list_of_tasks[task_idx].validation.features
                features_val = lag_features(x_validation, self.lags)

                x_test = list_of_tasks[task_idx].test.features
                features_ts = lag_features(x_test, self.lags)
            else:
                features_tr = lag_features(raw_time_series_tr, self.lags, keep_original=False)
                features_val = lag_features(raw_time_series_val, self.lags, keep_original=False)
                features_ts = lag_features(raw_time_series_ts, self.lags, keep_original=False)

            list_of_tasks[task_idx].training.features, list_of_tasks[task_idx].training.labels = prune_data(features_tr, y_train)
            list_of_tasks[task_idx].validation.features, list_of_tasks[task_idx].validation.labels = prune_data(features_val, y_validation)
            list_of_tasks[task_idx].test.features, list_of_tasks[task_idx].test.labels = prune_data(features_ts, y_test)

            # Normalise the features
            # list_of_tasks[task_idx].training.features = list_of_tasks[task_idx].training.features / norm(list_of_tasks[task_idx].training.features, axis=1, keepdims=True)
            # list_of_tasks[task_idx].validation.features = list_of_tasks[task_idx].validation.features / norm(list_of_tasks[task_idx].validation.features, axis=1, keepdims=True)
            # list_of_tasks[task_idx].test.features = list_of_tasks[task_idx].test.features / norm(list_of_tasks[task_idx].test.features, axis=1, keepdims=True)

        return list_of_tasks
