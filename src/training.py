import numpy as np
import pandas as pd
import xgboost
import multiprocess as mp
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from src.utilities import lag_features, prune_data
from numpy.linalg.linalg import norm, pinv, matrix_power


class Sarimax:
    def __init__(self, settings):
        self.settings = settings
        self.ar_order = 2
        self.difference_order = 0
        self.ma_order = 1

        self.seasonal_ar_order = 1
        self.seasonal_difference_order = 1
        self.seasonal_ma_order = 1
        self.seasonal_period = 24

        self.model = None
        self.prediction = None

    def fit(self, time_series, exog_variables=None):
        if self.settings.training.use_exog is True:
            exog_variables = exog_variables  # FIXME.diff().fillna(method='bfill')
        else:
            exog_variables = None

        model = SARIMAX(time_series, exog=exog_variables,
                        order=(self.ar_order, self.difference_order, self.ma_order),
                        seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period))
        self.model = model.fit(dips=1, maxiter=150)

    def predict(self, exog_variables=None, foreward_periods=1):
        if self.settings.training.use_exog is True:
            exog_variables = exog_variables
        else:
            exog_variables = None

        preds = self.model.predict(steps=foreward_periods, exog=exog_variables)
        forecast_table = preds.summary_frame(alpha=0.10)
        self.prediction = forecast_table['mean']


class Xgboost:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.training.lags

        self.model = None
        self.prediction = None

    def fit(self, labels, x=None):
        if self.settings.training.use_exog is True:
            x = x  # TODO Try .diff().fillna(method='bfill')
            # x = self._time_features(x)
            features = lag_features(x, self.lags)
        else:
            features = lag_features(labels, self.lags, keep_original=False)
        features, labels = prune_data(features, labels)
        dmatrix = xgboost.DMatrix(features, labels)

        def heartbeat():
            def callback(env):
                print('%5d | %8s ' % (env.iteration, time.strftime("%H:%M:%S")))
            return callback

        # noinspection PyUnresolvedReferences
        params = {'max_depth': 6,
                  'colsample_bytree': 0.2,
                  'eta': 0.05,
                  'num_parallel_tree': 20,
                  'n_jobs': -2,
                  'obj': 'reg:squarederror',
                  'verbosity': 0,
                  'seed': 1,
                  'booster': 'dart',
                  'min_child_weight': 2,
                  'nthread': mp.cpu_count()-1}
        self.model = xgboost.train(params, dmatrix, num_boost_round=100, callbacks=[heartbeat()])

    def predict(self, x):
        if self.settings.training.use_exog is True:
            x = x  # TODO .diff().fillna(method='bfill')
            # x = self._time_features(x)
            features = lag_features(x, self.lags)
        else:
            features = lag_features(x, self.lags, keep_original=False)
        features = prune_data(features)
        prediction = pd.DataFrame(self.model.predict(xgboost.DMatrix(features)), index=features.index)
        self.prediction = prediction
        return prediction

    @staticmethod
    def _time_features(indicators):
        datetimes = indicators.index

        times = datetimes.hour

        F = np.zeros((len(times), 24))
        F[range(len(times)), times] = 1
        for idx in range(24):
            indicators['calendar_' + str(idx)] = F[:, idx]
        return indicators


class BiasLTL:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.training.lags

        self.all_metaparameters = None
        self.final_metaparameters = None
        self.prediction = None

    def fit(self, training_tasks, validation_tasks):
        training_tasks = self._handle_data(training_tasks)
        validation_tasks = self._handle_data(validation_tasks)

        dims = training_tasks[0].training.features.shape[1]

        best_val_performance = np.Inf

        for regularization_parameter in self.settings.training.regularization_parameter_range:
            validation_performances = []
            all_average_vectors = []

            # For the sake of faster training
            # mean_vector = best_mean_vector
            mean_vector = np.random.randn(dims) / norm(np.random.randn(dims))

            for task_idx in range(len(training_tasks)):
                #####################################################
                # Optimisation
                x_train = training_tasks[task_idx].training.features.values
                y_train = training_tasks[task_idx].training.labels.values.ravel()

                mean_vector = self._solve_wrt_h(mean_vector, x_train, y_train, regularization_parameter, curr_iteration=task_idx, inner_iter_cap=3)
                all_average_vectors.append(mean_vector)
            #####################################################
            # Validation only needs to be measured at the very end, after we've trained on all training tasks
            for validation_task_idx in range(len(validation_tasks)):
                x_train = validation_tasks[validation_task_idx].training.features.values
                y_train = validation_tasks[validation_task_idx].training.labels.values.ravel()
                w = self._solve_wrt_w(mean_vector, x_train, y_train, regularization_parameter)

                x_test = validation_tasks[validation_task_idx].test.features.values
                y_test = validation_tasks[validation_task_idx].test.labels.values.ravel()

                validation_perf = self._performance_check(y_test, x_test @ w)
                validation_performances.append(validation_perf)
            validation_performance = np.mean(validation_performances)
            print(f'lambda: {regularization_parameter:6e} | val MSE: {validation_performance:12.5f}')

            if validation_performance < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False

            if validation_criterion:
                best_val_performance = validation_performance

                best_average_vectors = all_average_vectors
                best_mean_vector = mean_vector
        print(f'lambda: {np.nan:6e} | val MSE: {best_val_performance:12.5f}')
        self.all_metaparameters = best_average_vectors
        self.final_metaparameters = best_mean_vector

    @staticmethod
    def _solve_wrt_h(h, x, y, param, curr_iteration=0, inner_iter_cap=10):
        step_size_bit = 1e+3
        n = len(y)

        def grad(curr_h):
            return 2 * param**2 * n * x.T @ matrix_power(pinv(x @ x.T + param * n * np.eye(n)), 2) @ ((x @ curr_h).ravel() - y)

        i = 0
        curr_iteration = curr_iteration * inner_iter_cap
        while i < inner_iter_cap:
            i = i + 1
            prev_h = h
            curr_iteration = curr_iteration + 1
            step_size = np.sqrt(2) * step_size_bit / ((step_size_bit + 1) * np.sqrt(curr_iteration))
            h = prev_h - step_size * grad(prev_h)

        return h

    @staticmethod
    def _solve_wrt_w(h, x, y, param):
        n = len(y)
        dims = x.shape[1]

        c_n_lambda = x.T @ x / n + param * np.eye(dims)
        w = pinv(c_n_lambda) @ (x.T @ y / n + param * h).ravel()

        return w

    @staticmethod
    def _performance_check(y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)

    def _handle_data(self, list_of_tasks):
        for task_idx in range(len(list_of_tasks)):
            # The features are based just on the percentage difference of values of the time series
            raw_time_series_tr = list_of_tasks[task_idx].training.raw_time_series.diff()
            raw_time_series_val = list_of_tasks[task_idx].validation.raw_time_series.diff()
            raw_time_series_ts = list_of_tasks[task_idx].test.raw_time_series.diff()

            y_train = list_of_tasks[task_idx].training.labels
            y_validation = list_of_tasks[task_idx].validation.labels
            y_test = list_of_tasks[task_idx].test.labels
            if self.settings.training.use_exog is True:
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
