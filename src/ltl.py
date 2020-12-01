import numpy as np
import pandas as pd
from src.utilities import handle_data, labels_to_raw
from time import time
from numpy.linalg.linalg import norm, pinv, matrix_power


class BiasLTL:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.lags

        self.all_metaparameters = None
        self.final_metaparameters = None

        self.all_predictions = None
        self.all_raw_predictions = None
        self.test_per_per_training_task = None

    def fit(self, training_tasks, validation_tasks):
        training_tasks = handle_data(training_tasks, self.lags, self.settings.use_exog)
        validation_tasks = handle_data(validation_tasks, self.lags, self.settings.use_exog)

        dims = training_tasks[0].training.features.shape[1]

        best_val_performance = np.Inf
        mean_vector = np.random.randn(dims) / norm(np.random.randn(dims))
        for regularization_parameter in self.settings.regularization_parameter_range:
            validation_performances = []
            all_average_vectors = []

            # For the sake of faster training
            # mean_vector = best_mean_vector
            for task_idx in range(len(training_tasks)):
                #####################################################
                # Optimisation
                x_train = training_tasks[task_idx].training.features.values
                y_train = training_tasks[task_idx].training.labels.values.ravel()

                mean_vector = self._solve_wrt_h(mean_vector, x_train, y_train, regularization_parameter, curr_iteration=task_idx, inner_iter_cap=10)
                # print(mean_vector)
                all_average_vectors.append(mean_vector)
            #####################################################
            if np.all(np.isnan(mean_vector)):
                print('mean_vector:', mean_vector)
                continue
            # Validation only needs to be measured at the very end, after we've trained on all training tasks
            for validation_task_idx in range(len(validation_tasks)):
                x_train = validation_tasks[validation_task_idx].training.features.values
                y_train = validation_tasks[validation_task_idx].training.labels.values.ravel()
                # FIXME The validation of the validation tasks is going unused here
                x_test = validation_tasks[validation_task_idx].test.features.values
                y_test = validation_tasks[validation_task_idx].test.labels.values.ravel()

                temp_best_val_perf = np.Inf
                for temp_regul_param in self.settings.regularization_parameter_range:
                    w = self._solve_wrt_w(mean_vector, x_train, y_train, temp_regul_param)

                    curr_predictions = pd.Series(x_test @ w, index=validation_tasks[validation_task_idx].test.labels.index)
                    raw_predictions = labels_to_raw(curr_predictions, validation_tasks[validation_task_idx].test.raw_time_series, self.settings.horizon)
                    raw_labels = validation_tasks[validation_task_idx].test.raw_time_series.loc[raw_predictions.index]
                    temp_val_perf = self._performance_check(raw_labels, raw_predictions)
                    if temp_val_perf < temp_best_val_perf:
                        temp_best_val_perf = temp_val_perf
                validation_performances.append(temp_best_val_perf)
            validation_performance = np.mean(validation_performances)
            print(f'LTL | lambda: {regularization_parameter:6e} | val performance: {validation_performance:12.5f}')

            if validation_performance < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False

            if validation_criterion:
                best_val_performance = validation_performance
                best_param = regularization_parameter

                best_average_vectors = all_average_vectors
                best_mean_vector = mean_vector
        print(f'LTL | lambda: {best_param:6e} | val performance: {best_val_performance:20.16f}')
        self.all_metaparameters = best_average_vectors
        self.final_metaparameters = best_mean_vector

    def predict(self, test_tasks):
        test_tasks = handle_data(test_tasks, self.lags, self.settings.use_exog)
        test_per_per_training_task = []
        tt = time()
        for meta_param_idx in [len(self.all_metaparameters)-1]:
        # for meta_param_idx in range(len(self.all_metaparameters)):
            meta_param = self.all_metaparameters[meta_param_idx]
            all_test_perf = []
            predictions = []
            all_raw_predictions = []
            for task_idx in range(len(test_tasks)):
                x_train = test_tasks[task_idx].training.features.values
                y_train = test_tasks[task_idx].training.labels.values.ravel()

                x_val = test_tasks[task_idx].validation.features.values
                y_val = test_tasks[task_idx].validation.labels.values.ravel()

                best_val_performance = np.Inf
                for regularization_parameter in self.settings.regularization_parameter_range:
                    w = self._solve_wrt_w(meta_param, x_train, y_train, regularization_parameter)

                    curr_predictions = pd.Series(x_val @ w, index=test_tasks[task_idx].validation.labels.index)
                    raw_predictions = labels_to_raw(curr_predictions, test_tasks[task_idx].validation.raw_time_series, self.settings.horizon)
                    raw_labels = test_tasks[task_idx].validation.raw_time_series.loc[raw_predictions.index]
                    validation_performance = self._performance_check(raw_labels, raw_predictions)

                    if validation_performance < best_val_performance:
                        validation_criterion = True
                    else:
                        validation_criterion = False

                    if validation_criterion:
                        best_val_performance = validation_performance
                        best_w = w

                x_test = test_tasks[task_idx].test.features.values
                y_test = test_tasks[task_idx].test.labels.values.ravel()
                curr_predictions = pd.Series(x_test @ best_w, index=test_tasks[task_idx].test.labels.index)
                raw_predictions = labels_to_raw(curr_predictions, test_tasks[task_idx].test.raw_time_series, self.settings.horizon)
                raw_labels = test_tasks[task_idx].test.raw_time_series.loc[raw_predictions.index]
                all_raw_predictions.append(raw_predictions)

                test_performance = self._performance_check(raw_labels, raw_predictions)

                all_test_perf.append(test_performance)
                predictions.append(curr_predictions)
            avg_perf = float(np.mean(all_test_perf))
            test_per_per_training_task.append(avg_perf)
            print('%3d/%3d | %5.2fsec' % (meta_param_idx, len(self.all_metaparameters), time() - tt))
        self.all_predictions = predictions
        self.all_raw_predictions = all_raw_predictions
        self.test_per_per_training_task = test_per_per_training_task
        print(f'LTL | lambda: {np.nan:6e} | test performance: {avg_perf:20.16f}')

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(y_test)
        # plt.plot(curr_prediction)
        # plt.pause(0.1)

        if len(test_per_per_training_task) >= 2:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(test_per_per_training_task)
            plt.ticklabel_format(useOffset=False)
            plt.pause(0.01)
            # plt.show()

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
        y_true = y_true.values.ravel()
        y_pred = y_pred.values.ravel()
        # Make sure that if y_true is 0 then you return 0
        rel_error = np.abs(np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true), where=(y_true != 0)))
        mape = (100 / len(y_true)) * np.sum(rel_error)
        return mape
