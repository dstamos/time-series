import numpy as np
import pandas as pd
from src.utilities import handle_data, labels_to_raw, performance_check
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
        self.all_all_test_perf = None
        self.test_per_per_training_task = None

    def fit(self, training_tasks, validation_tasks):
        training_tasks = handle_data(training_tasks, self.lags, self.settings.use_exog)
        validation_tasks = handle_data(validation_tasks, self.lags, self.settings.use_exog)

        dims = training_tasks[0].training.features.shape[1]

        best_val_performance = np.Inf
        mean_vector = np.random.randn(dims) / norm(np.random.randn(dims))
        tt = time()
        for regularization_parameter in self.settings.regularization_parameter_range:
            validation_performances = []
            all_average_vectors = []

            # For the sake of faster training
            # mean_vector = best_mean_vector
            for task_idx in range(len(training_tasks)):
                # prev_mean_vector = mean_vector
                #####################################################
                # Optimisation
                x_train = training_tasks[task_idx].training.features.values
                y_train = training_tasks[task_idx].training.labels.values.ravel()

                # mean_vector = self._solve_wrt_h(mean_vector, x_train, y_train, regularization_parameter, curr_iteration=task_idx, inner_iter_cap=10)
                #
                # if not all_average_vectors:
                #     all_average_vectors.append(mean_vector)
                # else:
                #     all_average_vectors.append(task_idx * np.nanmean(all_average_vectors, axis=0) + mean_vector / (task_idx + 1))

                mean_vector = self._solve_wrt_h(mean_vector, x_train, y_train, regularization_parameter, curr_iteration=task_idx, inner_iter_cap=10)
                if all_average_vectors:
                    mean_vector = (task_idx * all_average_vectors[-1] + mean_vector) / (task_idx + 1)
                all_average_vectors.append(mean_vector)
            # mean_vector = all_average_vectors[-1]
            #####################################################
            # Validation only needs to be measured at the very end, after we've trained on all training tasks
            for validation_task_idx in range(len(validation_tasks)):
                x_train = validation_tasks[validation_task_idx].training.features.values
                y_train = validation_tasks[validation_task_idx].training.labels.values.ravel()
                x_val = validation_tasks[validation_task_idx].validation.features.values
                y_val = validation_tasks[validation_task_idx].validation.labels

                w = self._solve_wrt_w(mean_vector, x_train, y_train, regularization_parameter)
                # w = mean_vector

                curr_predictions = pd.Series(x_val @ w, index=y_val.index)
                errors = performance_check(y_val, curr_predictions)
                temp_val_perf = errors['nmse']
                validation_performances.append(temp_val_perf)
            validation_performance = np.mean(validation_performances)
            print(f'LTL | lambda: {regularization_parameter:6e} | val performance: {validation_performance:12.5f} | {time() - tt:5.2f}sec')

            if validation_performance < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False

            if validation_criterion:
                best_val_performance = validation_performance
                best_param = regularization_parameter

                best_average_vectors = all_average_vectors
                best_mean_vector = mean_vector
        print(f'LTL | best lambda: {best_param:6e} | best val performance: {best_val_performance:12.5f} | {time() - tt:5.2f}sec')
        self.all_metaparameters = best_average_vectors
        self.final_metaparameters = best_mean_vector

    def predict(self, test_tasks):
        test_tasks = handle_data(test_tasks, self.lags, self.settings.use_exog)
        test_per_per_training_task = []
        all_all_test_perf = []
        tt = time()

        if len(test_tasks) < 100:
            params_to_check = range(len(self.all_metaparameters))
        else:
            # To speed up the process
            params_to_check = np.arange(0, 50, 1)
            params_to_check = np.concatenate((params_to_check, np.arange(50, len(self.all_metaparameters), 50)))
            if len(self.all_metaparameters) - 1 not in params_to_check:
                params_to_check = np.append(params_to_check, len(self.all_metaparameters) - 1)
        for meta_param_idx in params_to_check:
        # for meta_param_idx in [len(self.all_metaparameters) - 1]:
            meta_param = self.all_metaparameters[meta_param_idx]
            all_test_perf = []
            predictions = []
            all_raw_predictions = []
            for task_idx in range(len(test_tasks)):
                x_train = test_tasks[task_idx].training.features.values
                y_train = test_tasks[task_idx].training.labels.values.ravel()

                x_val = test_tasks[task_idx].validation.features.values
                y_val = test_tasks[task_idx].validation.labels

                # best_w = meta_param

                best_val_performance = np.Inf
                for regularization_parameter in self.settings.regularization_parameter_range:
                    w = self._solve_wrt_w(meta_param, x_train, y_train, regularization_parameter)

                    curr_predictions = pd.Series(x_val @ w, index=y_val.index)
                    errors = performance_check(y_val, curr_predictions)
                    validation_performance = errors['nmse']

                    if validation_performance < best_val_performance:
                        validation_criterion = True
                    else:
                        validation_criterion = False

                    if validation_criterion:
                        best_val_performance = validation_performance
                        best_w = w

                x_test = test_tasks[task_idx].test.features.values
                y_test = test_tasks[task_idx].test.labels
                curr_predictions = pd.Series(x_test @ best_w, index=y_test.index)
                all_raw_predictions.append(curr_predictions)

                errors = performance_check(y_test, curr_predictions)
                test_performance = errors['nmse']

                all_test_perf.append(test_performance)
                predictions.append(curr_predictions)
            avg_perf = float(np.mean(all_test_perf))
            all_all_test_perf.append(all_test_perf)
            test_per_per_training_task.append(avg_perf)
            print(f'LTL | {meta_param_idx:4d} |test performance: {avg_perf:12.5f} | {time() - tt:5.2f}sec')
        self.all_predictions = predictions
        # self.all_raw_predictions = all_raw_predictions
        self.test_per_per_training_task = test_per_per_training_task
        self.all_all_test_perf = all_all_test_perf
        print(f'LTL | lambda: {np.nan:6e} | test performance: {avg_perf:12.5f}')

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(y_test)
        # plt.plot(curr_prediction)
        # plt.pause(0.1)

        # if len(test_per_per_training_task) >= 2:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(test_per_per_training_task)
        #     plt.ticklabel_format(useOffset=False)
        #     plt.pause(3)
        #     plt.show()

    @staticmethod
    def _solve_wrt_h(h, x, y, param, curr_iteration=0, inner_iter_cap=10):
        step_size_bit = 1e+3
        n = len(y)

        def grad(curr_h):
            from scipy.linalg import lstsq

            c_n_hat = x.T @ x / n + param * np.eye(x.shape[1])
            x_n_hat = (param / np.sqrt(n) * lstsq(c_n_hat.T, x.T)[0]).T
            y_n_hat = 1 / np.sqrt(n) * (y - x @ lstsq(c_n_hat, x.T @ y)[0] / n)

            grad_h = x_n_hat.T @ (x_n_hat @ curr_h - y_n_hat)
            grad_h = np.clip(grad_h, a_min=-10**3, a_max=10**3)
            return grad_h
            # import warnings
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore")
            #     return 2 * param**2 * n * x.T @ matrix_power(pinv(x @ x.T + param * n * np.eye(n)), 2) @ ((x @ curr_h).ravel() - y)

        i = 0
        curr_iteration = curr_iteration * inner_iter_cap
        while i < inner_iter_cap:
            i = i + 1
            prev_h = h
            curr_iteration = curr_iteration + 1
            step_size = min(np.sqrt(2) * step_size_bit / ((step_size_bit + 1) * np.sqrt(curr_iteration)), 0.5)
            h = prev_h - step_size * grad(prev_h)

        return h

    @staticmethod
    def _solve_wrt_w(h, x, y, param):
        n = len(y)
        dims = x.shape[1]

        from scipy.linalg import lstsq
        w = lstsq(x.T @ x / n + param * np.eye(dims), (x.T @ y / n + param * h))[0]

        return w
