import numpy as np
import pandas as pd
from src.utilities import handle_data
import xgboost
import multiprocess as mp
import time


class Xgboost:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.lags

        self.all_models = None
        self.all_test_perf = None
        self.all_predictions = None

    def fit(self, test_tasks):
        test_tasks = handle_data(test_tasks, self.lags, self.settings.use_exog)

        all_models = []
        tt = time.time()
        for task_idx in range(len(test_tasks)):
            x_train = test_tasks[task_idx].training.features.values
            y_train = test_tasks[task_idx].training.labels.values.ravel()

            dmatrix = xgboost.DMatrix(x_train, y_train)

            def heartbeat():
                def callback(env):
                    print('%5d | %8s ' % (env.iteration, time.strftime("%H:%M:%S")))
                return callback

            # noinspection PyUnresolvedReferences
            params = {'max_depth': 6,
                      'colsample_bytree': 0.2,
                      'eta': 0.05,
                      'num_parallel_tree': 2,
                      'n_jobs': -2,
                      'obj': 'reg:squarederror',
                      'verbosity': 0,
                      'seed': 1,
                      'booster': 'dart',
                      'min_child_weight': 2,
                      'nthread': mp.cpu_count()-1}
            model = xgboost.train(params, dmatrix, num_boost_round=500, callbacks=[])
            all_models.append(model)
            print('xgboost | task: %3d | time: %8.5fsec' % (task_idx, time.time() - tt))
        self.all_models = all_models
        self._predict(test_tasks)

    def _predict(self, test_tasks):
        all_test_perf = []
        predictions = []
        for task_idx in range(len(test_tasks)):
            x_test = test_tasks[task_idx].test.features.values
            y_test = test_tasks[task_idx].test.labels.values.ravel()

            x_test = xgboost.DMatrix(x_test)

            curr_prediction = pd.Series(self.all_models[task_idx].predict(x_test), index=test_tasks[task_idx].test.labels.index)
            test_performance = self._performance_check(y_test, curr_prediction.values.ravel())
            all_test_perf.append(test_performance)
            predictions.append(curr_prediction)
        self.all_predictions = predictions
        self.all_test_perf = all_test_perf

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(y_test)
        # plt.plot(curr_prediction)
        # plt.pause(0.1)

    @staticmethod
    def _performance_check(y_true, y_pred):
        # Make sure that if y_true is 0 then you return 0
        rel_error = np.abs(np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true), where=(y_true != 0)))
        mape = (100 / len(y_true)) * np.sum(rel_error)
        return mape
