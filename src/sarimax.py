import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


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

        self.all_models = None
        self.all_predictions = None

    def fit(self, test_tasks, exog_variables=None):
        # if self.settings.use_exog is True:
        #     exog_variables = exog_variables
        # else:
        #     exog_variables = None

        exog_variables = None
        # We are not doing any validation and retraining at the moment
        all_models = []
        for task_idx in range(len(test_tasks)):
            tr_time_series = test_tasks[task_idx].training.raw_time_series
            val_time_series = test_tasks[task_idx].validation.raw_time_series
            time_series = pd.concat([tr_time_series, val_time_series])
            model = SARIMAX(time_series, exog=exog_variables,
                            order=(self.ar_order, self.difference_order, self.ma_order),
                            seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period))
            model = model.fit(dips=1, maxiter=5)
            all_models.append(model)
            self.all_models = all_models

    def predict(self, test_tasks, exog_variables=None, foreward_periods=1):
        # if self.settings.use_exog is True:
        #     exog_variables = exog_variables
        # else:
        #     exog_variables = None

        exog_variables = None
        all_predictions = []
        for task_idx in range(len(test_tasks)):
            test_time_series = test_tasks[task_idx].test.raw_time_series
            curr_predictions = self.all_models[task_idx].forecast(steps=len(test_time_series), exog_variables=exog_variables)
            test_performance = self._performance_check(test_time_series.values.ravel(), curr_predictions.values.ravel())
            all_predictions.append(curr_predictions)
            self.all_predictions = all_predictions
        import matplotlib.pyplot as plt
        plt.plot(test_time_series, 'tab:blue')
        plt.plot(curr_predictions, 'tab:red')
        plt.pause(0.01)
        plt.show()

    @staticmethod
    def _performance_check(y_true, y_pred):
        # Make sure that if y_true is 0 then you return 0
        rel_error = np.abs(np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true), where=(y_true != 0)))
        mape = (100 / len(y_true)) * np.sum(rel_error)
        return mape
