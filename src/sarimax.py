import time
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
        self.all_forecasts = None
        self.all_test_perf = None

    def fit(self, test_tasks, exog_variables=None):
        # if self.settings.use_exog is True:
        #     exog_variables = exog_variables
        # else:
        #     exog_variables = None

        exog_variables = None

        # We are not doing any validation and retraining at the moment
        all_models = []
        tt = time.time()
        for task_idx in range(len(test_tasks)):
            tr_time_series = test_tasks[task_idx].training.raw_time_series
            val_time_series = test_tasks[task_idx].validation.raw_time_series
            time_series = pd.concat([tr_time_series, val_time_series])
            model = SARIMAX(time_series, exog=exog_variables,
                            order=(self.ar_order, self.difference_order, self.ma_order),
                            seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period),
                            enforce_stationarity=False)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = model.fit(disp=0, maxiter=150)
            all_models.append(model)
            self.all_models = all_models
            print('SARIMAX | task: %3d | time: %8.2f sec' % (task_idx, time.time() - tt))

    def predict(self, test_tasks, exog_variables=None, foreward_periods=1):
        # if self.settings.use_exog is True:
        #     exog_variables = exog_variables
        # else:
        #     exog_variables = None

        exog_variables = None

        all_predictions = []
        all_forecasts = []
        all_test_perf = []
        for task_idx in range(len(test_tasks)):
            model = self.all_models[task_idx]
            test_time_series = test_tasks[task_idx].test.raw_time_series
            curr_forecast = model.forecast(steps=len(test_time_series), exog_variables=exog_variables)

            tr_time_series = test_tasks[task_idx].training.raw_time_series
            val_time_series = test_tasks[task_idx].validation.raw_time_series

            horizon = self.settings.horizon
            curr_predictions = pd.Series(index=test_time_series.index)
            for idx in range(1, len(test_time_series) - horizon):
                curr_test_time_series = test_time_series.iloc[:idx]
                time_series = pd.concat([tr_time_series, val_time_series, curr_test_time_series])

                mod = SARIMAX(time_series, exog=exog_variables,
                              order=(self.ar_order, self.difference_order, self.ma_order),
                              seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period))
                res = mod.filter(model.params)
                # forecast = res.forecast(steps=self.settings.horizon)
                forecast = res.forecast(steps=horizon).iloc[-1]
                curr_predictions.iloc[idx + horizon] = forecast
            curr_predictions.dropna(inplace=True)
            test_performance = self._performance_check(test_time_series.loc[curr_predictions.index], curr_predictions)
            all_predictions.append(curr_predictions)
            all_forecasts.append(curr_forecast)
            all_test_perf.append(test_performance)
        self.all_predictions = all_predictions
        self.all_forecasts = all_forecasts
        self.all_test_perf = all_test_perf
        print(f'test performance: {np.nanmean(all_test_perf):20.16f}')

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(len(test_tasks)):
        #     plt.plot(test_tasks[i].test.raw_time_series, 'k', label='raw ts')
        #     plt.plot(all_predictions[i], 'tab:red', label='predictions')
        #     plt.plot(all_forecasts[i], 'tab:blue', label='forecasts')
        #     plt.legend()
        #     plt.pause(0.01)
        #     plt.show()

    @staticmethod
    def _performance_check(y_true, y_pred):
        y_true = y_true.values.ravel()
        y_pred = y_pred.values.ravel()
        # Make sure that if y_true is 0 then you return 0
        rel_error = np.abs(np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true), where=(y_true != 0)))
        mape = (100 / len(y_true)) * np.sum(rel_error)
        return mape
