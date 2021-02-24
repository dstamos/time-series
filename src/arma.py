from src.timeseries_utilities import get_features_labels_from_timeseries
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from src.preprocessing import PreProcess
from time import time
from src.utilities import normalised_mse


def train_test_arma(data, settings):
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

        preprocessing = PreProcess(standard_scaling=False, inside_ball_scaling=False, add_bias=True)

        # Retrain on full training set
        x, y = preprocessing.transform(x, y, fit=True, multiple_tasks=False)
        x_test, y_test = preprocessing.transform(data['test_tasks_test_features'][task_idx], data['test_tasks_test_labels'][task_idx], fit=False, multiple_tasks=False)

        pd.options.display.float_format = '{:10.6f}'.format

        model_arma = ARMA()
        model_arma.fit(x.iloc[x.shape[1]-1:], y.iloc[x.shape[1]-1:])
        # FIXME Remove the above 'pruning'

        np.set_printoptions(suppress=True)
        pd.set_option('display.max_rows', 12)
        pd.set_option('display.max_columns', 5000)
        pd.set_option('display.width', 40000)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # from statsmodels.tsa.arima.model import ARIMA
            # model_arima = ARIMA(endog=y, order=(6, 0, 12), trend='c', enforce_stationarity=False, enforce_invertibility=False)
            # model_arima = model_arima.fit()
            # print('statsmodels:\n', model_arima.params.to_frame().drop('sigma2'))
            #
            # mod = ARIMA(y, order=(6, 0, 12))
            # res = mod.filter(model_arima.params)
            # test_predictions = res.forecast(steps=len(x_test) + 1).to_frame()
            # test_predictions.columns = ['labels']
            # test_predictions = test_predictions.loc[y_test.index]

            # from statsmodels.tsa.ar_model import AutoReg
            # model_arima = AutoReg(endog=y, lags=3, trend='c')
            # model_arima = model_arima.fit()
            # print('statsmodels:\n', model_arima.params.to_frame())

        # print('')
        # print('custom:\n', model_arma.weight_vector_arma)

        # Testing
        test_predictions = model_arma.predict(x_test)
        all_performances.append(normalised_mse(y_test, test_predictions))
    test_performance = np.mean(all_performances)
    print(f'{"ARMA":12s} | test performance: {test_performance:12.5f} | {time() - tt:6.1f}sec')

    return test_performance


class ARMA:
    def __init__(self):
        self.max_lag_ma = 12
        self.weight_vector_arma = None
        self.weight_vector_ar = None

    def fit(self, features, labels):
        w_ar = lstsq(features.T @ features, features.T @ labels)[0].ravel()

        y_tr_pred_ar = (features @ w_ar).to_frame()
        y_tr_pred_ar.columns = ['labels']

        tr_residuals = labels - y_tr_pred_ar
        tr_residuals = tr_residuals

        # import matplotlib.pyplot as plt
        # plt.plot(labels)
        # plt.plot(y_tr_pred_ar)
        # plt.pause(0.1)
        #
        # plt.plot(tr_residuals)
        # plt.pause(0.1)

        x_tr_ma, _, _ = get_features_labels_from_timeseries([tr_residuals], self.max_lag_ma, diff_order=0, shift=0)
        x_tr_ma = x_tr_ma[0]
        # x_tr_ma = x_tr_ma.shift() ######################################################

        if self.max_lag_ma != 0:
            # Merge MA and AR features. Fill in MA features with 0 backwards.
            features_merged = pd.concat([features, x_tr_ma], axis=1).replace(np.nan, 0)
        else:
            features_merged = features

        w_arma = lstsq(features_merged.T @ features_merged, features_merged.T @ labels)[0].ravel()

        w_arma = pd.DataFrame(w_arma, index=features_merged.columns)

        # import matplotlib.pyplot as plt
        # plt.plot(labels)
        # plt.plot((features_merged @ w_arma))
        # plt.pause(0.1)

        self.weight_vector_arma = w_arma
        self.weight_vector_ar = w_ar

    def predict(self, features):
        y_pred_ar = (features @ self.weight_vector_ar).to_frame()
        y_pred_ar.columns = ['labels']

        # MA pred
        labels = features['lag_0'].shift(-1).to_frame()
        labels.columns = ['labels']
        y_residuals = labels - y_pred_ar

        # import matplotlib.pyplot as plt
        # plt.plot(labels)
        # plt.plot(y_pred_ar)
        # plt.pause(0.1)
        #
        # plt.plot(y_residuals)
        # plt.pause(0.1)

        x_residuals_ma, _, _ = get_features_labels_from_timeseries([y_residuals], self.max_lag_ma, diff_order=0, shift=0)
        x_residuals_ma = x_residuals_ma[0]
        # x_residuals_ma = x_residuals_ma.shift() ######################################################

        if self.max_lag_ma != 0:
            # Merge MA and AR features. Fill in MA features with 0 backwards.
            features_merged = pd.concat([features, x_residuals_ma], axis=1).replace(np.nan, 0)
        else:
            features_merged = features

        y_pred_arma = features_merged @ self.weight_vector_arma
        y_pred_arma.columns = ['labels']

        # plt.plot(labels)
        # plt.plot(y_pred_arma)
        # plt.pause(0.1)

        return y_pred_arma
