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

        model_itl = ARMA()
        model_itl.fit(x.iloc[x.shape[1]-1:], y.iloc[x.shape[1]-1:])
        # FIXME Remove the above 'pruning'

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from statsmodels.tsa.arima.model import ARIMA

            # model_arima = ARIMA(endog=y, order=(3, 0, 2), trend='c', enforce_stationarity=False, enforce_invertibility=False)
            # model_arima = model_arima.fit()

            from statsmodels.tsa.ar_model import AutoReg
            model_arima = AutoReg(endog=y, lags=3, trend='c')
            model_arima = model_arima.fit()

        np.set_printoptions(suppress=True)
        pd.set_option('display.max_rows', 12)
        pd.set_option('display.max_columns', 5000)
        pd.set_option('display.width', 40000)

        print('custom:\n', model_itl.weight_vector)
        print('')
        print('statsmodels:\n', model_arima.params.to_frame())
        # print('statsmodels:\n', model_arima.params.to_frame().drop('sigma2'))

        # Testing
        test_predictions = model_itl.predict(x_test)
        all_performances.append(normalised_mse(y_test, test_predictions))
    test_performance = np.mean(all_performances)
    print(f'{"ARMA":12s} | test performance: {test_performance:12.5f} | {time() - tt:6.1f}sec')

    return test_performance


class ARMA:
    def __init__(self):
        self.max_lag_ma = 0
        self.weight_vector = None

    def fit(self, features, labels):
        w_ar = lstsq(features.T @ features, features.T @ labels)[0].ravel()

        y_tr_pred_ar = (features @ w_ar).to_frame()
        y_tr_pred_ar.columns = ['labels']

        tr_residuals = labels - y_tr_pred_ar
        tr_residuals = tr_residuals

        x_tr_ma, y_tr_ma, _ = get_features_labels_from_timeseries([tr_residuals], self.max_lag_ma, diff_order=0, shift=0)
        x_tr_ma, y_tr_ma = x_tr_ma[0], y_tr_ma[0]

        if self.max_lag_ma != 0:
            # Merge MA and AR features. Fill in MA features with 0 backwards.
            features_merged = pd.concat([features, x_tr_ma], axis=1).replace(np.nan, 0)
        else:
            features_merged = features

        w_ar = lstsq(features_merged.T @ features_merged, features_merged.T @ labels)[0].ravel()

        w_ar = pd.DataFrame(w_ar, index=features_merged.columns)

        # x_tr_ma = x_tr_ma.iloc[x_tr_ma.shape[1]-1:]
        # y_tr_ma = y_tr_ma.iloc[x_tr_ma.shape[1]-1:]

        # x_tr_ma = np.r_[np.zeros((self.max_lag_ma, x_tr_ma.shape[1])), x_tr_ma]
        # y_tr_ma = np.r_[np.zeros(self.max_lag_ma), y_tr_ma.values.ravel()]

        # w_ma = lstsq(x_tr_ma.T @ x_tr_ma, x_tr_ma.T @ y_tr_ma)[0].ravel()

        self.weight_vector = w_ar

    def predict(self, features):
        y_pred_ar = (features @ self.weight_vector).to_frame()
        y_pred_ar.columns = ['labels']

        # MA pred
        labels = features['lag_0'].shift(-1).to_frame()
        labels.columns = ['labels']
        y_residuals = labels - y_pred_ar
        # x_residuals_ma, _, _ = get_features_labels_from_timeseries([y_residuals], self.max_lag_ma)

        x_residuals_ma, _, _ = get_features_labels_from_timeseries([y_residuals], self.max_lag_ma, diff_order=None, shift=0)
        x_residuals_ma = x_residuals_ma[0]
        x_residuals_ma = pd.DataFrame(np.r_[np.zeros((self.max_lag_ma, x_residuals_ma.shape[1])), x_residuals_ma], index=features.index)

        y_pred_ma = (x_residuals_ma @ self.weight).to_frame()

        # y_pred_ma = (x_residuals_ma[0] @ self.weight_vector_ma).to_frame()
        y_pred_ma.columns = ['labels']

        if self.max_lag_ma:
            pred = y_pred_ar
        else:
            pred = y_pred_ar + y_pred_ma
        return pred
