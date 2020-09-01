import numpy as np
import pandas as pd
import xgboost
import multiprocess as mp
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

        self.model = None
        self.prediction = None

    def fit(self, data):
        horizon = self.settings.data.horizon
        n_horizons_lookback = self.settings.data.n_horizons_lookback
        if self.settings.training.use_exog is True:
            exog_variables = data.features.iloc[:horizon*n_horizons_lookback].diff().fillna(method='bfill')
        else:
            exog_variables = None
        model_exog = SARIMAX(data.labels.iloc[:horizon*n_horizons_lookback], exog=exog_variables,
                             order=(self.ar_order, self.difference_order, self.ma_order),
                             seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period))
        self.model = model_exog.fit(disp=True, maxiter=150)

    def forecast(self, data, period=1):
        horizon = self.settings.data.horizon
        n_horizons_lookback = self.settings.data.n_horizons_lookback
        if self.settings.training.use_exog is True:
            exog_variables = data.features.iloc[horizon*n_horizons_lookback:horizon*n_horizons_lookback+period].diff().fillna(method='bfill')
        else:
            exog_variables = None

        preds = self.model.get_forecast(steps=period, exog=exog_variables)
        forecast_table = preds.summary_frame(alpha=0.10)
        self.prediction = forecast_table['mean']


class Xgboost:
    def __init__(self, settings):
        self.settings = settings
        self.lags = 3

        self.model = None
        self.prediction = None

    def fit(self, data):
        horizon = self.settings.data.horizon
        n_horizons_lookback = self.settings.data.n_horizons_lookback
        labels = data.labels.iloc[:horizon*n_horizons_lookback]
        if self.settings.training.use_exog is True:
            x = data.features.iloc[:horizon*n_horizons_lookback].diff().fillna(method='bfill')
            # x = self._time_features(x)
            features = self._lag_features(x)
            features, labels = self._prune_data(features, labels)
        else:
            raise ValueError('Featureless xgboost not implemented.')
        import xgboost
        dmatrix = xgboost.DMatrix(features, labels)

        # noinspection PyUnresolvedReferences
        params = {'max_depth': 6,  # 6
                  'colsample_bytree': 0.2,
                  'eta': 0.05,
                  'num_parallel_tree': 20,  # 20
                  'n_jobs': -2,
                  'obj': 'reg:squarederror',
                  'verbosity': 0,
                  'seed': 2130,  # TODO This seed is fixed at the moment
                  # 'eval_metric': 'rmse',
                  'booster': 'dart',  # gbtree
                  'min_child_weight': 2,
                  'nthread': mp.cpu_count()-1}
        self.model = xgboost.train(params, dmatrix, num_boost_round=500, evals=[(dmatrix, "training")], verbose_eval=1)

    def forecast(self, data, period=1):
        horizon = self.settings.data.horizon
        n_horizons_lookback = self.settings.data.n_horizons_lookback
        if self.settings.training.use_exog is True:
            x = data.features.iloc[horizon*n_horizons_lookback:horizon*n_horizons_lookback+period].diff().fillna(method='bfill')
            # x = self._time_features(x)
            features = self._lag_features(x)
            features = self._prune_data(features)
        else:
            raise ValueError('Featureless xgboost not implemented.')

        self.prediction = pd.DataFrame(self.model.predict(xgboost.DMatrix(features)), index=features.index)

    @staticmethod
    def _time_features(indicators):
        datetimes = indicators.index

        times = datetimes.hour

        F = np.zeros((len(times), 24))
        F[range(len(times)), times] = 1
        for idx in range(24):
            indicators['calendar_' + str(idx)] = F[:, idx]
        return indicators

    def _lag_features(self, indicators):
        lagged_indicators = [None] * self.lags
        original_column_names_to_lag = indicators.columns.values.tolist()
        original_column_names_to_lag = [s for s in original_column_names_to_lag if 'calendar' not in s]     # for selective lags
        for idx, c_lag in enumerate(range(self.lags)):
            column_names = [column_name + '-lagged_' + str(c_lag) for column_name in original_column_names_to_lag]
            temp = indicators[original_column_names_to_lag].shift(c_lag)
            temp.columns = column_names
            lagged_indicators[idx] = temp

        lagged_indicators.insert(0, indicators)
        lagged_indicators = pd.concat(lagged_indicators, axis=1)

        return lagged_indicators

    @ staticmethod
    def _prune_data(features, labels=None):
        features = features.replace([np.inf, -np.inf], np.nan)
        indexes_to_drop = features.index[pd.isnull(features).any(1).to_numpy().nonzero()[0]]

        features = features.drop(indexes_to_drop)
        if labels is not None:
            labels = labels.drop(indexes_to_drop)
            return features, labels
        else:
            return features
