import numpy as np
import pandas as pd
import xgboost
import multiprocess as mp
import time
from src.utilities import lag_features, prune_data


class Xgboost:
    def __init__(self, settings):
        self.settings = settings
        self.lags = settings.lags

        self.model = None
        self.prediction = None

    def fit(self, labels, x=None):
        if self.settings.use_exog is True:
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
        if self.settings.use_exog is True:
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
