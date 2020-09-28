import numpy as np
import pandas as pd
import xgboost
import multiprocess as mp
from statsmodels.tsa.statespace.sarimax import SARIMAX

from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
from src.utilities import lag_features, prune_data
from src.nbeats.model import NBeatsNet
import torch
from torch import optim
from torch.nn import functional as F


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
        model_exog = SARIMAX(time_series, exog=exog_variables,
                             order=(self.ar_order, self.difference_order, self.ma_order),
                             seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period))
        self.model = model_exog.fit(disp=1, maxiter=150)

    def forecast(self, exog_variables=None, foreward_periods=1):
        if self.settings.training.use_exog is True:
            exog_variables = exog_variables
        else:
            exog_variables = None

        preds = self.model.get_forecast(steps=foreward_periods, exog=exog_variables)
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


class NBeats:
    def __init__(self, settings):
        import torch
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.settings = settings
        self.forecast_length = settings.training.forecast_length
        self.lookback_length = settings.training.lookback_length

        self.model = None
        self.prediction = None

    def fit(self, data):
        if self.settings.training.use_exog is True:
            features_tr = lag_features(data.features_tr.multioutput, self.lookback_length)
            features_tr, labels_tr = prune_data(features_tr, data.labels_tr.multioutput)
        else:
            raise ValueError('Featureless nbeats not implemented.')

        dataset = TensorDataset(tensor(features_tr.values, dtype=torch.float), tensor(labels_tr.values, dtype=torch.float))
        trainloader = DataLoader(dataset, shuffle=True, batch_size=self.settings.training.batch_size)

        print('--- Model ---')
        net = NBeatsNet(device=self.device,
                        stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK],
                        forecast_length=self.forecast_length,
                        thetas_dims=[2, 8, 3],
                        nb_blocks_per_stack=3,
                        backcast_length=self.lookback_length,
                        hidden_layer_units=64,  # 1024
                        share_weights_in_stack=False,
                        nb_harmonics=None)

        optimiser = optim.Adam(net.parameters())

        max_grad_steps = 10000

        initial_grad_step = 0
        max_epochs = 1
        losses = []
        for epoch in range(max_epochs):
            for grad_step, (x, target) in enumerate(trainloader):
                grad_step += initial_grad_step
                optimiser.zero_grad()
                net.train()
                backcast, forecast = net(x.to(self.device))
                loss = F.mse_loss(forecast, target.to(self.device))
                loss.backward()
                optimiser.step()
                print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
                losses.append(loss)
                # if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
                #     with torch.no_grad():
                #         save(net, optimiser, grad_step)
                #         if on_save_callback is not None:
                #             on_save_callback(x, target, grad_step)
                if grad_step > max_grad_steps:
                    print('Finished.')
                    break
            print(epoch, 'done')

        self.model = net

    def forecast(self, data, period=1):
        if self.settings.training.use_exog is True:
            features_ts = lag_features(data.features_ts.multioutput, self.lookback_length)
            features_ts = prune_data(features_ts)
        else:
            raise ValueError('Featureless nbeats not implemented.')

        torch.no_grad()
        _, forecast = self.model(torch.tensor(features_ts.values, dtype=torch.float).to(self.device))
        self.prediction = pd.DataFrame([np.array(forecast[i].data[0]) for i in range(len(forecast))], index=features_ts.index)