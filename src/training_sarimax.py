import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Sarimax:
    def __init__(self, settings):
        self.settings = settings
        self.ar_order = 3
        self.difference_order = 1
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
        self.model = model_exog.fit(disp=True, maxiter=250)

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
