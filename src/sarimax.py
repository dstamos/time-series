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

    def fit(self, time_series, exog_variables=None):
        if self.settings.use_exog is True:
            exog_variables = exog_variables  # FIXME.diff().fillna(method='bfill')
        else:
            exog_variables = None

        model = SARIMAX(time_series, exog=exog_variables,
                        order=(self.ar_order, self.difference_order, self.ma_order),
                        seasonal_order=(self.seasonal_ar_order, self.seasonal_difference_order, self.seasonal_ma_order, self.seasonal_period))
        self.model = model.fit(dips=1, maxiter=150)

    def predict(self, exog_variables=None, foreward_periods=1):
        if self.settings.use_exog is True:
            exog_variables = exog_variables
        else:
            exog_variables = None

        preds = self.model.predict(steps=foreward_periods, exog=exog_variables)
        forecast_table = preds.summary_frame(alpha=0.10)
        self.prediction = forecast_table['mean']
