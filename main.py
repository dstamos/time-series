from src.data_management import DataHandler, Settings
from src.training import Sarimax, Xgboost, NBeats


def main():
    data_settings = {'dataset': 'AirQualityUCI',
                     'label': 'CO(GT)',
                     'label_period': 1,
                     'training_percentage': 0.03}

    # data_settings = {'dataset': 'm4',
    #                  'm4_time_series_idx': 0,
    #                  'forecast_length': 5
    #                  }

    training_settings = {'method': 'SARIMAX',
                         'use_exog': True}

    # training_settings = {'method': 'NBeats',
    #                      'use_exog': True,
    #                      'lookback_length': 4,
    #                      'forecast_length': data_settings['forecast_length'],
    #                      'batch_size': 8
    #                      }

    # data_settings = {'dataset': 'm4',
    #                  'horizon': forecast_length,
    #                  'm4_time_series_idx': 0,
    #                  'n_horizons_lookback': 3}

    settings = Settings(data_settings, 'data')
    settings.add_settings(training_settings, 'training')
    data = DataHandler(settings)

    # TODO Get Sarimax to work with m4 + AirQualityUCI. (forecast/lookback etc)
    # TODO Get xgboost to work with m4 + AirQualityUCI
    # TODO Read up on ARIMA for meta-learning
    # TODO Move the bias here

    # TODO Get the training to work to output forecasting
    # TODO Implement the old dataset for NBeats as well. (special treatment for multivariate?) - None or whatever
    # TODO Revisit sarimax and xgboost
    # TODO Bring bias meta-learning

    # model = NBeats(settings)
    # model.fit(data)
    # model.forecast(data.features_ts.multioutput)
    #
    # plt.plot(data.labels_ts.single_output, 'k')
    # plt.plot(model.prediction)
    # plt.pause(0.1)

    model = Sarimax(settings)
    model.fit(data.labels_tr.single_output, exog_variables=data.features_tr.single_output)
    foreward_periods = 48 * 8
    model.forecast(exog_variables=data.features_ts.single_output.iloc[:foreward_periods], foreward_periods=foreward_periods)

    # model.fit(data.labels_tr.single_output)
    # foreward_periods = 48 * 8
    # model.forecast(foreward_periods=foreward_periods)

    import matplotlib.pyplot as plt
    plt.plot(data.labels_ts.single_output[:foreward_periods], 'k')
    plt.plot(model.prediction)
    plt.pause(0.1)
    plt.show()

    forecast_period = 24*7
    model.forecast(data, foreward_periods=forecast_period)
    #
    model_xgboost = Xgboost(settings)
    model_xgboost.fit(data)

    forecast_period = 24*7
    model_xgboost.forecast(data, period=forecast_period)
    #

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data.labels.iloc[:settings.data.horizon*settings.data.n_horizons_lookback], 'k')
    plt.plot(data.labels.iloc[settings.data.horizon*settings.data.n_horizons_lookback:settings.data.horizon*settings.data.n_horizons_lookback+forecast_period], 'tab:blue')

    plt.plot(model.prediction, 'tab:red')
    plt.plot(model_xgboost.prediction, 'tab:orange')

    plt.pause(0.1)
    plt.show()
    print('done')

    k = 1

    #
    # import os
    # import pickle
    # if not os.path.exists('results'):
    #     os.makedirs('results')
    # f = open('results' + '/' + settings.data.dataset + ".pckl", 'wb')
    # pickle.dump(results, f)
    # f.close()

    # results = pickle.load(open('results/' + str(settings.data.dataset) + '.pckl', "rb"))

    # TODO Train on a few epochs and see predictions
    # TODO Rework the NBeats strucutre, move stuff inside src etc

    # TODO Try SARIMAX without diff on indicators
    # TODO Try predicting the average co2 of the rolling 8 hours
    # TODO xgboost for prediction
    #   TODO Add one-hot encoding of hourly time features for xgboost and don't lag them
    # TODO NBeats for prediction
    # TODO Rework all classes in training.py to match the "model" superclass in pytorch

    # TODO Testing metrics

    # TODO Simple linear regression (AR(p))
    # TODO Work on the diff of the labels and recover the full label afterwards
    # TODO Multivariate NBEATS

    # TODO Read NBEATS experiments metalearning

    # Implement https://github.com/philipperemy/n-beats https://github.com/philipperemy/n-beats ( simpler )


if __name__ == "__main__":

    main()
