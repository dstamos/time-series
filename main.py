from src.data_management import DataHandler, Settings
from src.training import Sarimax, Xgboost, NBeats


def main():
    data_settings = {'dataset': 'AirQualityUCI',
                     'label': 'CO(GT)',
                     'label_period': 1,
                     'training_percentage': 0.03}

    # data_settings = {'dataset': 'm4',
    #                  'm4_time_series_idx': 100,
    #                  'forecast_length': 24}

    # training_settings = {'method': 'SARIMAX',
    #                      'use_exog': True}

    training_settings = {'method': 'xgboost',
                         'use_exog': True,
                         'lags': 3}

    # training_settings = {'method': 'NBeats',
    #                      'use_exog': True,
    #                      'lookback_length': 4,
    #                      'forecast_length': data_settings['forecast_length'],
    #                      'batch_size': 8
    #                      }


    settings = Settings(data_settings, 'data')
    settings.add_settings(training_settings, 'training')
    data = DataHandler(settings)

    # TODO Get xgboost to work with AirQualityUCI + m4
    # TODO Read up on ARIMA for meta-learning (what is X, why it helps etc)
    # TODO Move the bias here

    # model = NBeats(settings)
    # model.fit(data)
    # model.forecast(data.features_ts.multioutput)

    #############################################################################

    model = Sarimax(settings)

    if settings.training.use_exog is True:
        model.fit(data.labels_tr.single_output, exog_variables=data.features_tr.single_output)
        foreward_periods = 24 * 6
        model.forecast(exog_variables=data.features_ts.single_output.iloc[:foreward_periods], foreward_periods=foreward_periods)
    else:
        model.fit(data.labels_tr.single_output)
        foreward_periods = 24 * 6
        model.forecast(foreward_periods=foreward_periods)

    #############################################################################

    model_xgboost = Xgboost(settings)
    model_xgboost.fit(data)

    forecast_period = 24 * 6
    model_xgboost.forecast(data, period=forecast_period)

    import matplotlib.pyplot as plt
    prediction = model.model.get_prediction(start=data.labels_tr.single_output.index[0], end=data.labels_tr.single_output.index[-1])
    plt.plot(data.labels_tr.single_output, 'k')
    plt.plot(prediction.predicted_mean)
    plt.plot(data.labels_ts.single_output[:foreward_periods], 'tab:red')
    plt.plot(model.prediction, 'tab:blue')
    plt.pause(0.1)
    print('done')
    plt.show()
    exit()




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
