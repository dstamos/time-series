from src.data_management import DataHandler, Settings
from src.training import Sarimax, Xgboost


def main():
    data_settings = {'dataset': 'AirQualityUCI',
                     'label': 'CO(GT)',
                     'horizon': 24,
                     'n_horizons_lookback': 5}

    training_settings = {'method': 'SARIMAX',
                         'use_exog': True}

    settings = Settings(data_settings, 'data')
    settings.add_settings(training_settings, 'training')
    data = DataHandler(settings)

    model = Sarimax(settings)
    model.fit(data)

    forecast_period = 24*7
    model.forecast(data, period=forecast_period)
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
