import numpy as np
from src.data_management import DataHandler, Settings
from src.training_sarimax import Sarimax
import time


def main():
    tt = time.time()

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

    model.forecast(data, period=20*30)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data.labels.iloc[:settings.data.horizon*settings.data.n_horizons_lookback], 'k')
    plt.plot(data.labels.iloc[settings.data.horizon*settings.data.n_horizons_lookback:settings.data.horizon*settings.data.n_horizons_lookback+20*30], 'tab:blue')

    plt.plot(model.prediction, 'tab:red')

    plt.pause(0.1)

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


if __name__ == "__main__":

    main()
