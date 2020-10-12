from src.data_management import DataHandler, Settings, MealearningDataHandler
from src.training import Sarimax, Xgboost, BiasLTL
import numpy as np


def main():
    np.random.seed(999)
    # data_settings = {'dataset': 'AirQualityUCI',
    #                  'label': 'CO(GT)',
    #                  'label_period': 1,
    #                  'training_percentage': 0.03}

    # training_settings = {'method': 'SARIMAX',
    #                      'use_exog': False}

    training_settings = {'method': 'ltl',
                         'use_exog': False,
                         'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 36)],
                         'lags': 6}

    data_settings = {'dataset': 'sine',
                     'training_tasks_pct': 0.75,
                     'validation_tasks_pct': 0.05,
                     'test_tasks_pct': 0.2,
                     'training_points_pct': 0.3,
                     'validation_points_pct': 0.3,
                     'test_points_pct': 0.4,
                     'forecast_length': 6}

    # training_settings = {'method': 'NBeats',
    #                      'use_exog': True,
    #                      'lookback_length': 4,
    #                      'forecast_length': data_settings['forecast_length'],
    #                      'batch_size': 8
    #                      }

    settings = Settings(data_settings, 'data')
    settings.add_settings(training_settings, 'training')

    data = MealearningDataHandler(settings)

    # model = NBeats(settings)
    # model.fit(data)
    # model.predict(data.features_ts.multioutput)

    #############################################################################

    # model = Sarimax(settings)
    #
    # if settings.training.use_exog is True:
    #     model.fit(data.labels_tr.single_output, exog_variables=data.features_tr.single_output)
    #     foreward_periods = 24 * 6
    #     model.predict(exog_variables=data.features_ts.single_output.iloc[:foreward_periods], foreward_periods=foreward_periods)
    # else:
    #     model.fit(data.labels_tr.single_output)
    #     foreward_periods = 24 * 6
    #     model.predict(foreward_periods=foreward_periods)

    #     import matplotlib.pyplot as plt
    #     prediction = model.model.get_prediction(start=data.labels_tr.single_output.index[0], end=data.labels_tr.single_output.index[-1])
    #     plt.plot(data.labels_tr.single_output, 'k')
    #     plt.plot(prediction.predicted_mean)
    #     plt.plot(data.labels_ts.single_output[:foreward_periods], 'tab:red')
    #     plt.plot(model.prediction, 'tab:blue')
    #     plt.pause(0.1)
    #     print('done')
    #     plt.show()
    #     exit()
    ##################################################
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()
    #
    # ax1.plot(data.training_tasks[0].training.raw_time_series, 'tab:blue')
    # ax2.plot(data.training_tasks[0].training.labels, 'tab:red')
    # plt.pause(0.1)
    ##################################################
    from src.independent_learning import itl, ITL

    # itl(data)

    itl = ITL(settings)
    itl.fit(data.test_tasks)

    ##################################################
    model = BiasLTL(settings)
    model.fit(data.training_tasks, data.validation_tasks)

    exit()
    #############################################################################
    model = Xgboost(settings)
    model.fit(data.labels_tr.single_output, data.features_tr.single_output)

    forecast_period = 24 * 6
    if settings.training.use_exog is True:
        tr_pred = model.predict(data.features_tr.single_output)
        ts_pred = model.predict(data.features_ts.single_output.iloc[:forecast_period])
    else:
        tr_pred = model.predict(data.labels_tr.single_output)
        ts_pred = model.predict(data.labels_ts.single_output.iloc[:forecast_period])

    import matplotlib.pyplot as plt
    plt.plot(data.labels_tr.single_output, 'k')
    plt.plot(tr_pred)
    plt.plot(data.labels_ts.single_output.iloc[:forecast_period], 'tab:red')
    plt.plot(ts_pred, 'tab:blue')
    plt.pause(0.1)
    print('done')
    plt.show()
    exit()
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(data.labels.iloc[:settings.data.horizon*settings.data.n_horizons_lookback], 'k')
    # plt.plot(data.labels.iloc[settings.data.horizon*settings.data.n_horizons_lookback:settings.data.horizon*settings.data.n_horizons_lookback+forecast_period], 'tab:blue')
    #
    # plt.plot(model.prediction, 'tab:red')
    # plt.plot(model_xgboost.prediction, 'tab:orange')
    #
    # plt.pause(0.1)
    # plt.show()
    # print('done')


if __name__ == "__main__":

    main()
