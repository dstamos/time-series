from src.ltl import BiasLTL
from src.sarimax import Sarimax
from src.xgboost import Xgboost
from src.independent_learning import ITL


def training(data, training_settings):
    method = training_settings.method

    if method == 'ITL':
        model_itl = ITL(training_settings)
        model_itl.fit(data.test_tasks)
    elif method == 'BiasLTL':
        model = BiasLTL(training_settings)
        model.fit(data.training_tasks, data.validation_tasks)
    elif method == 'SARIMAX':
        pass
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
    elif method == 'Xgboost':
        pass
        # model = Xgboost(training_settings)
        # model.fit(data.labels_tr.single_output, data.features_tr.single_output)
        #
        # forecast_period = 24 * 6
        # if training_settings.training.use_exog is True:
        #     tr_pred = model.predict(data.features_tr.single_output)
        #     ts_pred = model.predict(data.features_ts.single_output.iloc[:forecast_period])
        # else:
        #     tr_pred = model.predict(data.labels_tr.single_output)
        #     ts_pred = model.predict(data.labels_ts.single_output.iloc[:forecast_period])
        #
        # import matplotlib.pyplot as plt
        # plt.plot(data.labels_tr.single_output, 'k')
        # plt.plot(tr_pred)
        # plt.plot(data.labels_ts.single_output.iloc[:forecast_period], 'tab:red')
        # plt.plot(ts_pred, 'tab:blue')
        # plt.pause(0.1)
        # print('done')
        # plt.show()
        # exit()

    else:
        raise ValueError('Unknown method', method)
    return
