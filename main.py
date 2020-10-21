from src.data_management import Settings, MealearningDataHandler
from src.training import training
import numpy as np
import pandas as pd


def main():
    data_settings = {'dataset': 'm4',
                     'use_exog': False,
                     'training_tasks_pct': 0.7,
                     'validation_tasks_pct': 0.1,
                     'test_tasks_pct': 0.2,
                     'training_points_pct': 0.05,
                     'validation_points_pct': 0.25,
                     'test_points_pct': 0.7,
                     'forecast_length': 1}

    data_settings = Settings(data_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'ITL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 6, 36)],
                                  'lags': 6,
                                  'horizon': data_settings.forecast_length})

    model_itl = training(data, training_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'BiasLTL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 6, 36)],
                                  'lags': 6,
                                  'horizon': data_settings.forecast_length})

    model_ltl = training(data, training_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'SARIMAX',
                                  'use_exog': False})

    model_sarimax = training(data, training_settings)
    #############################################################################
    # np.random.seed(999)
    # data = MealearningDataHandler(data_settings)
    # training_settings = Settings({'method': 'xgboost',
    #                               'use_exog': False,
    #                               'lags': 6,
    #                               'horizon': data_settings.forecast_length})
    #
    # model_xgboost = training(data, training_settings)
    #############################################################################

    import matplotlib.pyplot as plt
    for task_idx in range(len(data.test_tasks)):
        my_dpi = 100
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi)
        ax = fig.add_subplot(111)
        ax.plot(data.test_tasks[task_idx].test.raw_time_series, color='k', label='original time series')

        ax.plot(model_itl.all_raw_predictions[task_idx], color='tab:blue', label='ITL')

        ax.plot(model_sarimax.all_predictions[task_idx], color='tab:red', label='SARIMAX predictions')
        ax.plot(model_sarimax.all_forecasts[task_idx], color='tab:orange', label='SARIMAX forecasts')

        # ax.plot(model_xgboost.all_raw_predictions[task_idx], color='tab:green', label='Random Forest')

        ax.plot(model_ltl.all_raw_predictions[task_idx], color='tab:purple', label='BiasLTL')

        plt.title('test task #' + str(task_idx))
        plt.legend()
        plt.xlabel('time periods')
        plt.ylabel('time series values')
        plt.savefig('m4_' + str(task_idx))
    plt.show()
    # k = 1


if __name__ == "__main__":

    main()
