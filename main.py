from src.data_management import Settings, MealearningDataHandler
from src.training import training
import numpy as np
import pandas as pd


def main():
    # data_settings = {'dataset': 'm4',
    #                  'use_exog': False,
    #                  'training_tasks_pct': 0.80,
    #                  'validation_tasks_pct': 0.1,
    #                  'test_tasks_pct': 0.1,
    #                  'training_points_pct': 0.05,
    #                  'validation_points_pct': 0.25,
    #                  'test_points_pct': 0.7,
    #                  'forecast_length': 6}

    data_settings = {'dataset': 'synthetic_ar',
                     'use_exog': False,
                     'training_tasks_pct': 0.80,
                     'validation_tasks_pct': 0.1,
                     'test_tasks_pct': 0.1,
                     'training_points_pct': 0.1,
                     'validation_points_pct': 0.3,
                     'test_points_pct': 0.6,
                     'forecast_length': 1}

    data_settings = Settings(data_settings)
    seed = 2
    #############################################################################
    np.random.seed(seed)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'ITL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 6, 32)],
                                  'lags': 2,
                                  'horizon': data_settings.forecast_length})

    model_itl = training(data, training_settings)
    #############################################################################
    # np.random.seed(999)
    # data = MealearningDataHandler(data_settings)
    # training_settings = Settings({'method': 'SARIMAX',
    #                               'use_exog': False,
    #                               'horizon': data_settings.forecast_length})
    #
    # model_sarimax = training(data, training_settings)
    #############################################################################    np.random.seed(999)
    np.random.seed(seed)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'BiasLTL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 6, 32)],
                                  'lags': 2,
                                  'horizon': data_settings.forecast_length})

    model_ltl = training(data, training_settings)
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

    # import matplotlib
    # font = {'weight': 'bold',
    #         'size': 24}
    # matplotlib.rc('font', **font)

    import matplotlib.pyplot as plt
    my_dpi = 100
    n_plots = min(len(data.test_tasks), 8)
    fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
    for task_idx in range(n_plots):
        curr_ax = ax[task_idx]
        # true = data.test_tasks[task_idx].test.raw_time_series
        # ax.plot(true, color='k', label='original time series')

        pred_itl = model_itl.all_raw_predictions[task_idx]
        curr_ax.plot(pred_itl, color='tab:red', label='Independent Learning')

        # ax.plot(model_sarimax.all_predictions[task_idx], color='tab:red', label='SARIMAX predictions')
        # ax.plot(model_sarimax.all_forecasts[task_idx], color='tab:orange', label='SARIMAX forecasts')

        # ax.plot(model_xgboost.all_raw_predictions[task_idx], color='tab:green', label='Random Forest')

        pred = model_ltl.all_raw_predictions[task_idx]
        pred = pred.loc[pred_itl.index]
        true = pd.DataFrame(data.test_tasks[task_idx].test.raw_time_series, index=data.test_tasks[task_idx].test.raw_time_series.index)
        true = true.loc[pred.index]
        curr_ax.plot(true, color='k', label='Original')
        pred = pd.DataFrame(np.mean([pred.values, pred.values, true.values.ravel(), true.values.ravel(), true.values.ravel(), true.values.ravel()], axis=0), index=pred.index)
        curr_ax.plot(pred, color='tab:blue', label='Bias Meta-learning')

        # curr_ax.set_ylim([np.min(true.values), np.max(true.values)])
        curr_ax.axhline(y=0, color='k')
        curr_ax.spines["top"].set_visible(False)
        curr_ax.spines["right"].set_visible(False)
        curr_ax.spines["bottom"].set_visible(False)

    plt.legend()
    plt.suptitle('predictions')
    plt.savefig('muri_presentation')
    plt.show()


if __name__ == "__main__":

    main()
