from src.data_management import Settings, MealearningDataHandler
from src.training import training
import numpy as np
import pandas as pd


def main():
    data_settings = {'dataset': 'sine',
                     'use_exog': False,
                     'training_tasks_pct': 0.7,
                     'validation_tasks_pct': 0.1,
                     'test_tasks_pct': 0.2,
                     'training_points_pct': 0.5,
                     'validation_points_pct': 0.1,
                     'test_points_pct': 0.4,
                     'forecast_length': 1}

    data_settings = Settings(data_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'ITL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 16)],
                                  'lags': 6})

    model_itl = training(data, training_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'BiasLTL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 16)],
                                  'lags': 6})

    model_ltl = training(data, training_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'SARIMAX',
                                  'use_exog': False})

    model_sarimax = training(data, training_settings)
    #############################################################################
    np.random.seed(999)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'xgboost',
                                  'use_exog': False,
                                  'lags': 6})

    model_xgboost = training(data, training_settings)
    #############################################################################

    def labels_to_raw(labels, first):
        raw_predictions = pd.Series(index=labels.index)
        raw = first
        for idx in range(len(labels)):
            label = labels.iloc[idx]
            raw = raw + raw * label

            raw_predictions.iloc[idx] = raw
        return raw_predictions

    import matplotlib.pyplot as plt
    my_dpi = 100
    fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi)
    ax = fig.add_subplot(111)
    task_idx = 0
    ax.plot(data.test_tasks[task_idx].test.raw_time_series, color='k', label='original time series')

    lab = model_itl.all_predictions[task_idx]
    first_idx = data.test_tasks[task_idx].test.raw_time_series.index.get_loc(lab.index[0]) - 1
    first_value = data.test_tasks[task_idx].test.raw_time_series.iloc[first_idx].values[0]
    r = labels_to_raw(lab, first_value)
    ax.plot(r, color='tab:blue', label='ITL')

    ax.plot(model_sarimax.all_predictions[task_idx], color='tab:red', label='SARIMAX predictions')
    ax.plot(model_sarimax.all_forecasts[task_idx], color='tab:orange', label='SARIMAX forecasts')

    lab = model_xgboost.all_predictions[task_idx]
    first_idx = data.test_tasks[task_idx].test.raw_time_series.index.get_loc(lab.index[0]) - 1
    first_value = data.test_tasks[task_idx].test.raw_time_series.iloc[first_idx].values[0]
    r = labels_to_raw(lab, first_value)
    ax.plot(r, color='tab:green', label='Random Forest')

    lab = model_ltl.all_predictions[task_idx]
    first_idx = data.test_tasks[task_idx].test.raw_time_series.index.get_loc(lab.index[0]) - 1
    first_value = data.test_tasks[task_idx].test.raw_time_series.iloc[first_idx].values[0]
    r = labels_to_raw(lab, first_value)
    ax.plot(r, color='tab:purple', label='BiasLTL')

    plt.title('test task #' + str(task_idx))
    plt.legend()
    plt.xlabel('time periods')
    plt.ylabel('time series values')
    plt.show()
    # k = 1


if __name__ == "__main__":

    main()
