from src.data_management import Settings, MealearningDataHandler
from src.training import training
import numpy as np


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
    # np.random.seed(999)
    # data = MealearningDataHandler(data_settings)
    # training_settings = Settings({'method': 'xgboost',
    #                               'use_exog': False,
    #                               'lags': 3})
    #
    # model_xgboost = training(data, training_settings)
    #############################################################################

    def labels_to_raw(labels, first_value):
        raw_predictions = []
        raw = first_value
        for idx in range(len(labels)):
            label = labels[idx]
            raw = raw + raw * label

            raw_predictions.append(raw)
        return raw_predictions

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    task_idx = 0
    ax.plot(data.test_tasks[task_idx].test.raw_time_series.values, color='k', label='original time series')

    lab = model_itl.all_predictions[task_idx]
    first = data.test_tasks[task_idx].test.raw_time_series.values[2]
    r = labels_to_raw(lab, first)
    ax.plot(r, color='tab:blue', label='ITL')

    ax.plot(model_sarimax.all_predictions[task_idx].values, color='tab:red', label='SARIMAX')

    # lab = model_xgboost.all_predictions[task_idx]
    # first = data.test_tasks[task_idx].test.raw_time_series.values[2]
    # r = labels_to_raw(lab, first)
    # ax.plot(r, color='tab:green', label='Random Forest')

    lab = model_ltl.all_predictions[task_idx]
    first = data.test_tasks[task_idx].test.raw_time_series.values[2]
    r = labels_to_raw(lab, first)
    ax.plot(r, color='tab:orange', label='BiasLTL')

    plt.title('test task #' + str(task_idx))
    plt.legend()
    plt.xlabel('time periods')
    plt.ylabel('time series values')
    plt.show()
    # k = 1


if __name__ == "__main__":

    main()
