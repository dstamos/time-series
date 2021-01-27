from src.data_management import Settings, MealearningDataHandler
from src.training import training
from time import time
import numpy as np
import pandas as pd


# noinspection PyInterpreter
def main(curr_seed, n_tr_points, lags):
    # data_settings = {'dataset': 'm4',
    #                  'use_exog': False,
    #                  'training_tasks_pct': 0.80,
    #                  'validation_tasks_pct': 0.1,
    #                  'test_tasks_pct': 0.1,
    #                  'training_points_pct': 0.05,
    #                  'validation_points_pct': 0.25,
    #                  'test_points_pct': 0.7,
    #                  'forecast_length': 6}

    data_settings = {'dataset': 'air_quality_madrid',
                     'use_exog': False,
                     'training_tasks_pct': 0.70,
                     'validation_tasks_pct': 0.1,
                     'test_tasks_pct': 0.2,
                     'n_tr_points': n_tr_points,
                     'n_test_points': 24 * 28 * 6,
                     'forecast_length': 1}

    # data_settings = {'dataset': 'synthetic_ar',
    #                  'use_exog': False,
    #                  'n_time_series': 100,
    #                  'n_tr_points': n_tr_points,
    #                  'n_test_points': 100,7
    #                  'n_total_points': 2000,
    #                  'training_tasks_pct': 0.80,
    #                  'validation_tasks_pct': 0.1,
    #                  'test_tasks_pct': 0.1,
    #                  'forecast_length': 1}

    data_settings = Settings(data_settings)
    seed = curr_seed
    #############################################################################
    np.random.seed(seed)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'ITL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-6, 6, 36)],
                                  'lags': lags,
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
    #############################################################################
    np.random.seed(seed)
    data = MealearningDataHandler(data_settings)
    training_settings = Settings({'method': 'BiasLTL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-6, 6, 36)],
                                  'lags': lags,
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

    import pickle
    filename = 'seed_' + str(seed) + '_n_points_' + str(data_settings.n_tr_points) + '_lags_' + str(lags)
    full_path = './results/madrid/' + filename
    splits = {'training': data.training_tasks_indexes, 'validation': data.validation_tasks_indexes, 'test': data.test_tasks_indexes}
    pickle.dump([splits, data_settings, model_itl, model_ltl], open(full_path + '.pckl', "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    ##############################
    k = 1
    # import matplotlib.pyplot as plt
    # my_dpi = 100
    # n_plots = min(len(data.test_tasks), 8)
    # fig, ax = plt.subplots(figsize=(2 * 1920 / my_dpi, 2 * 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
    # lookback = 24 * 21
    # for task_idx in range(n_plots):
    #     curr_ax = ax[task_idx]
    #     true = data.test_tasks[task_idx].test.labels.iloc[-lookback:]
    #     curr_ax.plot(true, color='k', label='original labels')
    #
    #     pred_itl = model_itl.all_predictions[task_idx].iloc[-lookback:]
    #     curr_ax.plot(pred_itl, color='tab:red', label='Independent Learning')
    #
    #     pred = model_ltl.all_predictions[task_idx].iloc[-lookback:]
    #     curr_ax.plot(pred, color='tab:blue', label='Bias Meta-learning')
    #
    #     curr_ax.axhline(y=0, color='tab:gray', linestyle=':')
    #     curr_ax.spines["top"].set_visible(False)
    #     curr_ax.spines["right"].set_visible(False)
    #     curr_ax.spines["bottom"].set_visible(False)
    #
    # plt.legend()
    # plt.suptitle('predictions (number of training points: ' + str(data_settings.n_tr_points) + ')')
    # plt.savefig('result_mse_seed_' + str(seed) + '_tr_' + str(data_settings.n_tr_points) + '.png', pad_inches=0)
    # plt.pause(0.1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        seed_range = [int(sys.argv[1])]
        tr_points_range = [i * 24 for i in [2, 3, 4, 5, 6, 7] + list(range(7, 6*7+1, 7))]
        lags_range = [6]
        # lags_range = [int(sys.argv[2])]
        # tr_points_range = [int(sys.argv[2])]
    else:
        seed_range = [999]
        tr_points_range = [24 * 28]
        lags_range = [24]
        # lags_range = [2, 4, 12]
        # seed_range = range(1, 21)
        # tr_points_range = [24 * 7, 24 * 30, 24 * 90, 24 * 365]

    # seed_range = range(1, 21)
    # tr_points_range = range(12, 201, 2)

    tt = time()
    for seed in seed_range:
        for tr_points in tr_points_range:
            for lags in lags_range:
                main(seed, tr_points, lags)
                print(f'seed: {seed:3d} | #points: {tr_points:3d}| #lags: {lags:3d} | {time() - tt:5.2f}sec')
    print('done', f'{time() - tt:5.2f}sec')
