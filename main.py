import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from src.naive_baseline import train_test_naive
from src.ltl import train_test_meta
from src.independent_learning import train_test_itl
from src.arma import train_test_arma
from src.single_task import train_test_single_task
from src.utilities import save_results
from src.data_management_essex import load_data_essex_one, load_data_essex_two, split_data_essex
import pickle
import sys


# matplotlib.use('Qt5Agg')


def main(settings, seed):
    np.random.seed(seed)

    from src.eu_weather_data import load_eu_weather_data
    from src.timeseries_utilities import get_features_labels_from_timeseries
    from src.data_management import split_data
    all_raw_time_series = load_eu_weather_data()
    all_features, all_labels, all_station_ids = get_features_labels_from_timeseries(all_raw_time_series, settings['max_lag'])

    data = split_data(all_features, all_labels, all_station_ids, settings, verbose=True)

    # TODO Time series style KFold

    test_performance_naive = train_test_naive(data, settings)
    # test_performance_naive = np.nan

    test_performance_itl = train_test_itl(data, settings)
    # test_performance_itl = np.nan

    # test_performance_single_task = train_test_single_task(data, settings)
    test_performance_single_task = np.nan

    best_model_meta, test_performance_meta = train_test_meta(data, settings, verbose=True)
    # test_performance_meta = [np.nan]    # Because this output is a list by default

    # test_performance_arma = train_test_arma(data, settings)
    test_performance_arma = np.nan

    results = {'test_performance_naive': test_performance_naive,
               'test_performance_single_task': test_performance_single_task,
               'test_performance_arma': test_performance_arma,
               'test_performance_itl': test_performance_itl,
               'test_performance_meta': test_performance_meta,
               'settings': settings}

    save_results(results,
                 foldername='results-eu/',
                 filename='seed_' + str(seed) + '-tr_pct_{:0.4f}'.format(settings['test_tasks_tr_points_pct']))

    print(f'{"Naive":20s} {test_performance_naive:6.4f} \n'
          f'{"Single-task":20s} {test_performance_single_task:6.4f} \n'
          f'{"ARMA":20s} {test_performance_arma:6.4f} \n'
          f'{"ITL":20s} {test_performance_itl:6.4f} \n'
          f'{"Meta":20s} {test_performance_meta[-1]:6.4f}')


if __name__ == "__main__":

    """
    The BiasLTL metalearning pipeline:
    a) Take your T tasks. Split them into training/validation/test tasks.
    b) You train the "centroid"/metaparameter on the training tasks.
    c) You go to the validation tasks, fine-tune the model on each task (on training points) and check the performance (on test points).
    d) Pick the metaparameter that resulted in the best average performance on the validation tasks.
    e) Go to the test tasks using the optimal metaparameter, fine-tune on a small number of points (or don't) and test the performance.
    """

    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'

    # Parameters
    max_lag = 6

    tr_tasks_pct = 0.7
    test_tasks_pct = 1 - tr_tasks_pct
    assert tr_tasks_pct + test_tasks_pct == 1, 'Percentages need to add up to 1'

    # test_tasks_tr_split_range = np.linspace(0.00, 0.8, 30)
    test_tasks_tr_split_range = np.array([0.3])

    if len(sys.argv) > 1:
        # This is the case when main.py is called from a bash script with inputs
        seed_range = [int(sys.argv[1])]
        test_tasks_tr_split_range = np.array([test_tasks_tr_split_range[int(sys.argv[2])]])
    else:
        seed_range = [9999]
    regul_param_range = np.logspace(-16, 5, 36)

    fine_tune = True  # Fine-tuning is the process of customizing the metalearning model on the test tasks. That typically includes re-training on a small number of datapoints.

    # Dataset split for training tasks (only training points)
    tr_tasks_tr_points_pct = test_tasks_tr_split_range[0]

    val_tasks_tr_points_pct = test_tasks_tr_split_range[0]
    val_tasks_test_points_pct = 1 - val_tasks_tr_points_pct
    assert val_tasks_tr_points_pct + val_tasks_test_points_pct == 1, 'Percentages need to add up to 1'

    test_tasks_tr_points_pct_range = test_tasks_tr_split_range
    test_tasks_test_points_pct_range = 1 - test_tasks_tr_points_pct_range
    assert np.all(test_tasks_tr_points_pct_range + test_tasks_test_points_pct_range == 1), 'Percentages need to add up to 1'

    for curr_seed in seed_range:
        for test_tasks_tr_points_pct, test_tasks_test_points_pct in zip(test_tasks_tr_points_pct_range, test_tasks_test_points_pct_range):
            print(f'seed: {curr_seed:2d} | tr_pct: {test_tasks_tr_points_pct:5.3f}')
            options = {'max_lag': max_lag,
                       'regul_param_range': regul_param_range,
                       'fine_tune': fine_tune,
                       'tr_tasks_pct': tr_tasks_pct,
                       'test_tasks_pct': test_tasks_pct,
                       'tr_tasks_tr_points_pct': tr_tasks_tr_points_pct,
                       'val_tasks_tr_points_pct': val_tasks_tr_points_pct,
                       'val_tasks_test_points_pct': val_tasks_test_points_pct,
                       'test_tasks_tr_points_pct': test_tasks_tr_points_pct,
                       'test_tasks_test_points_pct': test_tasks_test_points_pct}
            main(options, curr_seed)
            print('\n')
