import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error
from src.utilities import normalised_mse


def train_test_naive(data, settings):
    # "Training"
    tt = time()
    all_performances = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        y_tr = data['test_tasks_tr_labels'][task_idx]
        y_test = data['test_tasks_test_labels'][task_idx]

        if len(y_tr) > 1:
            prediction_value = np.mean(y_tr.values.ravel())
        else:
            # In the case we have no data for training (cold start), just use random data.
            prediction_value = np.mean(np.random.uniform(0.1, 1.0, len(y_test)))

        # Testing
        test_predictions = pd.DataFrame(prediction_value * np.ones(len(y_test)), index=y_test.index)
        all_performances.append(normalised_mse(y_test, test_predictions))
    test_performance = np.mean(all_performances)
    print(f'{"Naive":12s} | test performance: {test_performance:12.5f} | {time() - tt:6.1f}sec')

    return test_performance
