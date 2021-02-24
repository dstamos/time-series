from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle
import os


def multiple_tasks_nmse(all_true_labels, all_predictions, error_progression=False):
    if error_progression is False:
        performances = []
        for idx in range(len(all_true_labels)):
            curr_perf = normalised_mse(all_true_labels[idx], all_predictions[-1][idx])
            performances.append(curr_perf)
        performance = np.mean(performances)
        return performance
    else:
        all_performances = []
        for metamodel_idx in range(len(all_predictions)):
            metamodel_performances = []
            for idx in range(len(all_true_labels)):
                curr_perf = normalised_mse(all_true_labels[idx], all_predictions[metamodel_idx][idx])
                metamodel_performances.append(curr_perf)
            curr_metamodel_performance = np.mean(metamodel_performances)
            all_performances.append(curr_metamodel_performance)
        return all_performances


def multiple_tasks_mae_clip(all_true_labels, all_predictions, error_progression=False):
    if error_progression is False:
        all_predictions = all_predictions[-1]

        performances = []
        for task_idx in range(len(all_true_labels)):
            curr_perf = mae_clip(all_true_labels[task_idx], all_predictions[task_idx])
            performances.append(curr_perf)
        performance = np.mean(performances)
        return performance
    else:
        all_performances = []
        for metamodel_idx in range(len(all_predictions)):
            metamodel_performances = []
            for task_idx in range(len(all_true_labels)):
                curr_perf = mae_clip(all_true_labels[task_idx], all_predictions[metamodel_idx][task_idx])
                metamodel_performances.append(curr_perf)
            curr_metamodel_performance = np.mean(metamodel_performances)
            all_performances.append(curr_metamodel_performance)
            # TODO Recover individual errors for each task as well. This way it can be investigate how the errors progress for each task
        return np.array(all_performances)


def mae_clip(labels, predictions):
    return np.median(np.clip(np.abs(labels - predictions), 0, 1))


def normalised_mse(labels, predictions):
    assert(len(labels) == len(predictions))
    nan_points_idx_a = labels.index[pd.isnull(labels).any(1).to_numpy().nonzero()[0]]
    nan_labels_idx_b = predictions.index[pd.isnull(predictions).any(1).to_numpy().nonzero()[0]]
    idx_to_drop = np.concatenate((nan_points_idx_a, nan_labels_idx_b))

    if len(idx_to_drop) > 0:
        labels = labels.drop(idx_to_drop)
        predictions = predictions.drop(idx_to_drop)

    mse = mean_squared_error(labels, predictions)
    nmse = mse / mean_squared_error(labels.values.ravel(), np.mean(labels.values.ravel()) * np.ones(len(labels)))
    return nmse


def save_results(results, foldername='results', filename='temp'):
    os.makedirs(foldername, exist_ok=True)
    filename = './' + foldername + '/' + filename + '.pckl'
    pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

