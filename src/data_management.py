import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'


def split_data(all_features, all_labels, all_dataset_names, settings, verbose=True):
    test_tasks_indexes, training_tasks_indexes = train_test_split(range(len(all_features)), test_size=settings['tr_tasks_pct'], shuffle=True)
    training_tasks_indexes, validation_tasks_indexes = train_test_split(training_tasks_indexes, test_size=0.2)

    tr_tasks_tr_points_pct = settings['tr_tasks_tr_points_pct']
    val_tasks_tr_points_pct = settings['val_tasks_tr_points_pct']
    val_tasks_test_points_pct = settings['val_tasks_test_points_pct']
    test_tasks_tr_points_pct = settings['test_tasks_tr_points_pct']
    test_tasks_test_points_pct = settings['test_tasks_test_points_pct']

    # Training tasks (only training data)
    tr_tasks_tr_features = []
    tr_tasks_tr_labels = []
    for counter, task_index in enumerate(training_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        n_tr_points = int(tr_tasks_tr_points_pct * n_all_points)
        training_features = x.iloc[:n_tr_points, :]
        training_labels = y.iloc[:n_tr_points]

        tr_tasks_tr_features.append(training_features)
        tr_tasks_tr_labels.append(training_labels)

        if verbose is True:
            print(f'task: {counter:4d} | points: {n_all_points:5d} | tr: {n_tr_points:5d}')

    # Validation tasks (training and test data)
    val_tasks_tr_features = []
    val_tasks_tr_labels = []
    val_tasks_test_features = []
    val_tasks_test_labels = []
    for counter, task_index in enumerate(validation_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        n_tr_points = int(val_tasks_tr_points_pct * n_all_points)
        n_test_points = int(val_tasks_test_points_pct * n_all_points)

        training_features = x.iloc[:n_tr_points, :]
        training_labels = y.iloc[:n_tr_points]
        test_features = x.iloc[n_tr_points + 1:n_tr_points + n_test_points, :]
        test_labels = y.iloc[n_tr_points + 1:n_tr_points + n_test_points]

        val_tasks_tr_features.append(training_features)
        val_tasks_tr_labels.append(training_labels)
        val_tasks_test_features.append(test_features)
        val_tasks_test_labels.append(test_labels)

        if verbose is True:
            print(f'task: {counter:4d} | points: {n_all_points:5d} | tr: {n_tr_points:5d} | test: {n_test_points:5d}')

    # Test tasks (training and test data)
    test_tasks_tr_features = []
    test_tasks_tr_labels = []
    test_tasks_test_features = []
    test_tasks_test_labels = []
    for counter, task_index in enumerate(test_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        n_tr_points = int(test_tasks_tr_points_pct * n_all_points)
        n_test_points = int(test_tasks_test_points_pct * n_all_points)

        training_features = x.iloc[:n_tr_points, :]
        training_labels = y.iloc[:n_tr_points]
        test_features = x.iloc[n_tr_points + 1:n_tr_points + n_test_points, :]
        test_labels = y.iloc[n_tr_points + 1:n_tr_points + n_test_points]

        test_tasks_tr_features.append(training_features)
        test_tasks_tr_labels.append(training_labels)
        test_tasks_test_features.append(test_features)
        test_tasks_test_labels.append(test_labels)

        if verbose is True:
            print(f'task: {counter:4d} | points: {n_all_points:5d} | tr: {n_tr_points:5d} | test: {n_test_points:5d}')
    data = {'training_tasks_indexes': training_tasks_indexes,
            'validation_tasks_indexes': validation_tasks_indexes,
            'test_tasks_indexes': test_tasks_indexes,
            'all_dataset_names': all_dataset_names,
            # Training tasks
            'tr_tasks_tr_features': tr_tasks_tr_features,
            'tr_tasks_tr_labels': tr_tasks_tr_labels,
            # Validation tasks
            'val_tasks_tr_features': val_tasks_tr_features,
            'val_tasks_tr_labels': val_tasks_tr_labels,
            'val_tasks_test_features': val_tasks_test_features,
            'val_tasks_test_labels': val_tasks_test_labels,
            # Test tasks
            'test_tasks_tr_features': test_tasks_tr_features,
            'test_tasks_tr_labels': test_tasks_tr_labels,
            'test_tasks_test_features': test_tasks_test_features,
            'test_tasks_test_labels': test_tasks_test_labels}
    return data


def concatenate_data(all_features, all_labels):
    point_indexes_per_task = []
    for counter in range(len(all_features)):
        point_indexes_per_task.append(counter + np.zeros(all_features[counter].shape[0]))
    point_indexes_per_task = np.concatenate(point_indexes_per_task).astype(int)

    all_features = pd.concat(all_features)
    all_labels = pd.concat(all_labels)
    return all_features, all_labels, point_indexes_per_task


def split_tasks(all_features, indexes, all_labels=None):
    # Split the blob/array of features into a list of tasks based on point_indexes_per_task
    all_features = [all_features[indexes == task_idx] for task_idx in np.unique(indexes)]
    if all_labels is None:
        return all_features
    all_labels = [all_labels[indexes == task_idx] for task_idx in np.unique(indexes)]
    return all_features, all_labels
