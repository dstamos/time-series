import numpy as np
import pandas as pd
from numpy.linalg.linalg import norm


def lag_features(indicators, lags, keep_original=True):
    lagged_indicators = [None] * lags
    original_column_names_to_lag = indicators.columns.values.tolist()
    for idx, c_lag in enumerate(range(1, lags)):
        column_names = [column_name + '-lagged_' + str(c_lag) for column_name in original_column_names_to_lag]
        temp = indicators[original_column_names_to_lag].shift(c_lag)
        temp.columns = column_names
        lagged_indicators[idx] = temp

    if keep_original is True:
        lagged_indicators.insert(0, indicators)
    lagged_indicators = pd.concat(lagged_indicators, axis=1)

    return lagged_indicators


def prune_data(features, labels=None):
    features = features.replace([np.inf, -np.inf], np.nan)
    nan_points_idx = features.index[pd.isnull(features).any(1).to_numpy().nonzero()[0]]
    # Why was this even added?
    # all_zeroes_idx = features.index[np.where(np.all(features == 0, axis=1))[0]]

    if labels is not None:
        nan_labels_idx = labels.index[pd.isnull(labels).any(1).to_numpy().nonzero()[0]]
        idx_to_drop = np.concatenate((nan_labels_idx, nan_points_idx))
        labels = labels.drop(idx_to_drop)
        labels.index.freq = labels.index.inferred_freq
        features = features.drop(idx_to_drop)
        return features, labels
    else:
        features = features.drop(nan_points_idx)
        return features


def forward_shift_ts(df, horizon_list):
    times_series_name = df.columns.values.tolist()[0]

    forwarded_values = []
    for idx, current_shift in enumerate(horizon_list):
        column_name = str(times_series_name) + '-horizon_' + str(current_shift)
        temp = df[[times_series_name]].shift(-current_shift)
        temp.columns = [column_name]
        forwarded_values.append(temp)
    df = pd.concat(forwarded_values, axis=1)
    df = df.dropna()
    return df


def handle_data(list_of_tasks, lags, use_exog):
    for task_idx in range(len(list_of_tasks)):
        # The features are based just on the percentage difference of values of the time series
        raw_time_series_tr = list_of_tasks[task_idx].training.raw_time_series
        raw_time_series_val = list_of_tasks[task_idx].validation.raw_time_series
        raw_time_series_ts = list_of_tasks[task_idx].test.raw_time_series

        y_train = list_of_tasks[task_idx].training.labels
        y_validation = list_of_tasks[task_idx].validation.labels
        y_test = list_of_tasks[task_idx].test.labels
        if use_exog is True:
            x_train = list_of_tasks[task_idx].training.features
            features_tr = lag_features(x_train, lags)

            x_validation = list_of_tasks[task_idx].validation.features
            features_val = lag_features(x_validation, lags)

            x_test = list_of_tasks[task_idx].test.features
            features_ts = lag_features(x_test, lags)
        else:
            # FIXME This used to have keep_original=False
            features_tr = lag_features(raw_time_series_tr, lags)
            features_val = lag_features(raw_time_series_val, lags)
            features_ts = lag_features(raw_time_series_ts, lags)

        list_of_tasks[task_idx].training.features, list_of_tasks[task_idx].training.labels = prune_data(features_tr, y_train)
        list_of_tasks[task_idx].validation.features, list_of_tasks[task_idx].validation.labels = prune_data(features_val, y_validation)
        list_of_tasks[task_idx].test.features, list_of_tasks[task_idx].test.labels = prune_data(features_ts, y_test)

        # Normalise the features
        # list_of_tasks[task_idx].training.features = list_of_tasks[task_idx].training.features / norm(list_of_tasks[task_idx].training.features, axis=1, keepdims=True)
        # list_of_tasks[task_idx].validation.features = list_of_tasks[task_idx].validation.features / norm(list_of_tasks[task_idx].validation.features, axis=1, keepdims=True)
        # list_of_tasks[task_idx].test.features = list_of_tasks[task_idx].test.features / norm(list_of_tasks[task_idx].test.features, axis=1, keepdims=True)

    return list_of_tasks


def labels_to_raw(labels, raw_times_series, horizon):
    """

    :param horizon:
    :param labels: Pandas Series
    :param raw_times_series:  Pandas Series
    :return:
    """
    # first_idx = raw_times_series.index.get_loc(labels.index[0])
    # first_value = raw_times_series.iloc[first_idx].values[0]

    raw_predictions = labels

    # raw_predictions = pd.Series(index=raw_times_series.index)
    # for idx, actual_index in enumerate(labels.index):
    #     ts_index = raw_times_series.index.get_loc(actual_index)
    #     ts_value = raw_times_series.iloc[ts_index].values[0]
    #
    #     # curr_pred = labels.loc[actual_index]
    #     # future_ts_value = ts_value + ts_value * curr_pred
    #
    #     future_ts_value = labels.loc[actual_index]
    #
    #     # raw_predictions.loc[actual_index + horizon] = future_ts_value
    #     raw_predictions.loc[actual_index + horizon * actual_index.freq] = future_ts_value
    #
    # raw_predictions.dropna(inplace=True)
    return raw_predictions


def performance_check(y_true, y_pred):
    y_true = y_true.values.ravel()
    y_pred = y_pred.values.ravel()
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Make sure that if y_true is 0 then you return 0
        rel_error = np.abs(np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true), where=(y_true != 0)))
        mape = (100 / len(y_true)) * np.sum(rel_error)

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)

        nmse = mse / np.var(y_true)

    errors = {'mse': mse, 'mape': mape, 'nmse': nmse}
    return errors
