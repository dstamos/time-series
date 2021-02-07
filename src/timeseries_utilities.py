import numpy as np
import pandas as pd
from copy import deepcopy


def get_features_labels_from_timeseries(all_raw_timeseries, max_lag=1):

    all_features = []
    all_labels = []
    all_station_ids = []
    for ts_idx in range(len(all_raw_timeseries)):
        ts = all_raw_timeseries[ts_idx]
        ts = ts.diff().dropna()

        station_id = ts.columns[0]

        # TODO Try pct_change instead of diff here
        integrated_ts = ts.diff()
        labels = integrated_ts.shift(-1)
        labels.columns = ['labels']

        features = deepcopy(integrated_ts)
        features.columns = ['lag_0']
        for lag in range(1, max_lag + 1):
            features['lag_' + str(lag)] = integrated_ts.shift(lag)

        features = features.replace([np.inf, -np.inf], np.nan)
        nan_points_idx = features.index[pd.isnull(features).any(1).to_numpy().nonzero()[0]]
        nan_labels_idx = labels.index[pd.isnull(labels).any(1).to_numpy().nonzero()[0]]
        idx_to_drop = np.concatenate((nan_labels_idx, nan_points_idx))

        labels = labels.drop(idx_to_drop)
        features = features.drop(idx_to_drop)
        labels.index.freq = labels.index.inferred_freq
        features.index.freq = features.index.inferred_freq

        all_features.append(features)
        all_labels.append(labels)
        all_station_ids.append(station_id)
        print(ts_idx, len(all_raw_timeseries))

    return all_features, all_labels, all_station_ids
