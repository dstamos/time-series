import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def lag_features(indicators, lags):
    lagged_indicators = [None] * lags
    original_column_names_to_lag = indicators.columns.values.tolist()
    for idx, c_lag in enumerate(range(1, lags)):
        column_names = [column_name + '-lagged_' + str(c_lag) for column_name in original_column_names_to_lag]
        temp = indicators[original_column_names_to_lag].shift(c_lag)
        temp.columns = column_names
        lagged_indicators[idx] = temp

    lagged_indicators.insert(0, indicators)
    lagged_indicators = pd.concat(lagged_indicators, axis=1)

    return lagged_indicators


def prune_data(features, labels=None):
    features = features.replace([np.inf, -np.inf], np.nan)
    nan_points_idx = features.index[pd.isnull(features).any(1).to_numpy().nonzero()[0]]

    if labels is not None:
        nan_labels_idx = labels.index[pd.isnull(labels).any(1).to_numpy().nonzero()[0]]
        idx_to_drop = np.concatenate((nan_labels_idx, nan_points_idx))
        labels = labels.drop(idx_to_drop)
        features = features.drop(idx_to_drop)
        return features, labels
    else:
        features = features.drop(nan_points_idx)
        return features


def forward_shift_ts(df, horizon_list):
    times_series_name = df.columns.values.tolist()[0]

    forwarded_values = [None] * len(horizon_list)
    for idx, current_shift in enumerate(horizon_list):
        column_name = str(times_series_name) + '-horizon_' + str(current_shift)
        temp = df[[times_series_name]].shift(-current_shift)
        temp.columns = [column_name]
        forwarded_values[idx] = temp
    # FIXME
    return pd.concat(forwarded_values, axis=1)


class PandasDataset(Dataset):
    # TODO Delete this
    def __init__(self, features_df, labels_df):
        self.features_df = features_df
        self.labels_df = labels_df

    def __len__(self):
        return self.features_df[0]

    def __getitem__(self, index):
        return self.features_df.iloc[index].values, self.labels_df.iloc[index].values
