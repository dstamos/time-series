import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_management import concatenate_data
from numpy.linalg.linalg import norm
from src.data_management import split_tasks
from copy import deepcopy


class PreProcess:
    def __init__(self, standard_scaling, inside_ball_scaling, add_bias=False):
        self.standard_scaling = standard_scaling
        self.standard_scaler = None
        self.inside_ball_scaling = inside_ball_scaling
        self.add_bias = add_bias

    def transform(self, all_features_og, all_labels_og, fit=False, multiple_tasks=True):
        all_features = deepcopy(all_features_og)
        all_labels = deepcopy(all_labels_og)
        if multiple_tasks is True:
            all_features, all_labels, point_indexes_per_task = concatenate_data(all_features, all_labels)
        else:
            # In the case you want to preprocess just a single dataset, pass multiple_tasks=False
            point_indexes_per_task = None

        # These two scalers technically should be somehow applied before merging.
        # The reasoning is that metalearning is done in an online fashion, without reusing past data.
        if fit is True:
            if self.standard_scaling is True:
                sc = StandardScaler()
                all_features = pd.DataFrame(sc.fit_transform(all_features), columns=all_features.columns, index=all_features.index)
                self.standard_scaler = sc
        else:
            if self.standard_scaling is True:
                all_features = pd.DataFrame(self.standard_scaler.transform(all_features), columns=all_features.columns, index=all_features.index)

        if self.inside_ball_scaling is True:
            all_features = all_features / norm(all_features, axis=0, keepdims=True)

        if self.add_bias is True:
            all_features.insert(0, 'bias', 1)

        if multiple_tasks is True:
            all_features, all_labels = split_tasks(all_features, point_indexes_per_task, all_labels)
        return all_features, all_labels
