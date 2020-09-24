import numpy as np
import pandas as pd
import csv
from src.utilities import forward_shift_ts, prune_data
from collections import namedtuple
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 40000)


class Settings:
    def __init__(self, dictionary, struct_name=None):
        if struct_name is None:
            self.__dict__.update(**dictionary)
        else:
            temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)

    def add_settings(self, dictionary, struct_name=None):
        if struct_name is None:
            self.__dict__.update(dictionary)
        else:
            if hasattr(self, struct_name):
                temp_settings = getattr(self, struct_name)
                temp_settings.__dict__.update(dictionary)
            else:
                temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)


class DataHandler:
    def __init__(self, settings):
        self.settings = settings
        self.labels_tr = namedtuple('labels', ['single_output', 'multioutput'])
        self.features_tr = namedtuple('features', ['single_output', 'multioutput'])
        self.labels_ts = namedtuple('labels', ['single_output', 'multioutput'])
        self.features_ts = namedtuple('features', ['single_output', 'multioutput'])

        if self.settings.data.dataset == 'AirQualityUCI':
            self.air_quality_gen()
        elif self.settings.data.dataset == 'm4':
            self.m4_gen()
        else:
            raise ValueError('Invalid dataset')

    def air_quality_gen(self):
        # TODO For air_quality_gen, split those manually at the beginning based on a new setting parameter (I mean split the training and test sets)
        # https://archive.ics.uci.edu/ml/datasets/Air+Quality
        df = pd.read_excel('data/AirQualityUCI.xlsx', sheet_name=None)['AirQualityUCI']
        # Missing values are marked with -200
        df.replace(to_replace=-200, value=np.nan, inplace=True)

        # Turning Time into Timestamp defaults to current date. So we replace dates with the given/correct ones
        df['Datetime'] = [pd.Timestamp(str(df['Time'].iloc[i])) for i in range(len(df))]

        for curr_index in range(len(df)):
            df['Datetime'].iloc[curr_index] = df['Datetime'].iloc[curr_index].replace(year=df['Date'].iloc[curr_index].year)
            df['Datetime'].iloc[curr_index] = df['Datetime'].iloc[curr_index].replace(month=df['Date'].iloc[curr_index].month)
            df['Datetime'].iloc[curr_index] = df['Datetime'].iloc[curr_index].replace(day=df['Date'].iloc[curr_index].day)

        df.drop(['Time', 'Date'], axis=1, inplace=True)
        df.index = pd.DatetimeIndex(df['Datetime'].values, freq='H')

        # NMHC(GT) Has no values for most of the rows
        df.drop(['Datetime', 'NMHC(GT)'], axis=1, inplace=True)
        # df.fillna(method='ffill', inplace=True)
        df.interpolate(method='linear', inplace=True)
        print(df)

        # Cince CO(GT) is the label, we need to make sure we are looking 'ahead' to define it
        df[self.settings.data.label] = df[self.settings.data.label].shift(-1)
        df = df.dropna()
        self.labels = df[self.settings.data.label]
        self.features = df.drop(self.settings.data.label, axis=1)

    def m4_gen(self):
        training_filename = 'src/nbeats_theirs/data/m4/train/Daily-train.csv'
        test_filename = 'src/nbeats_theirs/data/m4/val/Daily-test.csv'

        def _load_m4(filename, idx):
            with open(filename) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Skip the header
                next(csv_reader, None)
                rows = list(csv_reader)
            # The first column is also spam
            row = rows[idx][1:]
            return pd.DataFrame(np.array(row).astype(float), columns=['m4_' + str(idx)])

        training_time_series = _load_m4(training_filename, self.settings.data.m4_time_series_idx)
        test_time_series = _load_m4(test_filename, self.settings.data.m4_time_series_idx)

        # def get_features_labels(time_series):
        #     labels = namedtuple('labels', ['single_output', 'multioutput'])
        #     labels.single_output = forward_shift_ts(time_series, [1])
        #     labels.multioutput = forward_shift_ts(time_series, range(1, self.settings.data.horizon + 1))
        #     features, labels.multioutput = prune_data(time_series, labels.multioutput)
        #     features, labels.single_output = prune_data(time_series, labels.single_output)
        #     return features, labels

        # self.settings.data.horizon
        def get_features_labels(time_series, horizon=1):
            y = forward_shift_ts(time_series, range(1, horizon + 1))
            x, y = prune_data(time_series, y)
            return x, y

        self.features_tr.single_output, self.labels_tr.single_output = get_features_labels(training_time_series)
        self.features_tr.multioutput, self.labels_tr.multioutput = get_features_labels(training_time_series, self.settings.data.horizon)

        self.features_ts.single_output, self.labels_ts.single_output = get_features_labels(test_time_series)
        self.features_ts.multioutput, self.labels_ts.multioutput = get_features_labels(test_time_series, self.settings.data.horizon)
