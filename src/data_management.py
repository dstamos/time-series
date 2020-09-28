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


        training_df = df.iloc[:int(df.shape[0] * self.settings.data.training_percentage)]
        test_df = df.drop(training_df.index)
        print(training_df.shape)

        # create labels, create features for training and test
        def get_features_labels(mixed_df):
            # Cince CO(GT) is the label, we need to make sure we are looking 'ahead' to define it
            mixed_df[self.settings.data.label] = mixed_df[self.settings.data.label].shift(-self.settings.data.label_period)
            df = mixed_df.dropna()
            y = df[self.settings.data.label]
            x = df.drop(self.settings.data.label, axis=1)
            return x, y

        self.features_tr.single_output, self.labels_tr.single_output = get_features_labels(training_df)
        # self.features_tr.multioutput, self.labels_tr.multioutput = get_features_labels(training_df)

        self.features_ts.single_output, self.labels_ts.single_output = get_features_labels(test_df)
        # self.features_ts.multioutput, self.labels_ts.multioutput = get_features_labels(test_df)

    def m4_gen(self):
        training_filename = 'data/m4/train/Daily-train.csv'
        test_filename = 'data/m4/val/Daily-test.csv'

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
        # The m4 dataset doesn't seem to offer timestamps so we use integers as indexes
        test_time_series.index = pd.RangeIndex(start=training_time_series.index[-1] + 1, stop=training_time_series.index[-1] + 1 + len(test_time_series), step=1)

        def get_features_labels(time_series, horizon=1):
            y = forward_shift_ts(time_series, range(1, horizon + 1))
            x, y = prune_data(time_series, y)
            return x, y

        self.features_tr.single_output, self.labels_tr.single_output = get_features_labels(training_time_series)
        self.features_tr.multioutput, self.labels_tr.multioutput = get_features_labels(training_time_series, self.settings.data.forecast_length)

        self.features_ts.single_output, self.labels_ts.single_output = get_features_labels(test_time_series)
        self.features_ts.multioutput, self.labels_ts.multioutput = get_features_labels(test_time_series, self.settings.data.forecast_length)
