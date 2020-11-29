import numpy as np
import pandas as pd
import csv
from src.utilities import forward_shift_ts
from collections import namedtuple
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 40000)


class Settings:
    def __init__(self, dictionary):
        self.__dict__.update(**dictionary)


class DataHandler:
    def __init__(self, settings):
        self.settings = settings
        self.labels_tr = namedtuple('labels', ['single_output', 'multioutput'])
        self.features_tr = namedtuple('features', ['single_output', 'multioutput'])
        self.labels_ts = namedtuple('labels', ['single_output', 'multioutput'])
        self.features_ts = namedtuple('features', ['single_output', 'multioutput'])

        if self.settings.dataset == 'AirQualityUCI':
            # data_settings = {'dataset': 'AirQualityUCI',
            #                  'label': 'CO(GT)',
            #                  'label_period': 1,
            #                  'training_percentage': 0.03}

            self.air_quality_gen()
        elif self.settings.dataset == 'm4':
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
        df['Datetime'] = [pd.Timestamp(str(df['Date'].iloc[i].date()) + ' ' + str(df['Time'].iloc[i])) for i in range(len(df))]
        df.index = df['Datetime'].values
        df.index = pd.DatetimeIndex(df['Datetime'].values, freq='1H')

        df.drop(['Time', 'Date'], axis=1, inplace=True)
        df.index = pd.DatetimeIndex(df['Datetime'].values, freq='1H')

        # NMHC(GT) Has no values for most of the rows
        df.drop(['Datetime', 'NMHC(GT)'], axis=1, inplace=True)
        # df.fillna(method='ffill', inplace=True)
        df.interpolate(method='linear', inplace=True)

        training_df = df.iloc[:int(df.shape[0] * self.settings.training_percentage)]
        test_df = df.drop(training_df.index)
        # print(training_df.shape)

        # create labels, create features for training and test
        def get_features_labels(mixed_df):
            # Cince CO(GT) is the label, we need to make sure we are looking 'ahead' to define it
            mixed_df[self.settings.label] = mixed_df[self.settings.label].shift(-self.settings.label_period)
            mixed_df = mixed_df.dropna()
            y = pd.DataFrame(mixed_df[self.settings.label])
            if self.settings.use_exog is True:
                x = mixed_df.drop(self.settings.label, axis=1)
            else:
                x = None
            return x, y

        self.features_tr.single_output, self.labels_tr.single_output = get_features_labels(training_df)
        # self.features_tr.multioutput, self.labels_tr.multioutput = get_features_labels(training_df)

        self.features_ts.single_output, self.labels_ts.single_output = get_features_labels(test_df)
        # self.features_ts.multioutput, self.labels_ts.multioutput = get_features_labels(test_df)

    def m4_gen(self):
        if self.settings.use_exog is True:
            raise ValueError('No exogenous variables available for the m4 datasets')
        training_filename = 'data/m4/Hourly-train.csv'
        test_filename = 'data/m4/Hourly-test.csv'

        def _load_m4(filename, idx):
            with open(filename) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Skip the header
                next(csv_reader, None)
                rows = list(csv_reader)
            # The first column is the time series identifier
            row = rows[idx][1:]
            # Remove empty strings from the end of the list
            row = list(filter(None, row))
            return pd.DataFrame(np.array(row).astype(float), columns=['m4_' + str(idx)])

        training_time_series = _load_m4(training_filename, self.settings.m4_time_series_idx)
        test_time_series = _load_m4(test_filename, self.settings.m4_time_series_idx)
        # The m4 dataset doesn't seem to offer timestamps so we use integers as indexes
        test_time_series.index = pd.RangeIndex(start=training_time_series.index[-1] + 1, stop=training_time_series.index[-1] + 1 + len(test_time_series), step=1)

        def get_features_labels(time_series, horizon=1):
            y = forward_shift_ts(time_series, range(1, horizon + 1))
            # x, y = prune_data(time_series, y)
            return None, y

        self.features_tr.single_output, self.labels_tr.single_output = get_features_labels(training_time_series)
        # self.features_tr.multioutput, self.labels_tr.multioutput = get_features_labels(training_time_series, self.settings.forecast_length)

        self.features_ts.single_output, self.labels_ts.single_output = get_features_labels(test_time_series)
        # self.features_ts.multioutput, self.labels_ts.multioutput = get_features_labels(test_time_series, self.settings.forecast_length)


class MealearningDataHandler:
    def __init__(self, settings):
        self.settings = settings

        self.training_tasks = None
        self.validation_tasks = None
        self.test_tasks = None

        self.training_tasks_indexes = None
        self.validation_tasks_indexes = None
        self.test_tasks_indexes = None

        if self.settings.dataset == 'm4':
            self.m4_gen()
        elif self.settings.dataset == 'sine':
            self.synthetic_sine()
        elif self.settings.dataset == 'synthetic_ar':
            self.synthetic_ar()
        else:
            raise ValueError('Invalid dataset')

    def m4_gen(self):
        if self.settings.use_exog is True:
            raise ValueError('No exogenous variables available for the m4 datasets')
        training_filename = 'data/m4/Hourly-train.csv'
        test_filename = 'data/m4/Hourly-test.csv'

        # Skip first row
        assert(len(list(csv.reader(open(training_filename)))) - 1 == len(list(csv.reader(open(test_filename)))) - 1)
        n_rows = len(list(csv.reader(open(training_filename)))) - 1
        # n_rows = 00

        def _load_m4(filename, idx):
            with open(filename) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Skip the header
                next(csv_reader, None)
                rows = list(csv_reader)
            # The first column is the time series identifier
            row = rows[idx][1:]
            # Remove empty strings from the end of the list
            row = list(filter(None, row))
            row = row[:min(len(row), 24 * 14)]
            return pd.DataFrame(np.array(row).astype(float), columns=['m4_' + str(idx)])

        all_full_time_series = []
        for time_series_idx in range(n_rows):
            raw_training_time_series = _load_m4(training_filename, time_series_idx)
            raw_test_time_series = _load_m4(test_filename, time_series_idx)

            # The m4 dataset doesn't seem to offer timestamps so we use integers as indexes
            raw_test_time_series.index = pd.RangeIndex(start=raw_training_time_series.index[-1] + 1, stop=raw_training_time_series.index[-1] + 1 + len(raw_test_time_series), step=1)
            full_time_series = pd.concat([raw_training_time_series, raw_test_time_series])
            full_time_series = full_time_series[:500]
            all_full_time_series.append(full_time_series)
        # exit()
        # Split the tasks _indexes_ into training/validation/test
        training_tasks_pct = self.settings.training_tasks_pct
        validation_tasks_pct = self.settings.validation_tasks_pct
        test_tasks_pct = self.settings.test_tasks_pct
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_full_time_series)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        training_points_pct = self.settings.training_points_pct
        validation_points_pct = self.settings.validation_points_pct
        test_points_pct = self.settings.test_points_pct

        def dataset_splits(task_indexes):
            def get_labels(time_series, horizon=1):
                y = (time_series.shift(-horizon) - time_series) / time_series
                # y = time_series.pct_change().shift(-horizon)
                # Will dropna later in the feature generation etc
                # y = y.dropna()
                return y

            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                training_time_series, temp_time_series = train_test_split(all_full_time_series[task_index], test_size=1 - training_points_pct, shuffle=False)
                validation_time_series, test_time_series = train_test_split(temp_time_series, test_size=test_points_pct / (test_points_pct + validation_points_pct), shuffle=False)

                # Features will be filled later within the method, if it requires lagging etc, which is a parameter

                training = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                training.features = None
                training.labels = get_labels(training_time_series, horizon=self.settings.forecast_length)
                training.raw_time_series = training_time_series
                training.n_points = len(training.labels)

                validation = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                validation.features = None
                validation.labels = get_labels(validation_time_series, horizon=self.settings.forecast_length)
                validation.raw_time_series = validation_time_series
                validation.n_points = len(validation.labels)

                test = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                test.features = None
                test.labels = get_labels(test_time_series, horizon=self.settings.forecast_length)
                test.raw_time_series = test_time_series
                test.n_points = len(test.labels)

                SetType = namedtuple('SetType', ['training', 'validation', 'test', 'n_tasks'])
                data = SetType(training, validation, test, len(task_indexes))

                bucket.append(data)
            return bucket

        self.training_tasks = dataset_splits(training_tasks_indexes)
        self.validation_tasks = dataset_splits(validation_tasks_indexes)
        self.test_tasks = dataset_splits(test_tasks_indexes)

        self.training_tasks_indexes = training_tasks_indexes
        self.validation_tasks_indexes = validation_tasks_indexes
        self.test_tasks_indexes = test_tasks_indexes

        """
        I need to split the tasks into training/validation/test tasks
        I need to split each task into training/validation/test time series
        I need to save those as raw time series
        I need to extract the labels out of the raw time series
        
        I need to confirm that the labels don't snoop forward in anyway and have the correct indexing
        """

    def synthetic_sine(self):
        if self.settings.use_exog is True:
            raise ValueError('No exogenous variables available for the sine dataset')

        n_time_series = 100
        n_points = 200

        x = np.linspace(0, 20 * np.pi, n_points)
        ts = pd.DataFrame(np.sin(x), columns=['sine'])
        # ts = ts[:200]

        all_full_time_series = []
        for time_series_idx in range(n_time_series):
            new_level = np.random.randint(1, 100000)
            amplitude = np.sqrt(new_level)
            curr_ts = new_level + amplitude * ts
            # Adding noise
            curr_ts = curr_ts + (amplitude / 10) * np.random.randn(len(curr_ts)).reshape(-1, 1)
            all_full_time_series.append(curr_ts)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(all_full_time_series[0])
        # plt.pause(0.01)

        # Split the tasks _indexes_ into training/validation/test
        training_tasks_pct = self.settings.training_tasks_pct
        validation_tasks_pct = self.settings.validation_tasks_pct
        test_tasks_pct = self.settings.test_tasks_pct
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_full_time_series)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        training_points_pct = self.settings.training_points_pct
        validation_points_pct = self.settings.validation_points_pct
        test_points_pct = self.settings.test_points_pct

        def dataset_splits(task_indexes):
            def get_labels(time_series, horizon=1):
                # y = (time_series - time_series.shift(-horizon)) / time_series
                # y = time_series.shift(-horizon).pct_change()

                # y = time_series.shift(-horizon)
                # y = time_series.diff()
                y = time_series.pct_change().shift(-1)

                # Will dropna later in the feature generation etc
                # y = y.dropna()
                return y

            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                training_time_series, temp_time_series = train_test_split(all_full_time_series[task_index], test_size=1 - training_points_pct, shuffle=False)
                validation_time_series, test_time_series = train_test_split(temp_time_series, test_size=test_points_pct / (test_points_pct + validation_points_pct), shuffle=False)

                # Features will be filled later within the method, if it requires lagging etc, which is a parameter
                training = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                training.features = None
                training.labels = get_labels(training_time_series, horizon=1)
                training.raw_time_series = training_time_series
                training.n_points = len(training.labels)

                validation = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                validation.features = None
                validation.labels = get_labels(validation_time_series, horizon=1)
                validation.raw_time_series = validation_time_series
                validation.n_points = len(validation.labels)

                test = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                test.features = None
                test.labels = get_labels(test_time_series, horizon=1)
                test.raw_time_series = test_time_series
                test.n_points = len(test.labels)

                SetType = namedtuple('SetType', ['training', 'validation', 'test', 'n_tasks'])
                data = SetType(training, validation, test, len(task_indexes))

                bucket.append(data)
            return bucket

        self.training_tasks = dataset_splits(training_tasks_indexes)
        self.validation_tasks = dataset_splits(validation_tasks_indexes)
        self.test_tasks = dataset_splits(test_tasks_indexes)

        self.training_tasks_indexes = training_tasks_indexes
        self.validation_tasks_indexes = validation_tasks_indexes
        self.test_tasks_indexes = test_tasks_indexes

    def synthetic_ar(self):
        if self.settings.use_exog is True:
            raise ValueError('No exogenous variables available for the sine dataset')

        def ar_constraints(w_1_coeff, w_2_coeff, std=0.0):
            coeff_2 = np.array(w_2_coeff)
            w_2_std = std * coeff_2
            coeff_2 = coeff_2 + w_2_std * np.random.randn()
            coeff_2 = np.clip(coeff_2, -0.99, 0.99)

            w_1_upper = 1 - coeff_2
            w_1_lower = -1 + coeff_2
            coeff_1 = np.array(w_1_coeff)
            w_1_std = std * coeff_1
            coeff_1 = coeff_1 + w_1_std * np.random.randn()
            coeff_1 = np.clip(coeff_1, w_1_lower, w_1_upper)
            return coeff_1, coeff_2

        n_time_series = 100
        n_points = 60
        lags = 2

        w_2_centroid = np.random.uniform(-0.99, 0.99)  # |φ_2| < 1
        w_1_centroid = np.random.uniform(w_2_centroid - 1, 1 - w_2_centroid)  # φ_1 + φ_2 < 1 and φ_2 - φ_1 < 1
        # Make sure the centroids themselves satisfy the ar conditions
        w_1_centroid, w_2_centroid = ar_constraints(w_1_centroid, w_2_centroid)
        w_std_wrt_w = 0.1

        all_full_time_series = []
        signal_magnitude = 1
        signal_std = 0.1 * signal_magnitude
        noise_std = 0.01 * signal_magnitude

        all_w = []
        for time_series_idx in range(n_time_series):
            # signal_magnitude = np.random.uniform(1, 100)
            # signal_std = 0.1 * signal_magnitude
            # noise_std = 0.01 * signal_magnitude

            if lags == 1:
                weight_mean = np.array([w_1_centroid])
                weight_std = 0.05 * weight_mean
                weight_vector = weight_mean + weight_std * np.random.randn(lags)
                w = np.clip(weight_vector, -0.99, 0.99)
            elif lags == 2:
                w_1, w_2 = ar_constraints(w_1_centroid, w_2_centroid, w_std_wrt_w)
                w = np.array([w_1, w_2])
            else:
                raise ValueError('Generation based on AR(p) for p > 2 not implemented')
            print(w.ravel())
            all_w.append(w)
            previous_values = list(signal_std * np.random.randn(lags))

            from copy import deepcopy
            curr_ts = deepcopy(previous_values)
            # n_points * 3 to allow the series to "warmup" and stabilize
            for idx in range(5 * n_points - lags):
                ar_value = previous_values @ w + noise_std * np.random.randn()

                curr_ts.append(ar_value)
                previous_values = [ar_value] + previous_values[:-1]
            curr_ts = pd.DataFrame(curr_ts[-n_points:], columns = ['ts_' + str(time_series_idx)])
            all_full_time_series.append(curr_ts[-n_points:])

        import matplotlib.pyplot as plt
        my_dpi = 100
        n_plots = min(n_time_series, 8)
        fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
        for time_series_idx in range(n_plots):
            curr_ax = ax[time_series_idx]
            curr_ax.plot(all_full_time_series[time_series_idx])
            curr_ax.axhline(y=0, color='k')

            curr_ax.spines["top"].set_visible(False)
            curr_ax.spines["right"].set_visible(False)
            curr_ax.spines["bottom"].set_visible(False)

        w_mean = np.mean(all_w, axis=0)
        w_std = np.std(all_w, axis=0)
        title = 'w_center = (' + "{:6.4f}".format(w_mean[0]) + ', ' + "{:6.4f}".format(w_mean[1]) + ')' + '     w_std = ' + '(' + "{:6.4f}".format(w_std[0]) + ', ' + "{:6.4f}".format(w_std[1]) + ')'
        plt.suptitle(title)
        plt.savefig(title + ".jpg")
        plt.pause(0.1)
        # plt.show()
        # exit()

        # Split the tasks _indexes_ into training/validation/test
        training_tasks_pct = self.settings.training_tasks_pct
        validation_tasks_pct = self.settings.validation_tasks_pct
        test_tasks_pct = self.settings.test_tasks_pct
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_full_time_series)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        training_points_pct = self.settings.training_points_pct
        validation_points_pct = self.settings.validation_points_pct
        test_points_pct = self.settings.test_points_pct

        def dataset_splits(task_indexes):
            def get_labels(time_series, horizon=1):
                # y = (time_series - time_series.shift(-horizon)) / time_series
                # y = time_series.shift(-horizon).pct_change()

                # y = time_series.shift(-horizon)
                # y = time_series.diff()
                y = time_series.pct_change().shift(-horizon)
                # y = time_series.shift(-horizon)

                # Will dropna later in the feature generation etc
                # y = y.dropna()
                return y

            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                training_time_series, temp_time_series = train_test_split(all_full_time_series[task_index], test_size=1 - training_points_pct, shuffle=False)
                validation_time_series, test_time_series = train_test_split(temp_time_series, test_size=test_points_pct / (test_points_pct + validation_points_pct), shuffle=False)

                # Features will be filled later within the method, if it requires lagging etc, which is a parameter
                training = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                training.features = None
                training.labels = get_labels(training_time_series, horizon=1)
                training.raw_time_series = training_time_series
                training.n_points = len(training.labels)

                validation = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                validation.features = None
                validation.labels = get_labels(validation_time_series, horizon=1)
                validation.raw_time_series = validation_time_series
                validation.n_points = len(validation.labels)

                test = namedtuple('Data', ['n_points', 'features', 'labels', 'raw_time_series'])
                test.features = None
                test.labels = get_labels(test_time_series, horizon=1)
                test.raw_time_series = test_time_series
                test.n_points = len(test.labels)

                SetType = namedtuple('SetType', ['training', 'validation', 'test', 'n_tasks'])
                data = SetType(training, validation, test, len(task_indexes))

                bucket.append(data)
            return bucket

        self.training_tasks = dataset_splits(training_tasks_indexes)
        self.validation_tasks = dataset_splits(validation_tasks_indexes)
        self.test_tasks = dataset_splits(test_tasks_indexes)

        self.training_tasks_indexes = training_tasks_indexes
        self.validation_tasks_indexes = validation_tasks_indexes
        self.test_tasks_indexes = test_tasks_indexes
