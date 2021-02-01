import numpy as np
import pandas as pd
import csv
import pickle
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

        self.extra_info = None

        if self.settings.dataset == 'm4':
            self.m4_gen()
        elif self.settings.dataset == 'sine':
            self.synthetic_sine()
        elif self.settings.dataset == 'synthetic_ar':
            self.synthetic_ar()
        elif self.settings.dataset == 'air_quality_eu':
            self.air_quality_eu()
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

        n_time_series = self.settings.n_time_series
        n_tr_points = self.settings.n_tr_points
        n_test_points = self.settings.n_test_points
        n_total_points = self.settings.n_total_points
        lags = 2

        w_2_centroid = np.random.uniform(-0.99, 0.99)
        w_1_centroid = np.random.uniform(w_2_centroid - 1, 1 - w_2_centroid)  
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
            # print(w.ravel())
            all_w.append(w)
            previous_values = list(signal_std * np.random.randn(lags))

            from copy import deepcopy
            curr_ts = deepcopy(previous_values)
            # n_points * 3 to allow the series to "warmup" and stabilize
            for idx in range(5 * n_total_points - lags):
                ar_value = previous_values @ w + noise_std * np.random.randn()

                curr_ts.append(ar_value)
                previous_values = [ar_value] + previous_values[:-1]
            curr_ts = pd.DataFrame(curr_ts[-n_total_points:], columns=['ts_' + str(time_series_idx)])
            all_full_time_series.append(curr_ts[-n_total_points:])

        # import matplotlib.pyplot as plt
        # my_dpi = 100
        # n_plots = min(n_time_series, 8)
        # fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
        # for time_series_idx in range(n_plots):
        #     curr_ax = ax[time_series_idx]
        #     curr_ax.plot(all_full_time_series[time_series_idx])
        #     curr_ax.axhline(y=0, color='k')
        #
        #     curr_ax.spines["top"].set_visible(False)
        #     curr_ax.spines["right"].set_visible(False)
        #     curr_ax.spines["bottom"].set_visible(False)
        #
        # w_mean = np.mean(all_w, axis=0)
        # w_std = np.std(all_w, axis=0)
        # title = 'w_center = (' + "{:6.4f}".format(w_mean[0]) + ', ' + "{:6.4f}".format(w_mean[1]) + ')' + '     w_std = ' + '(' + "{:6.4f}".format(w_std[0]) + ', ' + "{:6.4f}".format(w_std[1]) + ')'
        # plt.suptitle(title)
        # plt.savefig(title + ".jpg")
        # plt.pause(0.1)
        # plt.show()
        # exit()

        # Split the tasks _indexes_ into training/validation/test
        training_tasks_pct = self.settings.training_tasks_pct
        validation_tasks_pct = self.settings.validation_tasks_pct
        test_tasks_pct = self.settings.test_tasks_pct
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_full_time_series)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        def dataset_splits(task_indexes):
            def get_labels(time_series, horizon=1):
                # y = (time_series - time_series.shift(-horizon)) / time_series
                # y = time_series.shift(-horizon).pct_change()

                # y = time_series.shift(-horizon)
                # y = time_series.diff()
                # y = time_series.pct_change().shift(-horizon)
                y = time_series.shift(-horizon)

                # Will dropna later in the feature generation etc
                # y = y.dropna()
                return y

            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                fixed_val_pct = 0.2
                temp = all_full_time_series[task_index].iloc[-(n_tr_points + n_test_points):-n_test_points]
                training_time_series, validation_time_series = train_test_split(temp, test_size=fixed_val_pct, shuffle=False)
                test_time_series = all_full_time_series[task_index].iloc[-n_test_points:]

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

    def air_quality_madrid(self):
        import pickle

        station_info = pd.read_csv('./data/air_quality_madrid/stations.csv')
        station_info.index = station_info.id
        station_names = []
        for station_id in station_info.index:
            station_names.append(station_info.loc[int(station_id)]['name'])

        station_ids = station_info.index
        # split_datasets = []
        # measures = pickle.load(open('madrid_data.pckl', "rb"))
        # for station_id in station_ids:
        #     # Four major pollutants https://www.blf.org.uk/support-for-you/air-pollution/types
        #     # O3 (Ground-level Ozone)
        #     # PM10 (Particulate Matter (soot and dust))
        #     # SO2 (Sulphur Dioxide)
        #     # NO2 (Nitrogen Dioxide)
        #     df = measures.loc[measures.station.astype(int) == station_id].drop(['station'], axis=1).astype(np.float)
        #     df = df[['O_3', 'PM10', 'SO_2', 'NO_2']]
        #     split_datasets.append(df)

        try:
            split_datasets = pickle.load(open('split_stations.pckl', "rb"))
        except:
            pass
            # import pickle5 as pickle
            # split_datasets = pickle.load(open('split_stations.pckl', "rb"))

        all_full_time_series = []
        for idx in range(len(station_ids)):

            ts = np.sum(split_datasets[idx].fillna(method='ffill'), axis=1)
            ts = ts.resample('H').pad()

            ts = ts.diff().dropna()

            all_full_time_series.append(ts.to_frame('madrid_station_' + str(idx)))

        # import matplotlib.pyplot as plt
        # for idx in range(len(all_time_series)):
        #     fig, ax = plt.subplots(figsize=(1920 / 100, 1080 / 100), facecolor='white', dpi=100, nrows=1, ncols=1)
        #     print(len(all_time_series[idx]))
        #     ax.plot(all_time_series[idx])
        #     plt.show()

        n_tr_points = self.settings.n_tr_points
        n_test_points = self.settings.n_test_points

        n_total = n_tr_points + n_test_points
        import matplotlib.pyplot as plt
        my_dpi = 100
        n_plots = min(len(all_full_time_series), 338)
        fig, ax = plt.subplots(figsize=(1920 / my_dpi, 3 * 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
        fig.subplots_adjust(hspace=.5)
        for time_series_idx in range(n_plots):
            curr_ax = ax[time_series_idx]
            curr_ax.plot(all_full_time_series[time_series_idx].iloc[-n_total:])
            curr_ax.axhline(y=0, color='tab:gray', linestyle=':')
            curr_ax.set_ylabel(station_names[time_series_idx], fontsize=8)

            curr_ax.spines["top"].set_visible(False)
            curr_ax.spines["right"].set_visible(False)
            curr_ax.spines["bottom"].set_visible(False)

        fig.align_ylabels()
        plt.suptitle('Madrid Air Quality')
        plt.savefig("madrid_data_raw.jpg")
        plt.pause(0.1)
        plt.show()
        exit()
        #
        # print(np.min(station_info['lon']))
        # print(np.max(station_info['lon']))
        #
        # print(np.min(station_info['lat']))
        # print(np.max(station_info['lat']))
        #
        # -3.77461
        # -3.58003
        # 40.34713
        # 40.51805
        #
        # new:
        # -3.7995
        # -3.5505
        # 40.3370
        # 40.5281

        # BBox = ((-3.8311, -3.5551,
        #          40.2950, 40.5190,))
        #
        # my_dpi = 100
        # fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1920 / my_dpi), facecolor='white', dpi=my_dpi, nrows=1, ncols=1)
        #
        # ruh_m = plt.imread('./data/air_quality_madrid/stations.png')
        # ax.imshow(ruh_m, zorder=0, extent=BBox)
        # ax.scatter(station_info['lon'].values, station_info['lat'].values, c='r', s=200)
        # # ax.set_title('Plotting Spatial Data on Riyadh Map')
        # ax.set_xlim(BBox[0], BBox[1])
        # ax.set_ylim(BBox[2], BBox[3])
        # plt.show()

        training_tasks_pct = self.settings.training_tasks_pct
        validation_tasks_pct = self.settings.validation_tasks_pct
        test_tasks_pct = self.settings.test_tasks_pct
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_full_time_series)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        def dataset_splits(task_indexes):
            def get_labels(time_series, horizon=1):
                # y = (time_series - time_series.shift(-horizon)) / time_series
                # y = time_series.shift(-horizon).pct_change()

                # y = time_series.shift(-horizon)
                # y = time_series.diff()
                # y = time_series.pct_change().shift(-horizon)
                y = time_series.shift(-horizon)

                # Will dropna later in the feature generation etc
                # y = y.dropna()
                return y

            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                fixed_val_pct = 0.4
                temp = all_full_time_series[task_index].iloc[-(n_tr_points + n_test_points):-n_test_points]
                training_time_series, validation_time_series = train_test_split(temp, test_size=fixed_val_pct, shuffle=False)
                test_time_series = all_full_time_series[task_index].iloc[-n_test_points:]

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
        self.extra_info = station_info

    def air_quality_eu(self):
        import airbase
        import dask
        import os
        import time
        import glob
        import requests

        download = False
        merge = False
        load_precooked = True

        if download is True:
            print('Downloading the raw data.')
            if not os.path.exists('./data/airbase_data'):
                os.makedirs('./data/airbase_data')

            client = airbase.AirbaseClient()

            all_countries = client.all_countries
            for curr_country in all_countries:
                tt = time.time()

                if not os.path.exists('./data/airbase_data/' + curr_country):
                    os.makedirs('./data/airbase_data/' + curr_country)

                r = client.request(country=curr_country, pl=['NO2', 'O3', 'PM10', 'SO2'], year_from=2015, preload_csv_links=True, verbose=False)
                all_csv_links = r._csv_links
                print(f'{curr_country} | {len(all_csv_links):5d} csv files')

                def download_csv_link(url):
                    filename = url[url.rfind('/')+1:]
                    fullpath = './data/airbase_data/' + curr_country + '/' + filename
                    if os.path.exists(fullpath):
                        return

                    with requests.Session() as s:
                        attempts = 0
                        while True:
                            try:
                                download = s.get(url)
                                break
                            except Exception as e:
                                attempts = attempts + 1
                                time.sleep(1)
                                if attempts > 5:
                                    print('Failed to download', url)
                                    return
                    try:
                        decoded_content = download.content.decode('utf-8')
                    except Exception as e:
                        try:
                            decoded_content = download.content.decode('utf-16')
                        except Exception as e:
                            print('Failed to decode.', url)
                            return
                    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                    my_list = list(cr)

                    with open(fullpath, "w", newline="") as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerows(my_list)
                    return

                parallel_output = []
                parallel_inputs = zip(all_csv_links)
                for parameters in parallel_inputs:
                    lazy_result = dask.delayed(download_csv_link)(*parameters)
                    parallel_output.append(lazy_result)

                n_workers = 8  # Set this to the number of cpus you have.
                dask.compute(*parallel_output, scheduler='processes', num_workers=n_workers)
                print(f'{curr_country} | {int(time.time() - tt):5d} sec')

        all_countries = ['HR', 'ES', 'MT', 'CZ', 'CY', 'FI', 'PL', 'NO', 'SE', 'XK', 'BA', 'EE', 'IE', 'TR', 'RS', 'GI',
                         'NL', 'SK', 'RO', 'AD', 'AL', 'IS', 'LU', 'CH', 'MK', 'IT', 'AT', 'DK', 'GR', 'FR', 'PT', 'LT', 'DE', 'BG']

        if merge is True:
            print('Merging the individual csv files per country.')
            for curr_country in all_countries:
                tt = time.time()
                all_filenames = glob.glob(os.path.join("./data/airbase_data/" + curr_country, '*.{}'.format('csv')))
                print(f'{curr_country} | {len(all_filenames):5d} csv files')

                def merge_station_data(station_filenames):
                    idx = pd.date_range(start="2015-01-01 00:00:00", end="2020-12-31 00:00:00", freq='1H')
                    big_boy_df = pd.DataFrame(columns=['start_time', 'station_id', 'NO2', 'O3', 'PM10', 'SO2'], index=idx)
                    big_boy_df.index = big_boy_df.index.astype(str)

                    for i in range(len(station_filenames)):
                        df = pd.read_csv(station_filenames[i], delimiter='\t')

                        if df.shape[1] == 1:
                            # Busted station
                            print('Busted station.')
                            return

                        if df['AveragingTime'].iloc[0] != 'hour':
                            continue

                        pollutant = df['AirPollutant'].unique()
                        if len(pollutant) > 1:
                            if len(pollutant) == 2 and pd.isnull(pollutant).any():
                                df = df.drop(np.where(pd.isnull(df['AirPollutant']))[0])
                            else:
                                raise ValueError('More than one pollutant per csv file', station_filenames[i])

                        df['DatetimeBegin'] = df['DatetimeBegin'].str.slice(0, 19)
                        df.index = df['DatetimeBegin'].values

                        # Making the two dataframes compatible
                        df = df.rename(columns={"DatetimeBegin": "start_time", "AirQualityStationEoICode": "station_id", "Concentration": pollutant[0]})

                        big_boy_df = big_boy_df.combine_first(df[['start_time', 'station_id', pollutant[0]]])

                    if pd.isnull(big_boy_df).all().all():
                        return

                    station_id = big_boy_df['station_id'].iloc[0]
                    if len(big_boy_df['station_id'].unique()) > 1:
                        if len(big_boy_df['station_id'].unique()) == 2 and pd.isnull(big_boy_df['station_id'].unique()).any():
                            # Some stations miss the exact station name (but the filename "id" matching confirms it)
                            station_id = big_boy_df['station_id'].unique()[~pd.isnull(big_boy_df['station_id'].unique())][0]
                        else:
                            print('one', len(big_boy_df['station_id'].unique()) == 2)
                            print('two', pd.isnull(big_boy_df['station_id'].unique()).any())
                            print('three', big_boy_df['station_id'].unique())
                            raise ValueError('More than one station in the set of csv files', station_filenames)

                    os.makedirs('./data/airbase_data_merged_stations', exist_ok=True)
                    full_path = os.path.join('./data/airbase_data_merged_stations/', curr_country + '_' + str(station_id) + '.csv')
                    big_boy_df.to_csv(full_path, sep='\t')

                unique_station_file_identifiers = list(set([s.split('_')[-3] for s in all_filenames]))
                filenames_grouped_per_station = []
                for identifier in unique_station_file_identifiers:
                    relevant_filenames = [filename if identifier == filename.split('_')[-3] else None for filename in all_filenames]
                    relevant_filenames = list(filter(None, relevant_filenames))

                    filenames_grouped_per_station.append(relevant_filenames)

                parallel_output = []
                parallel_inputs = zip(filenames_grouped_per_station)
                for parameters in parallel_inputs:
                    lazy_result = dask.delayed(merge_station_data)(*parameters)
                    parallel_output.append(lazy_result)

                n_workers = 8  # Set this to the number of cpus you have.
                dask.compute(*parallel_output, scheduler='processes', num_workers=n_workers)
                # dask.compute(*parallel_output, scheduler='single-threaded', num_workers=n_workers)

                print(f'{curr_country} | {int(time.time() - tt):5d} sec')

        ##############################################################################################
        ##############################################################################################
        ##############################################################################################
        load_precooked = True
        if load_precooked is False:
            all_station_filenames = glob.glob(os.path.join("./data/airbase_data_merged_stations/", '*.{}'.format('csv')))
            all_full_time_series = []
            for idx, filename in enumerate(all_station_filenames):
                ts = pd.read_csv(filename, delimiter='\t', index_col=0, error_bad_lines=False, verbose=False)
                station_id = ts['station_id'].unique()
                station_id = station_id[~pd.isnull(station_id)][0]
                # Because of the way I constructed the data, it might have some nan values before and after the actual data
                ts = ts.loc[ts.first_valid_index():ts.last_valid_index()]
                ts = ts[~ts.index.duplicated(keep='last')]

                ts = ts[['NO2', 'O3', 'PM10', 'SO2']].sum(axis=1, min_count=1)

                n_nans = len(np.where(ts.isnull())[0])
                if n_nans / len(ts) > 0.05:
                    print(f'{idx:5d} | {filename:s} | {n_nans / len(ts):4.2f}')
                    continue
                # print(f'{idx:5d} | {filename:s} | {n_nans:5d}')
                # TODO If more than say 10% is nan, remove the station
                ts = ts.fillna(method='ffill')
                # TODO Fill backwards as well?

                ts.index = pd.to_datetime(ts.index, format='%Y-%m-%d %H:%M:%S')
                unique_idx = ts.index.drop_duplicates(keep='last')
                ts = ts.loc[unique_idx]

                ts = ts.resample('H').pad()
                ts = ts.diff().dropna()

                all_full_time_series.append(ts.to_frame(filename.split('/')[-1] + '__' + str(station_id) + str(idx)))

            pickle.dump(all_full_time_series, open('./data/airbase_data_merged_stations/eu_data.pckl', "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            all_full_time_series = pickle.load(open('./data/airbase_data_merged_stations/eu_data.pckl', "rb"))

        # Remove time series that don't have the "right" history
        # all_full_time_series = all_full_time_series[:400]
        n_ts = len(all_full_time_series)
        bad_indexes = []
        for idx in range(n_ts):
            ts = all_full_time_series[idx]
            if len(ts) == 0:
                bad_indexes.append(idx)
                continue

            if 'DE_DESN093.csv__DESN0933880' == ts.columns[0]:
                k = 1

            end_date = ts.index[-1]
            start_date = ts.index[0]
            # FIXME Super hardcoded filter.
            if end_date.year != 2020 or start_date.year == 2020:
                # Because we don't want to want to train past a certain day.
                # This is ensures that all times series at least "reach" 2020.
                bad_indexes.append(idx)
                continue
            elif end_date.year == 2020 and end_date.month < 11:
                bad_indexes.append(idx)
                continue

        # Remove the problematic time series
        for index in sorted(bad_indexes, reverse=True):
            del all_full_time_series[index]

        n_tr_points = self.settings.n_tr_points
        n_test_points = self.settings.n_test_points

        n_total = n_tr_points + n_test_points
        # import matplotlib.pyplot as plt
        # my_dpi = 100
        # n_plots = min(len(all_full_time_series), 338)
        # fig, ax = plt.subplots(figsize=(1920 / my_dpi, 3 * 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
        # fig.subplots_adjust(hspace=.5)
        # for time_series_idx in range(n_plots):
        #     curr_ax = ax[time_series_idx]
        #     curr_ax.plot(all_full_time_series[time_series_idx].iloc[-n_total:])
        #     curr_ax.axhline(y=0, color='tab:gray', linestyle=':')
        #     curr_ax.set_ylabel(station_names[time_series_idx], fontsize=8)
        #
        #     curr_ax.spines["top"].set_visible(False)
        #     curr_ax.spines["right"].set_visible(False)
        #     curr_ax.spines["bottom"].set_visible(False)
        #
        # fig.align_ylabels()
        # plt.suptitle('Madrid Air Quality')
        # plt.savefig("madrid_data_raw.jpg")
        # plt.pause(0.1)
        # plt.show()
        # exit()

        training_tasks_pct = self.settings.training_tasks_pct
        validation_tasks_pct = self.settings.validation_tasks_pct
        test_tasks_pct = self.settings.test_tasks_pct
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_full_time_series)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        def dataset_splits(task_indexes):
            def get_labels(time_series, horizon=1):
                y = time_series.shift(-horizon)
                return y

            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                fixed_val_pct = 0.4
                temp = all_full_time_series[task_index].iloc[-(n_tr_points + n_test_points):-n_test_points]
                if len(temp) == 0:
                    task_indexes.remove(task_index)
                    continue
                training_time_series, validation_time_series = train_test_split(temp, test_size=fixed_val_pct, shuffle=False)
                test_time_series = all_full_time_series[task_index].iloc[-n_test_points:]

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

                if test.n_points < 12:
                    print('k')

                SetType = namedtuple('SetType', ['training', 'validation', 'test', 'n_tasks'])
                data = SetType(training, validation, test, len(task_indexes))

                bucket.append(data)
            return bucket, task_indexes

        self.training_tasks, training_tasks_indexes = dataset_splits(training_tasks_indexes)
        self.validation_tasks, validation_tasks_indexes = dataset_splits(validation_tasks_indexes)
        self.test_tasks, test_tasks_indexes = dataset_splits(test_tasks_indexes)
        self.training_tasks_indexes = training_tasks_indexes
        self.validation_tasks_indexes = validation_tasks_indexes
        self.test_tasks_indexes = test_tasks_indexes
