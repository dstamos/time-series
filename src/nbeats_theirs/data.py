import csv
from os import listdir

import numpy as np


def get_m4_data(backcast_length, forecast_length, is_training=True):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    if is_training:
        filename = 'src/nbeats_theirs/data/m4/train/Daily-train.csv'
        # (99, 999)
    else:
        filename = 'src/nbeats_theirs/data/m4/val/Daily-test.csv'
        # (99, 14)

    all_time_series = []
    headers = True
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                all_time_series.append(line)
            if headers:
                headers = False
    all_time_series = np.array(all_time_series)

    time_series_idx = 0
    time_series = all_time_series[time_series_idx, :].astype(float)

    # TODO
    #  First split the time series into training and testing with appropriate lookback and lookahead
    #  Create the training and test batches
    #  Call the dataloader (outside of the generic generation here)



    x_batches, y_batches = [], []

    raw_time_series_training = []
    raw_time_series_test = []

    time_series = time_series.reshape((len(time_series), 1))

    # Start the batching with enough room to look back and forward for the features and labels
    for i in range(backcast_length, len(time_series) - forecast_length + 1):
        x_batches.append(time_series[i - backcast_length:i])
        y_batches.append(time_series[i:i + forecast_length])

    x_batches = np.array(x_batches).squeeze()
    y_batches = np.array(y_batches).squeeze()

    train_test_cutoff = int(len(x_batches) * 0.8)
    x_train_batches, y_train_batches = x_batches[:train_test_cutoff], y_batches[:train_test_cutoff]
    x_test_batches, y_test_batches = x_batches[train_test_cutoff:], y_batches[train_test_cutoff:]

    # Normalize features and labels to [0, 1] (since all time series are positive in this scenario)
    norm_constant = np.max(x_train_batches)
    x_train_batches = x_train_batches/norm_constant
    x_test_batches = x_test_batches/norm_constant
    y_train_batches = y_train_batches/norm_constant
    y_test_batches = y_test_batches/norm_constant

    for i in range(backcast_length):
        raw_time_series_training.append(x_train_batches[i][0])

    for i in range(len(y_train_batches)):
        raw_time_series_training.append(y_train_batches[i][0])

    for i in range(len(y_test_batches)):
        raw_time_series_test.append(y_test_batches[i][0])

    for i in range(1,forecast_length):
        raw_time_series_test.append(y_test_batches[-1][i])
    return x_train_batches, x_test_batches, y_train_batches, y_test_batches, x_tr, x_te

    #########################################################################################################
    #########################################################################################################
    #########################################################################################################

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
    df[self.settings.data.label] = df[self.settings.data.label].shift(-self.settings.data.horizon)
    df = df.dropna()
    self.labels = df[self.settings.data.label]
    self.features = df.drop(self.settings.data.label, axis=1)




# def dummy_data_generator_multivariate(backcast_length, forecast_length, signal_type='seasonality', random=False,
#                                       batch_size=32):
#     def get_x_y():
#         lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
#         if random:
#             offset = np.random.standard_normal() * 0.1
#         else:
#             offset = 1
#         if signal_type == 'trend':
#             a = lin_space + offset
#         elif signal_type == 'seasonality':
#             a = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
#             a += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
#             a += lin_space * offset + np.random.rand() * 0.1
#         elif signal_type == 'cos':
#             a = np.cos(2 * np.pi * lin_space)
#         else:
#             raise Exception('Unknown signal type.')
#
#         x = a[:backcast_length]
#         y = a[backcast_length:]
#
#         min_x, max_x = np.minimum(np.min(x), 0), np.max(np.abs(x))
#
#         x -= min_x
#         y -= min_x
#
#         x /= max_x
#         y /= max_x
#
#         return x, y
#
#     def gen():
#         while True:
#             xx = []
#             yy = []
#             for i in range(batch_size):
#                 x, y = get_x_y()
#                 xx.append(x)
#                 yy.append(y)
#             yield np.array(xx), np.array(yy)
#
#     return gen()
#
#
# def get_m4_data_multivariate(backcast_length, forecast_length, is_training=True):
#     # to be downloaded from https://www.mcompetitions.unic.ac.cy/the-dataset/
#
#     filename = '../examples/data/m4/train/Daily-train.csv'
#     x_tl = []
#     x_max = []
#     headers = True
#     with open(filename, "r") as file:
#         reader = csv.reader(file, delimiter=',')
#         for line in reader:
#             line = line[1:]
#             if not headers:
#                 x_tl.append(line)
#             if headers:
#                 headers = False
#     x_tl_tl = np.array(x_tl)
#     for i in range(x_tl_tl.shape[0]):
#         if len(x_tl_tl[i]) < backcast_length + forecast_length:
#             continue
#         time_series = np.array(x_tl_tl[i])
#         time_series = [float(s) for s in time_series if s != '']
#         x_max.append(np.max(time_series))
#     x_max = np.max(x_max)
#
#     if is_training:
#         filename = '../examples/data/m4/train/Daily-train.csv'
#     else:
#         filename = '../examples/data/m4/val/Daily-test.csv'
#
#     x = np.array([]).reshape(0, backcast_length)
#     y = np.array([]).reshape(0, forecast_length)
#     x_tl = []
#     headers = True
#     with open(filename, "r") as file:
#         reader = csv.reader(file, delimiter=',')
#         for line in reader:
#             line = line[1:]
#             if not headers:
#                 x_tl.append(line)
#             if headers:
#                 headers = False
#     x_tl_tl = np.array(x_tl)
#     for i in range(x_tl_tl.shape[0]):
#         if len(x_tl_tl[i]) < backcast_length + forecast_length:
#             continue
#         time_series = np.array(x_tl_tl[i])
#         time_series = [float(s) for s in time_series if s != '']
#         time_series = time_series / x_max
#         if is_training:
#             time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
#             time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
#             j = np.random.randint(backcast_length, time_series.shape[0] + 1 - forecast_length)
#             time_series_cleaned_forlearning_x[0, :] = time_series[j - backcast_length: j]
#             time_series_cleaned_forlearning_y[0, :] = time_series[j:j + forecast_length]
#         else:
#             time_series_cleaned_forlearning_x = np.zeros(
#                 (time_series.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
#             time_series_cleaned_forlearning_y = np.zeros(
#                 (time_series.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
#             for j in range(backcast_length, time_series.shape[0] + 1 - forecast_length):
#                 time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series[j - backcast_length:j]
#                 time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series[j: j + forecast_length]
#         x = np.vstack((x, time_series_cleaned_forlearning_x))
#         y = np.vstack((y, time_series_cleaned_forlearning_y))
#
#     return x.reshape(x.shape[0], x.shape[1], 1), None, y.reshape(y.shape[0], y.shape[1], 1)
#
#
# def process_data(filename):
#     import wfdb
#     ecg_list = listdir(filename)
#     sample_list = [ecg[:-4] for ecg in ecg_list]
#     clean_sample_list = [ecg for ecg in sample_list if
#                          ecg not in ['102-0', 'ANNOTA', 'REC', 'SHA256SUMS', 'mitd', 'x_m']]
#     all_samples = np.zeros((len(clean_sample_list), 650000, 2))
#     for idx, ecg in enumerate(clean_sample_list):
#         record = wfdb.rdrecord(filename + ecg)
#         all_samples[idx] = record.p_signal
#
#     return all_samples
#
#
# def get_kcg_data(backcast_length, forecast_length, is_training=True):
#     # to be downloaded from https://physionet.org/content/mitdb/1.0.0/
#     # once downloaded should be put in ../examples/data/kcg/
#
#     dataset = process_data(filename='../examples/data/kcg/')
#     x_max = np.amax(np.abs(dataset[:195, :, :]), axis=(0, 1))
#
#     if is_training:
#         dataset = dataset[:195, :, :]
#     else:
#         dataset = dataset[195:, 30000:30000 + backcast_length + forecast_length + 10, :]
#
#     x = np.array([]).reshape(0, backcast_length, 2)
#     y = np.array([]).reshape(0, forecast_length, 2)
#
#     for i in range(dataset.shape[0]):
#         if (dataset[i].shape[0] < backcast_length + forecast_length):
#             continue
#         time_series = dataset[i]
#         time_series = time_series / x_max
#         if is_training:
#             time_series_cleaned_forlearning_x = np.zeros((1, backcast_length, 2))
#             time_series_cleaned_forlearning_y = np.zeros((1, forecast_length, 2))
#             j = np.random.randint(backcast_length, time_series.shape[0] + 1 - forecast_length)
#             time_series_cleaned_forlearning_x[0] = time_series[j - backcast_length: j, :]
#             time_series_cleaned_forlearning_y[0] = time_series[j:j + forecast_length, :]
#         else:
#             time_series_cleaned_forlearning_x = np.zeros(
#                 (time_series.shape[0] + 1 - (backcast_length + forecast_length), backcast_length, 2))
#             time_series_cleaned_forlearning_y = np.zeros(
#                 (time_series.shape[0] + 1 - (backcast_length + forecast_length), forecast_length, 2))
#             for j in range(backcast_length, time_series.shape[0] + 1 - forecast_length):
#                 time_series_cleaned_forlearning_x[j - backcast_length] = time_series[j - backcast_length:j, :]
#                 time_series_cleaned_forlearning_y[j - backcast_length] = time_series[j: j + forecast_length, :]
#
#         x = np.vstack((x, time_series_cleaned_forlearning_x))
#         y = np.vstack((y, time_series_cleaned_forlearning_y))
#
#     return x, None, y
#
#
# def process_data_price():
#     filename = '../examples/data/nrj/EPEX_spot_DA_auction_hour_prices_20070720-20170831.csv'
#
#     x_tl = []
#     headers = True
#     with open(filename, "r") as file:
#         reader = csv.reader(file, delimiter=',')
#         for line in reader:
#             if not headers:
#                 x_tl.append(line)
#             if headers:
#                 headers = False
#     x_tl = [float(x_tl[i][1]) for i in range(len(x_tl)) if '00:00:00' in x_tl[i][0]]
#     x_tl = np.array(x_tl)
#
#     return x_tl
#
#
# def process_data_load():
#     filename = '../examples/data/nrj/20150101-20170830-forecast_load_renewable_gen.csv'
#
#     x_tl = []
#     headers = True
#     with open(filename, "r") as file:
#         reader = csv.reader(file, delimiter=',')
#         for line in reader:
#             if not headers:
#                 x_tl.append(line)
#             if headers:
#                 headers = False
#
#     x_tl = [x_tl[i][1] for i in range(len(x_tl)) if '00:00:00' in x_tl[i][0]]
#     x_tl = [float(x_tl[i]) if x_tl[i] != '' else 0. for i in range(len(x_tl))]
#     x_tl[x_tl == 0] = np.mean(x_tl)
#     x_tl = np.array(x_tl)
#
#     return x_tl
#
#
# def process_data_gen():
#     filename = '../examples/data/nrj/20150101-20170830-gen_per_prod_type.csv'
#
#     x_tl = []
#     headers = True
#     with open(filename, "r") as file:
#         reader = csv.reader(file, delimiter=',')
#         for line in reader:
#             if not headers:
#                 x_tl.append(line)
#             if headers:
#                 headers = False
#
#     x_tl = [x_tl[i][1] for i in range(len(x_tl)) if '00:00:00' in x_tl[i][0]]
#     x_tl = [float(x_tl[i]) if x_tl[i] != '' else 0. for i in range(len(x_tl))]
#     x_tl[x_tl == 0] = np.mean(x_tl)
#     x_tl = np.array(x_tl)
#
#     return x_tl
#
#
# def get_x_y_data(backcast_length, forecast_length):
#     x = np.array([]).reshape(0, backcast_length)
#     y = np.array([]).reshape(0, forecast_length)
#
#     time_series = process_data_price()[:-1]
#
#     time_series_cleaned_forlearning_x = np.zeros(
#         (time_series.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
#     time_series_cleaned_forlearning_y = np.zeros(
#         (time_series.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
#     for j in range(backcast_length, time_series.shape[0] + 1 - forecast_length):
#         time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series[j - backcast_length:j]
#         time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series[j: j + forecast_length]
#     x = np.vstack((x, time_series_cleaned_forlearning_x))
#     y = np.vstack((y, time_series_cleaned_forlearning_y))
#
#     return x.reshape((x.shape[0], x.shape[1], 1)), y.reshape((y.shape[0], y.shape[1], 1))
#
#
# def get_exo_var_data(backcast_length, forecast_length):
#     e1 = np.array([]).reshape(0, backcast_length)
#     e2 = np.array([]).reshape(0, backcast_length)
#
#     time_series_1 = process_data_gen()
#     time_series_2 = process_data_load()
#     time_series_cleaned_forlearning_1 = np.zeros(
#         (time_series_1.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
#     time_series_cleaned_forlearning_2 = np.zeros(
#         (time_series_1.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
#     for j in range(backcast_length, time_series_1.shape[0] + 1 - forecast_length):
#         time_series_cleaned_forlearning_1[j - backcast_length, :] = time_series_1[j - backcast_length:j]
#         time_series_cleaned_forlearning_2[j - backcast_length, :] = time_series_2[j - backcast_length:j]
#     e1 = np.vstack((e1, time_series_cleaned_forlearning_1))
#     e2 = np.vstack((e2, time_series_cleaned_forlearning_2))
#
#     return e1, e2
#
#
# def get_nrj_data(backcast_length, forecast_length, is_training=True):
#     x, y = get_x_y_data(backcast_length, forecast_length)
#     e1, e2 = get_exo_var_data(backcast_length, forecast_length)
#
#     x_max = np.amax(np.abs(x[:90 * x.shape[0] // 100, :, :]), axis=(0, 1))
#     e1_max = np.amax(np.abs(e1[:90 * x.shape[0] // 100, :]), axis=(0, 1))
#     e2_max = np.amax(np.abs(e2[:90 * x.shape[0] // 100, :]), axis=(0, 1))
#
#     x = x / x_max
#     y = y / x_max
#     e1 = e1 / e1_max
#     e2 = e2 / e2_max
#
#     e = np.concatenate((e1.reshape((e1.shape[0], e1.shape[1], 1)), e2.reshape((e2.shape[0], e2.shape[1], 1))), axis=-1)
#
#     if is_training:
#         return x[:90 * x.shape[0] // 100], e[:90 * x.shape[0] // 100], y[:90 * x.shape[0] // 100]
#     else:
#         return x[90 * x.shape[0] // 100:], e[90 * x.shape[0] // 100:], y[90 * x.shape[0] // 100:]
