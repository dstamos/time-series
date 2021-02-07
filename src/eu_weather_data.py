import numpy as np
import pandas as pd
import csv
import pickle
from datetime import datetime
import dask
import os
import time
import glob
import requests
import airbase

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 40000)


def download_raw_data():
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
            filename = url[url.rfind('/') + 1:]
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


def load_and_merge_individual_station_files():
    print('Merging the individual csv files per country.')
    all_countries = ['HR', 'ES', 'MT', 'CZ', 'CY', 'FI', 'PL', 'NO', 'SE', 'XK', 'BA', 'EE', 'IE', 'TR', 'RS', 'GI', 'NL',
                     'SK', 'RO', 'AD', 'AL', 'IS', 'LU', 'CH', 'MK', 'IT', 'AT', 'DK', 'GR', 'FR', 'PT', 'LT', 'DE', 'BG']
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

        print(f'{curr_country} | {int(time.time() - tt):5d} sec')


def load_and_process_station_files(final_dataset_path):
    all_station_filenames = glob.glob(os.path.join("./data/airbase_data_merged_stations/", '*.{}'.format('csv')))

    def load_and_process_series_wrapper(fn):
        timeseries = pd.read_csv(fn, delimiter='\t', index_col=0, error_bad_lines=False, verbose=False)
        station_id = timeseries['station_id'].unique()
        station_id = station_id[~pd.isnull(station_id)][0]
        # Because of the way I constructed the data, it might have some nan values before and after the actual data
        timeseries = timeseries.loc[timeseries.iloc[1:].first_valid_index():timeseries.iloc[:-1].last_valid_index()]
        timeseries = timeseries[~timeseries.index.duplicated(keep='last')]

        timeseries = timeseries[['NO2', 'O3', 'PM10', 'SO2']].sum(axis=1, min_count=1)

        # FIXME Need to resample first
        nan_ratio = len(np.where(pd.isnull(timeseries))[0]) / len(timeseries)

        if nan_ratio > 0.05:
            print(f'{fn:s} | {"nan ratio:":18s} {100 * nan_ratio:6.3f}%')
            return None

        same_value_ratio = len(np.where(timeseries.diff() == 0)[0]) / len(timeseries)
        if same_value_ratio > 0.10:
            # Some stations have the same value filled forward. Probably lazy/incosiderate data collectors.
            print(f'{fn:s} | {"same value ratio:":18s} {100 * same_value_ratio:6.3f}%')
            return None

        timeseries = timeseries.dropna()

        earliest_required_timestamp = datetime.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000
        latest_required_timestamp = datetime.strptime('2020-12-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000

        earliest_timestamp = datetime.strptime(timeseries.index[0], '%Y-%m-%d %H:%M:%S').timestamp() * 1000
        latest_timestamp = datetime.strptime(timeseries.index[-1], '%Y-%m-%d %H:%M:%S').timestamp() * 1000

        if earliest_timestamp > earliest_required_timestamp or latest_timestamp < latest_required_timestamp:
            return None

        timeseries.index = pd.to_datetime(timeseries.index, format='%Y-%m-%d %H:%M:%S')
        unique_idx = timeseries.index.drop_duplicates(keep='last')
        timeseries = timeseries.loc[unique_idx]

        timeseries = timeseries.resample('H').pad()
        return timeseries.to_frame(str(station_id))

    parallel_output = []
    parallel_inputs = zip(all_station_filenames)
    for parameters in parallel_inputs:
        lazy_result = dask.delayed(load_and_process_series_wrapper)(*parameters)
        parallel_output.append(lazy_result)
    n_workers = 6  # Set this to the number of cpus you have.
    parallel_output = dask.compute(*parallel_output, scheduler='processes', num_workers=n_workers)  # "threads", "processes", "single-threaded"

    parallel_output = [output for output in parallel_output if output is not None]

    pickle.dump(parallel_output, open(final_dataset_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_eu_weather_data():
    fresh_download_raw_data = False
    merge_raw_data_to_stations = False
    process_stations = False
    final_dataset_path = './data/airbase_data_merged_stations/eu_data_pruned.pckl'

    if fresh_download_raw_data is True:
        download_raw_data()

    if merge_raw_data_to_stations is True:
        load_and_merge_individual_station_files()

    if process_stations is True:
        load_and_process_station_files(final_dataset_path)

    if not os.path.exists(final_dataset_path):
        raise RuntimeError('You need to download, merge and process the raw data. '
                           '\nfresh_download_raw_data = True \nmerge_raw_data_to_stations = True \nprocess_stations = True')

    all_raw_time_series = pickle.load(open(final_dataset_path, "rb"))
    all_raw_time_series = all_raw_time_series[:20]
    return all_raw_time_series
