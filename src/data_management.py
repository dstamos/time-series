import numpy as np
import pandas as pd
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
        self.labels = None
        self.features = None

        if self.settings.data.dataset == 'AirQualityUCI':
            self.air_quality_gen()
        else:
            raise ValueError('Invalid dataset')

    def air_quality_gen(self):
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
