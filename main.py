import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 20000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 40000)


df = pd.read_excel('AirQualityUCI.xlsx', sheet_name=None)['AirQualityUCI']
df.replace(to_replace=-200, value=np.nan, inplace=True)

# Turning Time into Timestamp defaults to current date. So we replace dates with the given/correct ones
df['Datetime'] = [pd.Timestamp(str(df['Time'].iloc[i])) for i in range(len(df))]

for curr_index in range(len(df)):
    df['Datetime'].iloc[curr_index] = df['Datetime'].iloc[curr_index].replace(year=df['Date'].iloc[curr_index].year)
    df['Datetime'].iloc[curr_index] = df['Datetime'].iloc[curr_index].replace(month=df['Date'].iloc[curr_index].month)
    df['Datetime'].iloc[curr_index] = df['Datetime'].iloc[curr_index].replace(day=df['Date'].iloc[curr_index].day)

df.drop(['Time', 'Date'], axis=1, inplace=True)
df.index = pd.DatetimeIndex(df['Datetime'].values, freq='H')
# df.index = df['Datetime']

# Since CO(GT) is the label, we'll drop the rows that have CO(GT) nan
# df = df[df['CO(GT)'].notnull()]

# NMHC(GT) Has no values for most of the time series
df.drop(['Datetime', 'NMHC(GT)'], axis=1, inplace=True)
# df.fillna(method='ffill', inplace=True)
df.interpolate(method='linear', inplace=True)
print(df)

# Training stuff
horizon = 24 * 3
days = 15

# Cince CO(GT) is the label, we need to make sure we are looking 'ahead' to define it
df['CO(GT)'] = df['CO(GT)'].shift(-horizon)
df = df.dropna()

model = SARIMAX(df['CO(GT)'].iloc[:horizon*days], exog=None, order=(3, 1, 1), seasonal_order=(1, 1, 1, 24))
fit_res = model.fit(disp=True, maxiter=250)

fcast_res1 = fit_res.get_forecast(steps=horizon)
forecast_table = fcast_res1.summary_frame(alpha=0.10)

# exogenous
exog_variables = df.drop('CO(GT)', axis=1).iloc[:horizon*days].diff().fillna(method='bfill')
model_exog = SARIMAX(df['CO(GT)'].iloc[:horizon*days], exog=exog_variables, order=(3, 1, 1), seasonal_order=(1, 1, 1, 24))
fit_res_exog = model_exog.fit(disp=True, maxiter=250)

exog_forecast = df.drop('CO(GT)', axis=1).iloc[horizon*days:horizon*days+horizon].diff().fillna(method='bfill')
fcast_res1_exog = fit_res_exog.get_forecast(steps=horizon, exog=exog_forecast)
forecast_table_exog = fcast_res1_exog.summary_frame(alpha=0.10)
#######################################

plt.figure()
plt.plot(df['CO(GT)'].iloc[:horizon*days], 'k')
plt.plot(df['CO(GT)'].iloc[horizon*days:horizon*days+horizon], 'tab:blue')

plt.plot(forecast_table['mean'], 'tab:red')
plt.fill_between(forecast_table['mean'].index, forecast_table['mean_ci_lower'], forecast_table['mean_ci_upper'], alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True)


plt.plot(forecast_table_exog['mean'], 'tab:green')
plt.fill_between(forecast_table_exog['mean'].index, forecast_table_exog['mean_ci_lower'], forecast_table_exog['mean_ci_upper'], alpha=0.1, edgecolor='tab:green', facecolor='tab:green', antialiased=True)
plt.pause(0.1)

print('done')
k = 1

# TODO Create data_management file and copy paste shit from the parameter free paper
# done TODO Try SARIMAX on it
# done (seems slightly better) TODO Try SARIMAX on it on the diff of the indicaotrs
# done TODO Read NBEATS experiments single task (questions about i) testing without retraining ii) the metalearning intuition at the end of the paper iii) multivariate)

# TODO Data management
# TODO Testing metrics

# TODO Simple linear regression
# TODO Work on the diff of the labels and recover the full label afterwards
# TODO Try xgboost
# TODO Multivariate NBEATS

# TODO Read NBEATS experiments metalearning

# import matplotlib.pyplot as plt
# plt.figure()
# for col in df.keys():
#     df[col].plot(legend=True)
# plt.pause(0.1)
#
# fig = plt.figure()
# n_rows = 4
# n_cols = 4
# for idx, col in enumerate(df.keys()):
#     ax1 = fig.add_subplot(n_rows, n_cols, idx+1)
#     plt.plot(df[col])
#     plt.title(col)
#     plt.pause(0.1)
