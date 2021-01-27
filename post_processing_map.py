from src.data_management import Settings, MealearningDataHandler
from src.training import training
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
from matplotlib import colors
import matplotlib.ticker as mtick

font = {'size': 48}
matplotlib.rc('font', **font)


# seed_range = range(1, 21)
# tr_points_range = range(12, 201, 2)

seed_range = range(1, 301)
# tr_points_range = [72, 96, 120, 144, 168, 336, 504, 672, 840, 1008]
tr_points = 168
lags = 6

n_stations = 24

# all_improvements = np.zeros((len(seed_range), n_stations))
# all_improvements[:] = np.nan
# for seed_idx, seed in enumerate(seed_range):
#     print(seed, tr_points)
#     import pickle
#     filename = 'seed_' + str(seed) + '_n_points_' + str(tr_points) + '_lags_' + str(lags)
#     full_path = os.getcwd() + '/results/madrid/' + filename
#     try:
#         splits, data_settings, model_itl, model_ltl = pickle.load(open(full_path + '.pckl', "rb"))
#     except:
#         print('broken')
#         continue
#
#     ltl_errors = model_ltl.all_all_test_perf[-1]
#     itl_errors = model_itl.all_test_perf
#
#     test_stations = splits['test']
#
#     for idx, test_station in enumerate(test_stations):
#         improvement = (itl_errors[idx] - ltl_errors[idx]) / ltl_errors[idx]
#         all_improvements[seed_idx, test_station] = improvement * 100
#
# average_improvement = np.nanmean(all_improvements, axis=0)
# std_improvement = np.nanstd(all_improvements, axis=0)
#
# k = 1
#
# #############################################
# #############################################
# #############################################
# station_info = pd.read_csv('./data/air_quality_madrid/stations.csv')
#
# BBox = ((-3.7995, -3.5505,
#          40.3370, 40.5281,))
#
# my_dpi = 100
# fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=1, ncols=1)
# ruh_m = plt.imread('./data/air_quality_madrid/madrid_map_pure.png')
# ax.imshow(ruh_m, zorder=0, extent=BBox, aspect=1.2, alpha=0.5)
#
# ##################
#
#
# class MidpointNormalize(colors.Normalize):
#     def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
#         self.vcenter = vcenter
#         colors.Normalize.__init__(self, vmin, vmax, clip)
#
#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))
#
#
# midnorm = MidpointNormalize(vmin=-np.max(np.abs(average_improvement)), vcenter=0, vmax=np.max(np.abs(average_improvement)))
# ##################
# s = ax.scatter(station_info['lon'].values, station_info['lat'].values, c=average_improvement, cmap='PiYG', norm=midnorm, s=3000, marker='v')
# cb = fig.colorbar(s)
# cb.set_label('rel. improvement over ITL')
#
# ax.scatter(station_info['lon'].values, station_info['lat'].values, color='k', s=100, marker='x')
#
# cbar = ax.collections[0].colorbar
# cbar.set_ticks([-10.0, -7.5, -5.0, -2.5, 0, 2.5, 5, 7.5, 10])
# cbar.set_ticklabels(['-10%', '-7.5%', '-5%', '-2.5%', '0%', '2.5%', '5%', '7.5%', '10%'])
#
# # ax.set_title('Plotting Spatial Data on Riyadh Map')
# ax.set_xlim(BBox[0], BBox[1])
# ax.set_ylim(BBox[2], BBox[3])
# plt.axis('off')
# fig.tight_layout()
#
# plt.pause(0.1)
# plt.savefig('improvement_vs_itl_map.png', pad_inches=0)

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

all_improvements = np.zeros((len(seed_range), n_stations))
all_improvements[:] = np.nan
for seed_idx, seed in enumerate(seed_range):
    print(seed, tr_points)
    import pickle
    filename = 'seed_' + str(seed) + '_n_points_' + str(tr_points) + '_lags_' + str(lags)
    full_path = os.getcwd() + '/results/madrid/' + filename
    try:
        splits, data_settings, model_itl, model_ltl = pickle.load(open(full_path + '.pckl', "rb"))
    except:
        print('broken')
        continue

    # ltl_errors = model_ltl.all_all_test_perf[-1]
    itl_errors = model_itl.all_test_perf

    # Compute the errors of the "straight" metaparameter model
    np.random.seed(seed)
    data = MealearningDataHandler(data_settings)
    model_ltl.predict(data.test_tasks)
    meta_errors = model_ltl.all_all_test_perf[-1]

    test_stations = splits['test']

    for idx, test_station in enumerate(test_stations):
        improvement = (itl_errors[idx] - meta_errors[idx]) / meta_errors[idx]
        all_improvements[seed_idx, test_station] = improvement * 100

average_improvement = np.nanmean(all_improvements, axis=0)
std_improvement = np.nanstd(all_improvements, axis=0)

#############################################
#############################################
#############################################

station_info = pd.read_csv('./data/air_quality_madrid/stations.csv')

BBox = ((-3.7995, -3.5505,
         40.3370, 40.5281,))

my_dpi = 100
fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=1, ncols=1)
ruh_m = plt.imread('./data/air_quality_madrid/madrid_map_pure.png')
ax.imshow(ruh_m, zorder=0, extent=BBox, aspect=1.2, alpha=0.5)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


midnorm = MidpointNormalize(vmin=-np.max(np.abs(average_improvement)), vcenter=0, vmax=np.max(np.abs(average_improvement)))
##################
s = ax.scatter(station_info['lon'].values, station_info['lat'].values, c=average_improvement, cmap='PiYG', norm=midnorm, s=3000, marker='v')
cb = fig.colorbar(s)
cb.set_label('rel. improv (no fine-tuning)')

ax.scatter(station_info['lon'].values, station_info['lat'].values, color='k', s=100, marker='x')

cbar = ax.collections[0].colorbar
cbar.set_ticks([-10.0, -7.5, -5.0, -2.5, 0, 2.5, 5, 7.5, 10])
cbar.set_ticklabels(['-10%', '-7.5%', '-5%', '-2.5%', '0%', '2.5%', '5%', '7.5%', '10%'])

# ax.set_title('Plotting Spatial Data on Riyadh Map')
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
plt.axis('off')
fig.tight_layout()

plt.pause(0.1)
plt.savefig('improvement_meta_vs_ltl_map.png', pad_inches=0)
