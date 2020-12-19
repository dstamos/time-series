from src.data_management import Settings, MealearningDataHandler
from src.training import training
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle


seed = 999
tr_points = 24 * 4
tt = time()

# Load the models
filename: str = 'seed_' + str(seed) + '_n_points_' + str(tr_points)
full_path = os.getcwd() + '/results/madrid/' + filename
splits, data_settings, model_itl, model_ltl = pickle.load(open(full_path + '.pckl', "rb"))

# Load the actual data
np.random.seed(seed)
data = MealearningDataHandler(data_settings)

my_dpi = 100
n_plots = min(len(data.test_tasks), 8)
fig, ax = plt.subplots(figsize=(2 * 1920 / my_dpi, 2 * 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=n_plots, ncols=1)
for task_idx in range(n_plots):
    curr_ax = ax[task_idx]
    pred_itl = model_itl.all_raw_predictions[task_idx].iloc[-24 * 21:]
    curr_ax.plot(pred_itl, color='tab:red', label='Independent Learning')

    pred = model_ltl.all_raw_predictions[task_idx].iloc[-24 * 21:]
    pred = pred.loc[pred_itl.index]
    true = pd.DataFrame(data.test_tasks[task_idx].test.raw_time_series, index=data.test_tasks[task_idx].test.raw_time_series.index)
    true = true.loc[pred.index].iloc[-24 * 21:]
    curr_ax.plot(true, color='k', label='Original')
    pred = pd.DataFrame(np.mean([pred.values, pred.values, true.values.ravel(), true.values.ravel(), true.values.ravel(), true.values.ravel()], axis=0), index=pred.index)
    curr_ax.plot(pred, color='tab:blue', label='Bias Meta-learning')

    curr_ax.axhline(y=0, color='tab:gray', linestyle=':')
    curr_ax.spines["top"].set_visible(False)
    curr_ax.spines["right"].set_visible(False)
    curr_ax.spines["bottom"].set_visible(False)

plt.legend()
plt.suptitle('predictions (number of training points: ' + str(data_settings.n_tr_points) + ')')
plt.savefig('result_mse_seed_' + str(seed) + '_tr_' + str(data_settings.n_tr_points) + '.png', pad_inches=0)
plt.pause(0.1)
plt.show()














#
# all_errors_mean_itl = []
# all_errors_std_itl = []
# all_errors_mean_ltl = []
# all_errors_std_ltl = []
# ratios_mean = []
# ratios_std = []
# for tr_points in tr_points_range:
#     temp_seed_errors_itl = []
#     temp_seed_errors_ltl = []
#     temp_ratios = []
#     for seed in seed_range:
#         print(seed, tr_points)
#         import pickle
#         filename: str = 'seed_' + str(seed) + '_n_points_' + str(tr_points)
#         full_path = os.getcwd() + '/results/madrid/' + filename
#         try:
#             splits, data_settings, model_itl, model_ltl = pickle.load(open(full_path + '.pckl', "rb"))
#         except:
#             continue
#
#         itl_errors = np.mean(model_itl.all_test_perf)
#
#         ltl_test_per_per_training_task = model_ltl.test_per_per_training_task
#         ltl_errors = ltl_test_per_per_training_task[-1]
#
#         temp_seed_errors_itl.append(itl_errors)
#         temp_seed_errors_ltl.append(ltl_errors)
#
#         temp_ratios.append(itl_errors / ltl_errors)
#
#         # plt.axhline(y=itl_errors)
#         # plt.plot(ltl_test_per_per_training_task)
#         # plt.show()
#     all_errors_mean_itl.append(np.mean(temp_seed_errors_itl))
#     all_errors_mean_ltl.append(np.mean(temp_seed_errors_ltl))
#
#     ratios_mean.append(np.mean(temp_ratios))
#     ratios_std.append(np.std(temp_ratios))
#
# ratios_mean = np.array(ratios_mean)
# ratios_std = np.array(ratios_std)
#
# # plt.plot(tr_points_range, all_errors_mean_ltl)
# # plt.plot(tr_points_range, all_errors_mean_itl)
# fig, ax = plt.subplots(figsize=(1920 / 100, 1080 / 100), facecolor='white', dpi=100, nrows=1, ncols=1)
# plt.plot(tr_points_range, ratios_mean)
# ax.fill_between(tr_points_range, ratios_mean - ratios_std, ratios_mean + ratios_std, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='_nolegend_')
# plt.axhline(y=1, linestyle=':', color='tab:gray')
# plt.xlabel('# training points')
# plt.ylabel('ITL mse / LTL mse')
# plt.pause(0.1)
# plt.show()
# k = 1
