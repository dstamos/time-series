from src.data_management import Settings, MealearningDataHandler
from src.training import training
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
import matplotlib.ticker as mtick

font = {'size': 48}
matplotlib.rc('font', **font)


# seed_range = range(1, 21)
# tr_points_range = range(12, 201, 2)

seed_range = range(1, 301)
tr_points_range = [i * 24 for i in [3, 4, 5, 6, 7] + list(range(7, 6*7+1, 7))]

lags = 6

tt = time()

all_errors_mean_itl = []
all_errors_std_itl = []
all_errors_mean_ltl = []
all_errors_std_ltl = []
ratios_mean = []
ratios_std = []
for tr_points in tr_points_range:
    temp_seed_errors_itl = []
    temp_seed_errors_ltl = []
    temp_ratios = []
    for seed in seed_range:
        print(seed, tr_points)
        import pickle
        filename = 'seed_' + str(seed) + '_n_points_' + str(tr_points) + '_lags_' + str(lags)
        full_path = os.getcwd() + '/results/madrid/' + filename
        try:
            splits, data_settings, model_itl, model_ltl = pickle.load(open(full_path + '.pckl', "rb"))
        except:
            print('broken')
            continue

        itl_errors = np.mean(model_itl.all_test_perf)

        ltl_test_per_per_training_task = model_ltl.test_per_per_training_task
        ltl_errors = ltl_test_per_per_training_task[-1]

        temp_seed_errors_itl.append(itl_errors)
        temp_seed_errors_ltl.append(ltl_errors)

        # temp_ratios.append(itl_errors / ltl_errors)
        rel_improvement = (itl_errors - ltl_errors) / ltl_errors
        temp_ratios.append(rel_improvement * 100)

        # plt.axhline(y=itl_errors)
        # plt.plot(ltl_test_per_per_training_task)
        # plt.show()
    all_errors_mean_itl.append(np.mean(temp_seed_errors_itl))
    all_errors_mean_ltl.append(np.mean(temp_seed_errors_ltl))

    ratios_mean.append(np.mean(temp_ratios))
    ratios_std.append(np.std(temp_ratios))

ratios_mean = np.array(ratios_mean)
ratios_std = np.array(ratios_std)

fig, ax = plt.subplots(figsize=(1920 / 100, 1080 / 100), facecolor='white', dpi=100, nrows=1, ncols=1)
plt.plot(tr_points_range, ratios_mean)
ax.fill_between(tr_points_range, ratios_mean - ratios_std, ratios_mean + ratios_std, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='_nolegend_')

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.axhline(y=0, linestyle=':', color='tab:gray')
plt.xlabel('# training hours')
plt.ylabel('rel. improvement over ITL')
fig.tight_layout()
plt.pause(0.1)
plt.savefig('improvement_vs_tr_points.png', pad_inches=0)
plt.show()
k = 1
