from time import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import os
import matplotlib
import matplotlib.ticker as mtick


font = {'size': 48}
matplotlib.rc('font', **font)


# seed_range = range(1, 301)
# tr_points = 72
# lags = 6

seed_range = [9999]
tr_pct = 0.2

all_seeds_ltl = []
all_seeds_itl = []
for seed in seed_range:
    foldername = './results-eu/'
    filename = 'seed_' + str(seed) + '-tr_pct_{:0.4f}'.format(tr_pct)
    full_path = foldername + filename
    try:
        results = pickle.load(open(full_path + '.pckl', "rb"))
        test_performance_naive = results['test_performance_naive']
        test_performance_single_task = results['test_performance_single_task']
        test_performance_itl = results['test_performance_itl']
        test_performance_meta = results['test_performance_meta']
    except:
        print('broken')
        continue

    fig, ax = plt.subplots(figsize=(1920 / 100, 1080 / 100), facecolor='white', dpi=100, nrows=1, ncols=1)
    plt.plot(np.ones(len(test_performance_meta)) * [test_performance_naive], 'tab:gray', linewidth=2)
    plt.plot(np.ones(len(test_performance_meta)) * [test_performance_itl], 'tab:red', linewidth=2)
    plt.plot(test_performance_meta, 'tab:blue', linewidth=2)
    plt.pause(0.1)

    training_tasks = splits['training']
    ltl_errors = model_ltl.test_per_per_training_task
    all_seeds_ltl.append(ltl_errors)

    itl_errors = np.mean(model_itl.all_test_perf)
    all_seeds_itl.append(itl_errors)

all_seeds_ltl = np.array(all_seeds_ltl)
average_ltl = np.mean(all_seeds_ltl, axis=0)
standard_deviation_ltl = np.std(all_seeds_ltl, axis=0)

all_seeds_itl = np.array(all_seeds_itl)
average_itl = np.mean(all_seeds_itl, axis=0)
standard_deviation_itl = np.std(all_seeds_itl, axis=0)

if len(training_tasks) < 100:
    x_range = params_to_check = range(1, len(training_tasks) + 1)
else:
    # To speed up the process
    x_range = np.arange(0, 50, 1)
    x_range = np.concatenate((x_range, np.arange(50, len(training_tasks), 50)))
    if len(training_tasks) - 1 not in x_range:
        x_range = np.append(x_range, len(training_tasks) - 1)


fig, ax = plt.subplots(figsize=(1920 / 100, 1080 / 100), facecolor='white', dpi=100, nrows=1, ncols=1)
plt.plot(x_range, average_ltl, 'tab:blue')
ax.fill_between(x_range, average_ltl - standard_deviation_ltl, average_ltl + standard_deviation_ltl, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='LTL')

plt.plot(x_range, len(x_range) * [average_itl], 'tab:gray')
ax.fill_between(x_range, len(x_range) * [average_itl - standard_deviation_itl], len(x_range) * [average_itl + standard_deviation_itl], alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', antialiased=True, label='ITL')


ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
fig.tight_layout()
plt.xlabel('# training tasks')
plt.ylabel('NMSE')
plt.legend()
# plt.savefig('errors_vs_tr_tasks.png', pad_inches=0)
plt.pause(0.1)

k = 1
