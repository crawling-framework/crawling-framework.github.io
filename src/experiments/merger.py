import json
import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from utils import CENTRALITIES
from utils import RESULT_DIR

np.set_printoptions(precision=2)
# everyone of this is iterable
graph_name = 'petster-hamster'  # TODO now need to change by hands
# 'dolphins'
colors = list({'MOD': 'red', 'POD': 'm', 'BFS': 'b', 'DFS': 'k',
               'RW_': 'green', 'RC_': 'gray', 'FFC': 'orange'}.values())

fig, axs = plt.subplots(3, 2, figsize=(16, 9))
for metr_id, metric_name in enumerate(CENTRALITIES):
    subplot_x, subplot_y = metr_id % 3, metr_id // 3
    budget = int(glob(os.path.join(RESULT_DIR, graph_name, 'budget=*'))[0].split('=')[1])
    x_ticks = [round(fl, 2) for fl in np.linspace(0, 1, budget + 1)]  # for plotting ticks from 0 to 1
    # taking all crawlers from it
    files_path = os.path.join(RESULT_DIR, graph_name, 'budget={}'.format(budget)) + '/'
    crawlers = [file.replace(files_path, '') for file in glob(files_path + '*')]

    # maybe to make several for every metric
    average_plot = dict([(c, dict()) for c in crawlers])

    for i, crawler_name in enumerate(crawlers):
        experiments_path = glob(os.path.join(files_path, crawler_name) + '/*' + metric_name + '*.json')
        count = len(experiments_path)
        average_plot[crawler_name] = np.zeros(budget + 1)
        for experiment in experiments_path:
            with open(experiment, 'r') as f:
                imported = json.load(f)
                print(len(np.array(imported)), crawler_name, experiment, np.array(imported))
                average_plot[crawler_name] += (np.array(imported[:budget + 1])) / count
                axs[subplot_x, subplot_y].plot(imported, color=colors[i],
                                               linewidth=1, linestyle='dashed')

        axs[subplot_x, subplot_y].set_title(metric_name)
        # axs[subplot_x, subplot_y].set_xticks(range(0, budget+1), x_ticks)
        axs[subplot_x, subplot_y].plot(average_plot[crawler_name], label=crawler_name, color=colors[i], linewidth=3)

    # plt.xscale(budget)

plt.xticks(range(budget + 1), )
plt.suptitle('Graph: {}, iterations={}'.format(graph_name, budget))
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULT_DIR, graph_name, 'total_plot_iter={}.png'.format(budget)))
plt.show()
