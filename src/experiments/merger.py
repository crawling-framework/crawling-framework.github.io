import json
import os
import re
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from utils import CENTRALITIES
from utils import RESULT_DIR

np.set_printoptions(precision=2)
colors = list({'MOD': 'red', 'POD': 'm', 'BFS': 'b', 'DFS': 'k',
               'RW_': 'green', 'RC_': 'gray', 'FFC': 'orange'}.values())

# everyone of this is iterable
graph_name = 'petster-hamster'  # 'dolphins' # TODO now need to change by hands

budget_path = glob(os.path.join(RESULT_DIR, graph_name, 'step=*,budget=*'))  # TODO take first founded budget, step
for j, path_name in enumerate(budget_path):
    budget = int(re.search('budget=(\d+)$', path_name).group(1))
    step = int(re.search('step=(\d+),', path_name).group(1))
    print('Running file {}, with step={} and {} iterations'.format(j, budget, step))
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 9), )

    for metr_id, metric_name in enumerate(CENTRALITIES):
        subplot_x, subplot_y = metr_id // 3, metr_id % 3

        # file_path = ../results/Graph_name/step=10,budget=10/MOD/crawled_centrality[NUM].json
        files_path = os.path.join(RESULT_DIR, graph_name, 'step={},budget={}'.format(step, budget)) + '/'
        crawlers = [file.replace(files_path, '') for file in glob(files_path + '*')]

        # maybe to make several for every metric
        average_plot = dict([(c, dict()) for c in crawlers])

        for i, crawler_name in enumerate(crawlers):
            experiments_path = glob(os.path.join(files_path, crawler_name) + '/*' + metric_name + '*.json')
            count = len(experiments_path)
            average_plot[crawler_name] = np.zeros(budget)
            for experiment in experiments_path:
                with open(experiment, 'r') as f:
                    imported = json.load(f)
                    # print(len(np.array(imported)), crawler_name, experiment, np.array(imported))
                    average_plot[crawler_name] += (np.array(imported[:budget])) / count
                    axs[subplot_x, subplot_y].plot(imported, color=colors[i],
                                                   linewidth=0.5, linestyle=':')

            axs[subplot_x, subplot_y].set_title(metric_name)
            axs[subplot_x, subplot_y].plot(average_plot[crawler_name], label=crawler_name, color=colors[i], linewidth=2)
            axs[subplot_x, subplot_y].legend(loc='lower right')
            axs[subplot_x, subplot_y].grid()

    plt.xticks(range(budget), [i * step for i in range(budget)])
    fig.add_subplot(111, frameon=False)  # adding big frame to place names
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('crawling iterations')
    plt.ylabel('share of target set crawled')
    # plt.xticks(range(budget + 1), )
    plt.suptitle('Graph: {}, iterations={}'.format(graph_name, budget))
    plt.savefig(os.path.join(RESULT_DIR, graph_name, 'total_plot_step={},iter={}.pdf'.format(step, budget)))

plt.show()
