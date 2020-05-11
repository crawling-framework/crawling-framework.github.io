import json
import os
import re
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from statistics import Stat
# from utils import CENTRALITIES
from utils import RESULT_DIR


def merge(graph_name, show=True):
    np.set_printoptions(precision=2)
    # colors = list({'MOD': 'red', 'POD': 'm', 'BFS': 'b', 'DFS': 'k',
    #               'RW_': 'green', 'RC_': 'gray', 'FFC': 'orange'}.values(),)
    colors = ['r', 'g', 'b', 'k', 'gray', 'orange', 'magenta', 'darkviolet', 'cyan', 'gold', 'brown', 'lime'] * 2

    # everyone of this is iterable
    if graph_name is None:
        graph_name = 'petster-hamster'  # 'digg-friends'  # 'dolphins' # TODO now need to change by hands

    budget_path = glob(os.path.join(RESULT_DIR, graph_name, 'step=*,budget=*'))  # TODO take first founded budget, step
    for j, path_name in enumerate(budget_path):
        budget = int(re.search('budget=(\d+)$', path_name).group(1))
        step = int(re.search('step=(\d+),', path_name).group(1))
        print('Running file {}, with step={} and {} iterations'.format(j, budget, step))
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 9), )
        x_arr = [int(i * step) for i in range(budget)]
        # FIXME parse all names of files to get statistics (centralities)
        statistics = [s for s in Stat if 'DISTR' in s.name]
        print('Statistics', [s.name for s in statistics])
        for metr_id, stat in enumerate(statistics):
            subplot_x, subplot_y = metr_id // 3, metr_id % 3

            # file_path = ../results/Graph_name/step=10,budget=10/MOD/crawled_centrality[NUM].json
            files_path = os.path.join(RESULT_DIR, graph_name, 'step={},budget={}'.format(step, budget)) + '/'
            crawlers = [file.replace(files_path, '') for file in glob(files_path + '*')]

            # maybe to make several for every metric
            average_plot = dict([(c, dict()) for c in crawlers])

            for i, crawler_name in enumerate(crawlers):
                experiments_path = glob(os.path.join(files_path, crawler_name) + '/*' + stat.name + '*.json')
                count = len(experiments_path)
                average_plot[crawler_name] = np.zeros(budget)
                if metr_id == 1:
                    print(crawler_name, 'total experiments:', count)
                for experiment in experiments_path:
                    with open(experiment, 'r') as f:
                        imported = json.load(f)

                        # print(len(np.array(imported)), crawler_name, experiment, np.array(imported))
                        average_plot[crawler_name] += (np.array(imported[:budget])) / count
                        if '' in crawler_name:  # TODO normal filter
                            axs[subplot_x, subplot_y].plot(x_arr, imported, color=colors[i],
                                                           linewidth=0.5, linestyle=':')

                axs[subplot_x, subplot_y].set_title(stat.description)
                if '' in crawler_name:  # TODO normal filter
                    axs[subplot_x, subplot_y].plot(x_arr, average_plot[crawler_name],
                                                   label='[' + str(count) + '] ' + crawler_name,
                                                   color=colors[i], linewidth=2)
                axs[subplot_x, subplot_y].legend(loc='lower right')
                axs[subplot_x, subplot_y].grid()
                axs[subplot_x, subplot_y].set_yscale('log')
                axs[subplot_x, subplot_y].set_xscale('log')

        print()
        # plt.xticks(range(budget), np.arange(0, budget, 10))#, [int(i * step * budget/10) for i in range(10)])
        fig.add_subplot(111, frameon=False)  # adding big frame to place names
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('crawling iterations')
        plt.ylabel('fraction of target set crawled \n\n\n')
        plt.suptitle('Graph: {}, batches={}, step={}'.format(graph_name, budget, step))
        fig_name = os.path.join(RESULT_DIR, graph_name, 'total_plot_step={},iter={}logy'.format(step, budget))
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')

    if show:
        plt.show()


if __name__ == '__main__':
    merge('petster-hamster', show=True)
