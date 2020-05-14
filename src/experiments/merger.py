import json
import os
import re
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from statistics import Stat
# from utils import CENTRALITIES
from utils import RESULT_DIR, REMAP_ITER


def merge(graph_name, show=True, filter_only='', set_x_scale='', filter_budget=0, without_crawlers=set()):
    np.set_printoptions(precision=2)
    # colors = list({'MOD': 'red', 'POD': 'm', 'BFS': 'b', 'DFS': 'k',
    #               'RW_': 'green', 'RC_': 'gray', 'FFC': 'orange'}.values(),)
    colors = ['r', 'darkgreen', 'b', 'k', 'gray', 'magenta', 'darkviolet', 'cyan', 'gold', 'brown', 'lime',
              'darkblue', 'sienna', 'teal', 'peru'] * 2

    # everyone of this is iterable
    if graph_name is None:
        graph_name = 'petster-hamster'  # 'digg-friends'  # 'dolphins' # TODO now need to change by hands

    budget_path = glob(os.path.join(RESULT_DIR, graph_name, 'budget=*'))  # TODO take first founded budget, step
    for j, path_name in enumerate(budget_path):
        budget = int(re.search('budget=(\d+)$', path_name).group(1))
        print('budget=', budget, budget_path)
        # step = int(re.search('step=(\d+),', path_name).group(1))
        if ((filter_budget != 0) and (budget == filter_budget)) or (filter_budget == 0):
            print('Running file {}, with {} iterations'.format(j, budget))
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 9), )
            fig.text(0.5, 0.02, 'crawling iterations X', ha='center')
            fig.text(0.02, 0.5, 'fraction of target set crawled', va='center', rotation='vertical')
            # x_arr = [int(i * step) for i in range(budget)]  # it was before removing step
            # FIXME parse all names of files to get statistics (centralities)
            statistics = [s for s in Stat if 'DISTR' in s.name]
            print('Statistics', [s.name for s in statistics])
            for metr_id, stat in enumerate(statistics):
                subplot_x, subplot_y = metr_id // 3, metr_id % 3
                crawler_color_counter = 0  #
                # file_path = ../results/Graph_name/step=10,budget=10/MOD/crawled_centrality[NUM].json
                files_path = os.path.join(RESULT_DIR, graph_name, 'budget={}'.format(budget)) + '/'
                crawlers = [file.replace(files_path, '') for file in glob(files_path + '*')]

                # maybe to make several for every metric
                average_plot = dict([(c, dict()) for c in crawlers])

                for i, crawler_name in enumerate(crawlers):
                    if (filter_only in crawler_name) and (crawler_name not in without_crawlers):
                        crawler_color_counter += 1
                        experiments_path = glob(os.path.join(files_path, crawler_name) + '/*' + stat.name + '*.json')
                        count = len(experiments_path)

                        first_iter = True
                        if metr_id == 1:
                            print(crawler_name, 'total experiments:', count)
                        for experiment in experiments_path:
                            with open(experiment, 'r') as f:
                                imported = json.load(f)

                                x_arr = [int(x) for x in list(imported.keys())]

                                y_arr = [float(x) for x in list(imported.values())]

                                if first_iter:
                                    average_plot[crawler_name] = np.zeros(len(x_arr))
                                    first_iter = False
                                # print(i, crawler_name, len(x_arr), len(y_arr))
                                # print(len(np.array(imported)), crawler_name, experiment, np.array(imported))
                                average_plot[crawler_name] += (np.array(y_arr)) / count
                                axs[subplot_x, subplot_y].plot(x_arr, y_arr, color=colors[crawler_color_counter],
                                                               linewidth=0.2, linestyle=':')

                    axs[subplot_x, subplot_y].set_title(stat.description)
                    if (filter_only in crawler_name) and (crawler_name not in without_crawlers):
                        axs[subplot_x, subplot_y].plot(x_arr, average_plot[crawler_name],
                                                       label='[' + str(count) + '] ' + crawler_name,
                                                       color=colors[crawler_color_counter],
                                                       linewidth=2.5)
                    # TODO something normal with xticks and labels
                    # axs[subplot_x, subplot_y].legend(loc='lower right', mode='expand')
                    # xticks = np.arange(0, budget+10, int(budget / 10))
                    # axs[subplot_x, subplot_y].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    # axs[subplot_x, subplot_y].set_xticks(range(budget), list(xticks))
                    # axs[subplot_x, subplot_y].grid()
                    # axs[subplot_x, subplot_y].set_yscale('log')
                    if set_x_scale == 'log':
                        axs[subplot_x, subplot_y].set_xscale('log')

            # , [int(i * step * budget/10) for i in range(10)])
            axs[subplot_x, subplot_y].legend()  # loc='lower right', mode='expand')  # only on last iter
            # fig.add_subplot(111, frameon=False, xlabel=None)  # adding big frame to place names
            plt.subplots_adjust(left=0.06, bottom=0.05, right=0.98, top=0.93, wspace=0.03, hspace=0.08)
            plt.suptitle('Graph: {}, batches={}'.format(graph_name, budget))
            fig_name = os.path.join(RESULT_DIR, graph_name, 'total_plot_iter={}'.format(budget) + set_x_scale)
            if filter_only != '':
                fig_name += 'only_' + filter_only
            plt.savefig(fig_name + '.pdf')
            plt.savefig(fig_name + '.png')

    if (show):
        plt.show()


if __name__ == '__main__':
    # graph_name = 'digg-friends' # with 261489 nodes and 1536577 edges
    # graph_name = 'douban' # http://konect.uni-koblenz.de/networks/douban
    # graph_name = 'ego-gplus' # http://konect.uni-koblenz.de/networks/ego-gplus
    # graph_name = 'slashdot-threads' # N=51083, V=116573.  use step=100,
    # graph_name = 'facebook-wosn-links'
    graph_name = 'slashdot-threads'  # with 2000 nodes and 16098 edges
    merge(graph_name, show=True, set_x_scale='', filter_budget=0,
          filter_only='POD',
          without_crawlers={'RW_', 'DFS', 'RC_', 'SBS10', 'POD100', 'POD', 'POD10'}
          )
