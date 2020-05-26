import json
import os
import re
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
# from numpy import trapz
from sklearn import metrics  # import auc
from tqdm import tqdm

from statistics import Stat
# from utils import CENTRALITIES
from utils import RESULT_DIR

graphs_sizes = {'mipt': 14313,
                'ego-gplus': 23613,
                'facebook-wosn-links': 63392,
                'slashdot-threads': 51083,
                'digg-friends': 261489,
                'douban': 154908,
                'petster-hamster': 2000}

#
all_methods_list = ["MOD", "MOD10", "MOD100", "MOD1000", "MOD10000",
                    "POD", "POD10", "POD100", "POD1000", "POD10000",
                    "BFS", "SBS90", "SBS75", "SBS50", "SBS25", "SBS10",
                    "RW_", "RC_", "DFS"]


def merge(graph_name='petster-hamster', show=True, filter_only='', set_x_scale='', filter_budget=0,
          without_crawlers=set()):
    np.set_printoptions(precision=2)
    # colors = list({'MOD': 'red', 'POD': 'm', 'BFS': 'b', 'DFS': 'k',
    #               'RW_': 'green', 'RC_': 'gray', 'FFC': 'orange'}.values(),)
    colors = ['r', 'darkgreen', 'b', 'k', 'gray', 'magenta', 'darkviolet', 'cyan', 'gold', 'brown', 'lime',
              'darkblue', 'sienna', 'teal', 'peru'] * 2

    budget_path = glob(os.path.join(RESULT_DIR, graph_name, 'budget=*'))  # TODO take first founded budget, step
    for j, path_name in enumerate(budget_path):
        budget = int(re.search('budget=(\d+)$', path_name).group(1))
        print('budget=', budget, budget_path)

        # step = int(re.search('step=(\d+),', path_name).group(1))
        if ((filter_budget != 0) and (budget == filter_budget)) or (filter_budget == 0):
            # print('Running file {}, with {} iterations'.format(j, budget))
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 9), )
            fig.text(0.5, 0.00, 'crawling iterations X', ha='center')
            fig.text(0.02, 0.5, 'fraction of target set crawled', va='center', rotation='vertical')
            # x_arr = [int(i * step) for i in range(budget)]  # it was before removing step
            statistics = [s for s in Stat if 'DISTR' in s.name]
            areas = dict((stat.name, dict()) for stat in statistics)

            for metr_id, stat in enumerate(statistics):
                subplot_x, subplot_y = metr_id // 3, metr_id % 3
                plt.sca(axs[subplot_x][subplot_y])
                crawler_color_counter = 0  #
                # file_path = ../results/Graph_name/budget=10/MOD/crawled_centrality[NUM].json
                files_path = os.path.join(RESULT_DIR, graph_name, 'budget={}'.format(budget)) + '/'
                crawlers = [file.replace(files_path, '') for file in glob(files_path + '*')
                            if not file.replace(files_path, '') in without_crawlers]
                # areas like: areas['petster-hamster']['DEGREE_DISTR']['MOD100']
                areas[stat.name] = dict((crawler_name, dict()) for crawler_name in crawlers)
                average_plot = dict([(c, dict()) for c in crawlers])

                for i, crawler_name in enumerate(crawlers):
                    if (filter_only in crawler_name) and (crawler_name not in without_crawlers):
                        crawler_color_counter += 1
                        experiments_path = glob(os.path.join(files_path, crawler_name) + '/*' + stat.name + '*.json')[
                                           :8]  # TODO check only 8
                        count = len(experiments_path)
                        if set_x_scale == 'log':
                            plt.set_xscale('log')

                        first_iter = True
                        # if metr_id == 1: 
                        #     print(graph_name, crawler_name, 'total experiments:', count)
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
                                plt.plot(x_arr, y_arr, color=colors[crawler_color_counter],
                                         linewidth=0.2, linestyle=':')

                        plt.title(stat.description)
                        # if (filter_only in crawler_name) and (crawler_name not in without_crawlers):
                        plt.legend()
                        plt.plot(x_arr, average_plot[crawler_name],
                                 label='[' + str(count) + '] ' + crawler_name,
                                 color=colors[crawler_color_counter],
                                 linewidth=2.5)

                        areas[stat.name][crawler_name] = float(metrics.auc(x_arr, average_plot[crawler_name])) / budget

            # , [int(i * step * budget/10) for i in range(10)])
            # loc='lower right', mode='expand')  # only on last iter
            # fig.add_subplot(111, frameon=False, xlabel=None)  # adding big frame to place names
            plt.gca()
            plt.subplots_adjust(left=0.06, bottom=0.05, right=0.98, top=0.93, wspace=0.03, hspace=0.08)
            plt.suptitle('Graph: {}, budget={}'.format(graph_name, budget))
            fig_name = os.path.join(RESULT_DIR,
                                    'Crawling_curves_graph_{}_budget={}'.format(graph_name, budget) + set_x_scale)
            if filter_only != '':
                fig_name += 'only_' + filter_only
            plt.savefig(fig_name + '.pdf')
            plt.savefig(fig_name + '.png')

            # for stat_name in areas:
            #     for crawler_name in areas[stat_name]:
            #         print('stat {}, crawler {}, auc={}'.format(stat_name,crawler_name,areas[stat_name][crawler_name][graph_name]))
            # print('areas', areas)
            if os.path.exists(os.path.join(RESULT_DIR, 'auc.json')):
                with open(os.path.join(RESULT_DIR, 'auc.json'), 'r') as f:
                    calculated_aucs = json.load(f)
                    calculated_aucs[graph_name] = areas
                with open(os.path.join(RESULT_DIR, 'auc.json'), 'w') as f:
                    json.dump(calculated_aucs, f)
            else:
                # print('aaaaauc', areas)
                with open(os.path.join(RESULT_DIR, 'auc.json'), 'w') as f:
                    d = {graph_name: areas}
                    json.dump(d, f)
    if (show):
        plt.show()


def draw_auc(filter_only='', without_crawlers=set()):
    with open(os.path.join(RESULT_DIR, 'auc.json'), 'r') as f:
        calculated_aucs = json.load(f)

    with open(os.path.join(RESULT_DIR, 'all_methods_list.json'), 'r') as f:  # working with methods for x_axis of auc
        all_methods_list = json.load(f)
    method_names = [method_name for method_name in all_methods_list
                    if (filter_only in method_name) and (method_name not in without_crawlers)]

    graph_names = list(calculated_aucs.keys())
    print('graphs ', graph_names, len(graph_names) // 3, len(graph_names) % 3)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True,  # len(graph_names) // 3 + 1, len(graph_names) % 3 + 1,
                            figsize=(20, 10), )
    # fig.text(0.5, 0.00, 'method', ha='center')
    fig.text(0.02, 0.5, 'AUCC score', va='center', rotation='vertical')
    mini = 1
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.93, wspace=0.03, hspace=0.08)
    for i, graph_name in enumerate(graph_names):
        stat_names = list(calculated_aucs[graph_name].keys())
        subplot_x, subplot_y = i // 3, i % 3
        plt.sca(axs[subplot_x][subplot_y])
        for c, stat_name in enumerate(stat_names):
            auc_avg = []
            for method_name in method_names:
                if (filter_only in method_name) and (method_name not in without_crawlers):
                    if method_name in calculated_aucs[graph_name][stat_name].keys():
                        auc_avg.append(
                            calculated_aucs[graph_name][stat_name][method_name])  # for method_name in method_names]
                        mini = min(mini, calculated_aucs[graph_name][stat_name][method_name])
                    else:
                        auc_avg.append(0)  # calculated_aucs[graph_name][stat_name][method_name] = 0

            plt.ylim(mini, 1)
            plt.plot(method_names, auc_avg, color=['r', 'y', 'g', 'cyan', 'magenta', 'b'][c])
            plt.scatter(method_names, auc_avg, color=['r', 'y', 'g', 'cyan', 'magenta', 'b'][c], label=stat_name, )

        plt.legend()
        plt.title(graph_name + ' with {} nodes'.format(graphs_sizes[graph_name]))
        plt.xticks(rotation=75)
        plt.grid()
    plt.gca()

    plt.savefig(os.path.join(RESULT_DIR, 'aucs.png'))
    plt.savefig(os.path.join(RESULT_DIR, 'aucs.pdf'))
    plt.show()


if __name__ == '__main__':
    graphs = ['mipt', 'slashdot-threads', 'douban',
              'ego-gplus', 'facebook-wosn-links', 'digg-friends']  # in two rows one under another
    # graph_name = 'digg-friends' # with 261489 nodes and 1536577 edges
    # graph_name = 'douban' # http://konect.uni-koblenz.de/networks/douban         # !!!!!!
    # graph_name = 'ego-gplus' # http://konect.uni-koblenz.de/networks/ego-gplus   # !!!!!!!
    # graph_name = 'slashdot-threads' # N=51083, V=116573.  use step=100,
    graph_name = 'facebook-wosn-links'
    # graph_name = 'slashdot-threads'
    # graph_name ='mipt'
    # graph_name = 'petster-hamster'    with 2000 nodes and 16098 edges

    for graph_name in tqdm(graphs):
        merge(graph_name, show=False, set_x_scale='', filter_budget=0,
              # filter_only='POD',
              # without_crawlers={'MOD10', 'MOD100', 'MOD1000', 'MOD10000',
              #                   'POD10', 'POD100', 'POD1000', 'POD10000',
              #                   'SBS10', 'SBS90',  'SBS25',   'SBS75'},
              # {'RW_', 'DFS', 'RC_', 'SBS10', 'POD100', 'POD', 'POD10'}
              )

    draw_auc(
        # filter_only='',
        # without_crawlers={'MOD'}
    )
