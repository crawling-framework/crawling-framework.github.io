import json
import os
import re
from glob import glob
from tqdm import tqdm
from operator import itemgetter
import numpy as np
from matplotlib import pyplot as plt

from statistics import Stat
# from utils import CENTRALITIES
from utils import RESULT_DIR, REMAP_ITER
from scipy.integrate import simps  # integrating
# from numpy import trapz
from sklearn import metrics  # import auc


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
            print('Running file {}, with {} iterations'.format(j, budget))
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 9), )
            fig.text(0.5, 0.00, 'crawling iterations X', ha='center')
            fig.text(0.02, 0.5, 'fraction of target set crawled', va='center', rotation='vertical')
            # x_arr = [int(i * step) for i in range(budget)]  # it was before removing step
            # FIXME parse all names of files to get statistics (centralities)
            statistics = [s for s in Stat if 'DISTR' in s.name]
            print('Statistics', [s.name for s in statistics])
            areas = dict((stat.name, dict()) for stat in statistics)

            for metr_id, stat in enumerate(statistics):
                subplot_x, subplot_y = metr_id // 3, metr_id % 3
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
                        experiments_path = glob(os.path.join(files_path, crawler_name) + '/*' + stat.name + '*.json')
                        count = len(experiments_path)
                        if set_x_scale == 'log':
                            axs[subplot_x, subplot_y].set_xscale('log')

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
                        # if (filter_only in crawler_name) and (crawler_name not in without_crawlers):
                        axs[subplot_x, subplot_y].plot(x_arr, average_plot[crawler_name],
                                                       label='[' + str(count) + '] ' + crawler_name,
                                                       color=colors[crawler_color_counter],
                                                       linewidth=2.5)

                        areas[stat.name][crawler_name] = float(metrics.auc(x_arr, average_plot[crawler_name])) / budget

            # , [int(i * step * budget/10) for i in range(10)])
            axs[subplot_x, subplot_y].legend()  # loc='lower right', mode='expand')  # only on last iter
            # fig.add_subplot(111, frameon=False, xlabel=None)  # adding big frame to place names
            plt.subplots_adjust(left=0.06, bottom=0.05, right=0.98, top=0.93, wspace=0.03, hspace=0.08)
            plt.suptitle('Graph: {}, budget={}'.format(graph_name, budget))
            fig_name = os.path.join(RESULT_DIR, graph_name, 'total_plot_iter={}'.format(budget) + set_x_scale)
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
                    # for stat_name in areas:
                    #     for crawler_name in areas[stat_name]:
                    #         if not crawler_name in calculated_aucs[stat_name].keys():
                    #             calculated_aucs[stat_name][crawler_name] = dict()
                    #         # if not graph_name in calculated_aucs[stat_name][crawler_name].keys():
                    #         calculated_aucs[stat_name][crawler_name][graph_name] = \
                    #             areas[stat_name][crawler_name][graph_name]
                    #         # else:
                    #         #     calculated_aucs[stat_name][crawler_name][graph_name].append(
                    #         #     areas[stat_name][crawler_name][graph_name])

    if (show):
        plt.show()


def draw_auc():
    with open(os.path.join(RESULT_DIR, 'auc.json'), 'r') as f:
        calculated_aucs = json.load(f)
    method_set = set()
    graph_names = list(calculated_aucs.keys())
    print('graphs ', graph_names, len(graph_names) // 3, len(graph_names) % 3)
    fig, axs = plt.subplots(len(graph_names) // 3 + 1, len(graph_names) % 3 + 1, sharex=True, sharey=True,
                            figsize=(20, 10), )
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.93, wspace=0.03, hspace=0.08)
    for i, graph_name in enumerate(graph_names):
        stat_names = list(calculated_aucs[graph_name].keys())

        subplot_x, subplot_y = i // 3, i % 3
        plt.sca(axs[subplot_x][subplot_y])
        for stat_name in stat_names:
            method_set.update(calculated_aucs[graph_name][stat_name].keys())
            method_names = list(method_set)  # list(calculated_aucs[graph_name][stat_name].keys())
            auc_avg = []
            for method_name in method_names:
                if method_name in calculated_aucs[graph_name][stat_name].keys():
                    auc_avg.append(
                        calculated_aucs[graph_name][stat_name][method_name])  # for method_name in method_names]
                else:
                    auc_avg.append(0)  # calculated_aucs[graph_name][stat_name][method_name] = 0

            lists = sorted(zip(*[auc_avg, method_names]), reverse=True, key=itemgetter(1))
            auc_avg, method_names = list(zip(*lists))
            # auc_avg, method_names =  zip(*sorted(zip(method_names, auc_avg), reverse=True, key=itemgetter(1)))
            # print('meth', method_names)
            # print('avg',auc_avg)
            plt.plot(method_names, auc_avg, label=stat_name)
        plt.title(graph_name)
        plt.xticks(rotation=90)
        plt.grid()

    plt.gca()
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'aucs.png'))
    plt.savefig(os.path.join(RESULT_DIR, 'aucs.pdf'))
    plt.show()


if __name__ == '__main__':
    graphs = ['mipt', 'petster-hamster', 'facebook-wosn-links', 'slashdot-threads', 'digg-friends']
    # graph_name = 'digg-friends' # with 261489 nodes and 1536577 edges
    # graph_name = 'douban' # http://konect.uni-koblenz.de/networks/douban         # !!!!!!
    # graph_name = 'ego-gplus' # http://konect.uni-koblenz.de/networks/ego-gplus   # !!!!!!!
    # graph_name = 'slashdot-threads' # N=51083, V=116573.  use step=100,
    # graph_name = 'facebook-wosn-links'
    # graph_name = 'slashdot-threads'  # with 2000 nodes and 16098 edges
    # graph_name ='mipt'
    # graph_name = 'petster-hamster'

    for graph_name in tqdm(graphs):
        merge(graph_name, show=False, set_x_scale='', filter_budget=0,
              # filter_only='POD',
              without_crawlers='Multi_BFS;BFS',  # {'RW_', 'DFS', 'RC_', 'SBS10', 'POD100', 'POD', 'POD10'}
              )

    draw_auc()
