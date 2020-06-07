import json
import os
import re
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
# from numpy import trapz
from sklearn import metrics  # import auc
from tqdm import tqdm

# from statistics import Stat as file # for small graphs
Stat = ["DEGREE_DISTR", "BETWEENNESS_DISTR", "ECCENTRICITY_DISTR", "CLOSENESS_DISTR", "PAGERANK_DISTR",
        "K_CORENESS_DISTR"]
rus_stat = {"DEGREE_DISTR": "Степень",
            "BETWEENNESS_DISTR": "Посредничество",
            "ECCENTRICITY_DISTR": "Эксцентриситет",
            "CLOSENESS_DISTR": "Степень близости",
            "PAGERANK_DISTR": "Пейджранк",
            "K_CORENESS_DISTR": "K-ядерность"}
# Stat = ["DEGREE_DISTR", "PAGERANK_DISTR", "K_CORENESS_DISTR"]  # for big graphs
# from utils import CENTRALITIES
from utils import RESULT_DIR

graphs_sizes = {'mipt': 14313,
                'ego-gplus': 23613,
                'facebook-wosn-links': 63392,
                'slashdot-threads': 51083,
                'digg-friends': 261489,
                'douban': 154908,
                'petster-hamster': 2000,
                'petster-friendships-cat': 148826}

STATISTICS_COLOR = {'ECCENTRICITY_DISTR': 'r',
                    'BETWEENNESS_DISTR': 'y',
                    'DEGREE_DISTR': 'g',
                    'CLOSENESS_DISTR': 'cyan',
                    'PAGERANK_DISTR': 'magenta',
                    'K_CORENESS_DISTR': 'b'}

all_methods_list = ["MOD", "MOD10", "MOD100", "MOD1000", "MOD10000",
                    "POD", "POD10", "POD100", "POD1000", "POD10000",
                    "BFS", "SBS90", "SBS75", "SBS50", "SBS25", "SBS10",
                    "RW_", "RC_", "DFS"]


def get_methods_list(calculated_aucs=None, top_k_percent=0.01,
                     filter_only=''):  # get all needed methods with filter, that we need (for all graphs)
    if calculated_aucs == None:
        with open(os.path.join(RESULT_DIR + '/k={:1.2f}'.format(top_k_percent), 'auc_only{}.json'.format(filter_only)),
                  'r') as f:
            calculated_aucs = json.load(f)
    method_names_set = set()  # getting all methods for drawing
    graph_names = list(calculated_aucs.keys())
    for graph_name in graph_names:
        for stat_name in list(calculated_aucs[graph_name].keys()):
            for method_name in calculated_aucs[graph_name][stat_name].keys():
                if (filter_only in method_name) and (method_name not in without_crawlers):
                    method_names_set.add(method_name)
    method_names = list(method_names_set)
    method_names.sort()
    return method_names


def merge(graph_name='petster-hamster', show=True, top_k_percent=0.1, filter_only='', set_x_scale='', filter_budget=0,
          without_crawlers=set()):
    RESULT_DIR_BUDG = RESULT_DIR + '/k={:1.2f}'.format(top_k_percent)
    np.set_printoptions(precision=2)
    # colors = list({'MOD': 'red', 'POD': 'm', 'BFS': 'b', 'DFS': 'k',
    #               'RW_': 'green', 'RC_': 'gray', 'FFC': 'orange'}.values(),)
    colors = ['r', 'darkgreen', 'b', 'k', 'gray', 'magenta', 'darkviolet', 'cyan', 'gold', 'brown', 'lime',
              'darkblue', 'sienna', 'teal', 'peru'] * 2

    budget_path = glob(os.path.join(RESULT_DIR_BUDG, graph_name, 'budget=*'))  # TODO take first founded budget, step
    for j, path_name in enumerate(budget_path):
        budget = int(re.search('budget=(\d+)$', path_name).group(1))
        print('budget=', budget, budget_path)

        # step = int(re.search('step=(\d+),', path_name).group(1))
        if ((filter_budget != 0) and (budget == filter_budget)) or (filter_budget == 0):
            # print('Running file {}, with {} iterations'.format(j, budget))
            subplotcount = (3, 2)  # 2 3
            fig, axs = plt.subplots(*subplotcount, sharex=True, sharey=True, figsize=(20, 15), )
            fig.text(0.5, 0.02, 'Число итераций краулинга', ha='center')
            fig.text(0.02, 0.5, 'Доля собранных влиятельных вершин', va='center', rotation='vertical')
            # x_arr = [int(i * step) for i in range(budget)]  # it was before removing step
            statistics = Stat  # [s for s in Stat if 'DISTR' in s.name]
            areas = dict((stat, dict()) for stat in statistics)

            for metr_id, stat in enumerate(statistics):
                subplot_x, subplot_y = metr_id // 2, metr_id % 2
                if subplotcount[0] * subplotcount[1] > 1:
                    plt.sca(axs[subplot_x][subplot_y])
                crawler_color_counter = 0  #
                # file_path = ../results/Graph_name/budget=10/MOD/crawled_centrality[NUM].json
                files_path = os.path.join(RESULT_DIR_BUDG, graph_name, 'budget={}'.format(budget)) + '/'
                crawlers = [file.replace(files_path, '') for file in glob(files_path + '*')
                            if (not file.replace(files_path, '') in without_crawlers) and
                            (filter_only in file.replace(files_path, ''))]
                crawlers.sort()
                # print(budget,stat, crawlers)
                # areas like: areas['petster-hamster']['DEGREE_DISTR']['MOD100']
                areas[stat] = dict((crawler_name, dict()) for crawler_name in crawlers)
                average_plot = dict([(c, dict()) for c in crawlers])

                for i, crawler_name in enumerate(crawlers):
                    if (filter_only in crawler_name) and (crawler_name not in without_crawlers):

                        experiments_path = glob(
                            os.path.join(files_path, crawler_name) + '/*' + stat + '*.json')[:8]  # FIXME check only 8
                        count = len(experiments_path)

                        first_iter = True
                        # if metr_id == 1: 
                        #     print(graph_name, crawler_name, 'total experiments:', count)
                        for experiment in experiments_path:
                            with open(experiment, 'r') as f:
                                imported = json.load(f)
                                x_arr = [int(x) for x in list(imported.keys())]
                                if set_x_scale == 'log':
                                    x_arr_names = x_arr
                                    x_arr = np.log([x + 1 for x in x_arr])
                                y_arr = [float(x) for x in list(imported.values())]
                                if first_iter:
                                    average_plot[crawler_name] = np.zeros(len(x_arr))
                                    first_iter = False
                                # print(i, crawler_name, len(x_arr), len(y_arr))
                                # print(len(np.array(imported)), crawler_name, experiment, np.array(imported))
                                average_plot[crawler_name] += (np.array(y_arr)) / count
                                plt.plot(x_arr, y_arr, color=colors[crawler_color_counter],
                                         linewidth=0.5, linestyle=':')

                        maxauc = budget
                        if set_x_scale == 'log':
                            plt.xticks(x_arr[0::20], x_arr_names[0::20])
                            # plt.xscale('log')
                            maxauc = np.log(budget)

                        plt.title(rus_stat[stat])
                        # if (filter_only in crawler_name) and (crawler_name not in without_crawlers):

                        plt.plot(x_arr, average_plot[crawler_name],
                                 label='[' + str(count) + '] ' + crawler_name,
                                 color=colors[crawler_color_counter],
                                 linewidth=3)
                        # print('plotted ', '[' + str(count) + '] ' + crawler_name, colors[crawler_color_counter])
                        crawler_color_counter += 1
                        areas[stat][crawler_name] = float(metrics.auc(x_arr, average_plot[crawler_name])) / maxauc
                plt.legend()

            # , [int(i * step * budget/10) for i in range(10)])
            # loc='lower right', mode='expand')  # only on last iter
            # fig.add_subplot(111, frameon=False, xlabel=None)  # adding big frame to place names
            plt.gca()
            plt.subplots_adjust(left=0.07, bottom=0.08, right=0.98, top=0.93, wspace=0.03, hspace=0.08)
            # plt.suptitle('Graph: {}, budget={}'.format(graph_name, budget))
            plt.suptitle('Граф: {}, бюджет={}'.format(graph_name, budget))

            fig_name = os.path.join(RESULT_DIR_BUDG,
                                    'Crawling_curves_graph_{}_budget={}'.format(graph_name, budget) + set_x_scale)
            if filter_only != '':
                fig_name += 'only_' + filter_only
            if subplotcount[0] * subplotcount[1] > 1:
                plt.sca(axs[0][0])
            plt.savefig(fig_name + '_{}.pdf'.format(set_x_scale))
            plt.savefig(fig_name + '_{}.png'.format(set_x_scale))

            # for stat_name in areas:
            #     for crawler_name in areas[stat_name]:
            #         print('stat {}, crawler {}, auc={}'.format(stat_name,crawler_name,areas[stat_name][crawler_name][graph_name]))
            # print('areas', areas)
            if os.path.exists(os.path.join(RESULT_DIR_BUDG, '{}auc_only{}.json'.format(set_x_scale, filter_only))):
                with open(os.path.join(RESULT_DIR_BUDG, '{}auc_only{}.json'.format(set_x_scale, filter_only)),
                          'r') as f:
                    calculated_aucs = json.load(f)
                    calculated_aucs[graph_name] = areas
                with open(os.path.join(RESULT_DIR_BUDG, '{}auc_only{}.json'.format(set_x_scale, filter_only)),
                          'w') as f:
                    json.dump(calculated_aucs, f)
            else:
                # print('aaaaauc', areas)
                with open(os.path.join(RESULT_DIR_BUDG, '{}auc_only{}.json'.format(set_x_scale, filter_only)),
                          'w') as f:
                    d = {graph_name: areas}
                    json.dump(d, f)
    if (show):
        plt.show()
    else:
        plt.close()


def draw_auc(filter_only='', without_crawlers=set(), top_k_percent=0.1, set_x_scale='', show=False, ):
    # set_x_scale - 'log' or '' for 2 types of AUC
    RESULT_DIR_BUDG = RESULT_DIR + '/k={:1.2f}'.format(top_k_percent)
    with open(os.path.join(RESULT_DIR_BUDG, '{}auc_only{}.json'.format(set_x_scale, filter_only)), 'r') as f:
        calculated_aucs = json.load(f)

    graph_names = list(calculated_aucs.keys())
    method_names = get_methods_list(calculated_aucs, filter_only)
    statistics_names = Stat  # [s.name for s in Stat if 'DISTR' in s.name]
    aggregated_auc = dict(
        (method, dict((stat, 0) for stat in statistics_names)) for method in method_names)  # sum of real aucs values
    max_aggregated_auc = dict(
        (method, dict((stat, 0) for stat in statistics_names)) for method in method_names)  # sum of win counts
    print('graphs ', graph_names, len(graph_names) // 3, len(graph_names) % 3)
    fig, axs = plt.subplots((len(graph_names) + 1) // 2, 2, sharex=True, sharey=True,
                            # len(graph_names) // 3 + 1, len(graph_names) % 3 + 1,
                            figsize=(20, 15), )
    # fig.text(0.5, 0.00, 'method', ha='center')
    fig.text(0.02, 0.5, 'значение {}AUCC'.format(set_x_scale), va='center', rotation='vertical')
    mini = 1
    plt.subplots_adjust(left=0.04, bottom=0.12, right=0.98, top=0.92, wspace=0.15, hspace=0.18)
    for i, graph_name in enumerate(graph_names):
        stat_names = list(calculated_aucs[graph_name].keys())
        subplot_x, subplot_y = i // 2, i % 2
        # print(i, len(graph_names), subplot_x, subplot_y)
        plt.sca(axs[subplot_x][subplot_y])
        for c, stat_name in enumerate(stat_names):
            auc_avg = []
            max_auc_method, max_auc_method_value = method_names[0], 0
            for method_name in method_names:
                if (filter_only in method_name) and (method_name not in without_crawlers):
                    if method_name in calculated_aucs[graph_name][stat_name].keys():
                        auc_avg.append(
                            calculated_aucs[graph_name][stat_name][method_name])  # for method_name in method_names]
                        mini = min(mini, calculated_aucs[graph_name][stat_name][method_name])
                        aggregated_auc[method_name][stat_name] += auc_avg[-1]
                        # print('-',graph_name, stat_name, method_name, calculated_aucs[graph_name][stat_name][method_name])
                        if max_auc_method_value < auc_avg[-1]:
                            max_auc_method_value = auc_avg[-1]
                            max_auc_method = method_name
                    else:
                        auc_avg.append(0)  # calculated_aucs[graph_name][stat_name][method_name] = 0

            max_aggregated_auc[max_auc_method][stat_name] += 1

            # plt.ylim(mini, 1)
            plt.plot(method_names, auc_avg, color=STATISTICS_COLOR[stat_name])
            plt.scatter(method_names, auc_avg, color=STATISTICS_COLOR[stat_name], label=rus_stat[stat_name], )

        plt.legend()
        plt.title(graph_name + ' с {} вершинами'.format(graphs_sizes[graph_name]))
        plt.xticks(rotation=80)
        plt.grid()

    plt.gca()
    plt.title(graph_name + ' с {} вершинами'.format(graphs_sizes[graph_name]))
    plt.savefig(os.path.join(RESULT_DIR_BUDG, 'aucs_only_{},{}scale.png'.format(filter_only, set_x_scale)))
    plt.savefig(os.path.join(RESULT_DIR_BUDG, 'aucs_only_{},{}scale.pdf'.format(filter_only, set_x_scale)))
    if (show):
        plt.show()

    with open(os.path.join(RESULT_DIR_BUDG, 'max_aggregated_auc_only_{},{}scale.json'.format(filter_only, set_x_scale)),
              'w') as f:
        json.dump(max_aggregated_auc, f)
    with open(os.path.join(RESULT_DIR_BUDG, 'aggregated_auc_only_{},{}scale.json'.format(filter_only, set_x_scale)),
              'w') as f:
        json.dump(aggregated_auc, f)


def draw_aggregated_auc(count=6, top_k_percent=0.01, filter_only=''):
    RESULT_DIR_K = RESULT_DIR + '/k={:1.2f}'.format(top_k_percent)
    method_names = get_methods_list(filter_only=filter_only)
    statistics_names = Stat  # [s.name for s in Stat if 'DISTR' in s.name]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(10, 8), )
    plt.subplots_adjust(left=0.09, bottom=0.05, right=0.98, top=0.96, wspace=0.10, hspace=0.10)
    for log_num, set_x_scale in enumerate(['', 'log']):
        with open(
                os.path.join(RESULT_DIR_K, 'max_aggregated_auc_only_{},{}scale.json'.format(filter_only, set_x_scale)),
                'r') as f:
            max_aggregated_auc = json.load(f)
        with open(os.path.join(RESULT_DIR_K, 'aggregated_auc_only_{},{}scale.json'.format(filter_only, set_x_scale)),
                  'r') as f:
            aggregated_auc = json.load(f)
        for pl_num, aggr_auc in enumerate([max_aggregated_auc, aggregated_auc]):
            plt.sca(axs[pl_num][log_num])
            lines = []
            xs = np.array(list(range(len(method_names))))
            prev_bottom = np.array([0.0] * len(method_names))
            for p, prop in enumerate(statistics_names):
                h = []
                w = 0.8
                i = 0
                for method in method_names:
                    h.append(float(aggr_auc[method][prop]))
                    i += 1
                line = plt.bar(xs, h, w, bottom=prev_bottom, color=STATISTICS_COLOR[prop])
                lines.append(line)
                prev_bottom += h
            plt.subplots_adjust(bottom=0.20)
            plt.legend(lines, [rus_stat[stat_name] for stat_name in statistics_names])
            plt.xticks(xs, method_names)
            plt.xticks(rotation=65)
            plt.ylabel('Счёт у метода')
            if pl_num % 2 == 0:
                max_auc = 'побед по'
            else:
                max_auc = 'значений'

            # plt.title('Aggregated {}AUCC with {} in {} scale'.format(max_auc, count, set_x_scale))
            if set_x_scale == 'log':
                plt.title('Сумма {} wAUCC по {} графам'.format(max_auc, count))
            else:
                plt.title('Сумма {} AUCC по {} графам'.format(max_auc, count))

    plt.savefig(os.path.join(RESULT_DIR_K, 'total_aucs_vals_only_{}.png'.format(filter_only)))
    plt.savefig(os.path.join(RESULT_DIR_K, 'total_aucs_vals_only_{}.pdf'.format(filter_only)))

    plt.show()


if __name__ == '__main__':
    graphs = [
        #  'mipt',
        # 'slashdot-threads',
        # 'douban',
        # 'petster-hamster',
        # 'ego-gplus',
        # 'facebook-wosn-links',
        # 'digg-friends',
        'petster-friendships-cat',
    ]  # in two rows one under another

    # TODO normal arguments for function launch

    filter_only = 'MOD'
    without_crawlers = {'SB_', 'SB_10', 'SB_25', 'SB_50', 'SB_75', 'SB_90',
                        # 'SBS10', 'SBS25', 'SBS75','SBS90', 'POD10',
                        # 'MOD10', 'MOD100', 'MOD1000', 'MOD10000',

                        # 'Multi_10xMOD', 'Multi_30xMOD', 'Multi_2xMOD', 'Multi_3xMOD', 'Multi_4xMOD', 'Multi_100xMOD',
                        # 'Multi_50xMOD', 'Multi_5xMOD',
                        # 'Multi_10xBFS', 'Multi_30xBFS', 'Multi_2xBFS', 'Multi_3xBFS', 'Multi_4xBFS', 'Multi_500xBFS',
                        # 'Multi_50xBFS', 'Multi_5xBFS', 'Multi_1000xBFS','Multi_100xBFS',
                        # 'POD10', 'POD100', 'POD1000', 'POD10000',
                        # 'MOD10', 'MOD100', 'MOD1000', 'MOD10000',
                        }

    for graph_name in tqdm(graphs):
        merge(graph_name, show=True,
              filter_budget=0,
              filter_only=filter_only,
              without_crawlers=without_crawlers,
              top_k_percent=0.01,
              # set_x_scale='log',
              )

    # for graph_name in tqdm(graphs):
    #     merge(graph_name, show=False,
    #           filter_budget=0,
    #           filter_only=filter_only,
    #           without_crawlers=without_crawlers,
    #           top_k_percent=0.01,
    #           set_x_scale='log',
    #           )
    #
    # draw_auc(show=True,
    #          filter_only=filter_only,
    #          without_crawlers=without_crawlers,
    #          set_x_scale='log',
    #          top_k_percent=0.01,
    #          )
    # draw_auc(show=True,
    #          filter_only=filter_only,
    #          without_crawlers=without_crawlers,
    #          top_k_percent=0.01,
    #          # set_x_scale='log',
    #          )
    #
    # draw_aggregated_auc(
    #     filter_only=filter_only,
    #     top_k_percent=0.01,
    #     # without_crawlers=without_crawlers,
    #     count=len(graphs),
    # )
