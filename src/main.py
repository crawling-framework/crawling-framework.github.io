import matplotlib.pyplot as plt
import numpy as np
import re
import json
import os
from sklearn.metrics import auc
from utils import get_percentile, import_graph

METRICS_LIST = ['degrees', 'k_cores', 'eccentricity', 'betweenness_centrality']
graph_names = ['wikivote', 'hamsterster', 'DCAM', 'gnutella', 'github', 'dblp2010']
method_names = ['RC','RW','BFS','DFS','MOD','DE']
METHOD_COLOR = {
    'AFD': 'pink',
    'RC': 'grey',
    'RW': 'green',
    'DFS': 'black',
    'BFS': 'blue',
    'DE': 'darkmagenta',
    'MOD': 'orangered'
}

PROPERTIES_COLOR = {
    'nodes': 'k-',
    'degrees': 'g-',
    'k_cores': 'b-',
    'eccentricity': 'r-',
    'betweenness_centrality': 'y-'}
property_name = {
    'degrees': 'degrees',
    'k_cores': 'k-cores',
    'eccentricity': 'eccentricity',
    'betweenness_centrality': 'betweenness centrality'
}

DUMPS_DIR = '../results/dumps_shared'
SEED_COUNT = 8
TOP_PERCENTILE = 10


def load_results(base_dir, graph):
    file_pattern = graph + r'_results_budget(\d+).json'
    budget, filename = None, None

    for filename in os.listdir(base_dir):
        match = re.match(file_pattern, filename)
        if match:
            budget = int(match.group(1))
            break

    if not budget:
        raise FileNotFoundError

    with open(os.path.join(base_dir, filename), 'r') as result_file:
        data = json.load(result_file)
    return data['crawler_avg'], data['history'], int(budget)


# draw plot

def draw_nodes_history(history, crawler_avg, print_methods, graph_name, seed_count, budget):
    """
    Drawing history for every method(average are bold) and every seed(thin)
    """
    # TBD - ,graph_name,seed_count are only for filenames. need to clean them
    plt.figure(graph_name, figsize=(10, 10))
    # plt.grid()

    auc_res = {}

    # for method, method_data in history.items():
    #     if method in print_methods:
    #         auc_res[method] = {}
    #         for seed_num, seed_data in list(method_data.items())[:seed_count]:
    #             data = np.array(seed_data['nodes'][:budget])
    #             auc_res[method][seed_num] = auc(x=np.arange(len(data)), y=data)
    #             plt.plot(data,
    #                      linewidth=0.5, linestyle=':',
    #                      color=METHOD_COLOR[method])

    maxmethod = np.zeros(budget)
    for i in range(budget):
        for method in print_methods:
            if crawler_avg[method]['nodes'][i] > maxmethod[i]:
                maxmethod[i] = crawler_avg[method]['nodes'][i]

    for method, avg_data in crawler_avg.items():
        if method not in print_methods:
            continue
        data = np.array(avg_data['nodes'][:budget])
        # auc_res[method]['average'] = auc(x=np.arange(len(data)), y=data)
        plt.plot(data-maxmethod-1,
                 linewidth=3,
                 color=METHOD_COLOR[method],
                 label=method)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('iteration number')
    plt.ylabel('node count')
    plt.legend()
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig('../results/history/' + graph_name + '_history_' +
                str(seed_count) + '_seeds_' +
                str(budget) + 'iterations.png')
    # plt.show()
    # return auc_res


def draw_properties_history(percentile_set, crawler_avg, print_methods, graph_name, seed_count,
                            budget):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.title(graph_name)
    # plt.figure(figsize=(20, 20))

    # auc_res = {}

    for method in print_methods:
        # auc_res[method] = {}
        for prop in METRICS_LIST:
            # ремап для красивой отрисовки 2*2
            j = {'degrees': 0, 'k_cores': 1, 'eccentricity': 2, 'betweenness_centrality': 3}[prop]

            data = np.array(crawler_avg[method][prop][:budget]) / len(percentile_set[prop])
            # auc_res[method][prop] = auc(x=np.arange(len(data)), y=data)
            ax = axs[j // 2][j % 2]
            ax.plot(data, label=method, color=METHOD_COLOR[method])
            # ax.set_title(graph_name + '% nodes with ' + prop + ' from ' + str(len(percentile_set[prop])))
            ax.set_title(property_name[prop])
            ax.legend()

            ax.grid(linestyle=':')
            ax.set_xlabel('iteration number')
            ax.set_ylabel('% of top nodes')

            # ax.set_xscale('log')
            # ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig('../results/history/' + graph_name + '_scores_' +
                str(seed_count) + '_seeds_' +
                str(budget) + 'iterations.png')
    # plt.show()
    # return auc_res


def plot_graph(graph_name, print_methods, budget_slices):
    crawler_avg, history, max_budget = load_results(DUMPS_DIR, graph_name)
    budget_slices.append(max_budget)

    big_graph = import_graph(graph_name)
    # берём топ 10 процентов вершин
    percentile, percentile_set = get_percentile(big_graph, graph_name, TOP_PERCENTILE)

    if graph_name == 'gnutella':  # большой костыль.Мы брали не тот эксцентриситет
        percentile_set['eccentricity'] = set(big_graph.nodes()). \
            difference(percentile_set['eccentricity'])

    # Draw node coverage
    for budget_slice in budget_slices:
        draw_nodes_history(history, crawler_avg, print_methods, graph_name, SEED_COUNT,
                                     budget_slice)

    # Draw Props
    for budget_slice in budget_slices:
        draw_properties_history(percentile_set, crawler_avg, print_methods,
                                graph_name, SEED_COUNT, budget_slice)

    # Draw AUC
    for budget_slice in budget_slices:
        auc_res = {}
        for method in print_methods:
            auc_res[method] = {}
            for prop in METRICS_LIST:
                data = np.array(crawler_avg[method][prop][:budget_slice]) / len(percentile_set[prop])
                auc_res[method][prop] = auc(x=np.arange(len(data)), y=data)

            data = np.array(crawler_avg[method]['nodes'][:budget_slice]) / big_graph.number_of_nodes()
            auc_res[method]['nodes'] = auc(x=np.arange(len(data)), y=data)

        plt.figure("AUC res")
        for prop in ['degrees', 'k_cores', 'eccentricity', 'betweenness_centrality','nodes']:
            plt.plot([auc_res[i][prop]/big_graph.number_of_nodes() for i in print_methods], PROPERTIES_COLOR[prop], label=prop
                     ,linewidth=3)
            plt.xticks(range(len(print_methods)), print_methods)

        plt.xlabel('method')
        plt.ylabel('AUC value in [0,1] (epigraph square)')
        plt.tight_layout()
        plt.grid()
        plt.legend()
        print('Properties AUC: ' + str(auc_res))


plot_graph(graph_names[1], ['RC', 'RW', 'BFS', 'DFS', 'MOD', 'DE'], [])
#plot_graph(graph_names[2], ['RC','RW','BFS','DFS','MOD','DE'], [])
# plot_graph(graph_names[5], ['RC','RW','BFS','DFS','MOD','DE'], [])

plt.show()
