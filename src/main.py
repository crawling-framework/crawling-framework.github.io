from mpl_toolkits.axes_grid1 import ImageGrid
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

def draw_nodes_history(history, crawler_avg, print_methods, graph_name, seed_count, budget,
                       ax=None, draw_title=False, **plt_kw):
    """
    Drawing history for every method(average are bold) and every seed(thin)
    """
    # plt.figure(graph_name, figsize=(5.5, 5))
    if not ax:
        ax = plt.gca()

    if draw_title:
        ax.set_title(graph_name)

    normalize = True

    # for method, method_data in history.items():
    #     if method in print_methods:
    #         for seed_num, seed_data in list(method_data.items())[:seed_count]:
    #             data = np.array(seed_data['nodes'][:budget])
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
        if normalize:
            data /= budget
        xs = np.arange(0, 1, 1/len(data))
        ax.plot(xs, data, linewidth=2, color=METHOD_COLOR[method], label=method)

    # ax.xscale('log')
    # ax.yscale('log')
    # ax.set_xlim((0, 1 if normalize else budget))
    # ax.set_ylim((0, 1 if normalize else budget))
    # plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel('Fraction of nodes crawled')
    ax.set_ylabel(r"Fraction of nodes sampled, $|V'|/|V|$" if normalize else r"Nodes sampled, $|V'|$")
    ax.legend(loc=4)
    ax.grid(linestyle=':')
    # ax.savefig('../results/history/' + graph_name + '_history_' +
    #             str(seed_count) + '_seeds_' +
    #             str(budget) + 'iterations.png', dpi=300)


def draw_properties_history(percentile_set, crawler_avg, print_methods, graph_name, seed_count,
                            budget, **plt_kw):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # plt.title(graph_name)
    # plt.figure(figsize=(20, 20))

    # auc_res = {}

    for method in print_methods:
        # auc_res[method] = {}
        for prop in METRICS_LIST:
            # ремап для красивой отрисовки 2*2
            j = {'degrees': 0, 'k_cores': 1, 'eccentricity': 2, 'betweenness_centrality': 3}[prop]

            data = np.array(crawler_avg[method][prop][:budget]) / len(percentile_set[prop])

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
    plt.savefig('../results/history/' + graph_name + '_scores_' +
                str(seed_count) + '_seeds_' +
                str(budget) + 'iterations.png', dpi=300)


def draw_property_history(percentile_set, crawler_avg, print_methods, graph_name, seed_count,
                          budget, prop, hide_ylabels=False, ax=None, draw_title=False, **plt_kw):
    if not ax:
        ax = plt.gca()

    if draw_title:
        ax.set_title(graph_name)
    for method in print_methods:
        data = np.array(crawler_avg[method][prop][:budget]) / len(percentile_set[prop])

        xs = np.arange(0, 1, 1/len(data))
        ax.plot(xs, data, label=method, color=METHOD_COLOR[method])
    ax.legend()

    ax.grid(linestyle=':')
    if hide_ylabels:
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        # ax.set_ylabel(r"Target set coverage")
        ax.set_ylabel(property_name[prop])
    # ax.set_xlabel('iteration number')
    ax.set_xlabel('Fraction of nodes crawled')

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('../results/history/' + graph_name + '_scores_' +
                str(seed_count) + '_seeds_' +
                str(budget) + 'iterations.png')


def plot_graph(graph_name, print_methods, budget_slices, prop=None, **plt_kw):
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
                                     budget_slice, **plt_kw)

    # # Draw Props
    # for budget_slice in budget_slices:
    #     if prop is None:  # all 4 props
    #         draw_properties_history(percentile_set, crawler_avg, print_methods,
    #                                 graph_name, SEED_COUNT, budget_slice, **plt_kw)
    #     else:  # 1 specified prop
    #         draw_property_history(percentile_set, crawler_avg, print_methods,
    #                                 graph_name, SEED_COUNT, budget_slice, prop=prop, **plt_kw)

    # # Draw AUC
    # for budget_slice in budget_slices:
    #     auc_res = {}
    #     for method in print_methods:
    #         auc_res[method] = {}
    #         for prop in METRICS_LIST:
    #             data = np.array(crawler_avg[method][prop][:budget_slice]) / len(percentile_set[prop])
    #             auc_res[method][prop] = auc(x=np.arange(len(data)), y=data)
    #
    #     plt.figure("AUC res")
    #     for prop in METRICS_LIST:
    #         plt.plot([auc_res[i][prop] for i in print_methods], label=prop,
    #                  color=PROPERTIES_COLOR[prop])
    #         plt.xticks(range(len(print_methods)), print_methods)
    #
    #     plt.xlabel('method')
    #     plt.ylabel('AUC value')
    #     plt.tight_layout()
    #     plt.grid()
    #     print('Properties AUC: ' + str(auc_res))

        plt.figure("AUC res")
        for prop in METRICS_LIST:
            plt.plot([auc_res[i][prop] for i in print_methods], label=prop,
                     color=PROPERTIES_COLOR[prop])
            plt.xticks(range(len(print_methods)), print_methods)

        plt.xlabel('method')
        plt.ylabel('AUC value in [0,1] (epigraph square)')
        plt.tight_layout()
        plt.grid()
        plt.legend()
        print('Properties AUC: ' + str(auc_res))

### Several plots united in a common subplot

# grid = (1, 6)
# plt.subplots(*grid, figsize=(15, 5))
#
# for index, i in enumerate([0, 1, 2, 3, 4, 5]):
#     ax = plt.subplot(*grid, index+1)
#     plt.title(graph_names[i])
#     plot_graph(graph_names[i], ['RC','RW','BFS','DFS','MOD','DE'], [], prop='betweenness_centrality',
#                hide_ylabels=index > 0)
#
# plt.savefig('../results/history/all_props.png', dpi=300)
# plt.savefig('../results/history/all_props.pdf')


### Grid 4x6 of property X graph

fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 6),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 share_all=False,
                 aspect=False)
index = 0
for i in [0, 1, 2, 3, 4, 5]:
    ax = grid[index]
    plot_graph(graph_names[i], ['RC', 'RW', 'BFS', 'DFS', 'MOD', 'DE'], [],
               ax=ax, draw_title=True)
    index += 1

# for prop in METRICS_LIST:
#     for i in [0, 1, 2, 3, 4, 5]:
#         ax = grid[index]
#         # ax = plt.subplot(*grid, index+1)
#         # plt.suptitle(graph_names[i])
#         plot_graph(graph_names[i], ['RC','RW','BFS','DFS','MOD','DE'], [], prop=prop,
#                    ax=ax, draw_title=index<grid._ncols)
#         index += 1

plt.xlim((0, 1))
plt.ylim((0, 1))

# plt.layout(top=0.902, bottom=0.173, left=0.034, right=0.995, hspace=0.2, wspace=0.0)
plt.show()
