from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import os
from sklearn.metrics import auc
from old.utils import get_percentile, import_graph

METRICS_LIST = ['degrees', 'k_cores', 'eccentricity', 'betweenness_centrality']
graph_names = ['hamsterster', 'DCAM', 'facebook', 'slashdot', 'github', 'dblp2010']
method_names = ['RC', 'RW', 'BFS', 'DFS', 'MOD', 'DE']
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
    'nodes': 'nodes',
    'degrees': 'degrees',
    'k_cores': 'k-cores',
    'eccentricity': 'eccentricity',
    'betweenness_centrality': 'betweenness centrality'
}

# DUMPS_DIR = '../results/dumps_shared'
DUMPS_DIR = '../results/dumps_closed'
# DUMPS_DIR = '../results/dumps'
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
                       ax=None, draw_title=False, legend=False, **plt_kw):
    """
    Drawing history for every method(average are bold) and every seed(thin)
    """
    # plt.figure(graph_name, figsize=(5.5, 5))
    if not ax:
        ax = plt.gca()

    if draw_title:
        ax.set_title(graph_name)

    normalize = True
    show_gap = False
    show_seeds = True

    if show_seeds:
        for method, method_data in history.items():
            if method in print_methods:
                for seed_num, seed_data in list(method_data.items())[:seed_count]:
                    data = np.array(seed_data['nodes'][:budget])
                    if normalize:
                        data = data / budget
                    # xs = np.arange(0, 1, 1 / len(data))
                    xs = np.array([i / len(data) for i in range(len(data))])
                    ax.plot(xs, data, linewidth=0.5, linestyle=':', color=METHOD_COLOR[method])
    if show_gap:
        maxmethod = np.zeros(budget)
        for i in range(budget):
            for method in print_methods:
                val = crawler_avg[method]['nodes'][i]
                if val > maxmethod[i]:
                    maxmethod[i] = val

    for method, avg_data in crawler_avg.items():
        if method not in print_methods:
            continue
        data = np.array(avg_data['nodes'][:budget])

        if show_gap:
            data = data - maxmethod
            linewidth = 3
        else:
            linewidth = 3
        if normalize:
            data /= budget
        # xs = np.arange(0, 1, 1/len(data))
        xs = np.array([i/len(data) for i in range(len(data))])
        ax.plot(xs, data, linewidth=linewidth, color=METHOD_COLOR[method], label=method)


    # ax.xscale('log')
    # ax.yscale('log')
    ax.set_xlim((-0.01, 1.01))
    ax.set_xlabel('Fraction of nodes crawled')

    if show_gap:
        ax.set_ylim((-0.1, 0.003))
        ax.set_ylabel("Gap between the leader")
    else:
        ax.set_ylim((0.2, 1.01))
    ax.set_ylabel(r"Fraction of nodes sampled, $|V'|/|V|$" if normalize else r"Nodes sampled, $|V'|$")

    ax.set_xlim((0.17, 1.01))
    ax.set_ylim((0.87, 1.01))

    # plt.setp(ax.get_xticklabels(), visible=False)
    if legend:
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
                          budget, prop, hide_ylabels=False, ax=None, draw_title=False,
                          legend=False, **plt_kw):
    if not ax:
        ax = plt.gca()

    if draw_title:
        ax.set_title(graph_name)

    # maxmethod_index = np.zeros(budget, dtype=np.int)
    # for i in range(budget):
    #     max = 0
    #     for index, method in enumerate(print_methods):
    #         val = crawler_avg[method][prop][i] / len(percentile_set[prop])
    #         if val > max:
    #             maxmethod_index[i] = index
    #             max = val
    # xs = np.arange(0, 1, 1 / len(maxmethod_index))
    # segs = [[(x, 0), (x, 1)] for x in xs]
    # colors = [METHOD_COLOR[print_methods[i]] for i in maxmethod_index]
    #
    # line_segments = LineCollection(segs, colors=colors, linestyle='solid')
    # ax.add_collection(line_segments)

    maxmethod = np.zeros(budget)
    for i in range(budget):
        for method in print_methods:
            val = crawler_avg[method][prop][i] / len(percentile_set[prop])
            if val > maxmethod[i]:
                maxmethod[i] = val

    for method in print_methods:
        data = np.array(crawler_avg[method][prop][:budget]) / len(percentile_set[prop])

        data = data - maxmethod

        xs = np.arange(0, 1, 1/len(data))
        ax.plot(xs, data, linewidth=3, label=method, color=METHOD_COLOR[method])
    if legend:
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
    ax.set_ylim((-0.1, 0))

    plt.tight_layout()
    plt.savefig('../results/history/' + graph_name + '_scores_' +
                str(seed_count) + '_seeds_' +
                str(budget) + 'iterations.png')


def draw_auc(percentile_set, crawler_avg, print_methods, graph_name, SEED_COUNT, budget_slice,
             big_graph_nodes_count, ax=None, draw_title=False, legend=False, **plt_kw):
    if not ax:
        ax = plt.figure("AUC res").gca()
    if draw_title:
        ax.set_title(graph_name)

    auc_res = {}
    for method in print_methods:
        auc_res[method] = {}
        for prop in METRICS_LIST:
            data = np.array(crawler_avg[method][prop][:budget_slice]) / len(percentile_set[prop])
            auc_res[method][prop] = auc(x=np.arange(len(data)), y=data)

        data = np.array(crawler_avg[method]['nodes'][:budget_slice]) / big_graph_nodes_count
        auc_res[method]['nodes'] = auc(x=np.arange(len(data)), y=data)

    # plt.figure("AUC res")
    for prop in ['degrees', 'k_cores', 'eccentricity', 'betweenness_centrality','nodes']:
        ax.plot([auc_res[i][prop]/big_graph_nodes_count for i in print_methods],
                 PROPERTIES_COLOR[prop], label=property_name[prop], linewidth=1, marker='.',
                 linestyle='-')
    # ax.set_xticklabels(range(len(print_methods)), print_methods)
    # plt.xticks(range(len(print_methods)), print_methods)
    locs = ax.set_xticks(range(len(print_methods)))
    labels = ax.set_xticklabels(print_methods)

    # ax.set_xlabel('method')
    ax.set_ylabel('AUC value')
    # ax.tight_layout()
    ax.grid(linestyle=':')
    if legend:
        ax.legend()
    print('Properties AUC: ' + str(auc_res))


def plot_graph(graph_name, print_methods, budget_slices, prop=None, **plt_kw):
    crawler_avg, history, max_budget = load_results(DUMPS_DIR, graph_name)
    budget_slices.append(max_budget)

    big_graph = import_graph(graph_name)
    # берём топ 10 процентов вершин
    percentile, percentile_set = get_percentile(big_graph, graph_name, TOP_PERCENTILE)

    if graph_name == 'gnutella':  # большой костыль.Мы брали не тот эксцентриситет
        percentile_set['eccentricity'] = set(big_graph.nodes()). \
            difference(percentile_set['eccentricity'])

    # # Draw node coverage
    # for budget_slice in budget_slices:
    #     draw_nodes_history(history, crawler_avg, print_methods, graph_name, SEED_COUNT,
    #                                  budget_slice, **plt_kw)

    # # Draw Props
    # for budget_slice in budget_slices:
    #     if prop is None:  # all 4 props
    #         draw_properties_history(percentile_set, crawler_avg, print_methods,
    #                                 graph_name, SEED_COUNT, budget_slice, **plt_kw)
    #     else:  # 1 specified prop
    #         draw_property_history(percentile_set, crawler_avg, print_methods,
    #                                 graph_name, SEED_COUNT, budget_slice, prop=prop, **plt_kw)

    # Draw AUC
    for budget_slice in budget_slices:
        draw_auc(percentile_set, crawler_avg, print_methods,
                 graph_name, SEED_COUNT, budget_slice, big_graph.number_of_nodes(), **plt_kw)


# plt.figure(1, (4, 4))
SEED_COUNT = 56
plt.title(graph_names[5])
plot_graph(graph_names[5], ['RC','RW','BFS','DFS','MOD', 'DE'], [], legend=True, prop='nodes')
# plt.subplots_adjust(top=0.942, bottom=0.106, left=0.161, right=0.979, hspace=0.2, wspace=0.2)

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

# fig = plt.figure(1, (16., 4.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(1, 6),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  share_all=False,
#                  aspect=False)
# index = 0
# for i in [0, 1, 2, 3, 4, 5]:
#     ax = grid[index]
#     plot_graph(graph_names[i], ['RC', 'RW', 'BFS', 'DFS', 'MOD', 'DE'], [],
#                ax=ax, draw_title=True, prop='degrees', legend=index==0)
#     index += 1

# plt.subplots_adjust(top=0.899, bottom=0.195, left=0.047, right=0.996, hspace=0.2, wspace=0.2)
# for prop in METRICS_LIST:
#     for i in [0, 1, 2, 3, 4, 5]:
#         ax = grid[index]
#         # ax = plt.subplot(*grid, index+1)
#         # plt.suptitle(graph_names[i])
#         plot_graph(graph_names[i], ['RC','RW','BFS','DFS','MOD','DE'], [], prop=prop,
#                    ax=ax, draw_title=index<grid._ncols)
#         index += 1


plt.show()

