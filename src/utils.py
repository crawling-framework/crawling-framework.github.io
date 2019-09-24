#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

import matplotlib_venn # mb needed to be installed

import pip
from matplotlib.pyplot import figure
import json

DEFAULT_EDGE_LIST_FORMAT = 'ij'
WEIGHT_LABEL = 'w'  # weight attribute name in networkx graph
TMSTMP_LABEL = 't'  # timestamp attribute name in networkx graph
METRICS_LIST = ['degrees', 'k_cores', 'eccentricity', 'betweenness_centrality']
METHOD_COLOR = {'AFD':'pink','RC':'grey','RW':'green','DFS':'black', 'BFS':'blue', 'MED':'cyan','MOD':'red','DE':'magenta', 'MEUD':'indigo'}

def read_networkx_graph(path, directed=False, format=DEFAULT_EDGE_LIST_FORMAT):
    """
    Read graph from a specified file according to specified format. Ignores strings starting with '#'.
    Default format is "ij", (unweighted edges), or can be file extension.

    :param path: full path to edge list file
    :param directed: whether to create nx.DiGraph or nx.big_graph
    :param format: any combination of `i j w t` where i-source_id, j-target_id, w-weight, t-timestamp, 'GML', 'paj'
    :return: networkx graph
    """
    g = nx.Graph()
    if directed:
        g = nx.DiGraph()

    # if format is None:
    #     format = path.split('.')[-1]
    #     logging.warning("big_graph format extracted from file extension: '%s'" % format)

    # TODO make file extension correspond to format e.g. '.ijw'
    if set(format).issubset(set("ijwt")):
        # i-source_id, j-target_id, w-weight, t-timestamp
        source_index = format.index('i')
        target_index = format.index('j')
        weight_index = format.index('w') if 'w' in format else None
        tmstmp_index = format.index('t') if 't' in format else None

        def decode_line(line):
            elems = [x for x in line.split()]
            attr_dict = {}
            if weight_index is not None:
                attr_dict[WEIGHT_LABEL] = float(elems[weight_index])
            if tmstmp_index is not None:
                attr_dict[TMSTMP_LABEL] = float(elems[tmstmp_index])
            g.add_edge(int(elems[source_index]), int(elems[target_index]))#, attr_dict)

        with open(path) as graph_file:
            line = ''
            try:
                for line in graph_file:
                    # Filter comments.
                    # if not line.startswith("#"):
                    # if line.isalnum():
                    if line[0].isdigit():
                        decode_line(line)
            except Exception as e:
                raise IOError("Couldn't parse edge list file at line '%s' using format '%s': %s" % (line, format, e))
    # TODO we read just edges and weights here, no labels!
    # GML format
    elif format == "GML":
        g = nx.read_gml(path)
        return g
    # paj format
    elif format == "paj":
        with open(path) as graph_file:
            weight = 1
            pos = 0
            lines = graph_file.readlines()
            try:
                pos = lines.index("*arcs\r\n")  # FIXME \r\n is very bad
                pos += 1
                # Read edges.
                while pos < len(lines):
                    abw = [x for x in lines[pos].split()]
                    if len(abw) < 2:  # no more edges
                        break
                    # if weighted:
                    weight = float(abw[2])
                    g.add_edge(int(abw[0]), int(abw[1]), w=weight)
                    pos += 1
            except:
                raise IOError("Couldn't read data from file '%s', namely at string %d." % (path, pos))
    else:
        raise Exception("Unknown graph format '%s'" % format)
    return g

#def dump_history(): TBD - история экспорта


def import_graph(graph_name):
    # Работа с графами. Выбирается один из списка (всё определяет graph_name и *_graph) 
    with  open('../data/Graphs/' + graph_name + ".edges", 'rb') as fh:
        Graph = nx.read_edgelist(fh, delimiter=' ')  # ,create_using=nx.big_graph())
    Graph.remove_edges_from(Graph.selfloop_edges())  # удаляем петли
    return max(nx.connected_component_subgraphs(Graph), key=len)  # берём за граф только самую большую его компоненту


def draw_graph(Graph, title='graph'):
    if Graph.number_of_nodes() < 5000:
        t0 = time.time()
        plt.figure(figsize=(20, 20))
        nx.draw(Graph, with_labels=False, edge_cmap=plt.cm.Blues, node_size=10)
        plt.title(title)
        plt.savefig('../results/' + title + '.png')
        plt.show()
        print(int(time.time() - t0), 'sec')
    else:
        print("graph is too big")


# берём квантиль примерно на top_percent лучших вершин. 
# Квантиль это словарь percentile['degrees']=24 значит что топ вершин с макс степенью имеют степени 24 и больше
def get_percentile(Graph, graph_name, top_percent):
    properties = json.load(open('../data/Graphs/' + graph_name + '_properties.txt', 'r'))

    total = Graph.number_of_nodes()
    percentile = {'degrees': [], 'k_cores': [], 'eccentricity': [], 'betweenness_centrality': []}
    percentile_history = {'degrees': [], 'k_cores': [], 'eccentricity': [], 'betweenness_centrality': []}
    percentile_set = {}
    prop_color = {'degrees': 'green', 'k_cores': 'black', 'eccentricity': 'red', 'betweenness_centrality': 'blue'}
    for prop in METRICS_LIST:
        percentile[prop] = properties[prop][min(int(total * top_percent // 100), len(properties[prop])) - 1]
        percentile_set[prop] = set(
            [node for node in properties[prop + '_dict'] if properties[prop + '_dict'][node] >= percentile[prop]])
        percentile_history[prop].append(len(percentile_set[prop]))
    # print('percent:',top_percent,' percentile',percentile)
    return (percentile, percentile_set)


def draw_percentile_heatmap(percentile_set,graph_name,seed_count,b, normalized=True, venn_on=True):
    """
    Drawing and returning table (heatmap) based on intersection of metrics percentile_set (Jaccard coefficient).
    percentile_set is a dictionary (for every metric in METRICS_LIST) of sets (nodes, that are in percentile)
    normalized - if True, writes fraction of nodes, if False - total number of nodes in intersection
    ALSO: it draws Venn's 3-diagram from 1-3 METRICS.
    """
# tbd - последние три нужны лишь для названия
    table = np.zeros((len(METRICS_LIST), len(METRICS_LIST)))  # dict((i,dict((j,[]) for j in METRICS_LIST)) for i in METRICS_LIST)

    for prop1 in METRICS_LIST:
        for prop2 in METRICS_LIST:
            i = METRICS_LIST.index(prop1)
            j = METRICS_LIST.index(prop2)
            table[i][j] = len(percentile_set[prop1].intersection(percentile_set[prop2]))
            if normalized:  # если нормализуем
                table[i][j] /= len(percentile_set[prop1].union(percentile_set[prop2])) #Jaccard coefficient
                #table[i][j] /= max(len(percentile_set[prop1]), len(percentile_set[prop2]))


    fig, ax = plt.subplots()
    im = ax.imshow(np.array(table))
    ax.set_xticks(np.arange(len(METRICS_LIST)))
    ax.set_yticks(np.arange(len(METRICS_LIST)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(METRICS_LIST)
    ax.set_yticklabels(METRICS_LIST)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(METRICS_LIST)):
        for j in range(len(METRICS_LIST)):
            if normalized:
                text = ax.text(j, i, str(round(table[i, j] * 100, 1)) + '%', ha="center", va="center", color="black")
            else:
                text = ax.text(j, i, int(table[i, j]), ha="center", va="center", color="black")
    plt.show()
    fig.savefig('../results/' + graph_name + '_percentile.png')
    if venn_on:
        try:
            plt.figure(figsize=(10, 10))
            from matplotlib_venn import venn3
            venn3([percentile_set[metric] for metric in METRICS_LIST if metric != 'betweenness_centrality'],
                  set_labels=(METRICS_LIST))

            plt.show()
            fig.savefig('../results/' + graph_name + '_venn.png')
        except BaseException:
            print("need to install matplotlib_venn with command: !pip install  matplotlib_venn")
            #!pip install  matplotlib_venn

    return table


def draw_nodes_history(history,crawler_avg, methods,graph_name,seed_count,b):
    """
    Drawing history for every method(average are bold) and every seed(thin)
    """
    # TBD - ,graph_name,seed_count are only for filenames. need to clean them
    plt.figure(figsize=(16, 16))
    plt.grid()
    for method in methods:
        for seed_num in range(seed_count):
            plt.plot(history[method][seed_num]['nodes'], linewidth=0.5, color=METHOD_COLOR[method])
        plt.plot(crawler_avg[method]['nodes'] / seed_count, linewidth=5, color=METHOD_COLOR[method], label=method)

    plt.legend()
    plt.savefig('../results/' + graph_name + '_history_' + str(seed_count) + '_seeds_' + str(b) + 'iterations.png')
    plt.show()

def draw_scores_history(percentile_set,crawler_avg,methods,graph_name,seed_count,b):
# tbd - последние три нужны лишь для названия
# tbd - от percentile_set нужен только размер, это надо поменять
# tbd - method color это ремап цветов по названиям методов. надо в глобальные
    fig,axs = plt.subplots(2,2,figsize=(15,15))
    plt.figure(figsize=(30,30))

    for prop in METRICS_LIST:
        j = {'degrees':0,'k_cores':1,'eccentricity':2,'betweenness_centrality':3}[prop]  # ремап для красивой отрисовки 2*2
        for method in methods:
            axs[(j)//2][j%2].plot(crawler_avg[method][prop]/seed_count/len(percentile_set[prop]), label =method+' '+prop, color = METHOD_COLOR[method] )
            axs[(j)//2][j%2].set_title('fraction of found nodes with '+prop+ ' from '+str(len(percentile_set[prop])))
            axs[(j)//2][j%2].legend()

        axs[(j)//2][j%2].grid(True)
    fig.savefig('../results/'+graph_name+'_scores_'+str(seed_count) +'_seeds_'+str(b)+'iterations.png')
    plt.show()


def dump_results(graph_name,crawler_avg,history,b):
    """
    Dumping history of crawling results (graphics) into the ./results/dumps/' + graph_name + '_results_budget'+str(b)+'.json
    :param graph_name:
    :param crawler_avg:
    :param history:
    :param b:
    :return:
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    results = dict({'graph_name':graph_name, 'crawler_avg': crawler_avg,'history': history})
    json_file = open('../results/dumps/' + graph_name + '_results_budget'+str(b)+'.json', 'w+')
    json.dump(results, json_file, cls=NumpyEncoder)
    json_file.close()
    print('dumped '+str(b))
