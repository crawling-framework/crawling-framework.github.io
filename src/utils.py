#!/usr/bin/env python
# coding: utf-8

# Helloo!!!!!!!!!!! I have been commited

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json

DEFAULT_EDGE_LIST_FORMAT = 'ij'
WEIGHT_LABEL = 'w'  # weight attribute name in networkx graph
TMSTMP_LABEL = 't'  # timestamp attribute name in networkx graph
metrics_list = ['degrees', 'k_cores','eccentricity','betweenness_centrality']


def read_networkx_graph(path, directed=False, format=DEFAULT_EDGE_LIST_FORMAT):
    """
    Read graph from a specified file according to specified format. Ignores strings starting with '#'.
    Default format is "ij", (unweighted edges), or can be file extension.

    :param path: full path to edge list file
    :param directed: whether to create nx.DiGraph or nx.Graph
    :param format: any combination of `i j w t` where i-source_id, j-target_id, w-weight, t-timestamp, 'GML', 'paj'
    :return: networkx graph
    """
    g = nx.Graph()
    if directed:
        g = nx.DiGraph()

    # if format is None:
    #     format = path.split('.')[-1]
    #     logging.warning("Graph format extracted from file extension: '%s'" % format)

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
            g.add_edge(int(elems[source_index]), int(elems[target_index]), attr_dict)

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



def import_graph(graph_name):
    # Работа с графами. Выбирается один из списка (всё определяет graph_name и *_graph) 
    with  open('../data/Graphs/'+graph_name+ ".edges", 'rb') as fh:
        Graph=nx.read_edgelist(fh, delimiter = ' ')#,create_using=nx.Graph())
    Graph.remove_edges_from(Graph.selfloop_edges())  # удаляем петли
    return max(nx.connected_component_subgraphs(Graph), key=len) # берём за граф только самую большую его компоненту 

def draw_graph(Graph, title = 'graph'): 
    if Graph.number_of_nodes()<5000:
        t0 = time.time()
        plt.figure(figsize=(20,20))
        nx.draw(Graph, with_labels=False ,edge_cmap=plt.cm.Blues , node_size = 10)
        plt.title(graph_name)
        plt.savefig('../results/'+graph_name+'.png')
        plt.show()
        print(int(time.time() -t0) , 'sec')
    else:
        print("graph is too big")
        
# берём квантиль примерно на top_percent лучших вершин. 
# Квантиль это словарь percentile['degrees']=24 значит что топ вершин с макс степенью имеют степени 24 и больше
def get_percentile(Graph,graph_name,top_percent):
    properties = json.load(open('../data/Graphs/'+graph_name+'_properties.txt','r'))

    total = Graph.number_of_nodes()
    percentile = {'degrees':[],'k_cores':[],'eccentricity':[],'betweenness_centrality':[]}
    percentile_history = {'degrees':[],'k_cores':[],'eccentricity':[],'betweenness_centrality':[]}
    percentile_set = {}
    prop_color = {'degrees':'green','k_cores':'black','eccentricity':'red','betweenness_centrality':'blue'}
    for prop in metrics_list:
        percentile[prop] = properties[prop][min(int(total*top_percent//100),len(properties[prop]))-1]
        percentile_set[prop] = set([node for node in properties[prop+'_dict'] if properties[prop+'_dict'][node]>=percentile[prop]])
        percentile_history[prop].append(len(percentile_set[prop]))
    #print('percent:',top_percent,' percentile',percentile)
    return (percentile, percentile_set)





