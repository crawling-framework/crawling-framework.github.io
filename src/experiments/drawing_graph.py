import glob
import json
import os

import imageio
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


def snap_to_nx_graph(snap_graph):
    nx_graph = nx.Graph()
    for NI in snap_graph.Nodes():
        nx_graph.add_node(NI.GetId())
        for Id in NI.GetOutEdges():
            nx_graph.add_edge(NI.GetId(), Id)

    return nx_graph


def make_png_history(snap_graph, pngs_path, crawler_history_path, method_name='', labels=False, pos=None):
    # snap_graph = snap.LoadEdgeList(snap.PUNGraph, graph_path + 'orig_graph.txt', 0, 1, '\t')
    if os.path.exists(pngs_path):
        for file in glob.glob(pngs_path + method_name + "*.png"):
            os.remove(file)
    else:
        os.makedirs(pngs_path)

    with open(crawler_history_path + 'sequence.json', 'r') as f:
        sequence = json.load(f)

    nx_graph = snap_to_nx_graph(snap_graph)
    # node_list = list(nx_graph.nodes())
    for node in range(max(nx_graph.nodes())):  # filling empty indexes in graph
        if node not in nx_graph.nodes():
            nx_graph.add_node(1)
            print('i added node', node)

    if pos is None:
        pos = nx.spring_layout(nx_graph, iterations=100)

    iterator = 0
    for num in tqdm(sequence):
        iterator += 1
        gen_node_color = ['gray'] * nx_graph.number_of_nodes()

        # print("nodes in nx", nx_graph.nodes())

        with open(crawler_history_path + "observed{}{}.json".format(method_name, str(iterator).zfill(6)), 'r') as f:
            observed = json.load(f)
            for node in observed:
                gen_node_color[node] = 'y'

        with open(crawler_history_path + "crawled{}{}.json".format(method_name, str(iterator).zfill(6)), 'r') as f:
            crawled = json.load(f)
            for node in crawled:
                gen_node_color[node] = 'cyan'

        gen_node_color[num] = 'red'
        plt.title(str(iterator) + '/' + str(len(sequence)) + "cur:" + str(num) + " craw:" + str(crawled))
        nx.draw(nx_graph, pos=pos, with_labels=labels, node_size=80,
                node_color=[gen_node_color[node] for node in nx_graph.nodes])
        plt.savefig(pngs_path + '/gif{}{}.png'.format(method_name, str(iterator).zfill(3)))
        plt.clf()

    # def make_gif_from_png(gif_export_path, method_name, duration=5):
    images = []
    duration = 10
    filenames = glob.glob(pngs_path + "gif{}*.png".format(method_name))
    filenames.sort()
    print("adding")
    print(filenames)
    for filename in tqdm(filenames):
        for i in range(duration):
            images.append(imageio.imread(filename))
    print("compiling")
    imageio.mimsave(pngs_path + "result_{}.gif".format(method_name), images)
    print("done")

    return pos

# graph_path = './data/crawler_history/'
# crawler_history_path = "./data/crawler_history/sequence.json"
# pngs_path = "./data/gif_files/"
# gif_export_path = './data/graph_traversal.gif'


# def draw_crawling_curve(crawler_history_path, )
