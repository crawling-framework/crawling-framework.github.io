import glob
import json

import imageio
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from crawlers import test_graph

snap_graph = test_graph()


# sequence = [5, 4, 6, 7, 6, 7, 6, 7, 6, 7, 6, 1, 6, 1, 6, 7, 6, 1, 2, 3, 2, 1, 2, 1, 6, 1, 2, 3, 4, 5, 4, 5, 4, 3, 2, 3, 4, 5, 4, 2, 3, 4, 5, 4, 3, 2, 1, 2, 4, 5, 4, 3, 2, 4, 2, 4, 2, 1, 2, 4, 3, 2, 4, 3, 2, 4, 2, 1, 2, 3, 2, 1, 2, 1, 6, 1, 2, 1, 6, 7, 6, 1, 2, 1, 6, 1, 6, 1, 2, 4, 3, 2, 1, 6, 7, 6, 1, 6, 1, 6, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 1, 6, 1, 2, 4, 3, 4, 2, 4, 3, 2, 4, 2, 1, 6, 1, 2, 3, 4, 5, 4, 5, 4, 2, 4, 2, 1, 6, 1, 2, 3, 4, 5, 4, 2, 3, 2, 1, 6, 1, 6, 7, 6, 1, 6, 1, 6, 7, 6, 7, 6, 1, 2, 3, 4, 3, 4, 3, 2, 3, 4, 3, 2, 4, 5, 15, 5, 4, 5, 15, 13, 14, 12, 13, 14, 12, 13, 14, 13, 15, 5, 15, 5, 15, 5, 15, 13, 12, 13, 14, 12, 14, 13, 12, 13, 12, 14, 12, 14, 12, 13, 14, 13, 12, 11, 12, 11, 12, 14, 12, 14, 13, 12, 13, 15, 5, 15, 13, 15, 5, 15, 13, 12, 14, 12, 13, 12, 14, 12, 11, 16]
# sequence = [3, 16, 12, 13, 12, 11, 12, 13, 12, 11, 12, 14, 12, 13, 15, 5, 15, 5, 4, 3, 2, 4, 2, 3, 4, 5, 15, 13, 12, 11, 12, 13, 14, 12, 13, 12, 13, 12, 13, 14, 12, 11, 12, 14, 12, 14, 13, 12, 11, 16, 11, 12, 13, 15, 5, 4, 3, 2, 1, 6, 1, 2, 1, 2, 4, 3, 4, 2, 1, 2, 1, 6, 1, 6, 1, 2, 1, 6, 7]
# 79

def snap_to_nx_graph(snap_graph):
    nx_graph = nx.Graph()
    for NI in snap_graph.Nodes():
        nx_graph.add_node(NI.GetId())
        for Id in NI.GetOutEdges():
            nx_graph.add_edge(NI.GetId(), Id)

    return nx_graph


def make_png_history(snap_graph, sequence):
    nx_graph = snap_to_nx_graph(snap_graph)
    node_list = list(nx_graph.nodes())
    pos = nx.spring_layout(nx_graph, iterations=100)
    iterator = 0
    for num in tqdm(sequence):
        iterator += 1
        gen_node_color = ['gray' for node in nx_graph.nodes()]
        with open("../data/crawler_history/crawled{}.json".format(str(iterator).zfill(3)), 'r') as f:
            crawled = json.load(f)
        with open("../data/crawler_history/observed{}.json".format(str(iterator).zfill(3)), 'r') as f:
            observed = json.load(f)

        for node in observed:
            gen_node_color[node_list.index(node)] = 'y'
        for node in crawled:
            gen_node_color[node_list.index(node)] = 'purple'
        gen_node_color[node_list.index(num)] = 'r'
        plt.title(str(iterator) + '/' + str(len(sequence)) + " c:" + str(crawled))
        nx.draw(nx_graph, pos=pos, with_labels=True,
                node_color=gen_node_color)
        plt.savefig('../data/gif_files/gif{}.png'.format(str(iterator).zfill(3)))


def make_gif_from_png(duration=5):
    images = []
    filenames = glob.glob("../data/gif_files/gif*.png")
    filenames.sort()
    print("adding")
    for filename in tqdm(filenames):
        for i in range(duration):
            images.append(imageio.imread(filename))
    print("compiling")
    imageio.mimsave('../data/graph_traversal.gif', images)
    print("done")


import os  #

file_path = "../data/gif_files/"
if os.path.exists(file_path):
    for file in glob.glob("../data/gif_files/*.png"):
        os.remove(file)
else:
    os.makedirs(file_path)

with open("../data/crawler_history/sequence.json", 'r') as f:
    sequence = json.load(f)
make_png_history(snap_graph, sequence)
make_gif_from_png()
