import numpy as np
import matplotlib.pyplot as plt

from centralities import get_top_centrality_nodes
from graph_io import GraphCollections, MyGraph


def intersection(graph: MyGraph):
    import snap
    g = graph.snap
    n = g.GetNodes()
    deg = get_top_centrality_nodes(graph, 'degree')
    btw = get_top_centrality_nodes(graph, 'betweenness')
    pgr = get_top_centrality_nodes(graph, 'pagerank')
    ecc = get_top_centrality_nodes(graph, 'eccentricity')

    x = []
    b = []
    p = []
    e = []
    for t in range(1, int(n/10), 1):
        print(t)
        d = set(deg[:t])
        b.append(len(d.intersection(set(btw[:t])))/t)
        p.append(len(d.intersection(set(pgr[:t])))/t)
        e.append(len(d.intersection(set(ecc[-t:])))/t)
        x.append(t)

    plt.figure(figsize=(13, 9))
    plt.title("Graph N=%d E=%d max_deg=%d" % (g.GetNodes(), g.GetEdges(),
                                              g.GetNI(snap.GetMxDegNId(g)).GetDeg()))
    plt.xlabel("size")
    plt.ylabel("common fraction")
    plt.plot(x, b, '-', color='b', label='degree and betw')
    plt.plot(x, p, '-', color='r', label='degree and pagerank')
    plt.plot(x, e, '-', color='g', label='degree and ecc')
    plt.legend(loc=0)
    plt.tight_layout()


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    # name = 'libimseti'
    # name = 'loc-brightkite_edges'
    # name = 'facebook-wosn-links'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    name = 'digg-friends'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    graph = GraphCollections.get(name)
    # degs = get_top_centrality_nodes(graph, 'degree', 10)
    # print(degs)
    ecc = get_top_centrality_nodes(graph, 'eccentricity', 10)
    # print(btws)

    # intersection(graph)

    plt.grid()
    plt.show()
