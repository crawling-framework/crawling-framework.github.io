import numpy as np
import matplotlib.pyplot as plt

from graph_io import GraphCollections, MyGraph
from statistics import Stat, get_top_centrality_nodes


def intersection(graph: MyGraph):
    import snap
    g = graph.snap
    n = g.GetNodes()
    deg = graph[Stat.DEGREE_DISTR]
    btw = graph[Stat.BETWEENNESS_DISTR]
    pgr = graph[Stat.PAGERANK_DISTR]
    ecc = graph[Stat.ECCENTRICITY_DISTR]

    deg = sorted(deg, key=lambda x: deg[x], reverse=True)
    btw = sorted(btw, key=lambda x: btw[x], reverse=True)
    pgr = sorted(pgr, key=lambda x: pgr[x], reverse=True)
    ecc = sorted(ecc, key=lambda x: ecc[x], reverse=True)

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
    # name = 'digg-friends'
    name = 'ego-gplus'
    # name = 'petster-hamster'
    graph = GraphCollections.get(name, giant_only=True)
    # degs = get_top_centrality_nodes(graph, 'degree', 10)
    # print(degs)
    # ecc = get_top_centrality_nodes(graph, Stat.ECCENTRICITY_DISTR, 10)
    # print(btws)

    intersection(graph)

    plt.grid()
    plt.show()
