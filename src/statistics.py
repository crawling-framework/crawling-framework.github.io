import logging
from enum import Enum
from operator import itemgetter

import snap
from tqdm import tqdm

from graph_io import MyGraph


class Stat(Enum):
    def __init__(self, short, description, computer):
        self.short = short
        self.description = description
        self.computer = computer

    NODES = 'n', "number of nodes", lambda graph: graph.snap.GetNodes()
    EDGES = 'e', "number of edges", lambda graph: graph.snap.GetEdges()
    AVG_DEGREE = 'avg-deg', "average (out)degree", lambda graph: (1 if graph.directed else 2) * graph.snap.GetEdges() / graph.snap.GetNodes()
    MAX_DEGREE = 'max-deg', "maximal degree", lambda graph: graph.snap.GetNI(snap.GetMxDegNId(graph.snap)).GetDeg()
    # RECIPROCITY = 'reciprocity', "edge reciprocity"
    ASSORTATIVITY = 'ass', "nodes degree assortativity", lambda graph: assortativity(graph=graph)
    # GINI = 'gini', "degree distribution Gini"
    # DRE = 'dre', "degree distr. rel. entropy"
    AVG_CC = 'avg-cc', "average local clustering coeff.", lambda graph: snap.GetClustCf(graph.snap, -1)
    # TRANSITIVITY = 'trans', "transitivity (global clustering coeff.)"  # FIXME how to implement
    # SPEC_NORM = 'spec-norm', r"spectral norm, $||A||_2$"
    # ALGEBRAIC_CONNECTIVITY = 'alg-conn', r"algebraic connectivity of largest WCC, $\lambda_2[\mathbf{L}]$"
    # ASSORTATIVITY_IN_IN = 'ass-in-in', "in-in degree assortativity"
    # ASSORTATIVITY_IN_OUT = 'ass-in-out', "in-out degree assortativity"
    # ASSORTATIVITY_OUT_IN = 'ass-out-in', "out-in degree assortativity"
    # ASSORTATIVITY_OUT_OUT = 'ass-out-out', "out-out degree assortativity"
    # DRE_IN = 'dre-in', "in degree distr. rel. entropy"
    # DRE_OUT = 'dre-out', "out degree distr. rel. entropy"
    # GINI_IN = 'gini-in', "in degree distr. Gini"
    # GINI_OUT = 'gini-out', "out degree distr. Gini"
    # WCC_COUNT = 'wcc', "number of WCCs"
    # SCC_COUNT = 'scc', "number of SCCs"
    MAX_WCC = 'wcc-max', "relative size of largest WCC", lambda graph: snap.GetMxWccSz(graph.snap)
    # MAX_SCC = 'scc-max', "relative size of largest SCC"
    # RADIUS = 'rad', "radius of largest WCC"
    # DIAMETER = 'diam', "diameter of largest WCC"
    DIAMETER_90 = 'diam90', "90%eff. diam. of largest WCC", lambda graph: snap.GetBfsEffDiam(snap.GetMxWcc(graph.snap), min(1000, graph.snap.GetNodes()), False)
    # RADIUS_DIR = 'rad-dir', "directed radius of largest SCC"
    # DIAMETER_DIR = 'diam-dir', "directed diameter of largest SCC"
    # DIAMETER_90_DIR = 'diam90-dir', "90%eff. dir. diam. of largest SCC"

    DEGREE_DISTR = 'DegDistr', 'degree centrality', lambda graph: compute_nodes_centrality(graph, 'degree')
    BETWEENNESS_DISTR = 'BtwDistr', 'betweenness centrality', lambda graph: compute_nodes_centrality(graph, 'betweenness')
    ECCENTRICITY_DISTR = 'EccDistr', 'eccentricity centrality', lambda graph: compute_nodes_centrality(graph, 'eccentricity')
    CLOSENESS_DISTR = 'ClsnDistr', 'closeness centrality', lambda graph: compute_nodes_centrality(graph, 'closeness')
    PAGERANK_DISTR = 'PgrDistr', 'pagerank centrality', lambda graph: compute_nodes_centrality(graph, 'pagerank')
    # CLUSTERING_DISTR = 'ClustDistr', 'clustering centrality', lambda graph: compute_nodes_centrality(graph, 'clustering')
    K_CORENESS_DISTR = 'KCorDistr', 'k-coreness centrality', lambda graph: compute_nodes_centrality(graph, 'k-coreness')


def assortativity(graph):
    """ Degree assortativity -1<=r<=1"""
    g = graph.snap
    mul, sum2, sum_sq2 = 0, 0, 0
    m = g.GetEdges()
    for e in g.Edges():
        j, k = e.GetSrcNId(), e.GetDstNId()
        dj, dk = g.GetNI(j).GetDeg(), g.GetNI(k).GetDeg()
        mul += dj * dk
        sum2 += dj + dk
        sum_sq2 += dj*dj + dk*dk

    # print(mul/m, sum2/2/m, sum_sq2/2/m)
    r = (mul/m - (sum2/2/m) ** 2) / (sum_sq2/2/m - (sum2/2/m) ** 2)
    return r


def compute_nodes_centrality(graph: MyGraph, centrality, nodes_fraction_approximate=10000, only_giant=False):
    """
    Compute centrality value for each node of the graph.
    :param graph: MyGraph
    :param centrality: centrality name, one of utils.CENTRALITIES
    :param nodes_fraction_approximate: parameter for betweenness
    :param only_giant: compute for giant component only, note size of returned list will be less
    than the number of nodes
    :return: dict (node id -> centrality)
    """
    # assert centrality in CENTRALITIES

    s = graph.snap
    logging.info("Computing '%s' for graph N=%d, E=%d. Can take a while..." %
                 (centrality, s.GetNodes(), s.GetEdges()))

    if only_giant:
        s = snap.GetMxWcc(s)

    if centrality == 'degree':
        node_cent = {n.GetId(): n.GetDeg() for n in tqdm(s.Nodes())}

    elif centrality == 'betweenness':
        Nodes = snap.TIntFltH()
        Edges = snap.TIntPrFltH()
        if nodes_fraction_approximate is None and s.GetNodes() > 10000:
            nodes_fraction_approximate = 10000 / s.GetNodes()
        snap.GetBetweennessCentr(s, Nodes, Edges, nodes_fraction_approximate, graph.directed)
        node_cent = {node: Nodes[node] for node in Nodes}

    elif centrality == 'pagerank':
        PRankH = snap.TIntFltH()
        snap.GetPageRank(s, PRankH)
        node_cent = {item: PRankH[item] for item in PRankH}

    elif centrality == 'closeness':
        # FIXME seems to not distinguish edge directions
        # node_cent = []  # TODO :made it to see progrees, mb need to be optimized
        node_cent = {n.GetId(): snap.GetClosenessCentr(s, n.GetId(), graph.directed) for n in tqdm(s.Nodes())}

    elif centrality == 'eccentricity':
        node_cent = {n.GetId(): snap.GetNodeEcc(s, n.GetId(), graph.directed) for n in tqdm(s.Nodes())}

    elif centrality == 'clustering':
        NIdCCfH = snap.TIntFltH()
        snap.GetNodeClustCf(s, NIdCCfH)
        node_cent = {item: NIdCCfH[item] for item in NIdCCfH}

    elif centrality == 'k-coreness':  # TODO could be computed in networkx
        node_cent_dict = {}
        k = 0
        while True:
            k += 1
            KCore = snap.GetKCore(s, k)
            if KCore.Empty():
                break
            for node in KCore.Nodes():
                node_cent_dict[node.GetId()] = k
        node_cent = {node: k for node, k in node_cent_dict.items()}

    else:
        raise NotImplementedError("")

    logging.info(" done.")
    return node_cent


def get_top_centrality_nodes(graph: MyGraph, centrality, count=None, threshold=False):
    """
    Get top-count node ids of the graph sorted by centrality.
    :param graph: MyGraph
    :param centrality: centrality name, one of utils.CENTRALITIES
    :param count: number of nodes with top centrality to return. If None, return all nodes
    :param threshold # TODO make threshold cut
    :return: sorted list with top centrality
    """
    node_cent = list(graph[centrality].items())
    if centrality in [Stat.ECCENTRICITY_DISTR]:  # , Stat.CLOSENESS_DISTR
        reverse = False
    else:
        reverse = True
    sorted_node_cent = sorted(node_cent, key=itemgetter(1), reverse=reverse)

    # TODO how to choose nodes at the border centrality value?
    if not count:
        count = graph.snap.GetNodes()
    return [n for (n, d) in sorted_node_cent[:count]]


def test():
    # 1.
    graph = MyGraph.new_snap(name='test', directed=True)
    g = graph.snap
    g.AddNode(1)
    g.AddNode(2)
    g.AddNode(3)
    g.AddNode(4)
    g.AddNode(5)
    g.AddEdge(1, 2)
    g.AddEdge(3, 2)
    g.AddEdge(4, 2)
    g.AddEdge(4, 3)
    g.AddEdge(5, 4)
    print("N=%s E=%s" % (g.GetNodes(), g.GetEdges()))

    # for cent in CENTRALITIES[:-1]:
    #     print(cent)
    #     print(compute_nodes_centrality(graph, cent))
    #     # print(graph.get_node_property_dict(cent))

    # 2.
    from graph_io import GraphCollections
    graph = GraphCollections.get('ego-gplus', giant_only=True)
    node_prop = graph[Stat.BETWEENNESS_DISTR]
    print(node_prop)


def test_stats():
    # # imports workaround https://stackoverflow.com/questions/26589805/python-enums-across-modules
    # sys.modules['statistics'] = sys.modules['__main__']

    from graph_io import GraphCollections
    graph = GraphCollections.get('github', giant_only=True)

    for stat in Stat:
        print("%s = %s" % (stat.short, graph[stat]))


def main():
    import argparse
    stats = [s.name for s in Stat]
    parser = argparse.ArgumentParser(description='Compute centralities for graph nodes.')
    parser.add_argument('-p', '--path', required=True, help='path to input graph as edgelist')
    parser.add_argument('-d', action='store_true', help='specify if graph is directed')
    parser.add_argument('-s', '--stats', required=True, nargs='+', choices=stats,
                        help='node statistics to compute')

    args = parser.parse_args()
    # print(args)
    graph = MyGraph(path=args.path, name='', directed=args.d)
    for s in args.stats:
        assert s in stats, "Unknown statistics %s, available are: %s" % (s, stats)
        # print("Computing %s centrality for %s..." % (c, args.path))
        v = graph[s]
        print("%s: %s" % (s, v))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    from graph_io import MyGraph, MyGraph

    # test_stats()
    # test()
    main()
