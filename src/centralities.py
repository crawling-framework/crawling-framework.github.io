import logging
from operator import itemgetter

import snap
from tqdm import tqdm

from graph_io import MyGraph
from utils import CENTRALITIES


def get_top_centrality_nodes(graph: MyGraph, centrality, count=None, threshold=False):
    """
    Get top-count node ids of the graph sorted by centrality.
    :param graph: graph
    :param centrality: centrality name, one of utils.CENTRALITIES
    :param count: number of nodes with top centrality to return. If None, return all nodes
    :param threshold # TODO make threshold cut
    :return: sorted list with top centrality
    """
    node_cent = list(graph.get_node_property_dict(property=centrality).items())
    # node_cent = compute_nodes_centrality(graph, centrality)
    if centrality in ['eccentricity', 'closeness']:
        reverse = False
    else:
        reverse = True
    sorted_node_cent = sorted(node_cent, key=itemgetter(1), reverse=reverse)

    # TODO how to choose nodes at the border centrality value?
    if not count:
        count = graph.snap.GetNodes()
    return [n for (n, d) in sorted_node_cent[:count]]


def compute_nodes_centrality(graph: MyGraph, centrality, nodes_fraction_approximate=10000, only_giant=True):
    """
    Compute centrality value for each node of the graph.
    :param graph: snap graph
    :param centrality: centrality name, one of utils.CENTRALITIES
    :param nodes_fraction_approximate: parameter for betweenness
    :param only_giant: compute for giant component only, note size of returned list will be less
    than the number of nodes
    :return: list of pairs (node id, centrality)
    """
    assert centrality in CENTRALITIES

    s = graph.snap
    logging.info("Computing '%s' for graph N=%d, E=%d. Can take a while..." %
                 (centrality, s.GetNodes(), s.GetEdges()))

    if only_giant:
        s = snap.GetMxWcc(s)

    if centrality == 'degree':
        node_cent = [(n.GetId(), n.GetDeg()) for n in s.Nodes()]

    elif centrality == 'betweenness':
        Nodes = snap.TIntFltH()
        Edges = snap.TIntPrFltH()
        if nodes_fraction_approximate is None and s.GetNodes() > 10000:
            nodes_fraction_approximate = 10000 / s.GetNodes()
        snap.GetBetweennessCentr(s, Nodes, Edges, nodes_fraction_approximate, graph.directed)
        node_cent = [(node, Nodes[node]) for node in Nodes]

    elif centrality == 'pagerank':
        PRankH = snap.TIntFltH()
        snap.GetPageRank(s, PRankH)
        node_cent = [(item, PRankH[item]) for item in PRankH]

    elif centrality == 'closeness':
        # FIXME seems to not distinguish edge directions
        node_cent = []  # TODO :made it to see progrees, mb need to be optimized
        nodes = [node.GetId() for node in s.Nodes()]
        for i in tqdm(range(s.GetNodes())):  # fixed to see progress bars
            n = nodes[i]
            node_cent.append((n, snap.GetClosenessCentr(s, n, graph.directed)))

    elif centrality == 'eccentricity':
        node_cent = []
        nodes = [node.GetId() for node in s.Nodes()]
        for i in tqdm(range(s.GetNodes())):  # fixed to see progress bars
            n = nodes[i]
            node_cent.append((n, snap.GetNodeEcc(s, n, graph.directed)))

    elif centrality == 'clustering':
        NIdCCfH = snap.TIntFltH()
        snap.GetNodeClustCf(s, NIdCCfH)
        node_cent = [(item, NIdCCfH[item]) for item in NIdCCfH]

    elif centrality == 'k-coreness':  # TODO could be computed in networkx
        set_of_top_k = [0]
        node_cent = []
        node_cent_dict = {}
        k = 0
        while len(set_of_top_k) > 0:
            k += 1
            KCore = snap.GetKCore(s, k)
            if KCore.Empty():
                break
            for node in KCore.Nodes():
                node_cent_dict[node.GetId()] = k
        node_cent = [(node, k) for node, k in node_cent_dict.items()]

    else:
        raise NotImplementedError("")

    logging.info(" done.")
    return node_cent
    # sorted_node_cent = sorted(node_cent, key=itemgetter(1), reverse=True)


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
    graph = GraphCollections.get('digg-friends')
    node_prop = graph.get_node_property_dict('betweenness')
    print(node_prop)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute centralities for graph nodes.')
    parser.add_argument('-p', '--path', required=True, help='path to input graph as edgelist')
    parser.add_argument('-d', action='store_true', help='specify if graph is directed')
    parser.add_argument('-c', '--centralities', required=True, nargs='+', choices=CENTRALITIES,
                        help='node centralities to compute')

    args = parser.parse_args()
    # print(args)
    graph = MyGraph(path=args.path, name='', directed=args.d)
    for c in args.centralities:
        assert c in CENTRALITIES, "Unknown centrality %s, available are: %s" % (c, CENTRALITIES)
        # print("Computing %s centrality for %s..." % (c, args.path))
        graph.get_node_property_dict(property=c)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    test()
    # main()
