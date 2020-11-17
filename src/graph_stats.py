import logging
from enum import Enum
from operator import itemgetter

from base.cgraph import MyGraph

from graph_io import GraphCollections
from utils import USE_NETWORKIT

if USE_NETWORKIT:  # Use networkit library for community detection
    from networkit._NetworKit import PLM, Modularity


class Stat(Enum):
    """
    Graph statistics.
    Numeric, distributional, and other statistics are available.

    To add a new graph statistics, add its definition here and implement it in `cyth/cstatistics.pyx`.
    """

    def __init__(self, short: str, description: str):
        self.short = short
        self.description = description

    def __str__(self):
        return self.short

    # Numeric statistics
    NODES = 'n', "number of nodes"
    EDGES = 'e', "number of edges"
    AVG_DEGREE = 'avg-deg', "average (out)degree"
    MAX_DEGREE = 'max-deg', "maximal degree"
    ASSORTATIVITY = 'ass', "nodes degree assortativity"
    AVG_CC = 'avg-cc', "average local clustering coeff."
    MAX_WCC = 'wcc-max', "relative size of largest WCC"
    DIAMETER_90 = 'diam90', "90%eff. diam. of largest WCC"

    # Distributional statistics
    DEGREE_DISTR = 'DegDistr', 'degree centrality'
    BETWEENNESS_DISTR = 'BtwDistr', 'betweenness centrality'
    ECCENTRICITY_DISTR = 'EccDistr', 'eccentricity centrality'
    CLOSENESS_DISTR = 'ClsnDistr', 'closeness centrality'
    PAGERANK_DISTR = 'PgrDistr', 'pagerank centrality'
    K_CORENESS_DISTR = 'KCorDistr', 'k-coreness centrality'
    CLUSTERING_DISTR = 'ClustDistr', 'clustering centrality'

    # Other global statistics
    PLM_COMMUNITIES = 'PLM-comms', 'PLM communities'
    PLM_MODULARITY = 'PLM-modularity', 'PLM communities modularity'


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
        count = graph.nodes()
    return [n for (n, d) in sorted_node_cent[:count]]


def plm(graph: MyGraph):
    """
    Detect communities via PLM - Parallel Louvain Method and compute modularity.
    Set both the stats of the graph via setter.
    """
    node_map = {}
    nk = graph.networkit(node_map)
    plm = PLM(nk, refine=False, gamma=1)
    plm.run()
    partition = plm.getPartition()
    # for p in partition:
    comms_list = []
    for i in range(partition.numberOfSubsets()):
        nk_comm = partition.getMembers(i)
        comm = []
        for nk_i in nk_comm:
            i = node_map[nk_i]
            if i is not None:
                comm.append(i)
        if len(comm) > 0:
            comms_list.append(comm)

    mod = Modularity().getQuality(partition, nk)
    graph[Stat.PLM_COMMUNITIES] = comms_list
    graph[Stat.PLM_MODULARITY] = mod

    return comms_list, mod


def test_stats():
    from graph_io import GraphCollections
    graph = GraphCollections.get('dolphins', giant_only=True)

    for stat in Stat:
        print("%s = %s" % (stat.short, graph[stat]))


def main():
    import argparse
    stats = [s.name for s in Stat]
    parser = argparse.ArgumentParser(
        description='Compute statistics for graphs. Graph is specified via path (-p) or name in '
                    'available collections (-n).')
    parser.add_argument('-p', '--path', required=False, nargs='+', help='path to input graphs as edgelist')
    parser.add_argument('-n', '--name', required=False, nargs='+', help='names of input graphs in available collections')
    parser.add_argument('-c', '--collection', required=False, help="graphs collection: 'netrepo' or 'other'")
    parser.add_argument('-f', '--full', action='store_true', help='print full statistics value')
    parser.add_argument('-s', '--stats', required=True, nargs='+', choices=stats,
                        help='node statistics to compute')

    args = parser.parse_args()
    if (1 if args.path else 0) + (1 if args.name else 0) != 1:
        raise ValueError("Exactly one of '-p' and '-n' args must be specified.")

    for s in args.stats:
        assert s in stats, "Unknown statistics %s, available are: %s" % (s, stats)

    if args.path:
        graphs = [MyGraph(path=p, name='', directed=args.d) for p in args.path]
    else:
        collection = args.collection if args.collection else None
        graphs = [GraphCollections.get(n, collection, giant_only=True) for n in args.name]

    for graph in graphs:
        for s in args.stats:
            # print("Computing %s centrality for %s..." % (c, args.path))
            v = graph[s]
            if not args.full:  # short print
                v = (str(v)[:100] + '...') if len(str(v)) > 100 else str(v)
            logging.info("%s: %s" % (s, v))


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # imports workaround https://stackoverflow.com/questions/26589805/python-enums-across-modules
    import sys
    sys.modules['statistics'] = sys.modules['__main__']

    # test_stats()
    main()
