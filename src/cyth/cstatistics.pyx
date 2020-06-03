import logging
from tqdm import tqdm

from statistics import Stat, USE_NETWORKIT
from base.cgraph cimport CGraph, GetClustCf, GetMxWccSz, PUNGraph, TUNGraph, GetBfsEffDiam, \
    GetMxWcc, TIntFltH, GetBetweennessCentr, THashKeyDatI, TInt, TFlt, GetPageRank, \
    GetClosenessCentr, GetNodeEcc, GetNodeClustCf, GetKCore
from cython.operator cimport dereference as deref, postincrement as pinc

if USE_NETWORKIT:
    from networkit._NetworKit import Betweenness, ApproxBetweenness, EstimateBetweenness, ApproxCloseness

stat_computer = {
    Stat.NODES: lambda graph: graph.nodes(),
    Stat.EDGES: lambda graph: graph.edges(),
    Stat.AVG_DEGREE: lambda graph: (1 if graph.directed else 2) * graph.edges() / graph.nodes(),
    Stat.MAX_DEGREE: lambda graph: graph.max_deg(),
    Stat.ASSORTATIVITY: assortativity,
    Stat.AVG_CC: avg_cc,
    Stat.MAX_WCC: max_wcc_size,
    Stat.DIAMETER_90: diam_90,

    Stat.DEGREE_DISTR: lambda graph: compute_nodes_centrality(graph, 'degree'),
    Stat.BETWEENNESS_DISTR: lambda graph: compute_nodes_centrality(graph, 'betweenness'),
    Stat.ECCENTRICITY_DISTR: lambda graph: compute_nodes_centrality(graph, 'eccentricity'),
    Stat.CLOSENESS_DISTR: lambda graph: compute_nodes_centrality(graph, 'closeness'),
    Stat.PAGERANK_DISTR: lambda graph: compute_nodes_centrality(graph, 'pagerank'),
    # Stat.CLUSTERING_DISTR: lambda graph: compute_nodes_centrality(graph, 'clustering'),
    Stat.K_CORENESS_DISTR: lambda graph: compute_nodes_centrality(graph, 'k-coreness'),
}


cdef double avg_cc(CGraph graph):
    # cdef PUNGraph p = PUNGraph(&graph._snap_graph)
    cdef PUNGraph p = graph.snap_graph_ptr()
    cdef double res = GetClustCf[PUNGraph](p, -1)
    return res


cdef double max_wcc_size(CGraph graph):
    cdef PUNGraph p = graph.snap_graph_ptr()
    return GetMxWccSz[PUNGraph](p)


cdef double diam_90(CGraph graph):
    cdef int n_approx = 1000
    cdef PUNGraph p = graph.snap_graph_ptr()
    return GetBfsEffDiam[PUNGraph](GetMxWcc[PUNGraph](p), min(n_approx, graph.nodes()), False)


cdef float assortativity(CGraph graph):
    """ Degree assortativity -1<=r<=1"""
    cdef double mul = 0
    cdef double sum2 = 0
    cdef double sum_sq2 = 0
    cdef int m = graph.edges()
    cdef int j, k, dj, dk
    for j, k in graph.iter_edges():
        dj, dk = graph.deg(j), graph.deg(k)
        mul += dj * dk
        sum2 += dj + dk
        sum_sq2 += dj*dj + dk*dk

    cdef float r = (mul/m - (sum2/2/m) ** 2) / (sum_sq2/2/m - (sum2/2/m) ** 2)
    return r


cdef dict compute_nodes_centrality(CGraph graph, str centrality, nodes_fraction_approximate=10000, only_giant=False):
    """
    Compute centrality value for each node of the graph.
    :param graph: CGraph
    :param centrality: centrality name, one of utils.CENTRALITIES
    :param nodes_fraction_approximate: parameter for betweenness
    :param only_giant: compute for giant component only, note size of returned list will be less
    than the number of nodes
    :return: dict (node id -> centrality)
    """
    logging.info("Computing '%s' for graph '%s' with N=%d, E=%d. Can take a while..." %
                 (centrality, graph.name, graph.nodes(), graph.edges()))

    cdef TUNGraph.TNodeI ni
    cdef PUNGraph p = graph.snap_graph_ptr()
    cdef TUNGraph g = deref(p)
    cdef int n = g.GetNodes(), node, k, i
    cdef TIntFltH Nodes
    cdef THashKeyDatI[TInt, TFlt] if_iter
    cdef TUNGraph KCore

    if only_giant:
        p = GetMxWcc[PUNGraph](p)

    node_cent = {}
    if centrality == 'degree':
        ni = g.BegNI()
        for i in tqdm(range(n)):
            node_cent[ni.GetId()] = ni.GetDeg()
            pinc(ni)

    elif centrality == 'betweenness':
        if not USE_NETWORKIT:
            # cdef TIntPrFltH Edges
            Nodes = TIntFltH()
            # Edges = TIntPrFltH()
            if nodes_fraction_approximate is None and n > 10000:
                nodes_fraction_approximate = 10000 / n
            GetBetweennessCentr[PUNGraph](p, Nodes, nodes_fraction_approximate, graph.directed)
            if_iter = Nodes.BegI()
            while not if_iter == Nodes.EndI():
                node_cent[if_iter.GetKey()()] = if_iter.GetDat()()
                pinc(if_iter)

        else:  # networkit
            # Based on the paper:
            # Sanders, Geisberger, Schultes: Better Approximation of Betweenness Centrality
            node_map = {}
            nk_graph = graph.networkit(node_map)
            centr = EstimateBetweenness(nk_graph, nSamples=1000, normalized=False, parallel=True)
            centr.run()
            scores = centr.scores()
            node_cent = {node_map[i]: score for i, score in enumerate(scores)}
            if None in node_cent:
                del node_cent[None]

    elif centrality == 'pagerank':
        Nodes = TIntFltH()
        GetPageRank[PUNGraph](p, Nodes)
        if_iter = Nodes.BegI()
        while not if_iter == Nodes.EndI():
            node_cent[if_iter.GetKey()()] = if_iter.GetDat()()
            pinc(if_iter)

    elif centrality == 'closeness':
        if not USE_NETWORKIT:  # snap
            # FIXME seems to not distinguish edge directions
            ni = g.BegNI()
            for i in tqdm(range(n)):
                node_cent[ni.GetId()] = GetClosenessCentr[PUNGraph](p, ni.GetId(), graph.directed)
                pinc(ni)
        else:  # networkit
            # Based on the paper:
            # Cohen et al., Computing Classic Closeness Centrality, at Scale.
            node_map = {}
            nk_graph = graph.networkit(node_map)
            # TODO for directed graphs see documentation
            centr = ApproxCloseness(nk_graph, nSamples=min(nk_graph.numberOfNodes(), 300), epsilon=0.1, normalized=False)
            centr.run()
            scores = centr.scores()
            node_cent = {node_map[i]: score for i, score in enumerate(scores)}
            if None in node_cent:
                del node_cent[None]

    elif centrality == 'eccentricity':
        ni = g.BegNI()
        for i in tqdm(range(n)):
            node_cent[ni.GetId()] = GetNodeEcc[PUNGraph](p, ni.GetId(), graph.directed)
            pinc(ni)

    elif centrality == 'clustering':
        Nodes = TIntFltH()
        GetNodeClustCf[PUNGraph](p, Nodes)
        if_iter = Nodes.BegI()
        while not if_iter == Nodes.EndI():
            node_cent[if_iter.GetKey()()] = if_iter.GetDat()()
            pinc(if_iter)

    elif centrality == 'k-coreness':  # TODO could be computed in networkx
        k = 0
        while True:
            k += 1
            KCore = deref(GetKCore[PUNGraph](p, k))
            if KCore.GetNodes() == 0:
                break
            ni = KCore.BegNI()
            for i in range(KCore.GetNodes()):
                node_cent[ni.GetId()] = k
                pinc(ni)

    else:
        raise NotImplementedError("")

    logging.info(" done.")
    return node_cent


cpdef int test_cstats() except -1:
    from graph_io import GraphCollections
    from utils import USE_CYTHON_CRAWLERS
    assert USE_CYTHON_CRAWLERS == True
    cdef CGraph g = GraphCollections.get("petster-hamster")

    for s in [Stat.ECCENTRICITY_DISTR]:
        v = stat_computer[s](g)
        print(s, v)

    return 0
