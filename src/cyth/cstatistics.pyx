import logging
import os
import subprocess
import sys

from tqdm import tqdm

from utils import USE_NETWORKIT, USE_LIGRA, LIGRA_DIR
from statistics import Stat, plm
from base.cgraph cimport MyGraph, GetClustCf, GetMxWccSz, PUNGraph, TUNGraph, GetBfsEffDiam, \
    GetMxWcc, TIntFltH, GetBetweennessCentr, THashKeyDatI, TInt, TFlt, GetPageRank, \
    GetClosenessCentr, GetNodeEcc, GetNodesClustCf, GetKCore
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

    Stat.PLM_COMMUNITIES: (lambda graph: plm(graph)[0]),
    Stat.PLM_MODULARITY: (lambda graph: plm(graph)[1]),

    Stat.LFR_COMMUNITIES: (lambda _: []),  # can be pre-defined only, not computable
}


cdef double avg_cc(MyGraph graph):
    return GetClustCf[PUNGraph](graph.snap_graph_ptr(), -1)


cdef double max_wcc_size(MyGraph graph):
    return GetMxWccSz[PUNGraph](graph.snap_graph_ptr())


cdef double diam_90(MyGraph graph):
    cdef int n_approx = 1000
    return GetBfsEffDiam[PUNGraph](GetMxWcc[PUNGraph](graph.snap_graph_ptr()), min(n_approx, graph.nodes()), False)


cdef float assortativity(MyGraph graph):
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


cdef dict compute_nodes_centrality(MyGraph graph, str centrality, nodes_fraction_approximate=10000, only_giant=False):
    """
    Compute centrality value for each node of the graph.
    :param graph: MyGraph
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
    cdef int n = graph.nodes(), node, k, i
    cdef TIntFltH Nodes
    cdef THashKeyDatI[TInt, TFlt] if_iter
    cdef TUNGraph KCore

    if only_giant:
        p = GetMxWcc[PUNGraph](p)

    node_cent = {}
    if centrality == 'degree':
        ni = deref(p).BegNI()
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
            ni = deref(p).BegNI()
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
        if not USE_LIGRA or n < 1000:
            ni = deref(p).BegNI()
            for i in tqdm(range(n)):
                node_cent[ni.GetId()] = GetNodeEcc[PUNGraph](p, ni.GetId(), graph.directed)
                pinc(ni)

        else:  # Ligra
            # duplicate edges
            path = graph.path
            path_dup = path + '_dup'
            with open(path_dup, 'w') as out_file:
                for line in open(path, 'r'):
                    if len(line) < 3:
                        break
                    x, y = line.split()
                    out_file.write('%s %s\n' % (x, y))
                    out_file.write('%s %s\n' % (y, x))

            # convert to Adj
            path_lig = path + '_ligra'
            ligra_converter_command = "./utils/SNAPtoAdj '%s' '%s'" % (path_dup, path_lig)
            retcode = subprocess.Popen(ligra_converter_command, cwd=LIGRA_DIR, shell=True,
                                       stderr=sys.stderr).wait()
            if retcode != 0:
                raise RuntimeError("Ligra converter failed: '%s'" % ligra_converter_command)

            # Run Ligra kBFS
            path_lig_ecc = path + '_ecc'
            ligra_ecc_command = "./apps/eccentricity/kBFS-Ecc -s -rounds 0 -out '%s' '%s'" % (
            path_lig_ecc, path_lig)
            # ligra_ecc_command = "./apps/eccentricity/kBFS-Exact -s -rounds 0 -out '%s' '%s'" % (path_lig_ecc, path_lig)
            retcode = subprocess.Popen(ligra_ecc_command, cwd=LIGRA_DIR, shell=True,
                                       stdout=subprocess.DEVNULL, stderr=sys.stderr).wait()
            if retcode != 0:
                raise RuntimeError("Ligra kBFS-Ecc failed: %s" % ligra_ecc_command)

            # read and convert ecc
            node_cent = {}
            for n, line in enumerate(open(path_lig_ecc)):
                if graph.has_node(n):
                    node_cent[n] = int(line)
            assert len(node_cent) == graph.nodes()

            os.remove(path_dup)
            os.remove(path_lig)
            os.remove(path_lig_ecc)


    elif centrality == 'clustering':
        Nodes = TIntFltH()
        GetNodesClustCf[PUNGraph](p, Nodes)
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
    cdef MyGraph g = GraphCollections.get("petster-hamster")

    for s in [Stat.ECCENTRICITY_DISTR]:
        v = stat_computer[s](g)
        print(s, v)

    return 0
