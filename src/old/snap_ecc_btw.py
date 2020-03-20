def eccentricities(graph):
    """ Eccentricity for each node. Quite long to wait
    """
    s = graph.get_as_snap()
    # res = [(NI.GetId(), snap.GetNodeEcc(s, NI.GetId(), gr.directed)) for NI in s.Nodes()]
    res = {}
    for NI in s.Nodes():
        # print NI
        res[NI.GetId()] = snap.GetNodeEcc(s, NI.GetId(), graph.directed)
    return res


def betweenness_centrality(graph, nodes_fraction_approximate=None):
    """ Computes (approximate) Node and Edge Betweenness Centrality based on a sample of nodes.
    """
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    s = graph.get_as_snap()
    if nodes_fraction_approximate is None:
        nodes_fraction_approximate = 1.0
    snap.GetBetweennessCentr(s, Nodes, Edges, nodes_fraction_approximate, graph.directed)
    res_nodes = {}
    for node in Nodes:
        res_nodes[node] = Nodes[node]
        # print "node: %d centrality: %f" % (node, Nodes[node])
    res_edges = {}
    for edge in Edges:
        # res_edges[(edge.GetVal1(), edge.GetVal2())] = Edges[edge]
        res_edges["(%d, %d)" % (edge.GetVal1(), edge.GetVal2())] = Edges[edge]
        # print "edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge])
    return res_nodes, res_edges


def distance_test():
    import seaborn
    from graph_store import *

    gr = GRAPH_VAST_DIR

    import json
    # for gr in [GRAPH_VK_DCAM, GRAPH_HAMSTERSTER, GRAPH_GITHUB, GRAPH_DBLP2010,
    # GRAPH_SLASHDOT_THREADS, GRAPH_GNUTELLA_[7]]:
    for gr in [GRAPH_SLASHDOT_THREADS]:
        gr.directed = False
        print(gr.get_size())
        print("Computing eccentricities...")
        ecc = eccentricities(gr)
        with open(gr.path + "_ecc", 'w') as f:
            json.dump(ecc, f)

        print("Computing betweenness...")
        btw_nodes, btw_edges = betweenness_centrality(gr, 1.0)
        with open(gr.path + "_btw_nodes", 'w') as f:
            json.dump(btw_nodes, f)
        with open(gr.path + "_btw_edges", 'w') as f:
            json.dump(btw_edges, f)



