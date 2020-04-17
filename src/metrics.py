from enum import Enum
import snap

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


def assortativity(graph: MyGraph):
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


def test():
    import sys
    # imports workaround https://stackoverflow.com/questions/26589805/python-enums-across-modules
    sys.modules['metrics'] = sys.modules['__main__']

    from graph_io import GraphCollections, MyGraph
    # graph = GraphCollections.get('petster-hamster')
    graph = GraphCollections.get('ego-gplus')

    for stat in Stat:
        print("%s = %s" % (stat.short, graph[stat]))


if __name__ == '__main__':
    test()
