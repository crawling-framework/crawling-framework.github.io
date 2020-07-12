import glob
import os
from os.path import join as opj
from itertools import combinations

import numpy as np
from numpy import random
import scipy.stats as stats

from base.cgraph import MyGraph
from graph_models.cmodels import configuration_model, grid2d
from graph_io import temp_dir, GraphCollections
from statistics import Stat
from utils import LFR_DIR, GRAPHS_DIR

LFR_PATH = opj(LFR_DIR, 'benchmark')


def truncated_power_law(count, gamma, maximal):
    """
    Generate power law integer samples y ~ x^{-gamma} for 1<=x<=maximal.
    :param count: number of samples
    :param gamma: power
    :param maximal: max value
    :return:
    """
    if maximal >= count:
        maximal = count-1
    x = np.arange(1, maximal + 1, dtype='float')
    pmf = 1 / x ** gamma
    pmf /= pmf.sum()
    d = stats.rv_discrete(values=(range(1, maximal + 1), pmf))
    return d.rvs(size=count)


def truncated_normal(count, mean, variance, min, max):
    """
    Generate normal integer samples for min<=x<=maximal.
    :param count: number of samples
    :param mean:
    :param variance:
    :param min: min value
    :param max: max value
    :return:
    """
    assert min < max
    x = np.arange(min, max + 1, dtype='float')
    pmf = np.exp(-0.5*(x-mean)**2 / variance)
    pmf /= pmf.sum()
    d = stats.rv_discrete(values=(range(min, max + 1), pmf))
    return d.rvs(size=count)


# def ba_model(n, avg_deg, directed=False, random_seed=None) -> MyGraph:
#     assert directed is False
#
#     # g = snap.TNGraph.New() if directed else snap.TUNGraph.New()
#     #
#     # degs = np.zeros(n, dtype=int)
#     #
#     # # Initial component
#     # for i in range(avg_deg):
#     #     g.AddNode(i)
#     #     degs[i] = avg_deg-1
#     # print(list(combinations(range(avg_deg), 2)))
#     # for i, j in combinations(range(avg_deg), 2):
#     #     g.AddEdge(i, j)
#     #
#     # for i in range(avg_deg, n):
#     #     if i%1000 == 0:
#     #         print("BA: nodes", i)
#     #     g.AddNode(i)
#     #     s = sum(degs)
#     #     for j in range(0, i):
#     #         p = avg_deg * degs[j] / s
#     #         if np.random.random() < p:
#     #             g.AddEdge(i, j)
#     #             degs[i] += 1
#     #             degs[j] += 1
#     #
#     # print(degs)
#
#     Rnd = snap.TRnd(random_seed if random_seed else random.randint(1e9))
#     g = snap.GenPrefAttach(n, avg_deg, Rnd)
#     # for EI in g.Edges():
#     #     print("edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
#     print("N=%d, E=%d" % (g.GetNodes(), g.GetEdges()))
#     return MyGraph.new_snap(g, name='BA_model')


def LFR(nodes: int, avg_deg: float, max_deg: int, mixing: float, t1=None, t2=None, minc=None, maxc=None, on=None, om=None, C=None):
    """
    LFR benchmark. https://www.santofortunato.net/resources
    Default parameters: t1=2, t2=1, on=0, om=0, minc and maxc will be chosen close to the degree sequence extremes.

    :param nodes: number of nodes
    :param avg_deg: average degree
    :param max_deg: maximum degree
    :param mixing: mixing parameter
    :param t1: minus exponent for the degree sequence
    :param t2: minus exponent for the community size distribution
    :param minc: minimum for the community sizes
    :param maxc: maximum for the community sizes
    :param on: number of overlapping nodes
    :param om: number of memberships of the overlapping nodes
    :param C: [average clustering coefficient]
    :return:
    """
    kwargs = {
        'N': nodes,
        'k': avg_deg,
        'maxk': max_deg,
        'mu': mixing,
        't1': t1,
        't2': t2,
        'minc': minc,
        'maxc': maxc,
        'on': on,
        'om': om,
        'C': C,
    }
    commands = [LFR_PATH]
    for key, value in kwargs.items():
        if value is not None:
            commands += ['-%s' % key, str(value)]

    # Create path for graph_models new graph
    name = "LFR(%s)" % ",".join(
        "%s=%s" % (key, value) for key, value in sorted(kwargs.items()) if value is not None)
    path = opj(GRAPHS_DIR, 'synthetic', name, '*.ij')
    ix = len(glob.glob(path))
    path = path.replace('*', "%s" % ix)
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with temp_dir() as directory:
        # Run LFR
        retcode = os.system(" ".join(commands))
        if retcode != 0:
            raise RuntimeError("LFR benchmark failed with code %s" % retcode)

        # Handle graph
        os.rename(opj(directory, 'network.dat'), path)
        g = MyGraph(path=path, name=name).giant_component()
        assert g[Stat.MAX_WCC] == 1

        # Handle communities
        comms = {}
        with open(opj(directory, 'community.dat'), 'r') as f:
            for line in f.readlines():
                node, comm = line.split()
                node = int(node)
                comm = int(comm)
                if comm not in comms:
                    comms[comm] = []
                comms[comm].append(node)
        g[Stat.LFR_COMMUNITIES] = list(comms.values())
    return g


def test():
    import matplotlib.pyplot as plt

    deg_seq = truncated_power_law(1000, gamma=1.6, maximal=100)
    # deg_seq = truncated_normal(100, mean=10, variance=1000, min=1, max=100)
    # print(deg_seq)

    # plt.hist(sample, bins=np.arange(100) + 0.5)
    # plt.show()
    # graph = configuration_model(deg_seq)
    # g = ba_model(100, 10)

    g = LFR(500, 10, 100, 0.3)

    # from graph_io import GraphCollections
    # graph = GraphCollections.get('petster-hamster')
    # for stat in Stat:
    #     print(stat, g[stat])
    # print(graph[Stat.DIAMETER_90])


if __name__ == '__main__':
    test()
    g = LFR(500, 10, 30, 0.02)
    print(g.nodes())
