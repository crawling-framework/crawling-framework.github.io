from itertools import combinations

import numpy as np
from numpy import random
import scipy.stats as stats
import snap

from base.graph import MyGraph
from statistics import Stat


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


def configuration_model(deg_seq, random_seed=None) -> MyGraph:
    DegSeqV = snap.TIntV()
    for deg in deg_seq:
        DegSeqV.Add(deg)
    Rnd = snap.TRnd(random_seed if random_seed else random.randint(1e9))

    g = snap.GenConfModel(DegSeqV, Rnd)
    print("N=%d, E=%d" % (g.GetNodes(), g.GetEdges()))
    # for EI in g.Edges():
    #     print("edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
    return MyGraph.new_snap(g, name='Config_model')


def ba_model(n, avg_deg, directed=False, random_seed=None) -> MyGraph:
    assert directed is False

    # g = snap.TNGraph.New() if directed else snap.TUNGraph.New()
    #
    # degs = np.zeros(n, dtype=int)
    #
    # # Initial component
    # for i in range(avg_deg):
    #     g.AddNode(i)
    #     degs[i] = avg_deg-1
    # print(list(combinations(range(avg_deg), 2)))
    # for i, j in combinations(range(avg_deg), 2):
    #     g.AddEdge(i, j)
    #
    # for i in range(avg_deg, n):
    #     if i%1000 == 0:
    #         print("BA: nodes", i)
    #     g.AddNode(i)
    #     s = sum(degs)
    #     for j in range(0, i):
    #         p = avg_deg * degs[j] / s
    #         if np.random.random() < p:
    #             g.AddEdge(i, j)
    #             degs[i] += 1
    #             degs[j] += 1
    #
    # print(degs)

    Rnd = snap.TRnd(random_seed if random_seed else random.randint(1e9))
    g = snap.GenPrefAttach(n, avg_deg, Rnd)
    # for EI in g.Edges():
    #     print("edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
    print("N=%d, E=%d" % (g.GetNodes(), g.GetEdges()))
    return MyGraph.new_snap(g, name='BA_model')


def test():
    import matplotlib.pyplot as plt

    # deg_seq = truncated_power_law(100000, gamma=1.6, maximal=100)
    # deg_seq = truncated_normal(100, mean=10, variance=1000, min=1, max=100)
    # print(deg_seq)

    # plt.hist(sample, bins=np.arange(100) + 0.5)
    # plt.show()
    # g = configuration_model(deg_seq)
    # g = ba_model(100, 10)

    from graph_io import GraphCollections, MyGraph
    graph = GraphCollections.get('petster-hamster', giant_only=True)
    for name, obj in vars(Stat).items():
        if type(obj) == Stat:

            # print(stat)
            # print(stat.short, stat.description)
            print(name, graph[obj])


if __name__ == '__main__':
    test()
