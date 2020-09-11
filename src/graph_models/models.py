import glob
import os
from os.path import join as opj

import numpy as np
import scipy.stats as stats

from base.cgraph import MyGraph
from graph_io import temp_dir, GraphCollections
from graph_stats import Stat
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
    :return: MyGraph
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

    # Create path for a new graph
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
        if g.nodes() == 0:
            raise RuntimeError("LFR generated graph has no nodes.")
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


if __name__ == '__main__':
    g = LFR(500, 10, 30, 0.02)
    print(g.nodes())
