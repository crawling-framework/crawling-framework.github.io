import logging
from operator import itemgetter

import numpy as np
import snap
from numpy import random

from centralities import get_top_centrality_nodes
from graph_io import MyGraph


class Crawler(object):

    def __init__(self, graph: MyGraph):
        # original graph
        self.orig_graph = graph
        # observed graph
        self.observed_graph = MyGraph.new_snap(directed=graph.directed, weighted=graph.weighted)
        # observed snap graph
        # crawled ids set
        self.crawled_set = set()
        # observed ids set excluding crawled ones
        self.observed_set = set()

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_graph.snap.Nodes()])

    def crawl(self, seed: int):
        """
        Crawl specified nodes. The observed graph is updated, also crawled and observed set.
        :param seed: node id to crawl
        :return: whether the node was crawled
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            return False

        self.crawled_set.add(seed)
        g = self.observed_graph.snap
        if g.IsNode(seed):  # remove from observed set
            self.observed_set.remove(seed)
        else:  # add to observed graph
            g.AddNode(seed)

        # iterate over neighbours
        for n in self.orig_graph.snap.GetNI(seed).GetOutEdges():
            if not g.IsNode(n):  # add to observed graph and observed set
                g.AddNode(n)
                self.observed_set.add(n)
            g.AddEdge(seed, n)

        return True

    def next_seed(self):
        raise NotImplementedError()

    def crawl_budget(self, budget: int, *args):
        for _ in range(budget):
            seed = self.next_seed()
            self.crawl(seed)
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))


class RandomCrawler(Crawler):
    def __init__(self, graph: MyGraph):
        super().__init__(graph)

    def next_seed(self):
        return random.choice(tuple(self.observed_set))

    def crawl_budget(self, budget: int, initial_seed=1):
        self.observed_set.add(initial_seed)
        self.observed_graph.snap.AddNode(initial_seed)
        super().crawl_budget(budget)


class AvrachenkovCrawler(Crawler):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """
    def __init__(self, graph, n=1000, n1=500, k=100):
        super().__init__(graph)
        assert n1 <= n <= self.orig_graph.snap.GetNodes()
        assert k <= n-n1
        self.n1 = n1
        self.n = n
        self.k = k

    def first_step(self):
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        N = len(graph_nodes)

        i = 0
        while True:
            seed = graph_nodes[np.random.randint(N)]
            if seed in self.crawled_set:
                continue
            self.crawl(seed)
            i += 1
            if i == self.n1:
                break

    def second_step(self):
        observed_only = self.observed_set

        # Get n2 max degree observed nodes
        obs_deg = []
        g = self.observed_graph.snap
        for o_id in observed_only:
            deg = g.GetNI(o_id).GetDeg()
            obs_deg.append((o_id, deg))

        max_degs = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n-self.n1]

        # Crawl chosen nodes
        [self.crawl(n) for n, _ in max_degs]

        # assert len(self.crawled) == self.n
        # Take top-k of degrees
        hubs_detected = get_top_centrality_nodes(self.observed_graph.snap, 'degree', self.k)
        return hubs_detected


class TwoStageCrawler(Crawler):
    """
    todo

    """
    def __init__(self, graph: MyGraph, s=500, n=1000, p=0.1):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph)
        assert s <= n <= int(p*self.orig_graph.snap.GetNodes())
        self.s = s
        self.n = n
        self.pN = int(p*self.orig_graph.snap.GetNodes())

    def first_step(self, random_init=None):
        """ 1) Crawl s random seeds (same as Avrachenkov), obtain E1 - first neighbourhood.
        """
        if random_init is not None:
            np.random.seed(random_init)
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        N = len(graph_nodes)

        i = 0
        while True:
            seed = graph_nodes[np.random.randint(N)]
            if seed in self.crawled_set:
                continue
            self.crawl(seed)
            i += 1
            if i == self.s:
                break

    def second_step(self):
        """
        2) Detect (n-s) nodes by MOD from E1 -> E1*. Crawl E1* and obtain E2 - second neighbourhood.
        3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN
        """
        # Get E1 - first neighbourhood with, their degrees.
        e1 = []
        o = self.observed_graph.snap
        for o_id in self.observed_set:
            deg = o.GetNI(o_id).GetDeg()
            e1.append((o_id, deg))

        # Check that e1 size is more than (n-s)
        if self.n-self.s > len(e1):
            logging.warning("%s: (n-s) must not be > |E1|, n-s=%s, |E1|=%s" %
                            (type(self), self.n-self.s, len(e1)))

        # Get n-s max degree observed nodes
        self.e1s = [n for n, _ in sorted(e1, key=itemgetter(1), reverse=True)[:self.n-self.s]]

        # Crawl chosen nodes
        [self.crawl(n) for n in self.e1s]

        # Get E2 - second neighbourhood, with their degrees.
        e2 = []
        for o_id in self.observed_set:
            deg = o.GetNI(o_id).GetDeg()
            e2.append((o_id, deg))

        # Get v=(pN-n+s) max degree observed nodes
        self.e2s = [n for n, _ in sorted(e2, key=itemgetter(1), reverse=True)[:self.pN - self.n + self.s]]

        # Final answer - E* = E1* + E2*
        self.es = set(self.e1s + self.e2s)
        print("s=%s, n=%s" % (self.s, self.n))
        print("len e1", len(e1))
        print("len e1s", len(self.e1s))
        print("len e2", len(e2))
        print("len e2s", len(self.e2s))
        print("len es", len(self.es))
        print("pN", self.pN)
        # assert len(es) == self.pN
        return self.es
        # top_from_observed = list(self.observed_set)
        #top_from_observed = sorted(e2e1, key=itemgetter(1), reverse=True)[:self.pN - self.n + self.s]
        # [top_from_observed.append(n) for n, _ in e1s]

        # return set(top_from_observed)


def test():
    g = snap.TUNGraph.New()
    g.AddNode(1)
    g.AddNode(2)
    g.AddNode(3)
    g.AddNode(4)
    g.AddNode(5)
    g.AddEdge(1, 2)
    g.AddEdge(2, 3)
    g.AddEdge(4, 2)
    g.AddEdge(4, 3)
    g.AddEdge(5, 4)
    print("N=%s E=%s" % (g.GetNodes(), g.GetEdges()))
    graph = MyGraph.new_snap(name='test', directed=False)
    graph.snap_graph = g

    # crawler = Crawler(graph)
    # for i in range(1, 6):
    #     crawler.crawl(i)
    #     print("crawled:%s, observed:%s, all:%s" %
    #           (crawler.crawled_set, crawler.observed_set, crawler.nodes_set))

    crawler = RandomCrawler(graph)
    crawler.crawl_budget(15)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    test()
