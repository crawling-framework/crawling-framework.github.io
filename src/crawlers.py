import logging
from operator import itemgetter

import numpy as np
import snap
from numpy import random

from graph_io import MyGraph


class Crawler(object):

    def __init__(self, graph: MyGraph):
        # original graph
        self.orig_graph = graph
        # observed snap graph
        self.observed_g = snap.TNGraph.New() if graph.directed else snap.TUNGraph.New()
        # crawled ids set
        self.crawled_set = set()
        # observed ids set excluding crawled ones
        self.observed_set = set()

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_g.Nodes()])

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
        g = self.observed_g
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
        self.observed_g.AddNode(initial_seed)
        super().crawl_budget(budget)


def get_top_hubs(graph, count):
    """
    Get top-count hubs of the graph sorted by degree
    :param graph: snap graph
    :param count:
    :return:
    """
    # TODO how to choose nodes at the border degree?
    node_deg = [(n.GetId(), n.GetDeg()) for n in graph.Nodes()]
    sorted_node_deg = sorted(node_deg, key=itemgetter(1), reverse=True)
    # print(sorted_node_deg)
    return [n for (n, d) in sorted_node_deg[:count]]


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
        for o_id in observed_only:
            deg = self.observed_g.GetNI(o_id).GetDeg()
            obs_deg.append((o_id, deg))

        max_degs = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n-self.n1]

        # Crawl chosen nodes
        [self.crawl(n) for n, _ in max_degs]

        # assert len(self.crawled) == self.n
        # Take top-k of degrees
        hubs_detected = get_top_hubs(self.observed_g, self.k)
        return hubs_detected


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
    graph = MyGraph('', 'test', directed=False)
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
