import logging
# from abc import ABC
from abc import ABC
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
        for n in self.orig_graph.neighbors(seed):
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
            while not self.crawl(seed):
                continue
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))


class MultiSeedCrawler(Crawler, ABC):
    """
    great class to Avrachenkov and other crawlers starting with n1 seeds
    """

    def __init__(self, graph: MyGraph):
        super().__init__(graph)
        # assert n1 <= self.budget_left <= self.orig_graph.snap.GetNodes()
        # assert k <= self.budget_left - n1
        # self.n1 = n1  # n1 seeds crawled on first steps, then comes crawler
        self.seed_sequence_ = []  # sequence of tries to add nodes
        self.initial_seeds = []  # list of initial seeds to iter or jump into
        self.budget_left = 1  # how many iterations left. stops when 0
        # print("n1={}, budget={}, nodes={}".format(self.n1, self.budget_left, self.orig_graph.snap.GetNodes()))

    def crawl_multi_seed(self, n1):
        if n1 <= 1:  # if there is no parallel seeds, method do nothing
            return False
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        multi_seeds = [int(node) for node in np.random.choice(graph_nodes, n1)]
        print("seeds for multi crawl:", multi_seeds)
        for seed in multi_seeds:
            self.crawl(seed)

        print("observed set", list(self.observed_set))

    def crawl(self, seed):
        """
        Crawls given seed
        Decrease self.budget_left only if crawl is successful
        """
        self.seed_sequence_.append(seed)
        if super().crawl(seed):
            self.budget_left -= 1
        else:
            logging.debug("budget ={}, seed = {}, in crawled set={}".format(self.budget_left, seed, self.crawled_set))
            return False

        logging.debug("-- seed {}, crawled #{}: {}, observed #{},".format(seed,
                                                                          len(self.crawled_set), self.crawled_set,
                                                                          len(self.observed_set), self.observed_set))

    def crawl_budget(self, budget, p=0):
        """
        Crawl until done budget
        :param p: probability to jump into one of self.initial_seed nodes
        :param budget: how many iterations left
        :return:
        """
        self.budget_left = budget
        if random.randint(0, 100, 1) < p * 100:
            print("variety play")
            self.crawl(int(np.random.choice(self.initial_seeds, 1)[0]))
            self.budget_left -= 1

        while self.budget_left > 0:
            seed = int(self.next_seed())
            if not (self.crawl(seed)):
                seed = self.next_seed()
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))


class RandomWalk(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.prev_seed = 1  # previous node, that was already crawled

    def next_seed(self):
        node_neighbours = self.observed_graph.neighbors(self.prev_seed)
        # for walking we need to step on already crawled nodes too
        if len(node_neighbours) == 0:
            node_neighbours = tuple(self.observed_set)
        return random.choice(node_neighbours, 1)[0]

    def crawl(self, seed):
        super().crawl(seed)
        self.prev_seed = seed


class RandomCrawler(Crawler):  # TODO
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)

    def next_seed(self):
        return random.choice(tuple(self.observed_set))


class AvrachenkovCrawler(Crawler, ABC):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """

    def __init__(self, graph, n=1000, n1=500, k=100):
        super().__init__(graph)
        assert n1 <= n <= self.orig_graph.snap.GetNodes()
        assert k <= n - n1
        self.n1 = n1
        self.n = n
        self.k = k

    def first_step(self):
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        size = len(graph_nodes)

        i = 0
        while True:
            seed = graph_nodes[np.random.randint(size)]
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

        max_degs = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n - self.n1]

        # Crawl chosen nodes
        [self.crawl(n) for n, _ in max_degs]

        # assert len(self.crawled) == self.n
        # Take top-k of degrees
        hubs_detected = get_top_centrality_nodes(self.observed_graph, 'degree', self.k)
        return hubs_detected


def test_graph():
    g = snap.TUNGraph.New()
    g.AddNode(1)
    g.AddNode(2)
    g.AddNode(3)
    g.AddNode(4)
    g.AddNode(5)
    g.AddNode(6)
    g.AddNode(7)

    g.AddEdge(1, 2)
    g.AddEdge(2, 3)
    g.AddEdge(4, 2)
    g.AddEdge(4, 3)
    g.AddEdge(5, 4)
    g.AddEdge(1, 6)
    g.AddEdge(6, 7)

    g.AddNode(11)
    g.AddNode(12)
    g.AddNode(13)
    g.AddNode(14)
    g.AddNode(15)
    g.AddNode(16)

    g.AddEdge(11, 12)
    g.AddEdge(12, 13)
    g.AddEdge(14, 12)
    g.AddEdge(14, 13)
    g.AddEdge(15, 13)
    g.AddEdge(15, 5)
    g.AddEdge(11, 16)
    return g


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # test()
    test_g = test_graph()
    print("N=%s E=%s" % (test_g.GetNodes(), test_g.GetEdges()))
    Graph = MyGraph.new_snap(name='test', directed=False)
    Graph._snap_graph = test_g
    print(Graph.snap.GetNodes())
    # crawler = Crawler(graph)
    # for i in range(1, 6):
    #     crawler.crawl(i)
    #     print("crawled:%s, observed:%s, all:%s" %
    #           (crawler.crawled_set, crawler.observed_set, crawler.nodes_set))

    # crawler = RandomCrawler(graph)
    total_budget = 10

    crawler = RandomWalk(Graph)
    crawler.crawl_multi_seed(n1=3)
    print("after first: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
          " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))
    print("normal crawling")

    crawler.initial_seed = [crawler.prev_seed]  # TODO
    crawler.crawl_budget(total_budget)

    print("after second: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
          " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))

    print("Total iterations:", len(crawler.seed_sequence_))
    print("sequence of seeds:", crawler.seed_sequence_)
