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

    def __init__(self, orig_graph: MyGraph):
        # original graph
        self.orig_graph = orig_graph
        print("original graph nodes", self.orig_graph.snap.GetNodes())
        # observed graph
        self.observed_graph = MyGraph.new_snap(directed=orig_graph.directed, weighted=orig_graph.weighted)
        # observed snap graph
        # crawled ids set
        self.crawled_set = set()
        # observed ids set excluding crawled ones
        self.observed_set = set()
        self.prev_seed = 1  # previous node, that was already crawled
        self.budget_left = 1  # how many iterations left. stops when 0
        self.seed_sequence = []  # sequence of tries to add nodes

    def observed_neighbors(self, node):  # Denis's realisation
        """ returns set on neighbors of given node in OBSERVED graph only"""
        return tuple(self.observed_graph.snap.GetNI(node).GetOutEdges())

    def orig_neighbors(self, node):  # Denis's realisation
        """returns set on neighbors of given node in ORIGINAL graph (all real friends)"""
        return tuple(self.orig_graph.snap.GetNI(node).GetOutEdges())

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_graph.snap.Nodes()])

    def crawl(self, seed: int):
        """
        Crawl specified nodes. The observed graph is updated, also crawled and observed set.
        Decrease self.budget_left only if crawl is successful
        :param seed: node id to crawl
        :return: whether the node was crawled
        """
        self.seed_sequence.append(seed)
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            print("seed {} already in crawl set {}".format(seed, self.crawled_set))
            return False

        self.crawled_set.add(seed)
        g = self.observed_graph.snap
        if g.IsNode(seed):  # remove from observed set
            self.observed_set.remove(seed)
        else:  # add to observed graph
            # self.observed_set.add(seed)
            g.AddNode(seed)

        # iterate over neighbours
        for n in self.orig_neighbors(seed):
            if not g.IsNode(n):  # add to observed graph and observed set
                g.AddNode(n)
                self.observed_set.add(n)
            g.AddEdge(seed, n)
        print("-- seed {}, crawled #{}: {},".format(seed, len(self.crawled_set), self.crawled_set),
              " observed #{}: {}".format(len(self.observed_set), self.observed_set))
        self.budget_left -= 1
        return True

    def next_seed(self):
        raise NotImplementedError()

    def crawl_budget(self, *args):
        while self.budget_left > 0:
            seed = self.next_seed()
            print("budget left:", self.budget_left, "seed:", seed)
            self.crawl(seed)
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))


class MultiSeedCrawler(Crawler, ABC):
    """
    great class to Avrachenkov and other crawlers starting with n1 seeds
    """

    def __init__(self, orig_graph: MyGraph, n1: int):
        super().__init__(orig_graph)
        print(n1, self.budget_left, self.orig_graph.snap.GetNodes())
        # assert n1 <= self.budget_left <= self.orig_graph.snap.GetNodes()
        # assert k <= self.budget_left - n1
        self.n1 = n1  # n1 seeds crawled on first steps, then comes crawler

    def crawl_multi_seed(self):
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        # n = len(graph_nodes)

        first_stage_seeds = np.random.choice(graph_nodes, self.n1)

        for seed in first_stage_seeds:
            self.crawl(seed)

        print(list(self.observed_set))
        self.prev_seed = int(random.choice(list(self.observed_set), 1)[0])
        print("normal stage will start from seed", self.prev_seed)

        #
        # counter = 0
        # while True:
        #     seed = graph_nodes[np.random.randint(n)]
        #     if seed in self.crawled_set:
        #         continue
        #     self.crawl(seed)
        #     counter += 1
        #     if counter == self.n1:
        #         break
        # print(self.observed_set)


class RandomWalkMS(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph, n1):
        super().__init__(orig_graph, n1=n1)

    def crawl(self, seed: int):
        super(RandomWalkMS, self).crawl(seed)
        self.prev_seed = int(seed)

    def next_seed(self):
        node_neighbours = self.observed_neighbors(self.prev_seed)
        if len(node_neighbours) == 0:
            node_neighbours = tuple(self.observed_set)
        return random.choice(node_neighbours, 1)[0]

    def crawl_multi_seed(self):
        super(RandomWalkMS, self).crawl_multi_seed()
        self.prev_seed = int(random.choice(tuple(self.observed_set), 1)[0])

    def crawl_budget(self, t: int, initial_seed=1):
        # print("-- adding seed {} to observed {}".format(initial_seed, self.observed_set))
        # if not (initial_seed in self.observed_set):
        self.crawl(initial_seed)
        # self.observed_set.add(initial_seed)
        # self.observed_graph.snap.AddNode(initial_seed)
        super().crawl_budget()


class RandomCrawler(Crawler):
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)

    def next_seed(self):
        return random.choice(tuple(self.observed_set))

    def crawl_budget(self, initial_seed=3):
        self.observed_set.add(initial_seed)
        self.observed_graph.snap.AddNode(initial_seed)
        super().crawl_budget()


class AvrachenkovCrawler(MultiSeedCrawler, ABC):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """

    def __init__(self, orig_graph, n1=500, k=100):
        super().__init__(orig_graph, n1=n1)
        # assert k <= b-n1
        self.n1 = n1
        self.k = k

    def first_step(self):
        super().crawl_multi_seed()

    def second_step(self):
        observed_only = self.observed_set

        # Get n2 max degree observed nodes
        obs_deg = []
        g = self.observed_graph.snap
        for o_id in observed_only:
            deg = g.GetNI(o_id).GetDeg()
            obs_deg.append((o_id, deg))

        max_degs = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.budget_left - self.n1]

        # Crawl chosen nodes
        [self.crawl(node) for node, _ in max_degs]

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
    graph = MyGraph.new_snap(name='test', directed=False)
    graph._snap_graph = test_g
    print(graph.snap.GetNodes())
    # crawler = Crawler(graph)
    # for i in range(1, 6):
    #     crawler.crawl(i)
    #     print("crawled:%s, observed:%s, all:%s" %
    #           (crawler.crawled_set, crawler.observed_set, crawler.nodes_set))

    # crawler = RandomCrawler(graph)
    budget = 10

    crawler = RandomWalkMS(graph, n1=3)
    crawler.budget_left = budget
    crawler.crawl_multi_seed()
    print("after first: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
          " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))
    print("normal crawling")

    initial_seed = crawler.prev_seed
    crawler.crawl_budget(budget, initial_seed)

    print("after second: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
          " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))

    print("Total iterations:", len(crawler.seed_sequence))
    print("sequence of seeds:", crawler.seed_sequence)
