import heapq
import logging
import random
from abc import ABC

import numpy as np
from scipy import stats

from crawlers.basic import Crawler
from graph_io import MyGraph


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
        self.initial_seeds = []  # store n1 seeds # TODO maybe use it not only in Random Walk (Frontier Sampling)
        self.budget_left = 1  # how many iterations left. stops when 0
        # print("n1={}, budget={}, nodes={}".format(self.n1, self.budget_left, self.orig_graph.snap.GetNodes()))
        self.crawler_name = ""  # will be used in names of files
        # self.components_current_seeds

    def crawl_multi_seed(self, n1):
        if n1 <= 0:  # if there is no parallel seeds, method do nothing
            return False
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        # TODO do something if there are doublicates in initial seeds
        self.initial_seeds = [int(node) for node in np.random.choice(graph_nodes, n1)]
        for seed in self.initial_seeds:
            self.crawl(seed)
        print("observed set after multiseed", list(self.observed_set))

    def crawl(self, seed):
        """
        Crawls given seed
        Decrease self.budget_left only if crawl is successful
        """
        # http://conferences.sigcomm.org/imc/2010/papers/p390.pdf
        # TODO implement frontier choosing of component (one from m)
        self.seed_sequence_.append(seed)  # updates every TRY of crawling
        if super().crawl(seed):
            self.budget_left -= 1

            logging.debug("{}.-- seed {}, crawled #{}: {}, observed #{},".format(len(self.seed_sequence_), seed,
                                                                                 len(self.crawled_set),
                                                                                 self.crawled_set,
                                                                                 len(self.observed_set),
                                                                                 self.observed_set))
            return True
        else:
            logging.debug("{}. budget ={}, seed = {}, in crawled set={}, observed ={}".format(len(self.seed_sequence_),
                                                                                              self.budget_left, seed,
                                                                                              self.crawled_set,
                                                                                              self.observed_set))
            return False

    def crawl_budget(self, budget, p=0, file=False):
        """
        Crawl until done budget
        :param p: probability to jump into one of self.initial_seed nodes  # TODO do something with it. Mb delete?
        :param budget: how many iterations left
        :param file: - if you need to
        :return:
        """
        self.budget_left = min(budget, self.observed_graph.snap.GetNodes() - 1)
        if np.random.randint(0, 100, 1) < p * 100:  # TODO to play with this dead staff
            print("variety play")
            self.crawl(int(np.random.choice(self.initial_seeds, 1)[0]))
            self.budget_left -= 1

        while (self.budget_left > 0) and (len(self.observed_set) > 0) \
                and (self.observed_graph.snap.GetNodes() <= self.orig_graph.snap.GetNodes()):
            seed = self.next_seed()
            self.crawl(seed)

            # if file:
            logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
                          (seed, self.crawled_set, self.observed_set, self.nodes_set))


# TODO inherit all following crawlers from Crawler, NOT MultiSeedCrawler, and turn back to basic.py
class RandomWalkCrawler(MultiSeedCrawler):
    """
    Normal random work if n1=1. Otherwise it is Frontier Crawling, that chooses from self.initial_seeds ~ degree
    """

    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.prev_seed = 1  # previous node, that was already crawled
        self.crawler_name = 'RW_'

    def next_seed(self):
        # step 4 from paper about Frontier Sampling.  taken from POD.next_seed
        prob_func = {node: self.observed_graph.snap.GetNI(node).GetDeg() for node in self.initial_seeds}
        keys, values = zip(*prob_func.items())
        values = np.array(values) / sum(values)
        self.discrete_distribution = stats.rv_discrete(values=(keys, values))
        self.prev_seed = [node for node in self.discrete_distribution.rvs(size=1)].pop(0)
        # print("node degrees (probabilities)", prob_func)

        # original Random Walk
        node_neighbours = self.observed_graph.neighbors(self.prev_seed)
        # for walking we need to step on already crawled nodes too
        if len(node_neighbours) == 0:
            node_neighbours = tuple(self.observed_set)
        next_seed = int(np.random.choice(node_neighbours, 1)[0])
        self.initial_seeds[self.initial_seeds.index(self.prev_seed)] = next_seed
        return next_seed

    def crawl(self, seed):
        super().crawl(seed)
        self.prev_seed = seed


class BreadthFirstSearchCrawler(MultiSeedCrawler):  # в ширину
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.bfs_queue = []
        self.crawler_name = 'BFS'

    def next_seed(self):
        while self.bfs_queue[0] not in self.observed_set:
            self.bfs_queue.pop(0)
        return self.bfs_queue[0]

    def crawl(self, seed):
        for n in self.orig_graph.neighbors(seed):
            self.bfs_queue.append(n)
        return super(BreadthFirstSearchCrawler, self).crawl(seed)


class DepthFirstSearchCrawler(MultiSeedCrawler):  # TODO
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.dfs_queue = []
        self.dfs_counter = 0
        self.crawler_name = 'DFS'

    def next_seed(self):
        while self.dfs_queue[0] not in self.observed_set:
            self.dfs_queue.pop(0)
        return self.dfs_queue[0]

    def crawl(self, seed):
        self.dfs_counter = 0
        for n in self.orig_graph.neighbors(seed):
            self.dfs_counter += 1
            self.dfs_queue.insert(self.dfs_counter, n)
        return super(DepthFirstSearchCrawler, self).crawl(seed)


class RandomCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.crawler_name = 'RC_'

    def next_seed(self):
        return int(np.random.choice(tuple(self.observed_set)))


class MaximumObservedDegreeCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph, top_k=1):
        super().__init__(orig_graph)
        self.mod_queue = []
        self.top_k = top_k  # crawling by batches if > 1 # TODO make another MOD crawler with batches
        self.crawler_name = 'MOD'

    def next_seed(self):
        if len(self.mod_queue) == 0:  # making array of topk degrees
            deg_dict = {node.GetId(): node.GetDeg() for node in self.observed_graph.snap.Nodes()
                        if node.GetId() not in self.crawled_set}

            heap = [(-value, key) for key, value in deg_dict.items()]
            min_iter = min(self.top_k, len(deg_dict))
            self.mod_queue = [heapq.nsmallest(self.top_k, heap)[i][1] for i in range(min_iter)]
        return self.mod_queue.pop(0)


class PreferentialObservedDegreeCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph, top_k=1):
        super().__init__(orig_graph)
        self.discrete_distribution = None
        self.pod_queue = []  # queue of nodes to proceed in batch
        self.top_k = top_k  # crawling by batches if > 1 #
        self.crawler_name = 'POD'

    def next_seed(self):
        if len(self.pod_queue) == 0:  # when batch ends, we create another one
            prob_func = {node.GetId(): node.GetDeg() for node in self.observed_graph.snap.Nodes()
                         if node.GetId() not in self.crawled_set}
            keys, values = zip(*prob_func.items())
            values = np.array(values) / sum(values)

            self.discrete_distribution = stats.rv_discrete(values=(keys, values))
            self.pod_queue = [node for node in self.discrete_distribution.rvs(size=self.top_k)]

        # print("node degrees (probabilities)", prob_func)
        return self.pod_queue.pop(0)


class ForestFireCrawler(BreadthFirstSearchCrawler):
    """Algorythm from https://dl.acm.org/doi/abs/10.1145/1081870.1081893
    with my little modification - stuck_ends, it is like illegitimate son of BFS and RC
    :param p - forward burning probability of algorythm
    :param stuck_ends - if true, finishes when queue is empty, otherwise crawl random from observed
    """

    def __init__(self, orig_graph: MyGraph, p=0.35, stuck_ends=False):
        super().__init__(orig_graph)
        self.crawler_name = 'FFC'
        self.p = p

    # next_seed is the same with BFS, just choosing ambassador node w=seed, except empty queue
    def next_seed(self):
        while self.bfs_queue[0] not in self.observed_set:
            self.bfs_queue.pop(0)
            if len(self.bfs_queue) == 0:  # if we get stucked, choosing random from observed
                return int(np.random.choice(tuple(self.observed_set)))
        return self.bfs_queue[0]

    def crawl(self, seed):
        degree = self.orig_graph.snap.GetNI(seed).GetDeg()
        # computing x - number of friends to add
        # print("seed", seed, self.p, degree, (1 - self.p) ** (-1) / degree )
        # in paper (1-p)**(-1) == E == degree * bin_prob
        x = np.random.binomial(degree, self.p)  # (1 - self.p) ** (-1) / degree)
        x = max(1, min(x, len(self.orig_graph.neighbors(seed))))
        intersection = [n for n in self.orig_graph.neighbors(seed)]
        burning = [int(n) for n in random.sample(intersection, x)]
        # print("FF: queue:{},obs:{},crawl:{},x1={},burn={}".format(self.bfs_queue, self.observed_set,self.crawled_set,x, burning))
        for node in burning:
            self.bfs_queue.append(node)

        return super(BreadthFirstSearchCrawler, self).crawl(seed)


def test():
    import snap
    g = snap.TUNGraph.New()

    for i in range(19):
        g.AddNode(i)

    g.AddEdge(1, 2)
    g.AddEdge(2, 3)
    g.AddEdge(4, 2)
    g.AddEdge(4, 3)
    g.AddEdge(5, 4)
    g.AddEdge(1, 6)
    g.AddEdge(6, 7)
    g.AddEdge(8, 7)
    g.AddEdge(8, 16)
    g.AddEdge(8, 9)
    g.AddEdge(8, 10)
    g.AddEdge(8, 7)
    g.AddEdge(0, 10)
    g.AddEdge(0, 9)
    g.AddEdge(1, 17)
    g.AddEdge(7, 17)
    g.AddEdge(1, 18)
    g.AddEdge(7, 18)
    g.AddEdge(11, 15)

    g.AddEdge(11, 12)
    g.AddEdge(12, 13)
    g.AddEdge(14, 12)
    g.AddEdge(14, 13)
    g.AddEdge(15, 13)
    g.AddEdge(15, 5)
    g.AddEdge(11, 16)

    # for dfs

    g.AddEdge(14, 0)
    g.AddEdge(3, 0)

    return g


def test_carpet_graph(n, m):
    # special n*m graph for visual testing
    import snap
    g = snap.TUNGraph.New()
    pos = dict()
    for i in range(0, n * m):
        g.AddNodeUnchecked(i)
    for k in range(0, n):
        for i in range(0, m):
            node = i * n + k
            if (node > 0) and (node % n != 0):
                g.AddEdgeUnchecked(node, node - 1)
                g.AddEdgeUnchecked(node - 1, node)
            if node > n - 1:
                g.AddEdgeUnchecked(node, node - n)
                g.AddEdgeUnchecked(node - n, node)

            pos[node] = [float(k / n), float(i / m)]
    graph = MyGraph.new_snap(g, name='test', directed=False)
    return [graph, pos]


if __name__ == '__main__':
    test()
