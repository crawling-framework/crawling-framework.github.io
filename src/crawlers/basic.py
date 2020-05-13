import heapq
import logging
import random
from queue import deque  # here was a warning in pycharm

import numpy as np
import snap
from scipy import stats
from sortedcontainers import SortedKeyList

from graph_io import MyGraph


class Crawler(object):

    def __init__(self, graph: MyGraph, name=None, **kwargs):
        # original graph
        self.orig_graph = graph

        # observed graph
        if 'observed_graph' in kwargs:
            self.observed_graph = kwargs['observed_graph']
        else:
            self.observed_graph = MyGraph.new_snap(directed=graph.directed, weighted=graph.weighted)

        # crawled ids set
        if 'crawled_set' in kwargs:
            self.crawled_set = kwargs['crawled_set']
        else:
            self.crawled_set = set()

        # observed ids set excluding crawled ones
        if 'observed_set' in kwargs:
            self.observed_set = kwargs['observed_set']
        else:
            self.observed_set = set()

        self.seed_sequence_ = []  # D: sequence of tries to add nodes to draw history and debug
        self.name = name if name is not None else type(self).__name__

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_graph.snap.Nodes()])

    def crawl(self, seed: int) -> bool:
        """
        Crawl specified node. The observed graph is updated, also crawled and observed set.
        :param seed: node id to crawl
        :return: whether the node was crawled
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            logging.debug("Already crawled: %s" % seed)
            return False  # if already crawled - do nothing

        self.seed_sequence_.append(seed)
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

    def next_seed(self) -> int:
        """
        Core of the crawler - the algorithm to choose the next seed to be crawled.
        Seed must be a node of the original graph.

        :return: node id as int
        """
        raise CrawlerException("Not implemented")

    def crawl_budget(self, budget: int, *args):
        """
        Perform `budget` number of crawls according to the algorithm.
        Note that `next_seed()` may be called more times - some returned seeds may not be crawled.

        :param budget: so many nodes will be crawled. If can't crawl any more, raise CrawlerError
        :param args: customizable additional args for subclasses
        :return:
        """
        for _ in range(budget):
            while not self.crawl(self.next_seed()):
                continue


class CrawlerException(Exception):
    pass


class NoNextSeedError(CrawlerException):
    """ Can't get next seed: no more observed nodes."""
    def __init__(self, error_msg=None):
        super().__init__(self)
        self.error_msg = error_msg if error_msg else "Can't get next seed: no more observed nodes."

    def __str__(self):
        return self.error_msg


class RandomCrawler(Crawler):
    def __init__(self, graph: MyGraph, initial_seed=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name='RC_', **kwargs)

        if len(self.observed_set) == 0:
            if initial_seed is None:  # FIXME duplicate code in all basic crawlers?
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.observed_set.add(initial_seed)
            self.observed_graph.snap.AddNode(initial_seed)

    def next_seed(self):
        if len(self.observed_set) == 0:
            raise NoNextSeedError()
        return random.choice(tuple(self.observed_set))


class RandomWalkCrawler(Crawler):
    def __init__(self, graph: MyGraph, initial_seed=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name='RW_', **kwargs)

        if len(self.observed_set) == 0:
            if initial_seed is None:  # is not a duplicate code
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.initial_seed = initial_seed

        self.prev_seed = None

    def next_seed(self):
        if not self.prev_seed:  # first step
            self.prev_seed = self.initial_seed
            return self.initial_seed

        node_neighbours = self.observed_graph.neighbors(self.prev_seed)
        # for walking we need to step on already crawled nodes too
        if len(node_neighbours) == 0:
            raise NoNextSeedError("No neighbours to go next.")
            # node_neighbours = tuple(self.observed_set)

        # Since we do not check if seed is in crawled_set, many re-crawls will occur
        seed = (random.choice(node_neighbours))
        self.prev_seed = seed
        return seed


class BreadthFirstSearchCrawler(Crawler):
    def __init__(self, graph: MyGraph, initial_seed=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name='BFS', **kwargs)

        if len(self.observed_set) == 0:
            if initial_seed is None:  # FIXME duplicate code in all basic crawlers?
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.observed_set.add(initial_seed)
            self.observed_graph.snap.AddNode(initial_seed)

        self.bfs_queue = deque(self.observed_set)  # FIXME what if its size > 1 ?

    def next_seed(self):
        while len(self.bfs_queue) > 0:
            seed = self.bfs_queue.popleft()
            if seed not in self.crawled_set:
                return seed

        assert len(self.observed_set) == 0
        raise NoNextSeedError()

    def crawl(self, seed):
        res = super().crawl(seed)
        if res:
            [self.bfs_queue.append(n) for n in self.orig_graph.neighbors(seed)
             if n in self.observed_set]
            # if n not in self.crawled_set]  # not work in multiseed
        return res


class SnowBallSampling(Crawler):
    def __init__(self, graph: MyGraph, p=0.5, initial_seed=None, **kwargs):
        """
        Every step of BFS taking neighbors with probability p.
        http://www.soundarajan.org/papers/CrawlingAnalysis.pdf
        https://arxiv.org/pdf/1004.1729.pdf
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        :param p: probability of taking neighbor into queue
        """
        super().__init__(graph, name='SBS%s' % (int(p * 100) if p != 0.5 else ''), **kwargs)
        if len(self.observed_set) == 0:
            if initial_seed is None:
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.observed_set.add(initial_seed)
            self.observed_graph.snap.AddNode(initial_seed)
        self.p = p
        self.sbs_queue = deque(self.observed_set)  # FIXME what if its size > 1 ?
        self.sbs_backlog = set()

    def next_seed(self):
        while len(self.sbs_queue) > 0:
            seed = self.sbs_queue.popleft()
            if seed not in self.crawled_set:
                return seed

        while len(self.sbs_backlog) > 0:
            seed = self.sbs_backlog.pop()
            if seed not in self.crawled_set:
                return seed

        assert len(self.observed_set) == 0
        raise NoNextSeedError()

    def crawl(self, seed):
        res = super().crawl(seed)
        if res:
            neighbors = self.orig_graph.neighbors(seed)
            binomial_map = np.random.binomial(1, p=self.p, size=len(neighbors))
            # print('seed', seed, [(i, j) for i,j in zip(neighbors, binomial_map)], self.sbs_backlog)
            [self.sbs_queue.append(n) for n in self.orig_graph.neighbors(seed)
             if (n in self.observed_set) and (binomial_map[neighbors.index(n)] == 1)]

            # to store observed nodes
            [self.sbs_backlog.add(n) for n in self.orig_graph.neighbors(seed)
             if (n in self.observed_set) and (binomial_map[neighbors.index(n)] == 0)]
            # if n not in self.crawled_set]  # not work in multiseed
        return res


class DepthFirstSearchCrawler(Crawler):
    def __init__(self, orig_graph: MyGraph, initial_seed=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(orig_graph, name='DFS', **kwargs)

        if len(self.observed_set) == 0:
            if initial_seed is None:  # FIXME duplicate code in all basic crawlers?
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.observed_set.add(initial_seed)
            self.observed_graph.snap.AddNode(initial_seed)

        self.dfs_queue = deque(self.observed_set)  # FIXME what if its size > 1 ?

    def next_seed(self):
        while len(self.dfs_queue) > 0:
            seed = self.dfs_queue.pop()
            if seed not in self.crawled_set:
                return seed

        assert len(self.observed_set) == 0
        raise NoNextSeedError()

    def crawl(self, seed):
        res = super().crawl(seed)
        if res:
            [self.dfs_queue.append(n) for n in self.orig_graph.neighbors(seed)
             if n in self.observed_set]
             # if n not in self.crawled_set]  # not work in multiseed
        return res


class MaximumObservedDegreeCrawler(Crawler):
    def __init__(self, orig_graph: MyGraph, batch=1, initial_seed=None, skl_mode=False, **kwargs):
        """
        :param batch: batch size
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        :param skl_mode: if True, SortedKeyList is used and updated at each step. Use it if batch is
         small (<10). Do not use it in multiseed mode!
        """
        super().__init__(orig_graph, name='MOD%s' % (batch if batch > 1 else ''), **kwargs)

        if len(self.observed_set) == 0:
            if initial_seed is None:  # fixme duplicate code in all basic crawlers?
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.observed_set.add(initial_seed)
            self.observed_graph.snap.AddNode(initial_seed)

        self.batch = batch
        self.mod_queue = deque()

        if skl_mode:
            self.observed_skl = SortedKeyList(
                self.observed_set, key=lambda node: (self.observed_graph.snap.GetNI(node).GetDeg(), node))
            self.crawl = self.skl_crawl
            self.next_seed = self.skl_next_seed

    def skl_crawl(self, seed: int) -> bool:
        """ Crawl specified node and update observed SortedKeyList
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            return False  # if already crawled - do nothing
        self.seed_sequence_.append(seed)
        self.crawled_set.add(seed)
        g = self.observed_graph.snap
        if g.IsNode(seed):  # remove from observed set
            self.observed_set.remove(seed)
            self.observed_skl.discard(seed)
        else:  # add to observed graph
            g.AddNode(seed)

        # iterate over neighbours
        for n in self.orig_graph.neighbors(seed):
            n = int(n)
            if not g.IsNode(n):  # add to observed graph and observed set
                g.AddNode(n)
                self.observed_set.add(n)
            self.observed_skl.discard(n)
            g.AddEdge(seed, n)  # this is why we can't make it via super().crawl
            if n not in self.crawled_set:
                self.observed_skl.add(n)
        return True

    def skl_next_seed(self):
        """ Next node is taken as SortedKeyList top
        """
        if len(self.mod_queue) == 0:  # making array of top-k degrees
            if len(self.observed_skl) == 0:
                assert len(self.observed_set) == 0
                raise NoNextSeedError()
            self.mod_queue = deque(self.observed_skl[-self.batch:])
            logging.debug("MOD queue: %s" % self.mod_queue)
        return self.mod_queue.pop()

    def next_seed(self):
        """ Next node is taken by sorting degrees for the observed set
        """
        if len(self.mod_queue) == 0:  # making array of topk degrees
            if len(self.observed_set) == 0:
                raise NoNextSeedError()
            deg_dict = {node: self.observed_graph.snap.GetNI(node).GetDeg()
                        for node in self.observed_set}

            heap = [(-value, key) for key, value in deg_dict.items()]
            min_iter = min(self.batch, len(deg_dict))
            self.mod_queue = [heapq.nsmallest(self.batch, heap)[i][1] for i in range(min_iter)]  # FIXME check!
            logging.debug("MOD queue: %s" % self.mod_queue)
        return self.mod_queue.pop(0)


class PreferentialObservedDegreeCrawler(Crawler):
    def __init__(self, orig_graph: MyGraph, batch=10, initial_seed=None, **kwargs):
        super().__init__(orig_graph, name='POD%s' % (batch if batch > 1 else ''), **kwargs)

        if len(self.observed_set) == 0:
            if initial_seed is None:  # fixme duplicate code in all basic crawlers?
                initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
            self.observed_set.add(initial_seed)
            self.observed_graph.snap.AddNode(initial_seed)

        self.discrete_distribution = None
        self.pod_queue = [initial_seed]  # queue of nodes to proceed in batch
        self.batch = batch  # crawling by batches if > 1 #

    def next_seed(self):
        if len(self.pod_queue) == 0:  # when batch ends, we create another one
            prob_func = {node.GetId(): node.GetDeg() for node in self.observed_graph.snap.Nodes()
                         if node.GetId() not in self.crawled_set}
            if len(prob_func) == 0:
                return None
            keys, values = zip(*prob_func.items())
            values = np.array(values) / sum(values)
            # print(keys, values)
            self.discrete_distribution = stats.rv_discrete(values=(keys, values))
            self.pod_queue = [node for node in self.discrete_distribution.rvs(size=self.batch)]

        # print("node degrees (probabilities)", prob_func)
        return self.pod_queue.pop(0)


# class ForestFireCrawler(BreadthFirstSearchCrawler):  # TODO need testing and debug different p
#     """Algorythm from https://dl.acm.org/doi/abs/10.1145/1081870.1081893
#     with my little modification - stuck_ends, it is like illegitimate son of BFS and RC
#     :param p - forward burning probability of algorythm
#     :param stuck_ends - if true, finishes when queue is empty, otherwise crawl random from observed
#     """
#
#     def __init__(self, orig_graph: MyGraph, p=0.35, initial_seed=None, **kwargs):
#         super().__init__(orig_graph, **kwargs)
#         self.name = 'FFC_p=%s' % p  # unless doesnt work because BFS (super) has own name
#         if len(self.observed_set) == 0:
#             if initial_seed is None:  # fixme duplicate code in all basic crawlers?
#                 initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
#             self.observed_set.add(initial_seed)
#             self.observed_graph.snap.AddNode(initial_seed)
#
#         self.bfs_queue = [initial_seed]
#         self.p = p
#
#     # next_seed is the same with BFS, just choosing ambassador node w=seed, except empty queue
#     # def next_seed(self):
#     #     while self.bfs_queue[0] not in self.observed_set:
#     #         self.bfs_queue.pop(0)
#     #         if len(self.bfs_queue) == 0:  # if we get stucked, choosing random from observed
#     #             return int(np.random.choice(tuple(self.observed_set)))
#     #     return self.bfs_queue[0]
#
#     def crawl(self, seed):
#         degree = self.orig_graph.snap.GetNI(seed).GetDeg()
#         # computing x - number of friends to add
#         # print("seed", seed, self.p, degree, (1 - self.p) ** (-1) / degree )
#         # in paper (1-p)**(-1) == E == degree * bin_prob
#         x = np.random.binomial(degree, self.p)  # (1 - self.p) ** (-1) / degree)
#         x = max(1, min(x, len(self.orig_graph.neighbors(seed))))
#         intersection = [n for n in self.orig_graph.neighbors(seed)]
#         burning = [int(n) for n in random.sample(intersection, x)]
#         # print("FF: queue:{},obs:{},crawl:{},x1={},burn={}".format(self.bfs_queue, self.observed_set,self.crawled_set,x, burning))
#         for node in burning:
#             self.bfs_queue.append(node)
#
#         return super(BreadthFirstSearchCrawler, self).crawl(seed)


def test_crawlers():
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
    graph = MyGraph.new_snap(g, name='test', directed=False)

    # crawler = Crawler(graph)
    # for i in range(1, 6):
    #     crawler.crawl(i)
    #     print("crawled:%s, observed:%s, all:%s" %
    #           (crawler.crawled_set, crawler.observed_set, crawler.nodes_set))

    crawler = RandomCrawler(graph)
    crawler.crawl_budget(15)


if __name__ == '__main__':
    test_crawlers()
