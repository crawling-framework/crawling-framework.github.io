import logging
import random
from queue import Queue, deque

import snap
from sortedcontainers import SortedKeyList

from graph_io import MyGraph


class Crawler(object):

    def __init__(self, graph: MyGraph, name=None):
        # original graph
        self.orig_graph = graph
        # observed graph
        self.observed_graph = MyGraph.new_snap(directed=graph.directed, weighted=graph.weighted)
        # crawled ids set
        self.crawled_set = set()
        # observed ids set excluding crawled ones
        self.observed_set = set()
        self.name = name if name is not None else type(self).__name__

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_graph.snap.Nodes()])

    def crawl(self, seed: int) -> bool:
        """
        Crawl specified nodes. The observed graph is updated, also crawled and observed set.
        :param seed: node id to crawl
        :return: whether the node was crawled
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            return False  # if already crawled - do nothing

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
        raise CrawlerError("Not implemented")

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
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))


class CrawlerError(Exception):
    pass


class RandomCrawler(Crawler):
    def __init__(self, graph: MyGraph, initial_seed=None):
        """
        :param initial_seed: the node to start crawling from, by default a random graph node will be used
        """
        super().__init__(graph, name='RC_')
        if initial_seed is None:
            initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
        self.observed_set.add(initial_seed)
        self.observed_graph.snap.AddNode(initial_seed)

    def next_seed(self):
        if len(self.observed_set) == 0:
            raise CrawlerError("Can't get next seed: no more observed nodes.")
        return random.choice(tuple(self.observed_set))


class MaximumObservedDegreeCrawler(Crawler):
    def __init__(self, orig_graph: MyGraph, batch=1, initial_seed=None):
        """
        :param batch: batch size
        :param initial_seed: the node to start crawling from, by default a random graph node will be used
        """
        super().__init__(orig_graph, name='MOD%s' % (batch if batch > 1 else ''))
        if initial_seed is None:  # fixme duplicate code in all basic crawlers?
            initial_seed = random.choice([n.GetId() for n in self.orig_graph.snap.Nodes()])
        self.observed_set.add(initial_seed)
        self.observed_graph.snap.AddNode(initial_seed)

        self.batch = batch
        self.mod_queue = deque()
        self.observed_skl = SortedKeyList(
            self.observed_set, key=lambda node: self.observed_graph.snap.GetNI(node).GetDeg())

    def crawl(self, seed: int) -> bool:
        """ Crawl specified node and update observed SortedKeyList
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            return False  # if already crawled - do nothing

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
            g.AddEdge(seed, n)
            if n not in self.crawled_set:
                self.observed_skl.add(n)
        return True

    def next_seed(self):
        if len(self.mod_queue) == 0:  # making array of top-k degrees
            if len(self.observed_skl) == 0:
                assert len(self.observed_set) == 0
                raise CrawlerError("Can't get next seed: no more observed nodes.")
            self.mod_queue = deque(self.observed_skl[-self.batch:])
            logging.debug("MOD queue: %s" % self.mod_queue)
        return self.mod_queue.pop()


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
