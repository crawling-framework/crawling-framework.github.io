import logging
from abc import ABC

import numpy as np

from crawlers.basic import Crawler, NoNextSeedError, MaximumObservedDegreeCrawler, CrawlerUpdatable
from graph_io import MyGraph

logger = logging.getLogger(__name__)


class MultiCrawler(Crawler):
    """
    Runs several crawlers in parallel. Each crawler makes a step iteratively in a cycle.
    When the crawler can't get next seed it is discarded from the cycle.
    """
    def __init__(self, graph: MyGraph, crawlers, **kwargs):
        """
        :param crawlers: crawler instances to run in parallel
        """
        super().__init__(graph, name='Multi_%s' % ";".join([c.name for c in crawlers[:15]]),
                         **kwargs)  # taking only first 15 into name
        # assert len(crawlers) > 1
        self.crawlers = crawlers
        self.keep_node_owners = False  # True if any crawler is MOD or POD
        self.node_owner = {}  # node -> index of crawler who owns it. Need for MOD, POD

        # Merge observed graph and crawled set for all crawlers
        g = self.observed_graph.snap
        c = self.crawled_set
        o = self._observed_set
        for index, crawler in enumerate(crawlers):
            assert crawler.orig_graph == self.orig_graph
            if isinstance(crawler, CrawlerUpdatable):
                self.keep_node_owners = True

            # merge observed graph
            for n in crawler.observed_graph.snap.Nodes():
                n = n.GetId()
                if not g.IsNode(n):
                    g.AddNode(n)
            for edge in crawler.observed_graph.snap.Edges():
                i, j = edge.GetSrcNId(), edge.GetDstNId()
                if not g.IsEdge(i, j):
                    g.AddEdge(i, j)

            # merge crawled_set and observed_set
            c = c.union(crawler.crawled_set)
            assert o.isdisjoint(crawler.observed_set)  # making sure observed_sets are individual FIXME this is for debug, remove it to speedup
            o = o.union(crawler.observed_set)
            for n in crawler.observed_set:
                self.node_owner[n] = index

        self.crawled_set = c
        self._observed_set = o
        for crawler in crawlers:
            crawler.observed_graph = self.observed_graph
            crawler.crawled_set = self.crawled_set
            # crawler.observed_set is individual

        self.next_crawler = 0  # next crawler index to run

    @property
    def observed_set(self):
        return self._observed_set

    def crawl(self, seed: int) -> set:
        """ Run the next crawler.
        """
        if seed == 58:
            pass
        res = self.crawlers[self.next_crawler].crawl(seed)
        logger.debug("res of crawler[%s]: %s" % (self.next_crawler, res))
        assert seed in self.crawled_set
        self._observed_set.remove(seed)  # removed crawled node FIXME potentially error if node was already removed
        self._observed_set.update(res)  # add newly observed nodes

        # for c in self.crawlers:
        #     logger.debug("%s.ND_Set: %s" % (c.name, c.observed_skl))

        if self.keep_node_owners:
            # update owners dict
            del self.node_owner[seed]
            for n in res:
                self.node_owner[n] = self.crawlers[self.next_crawler]

            # distribute nodes with changed degree among instances to update their priority structures
            for n in self.observed_graph.neighbors(seed):
                if n in self.node_owner:
                    c = self.node_owner[n]
                    if c != self.crawlers[self.next_crawler] and isinstance(c, CrawlerUpdatable):
                        c.update([n])

        self.next_crawler = (self.next_crawler+1) % len(self.crawlers)
        self.seed_sequence_.append(seed)
        return res

    def next_seed(self) -> int:
        """ The next crawler makes a step. If impossible, it is discarded.
        """
        for _ in range(len(self.crawlers)):
            try:
                s = self.crawlers[self.next_crawler].next_seed()
            except NoNextSeedError as e:
                logger.debug("Run crawler[%s]: %s Removing it." % (self.next_crawler, e))
                del self.crawlers[self.next_crawler]
                # idea - create a new instance
                self.next_crawler = self.next_crawler % len(self.crawlers)
                continue

            logger.debug("Crawler[%s] next seed=%s" % (self.next_crawler, s))
            return s

        raise NoNextSeedError("None of %s subcrawlers can get next seed." % len(self.crawlers))


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
        print("observed set after multiseed", list(self._observed_set))

    def crawl(self, seed):
        """
        Crawls given seed
        Decrease self.budget_left only if crawl is successful
        """
        # http://conferences.sigcomm.org/imc/2010/papers/p390.pdf
        # TODO implement frontier choosing of component (one from m)
        raise NotImplementedError()
        # FIXME res is set now
        self.seed_sequence_.append(seed)  # updates every TRY of crawling
        if super().crawl(seed):
            self.budget_left -= 1

            logging.debug("{}.-- seed {}, crawled #{}: {}, observed #{},".format(len(self.seed_sequence_), seed,
                                                                                 len(self.crawled_set),
                                                                                 self.crawled_set,
                                                                                 len(self._observed_set),
                                                                                 self._observed_set))
            return True
        else:
            logging.debug("{}. budget ={}, seed = {}, in crawled set={}, observed ={}".format(len(self.seed_sequence_),
                                                                                              self.budget_left, seed,
                                                                                              self.crawled_set,
                                                                                              self._observed_set))
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

        while (self.budget_left > 0) and (len(self._observed_set) > 0) \
                and (self.observed_graph.snap.GetNodes() <= self.orig_graph.snap.GetNodes()):
            seed = self.next_seed()
            self.crawl(seed)

            # if file:
            logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
                          (seed, self.crawled_set, self._observed_set, self.nodes_set))


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
                # g.AddEdgeUnchecked(node - 1, node)
            if node > n - 1:
                g.AddEdgeUnchecked(node, node - n)
                # g.AddEdgeUnchecked(node - n, node)

            pos[node] = [float(k / n), float(i / m)]
    graph = MyGraph.new_snap(g, name='test', directed=False)
    return [graph, pos]


if __name__ == '__main__':
    test()
