import logging
from abc import ABC

import numpy as np

from utils import USE_CYTHON_CRAWLERS

if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
    from crawlers.cbasic import CCrawler as Crawler, CCrawlerUpdatable as CrawlerUpdatable, NoNextSeedError,\
        MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler, definition_to_filename
else:
    from base.graph import MyGraph
    from crawlers.basic import Crawler, CrawlerUpdatable, NoNextSeedError, PreferentialObservedDegreeCrawler, MaximumObservedDegreeCrawler

logger = logging.getLogger(__name__)


# class MultiSeedCrawler(Crawler, ABC):
#     """
#     great class to Avrachenkov and other crawlers starting with n1 seeds
#     """
#
#     def __init__(self, graph: MyGraph):
#         super().__init__(graph)
#         # assert n1 <= self.budget_left <= self.orig_graph.nodes()
#         # assert k <= self.budget_left - n1
#         # self.n1 = n1  # n1 seeds crawled on first steps, then comes crawler
#         self.seed_sequence_ = []  # sequence of tries to add nodes
#         self.initial_seeds = []  # store n1 seeds # TODO maybe use it not only in Random Walk (Frontier Sampling)
#         self.budget_left = 1  # how many iterations left. stops when 0
#         # print("n1={}, budget={}, nodes={}".format(self.n1, self.budget_left, self.orig_graph.nodes()))
#         self.crawler_name = ""  # will be used in names of files
#         # self.components_current_seeds
#
#     def crawl_multi_seed(self, n1):
#         if n1 <= 0:  # if there is no parallel seeds, method do nothing
#             return False
#         graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
#         # TODO do something if there are doublicates in initial seeds
#         self.initial_seeds = [int(node) for node in np.random.choice(graph_nodes, n1)]
#         for seed in self.initial_seeds:
#             self.crawl(seed)
#         print("observed set after multiseed", list(self._observed_set))
#
#     def crawl(self, seed):
#         """
#         Crawls given seed
#         Decrease self.budget_left only if crawl is successful
#         """
#         # http://conferences.sigcomm.org/imc/2010/papers/p390.pdf
#         # TODO implement frontier choosing of component (one from m)
#         raise NotImplementedError()
#         # FIXME res is set now
#         self.seed_sequence_.append(seed)  # updates every TRY of crawling
#         if super().crawl(seed):
#             self.budget_left -= 1
#
#             logging.debug("{}.-- seed {}, crawled #{}: {}, observed #{},".format(len(self.seed_sequence_), seed,
#                                                                                  len(self.crawled_set),
#                                                                                  self.crawled_set,
#                                                                                  len(self._observed_set),
#                                                                                  self._observed_set))
#             return True
#         else:
#             logging.debug("{}. budget ={}, seed = {}, in crawled set={}, observed ={}".format(len(self.seed_sequence_),
#                                                                                               self.budget_left, seed,
#                                                                                               self.crawled_set,
#                                                                                               self._observed_set))
#             return False
#
#     def crawl_budget(self, budget, p=0, file=False):
#         """
#         Crawl until done budget
#         :param p: probability to jump into one of self.initial_seed nodes  # TODO do something with it. Mb delete?
#         :param budget: how many iterations left
#         :param file: - if you need to
#         :return:
#         """
#         self.budget_left = min(budget, self.observed_graph.nodes() - 1)
#         if np.random.randint(0, 100, 1) < p * 100:  # TODO to play with this dead staff
#             print("variety play")
#             self.crawl(int(np.random.choice(self.initial_seeds, 1)[0]))
#             self.budget_left -= 1
#
#         while (self.budget_left > 0) and (len(self._observed_set) > 0) \
#                 and (self.observed_graph.nodes() <= self.orig_graph.nodes()):
#             seed = self.next_seed()
#             self.crawl(seed)
#
#             # if file:
#             logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
#                           (seed, self.crawled_set, self._observed_set, self.nodes_set))


class MultiCrawler(Crawler):
    """ General class for complex crawling strategies using several crawlers
    """
    pass


class MultiInstanceCrawler(MultiCrawler):
    """
    Runs several crawlers in parallel. Each crawler makes a step iteratively in a cycle.
    When the crawler can't get next seed it is discarded from the cycle.
    """
    short = 'MultiInstance'
    def __init__(self, graph: MyGraph, name: str=None, count: int=0, crawler_def=None):
        """
        :param graph:
        :param count: how many instances to use
        :param crawler_def: crawler instance definition as (class, kwargs)
        """
        assert crawler_def is not None
        assert count < graph.nodes()

        super().__init__(graph, name=name, count=count, crawler_def=crawler_def)

        self.crawler_def = crawler_def  # FIXME can we speed it up?
        self.crawlers = []
        self.keep_node_owners = False  # True if any crawler is MOD or POD
        self.node_owner = {}  # node -> index of crawler who owns it. Need for MOD, POD

        seeds = graph.random_nodes(count)
        [self.observe(s) for s in seeds]

        _class, kwargs = crawler_def

        # Create crawler instances and init them with different random seeds
        for i in range(count):
            crawler = _class(graph, initial_seed=seeds[i],
                             observed_graph=self._observed_graph, crawled_set=self._crawled_set,
                             observed_set={seeds[i]})
            self.crawlers.append(crawler)

            if isinstance(crawler, CrawlerUpdatable):
                n = seeds[i]
                self.node_owner[n] = crawler
                self.keep_node_owners = True

        if not name:
            self.name = 'Multi%s%s' % (count, self.crawlers[0].name)  # short name for pics
        self.next_crawler = 0  # next crawler index to run

    def crawl(self, seed: int) -> list:
        """ Run the next crawler.
        """
        c = self.crawlers[self.next_crawler]  # FIXME ref better?
        res = c.crawl(seed)
        logger.debug("res of crawler[%s]: %s" % (self.next_crawler, [n for n in res]))

        assert seed in self._crawled_set  # FIXME do we need it?
        assert seed in self._observed_set  # FIXME potentially error if node was already removed
        self._observed_set.remove(seed)  # removed crawled node
        for n in res:
            self._observed_set.add(n)  # add newly observed nodes

        if self.keep_node_owners:  # TODO can we speed it up?
            # update owners dict
            del self.node_owner[seed]
            for n in res:
                self.node_owner[n] = self.crawlers[self.next_crawler]

            # distribute nodes with changed degree among instances to update their priority structures
            for n in self._observed_graph.neighbors(seed):
                if n in self.node_owner:
                    c = self.node_owner[n]
                    if c != self.crawlers[self.next_crawler] and isinstance(c, CrawlerUpdatable):
                        c.update([n])

        self.next_crawler = (self.next_crawler+1) % len(self.crawlers)
        # self.seed_sequence_.append(seed)
        return res

    def next_seed(self) -> int:
        """ The next crawler makes a step. If impossible, it is discarded.
        """
        for _ in range(len(self.crawlers)):
            try:
                s = self.crawlers[self.next_crawler].next_seed()
            except NoNextSeedError as e:
                logger.debug("Run crawler[%s]: %s Removing it." % (self.next_crawler, e))
                # print("Run crawler[%s]: %s Removing it." % (self.next_crawler, e))
                del self.crawlers[self.next_crawler]
                # idea - create a new instance
                self.next_crawler = self.next_crawler % len(self.crawlers)
                continue

            logger.debug("Crawler[%s] next seed=%s" % (self.next_crawler, s))
            # print("Crawler[%s] next seed=%s" % (self.next_crawler, s))
            return s

        raise NoNextSeedError("None of %s subcrawlers can get next seed." % len(self.crawlers))


def test_carpet_graph(n, m):
    # special n*m graph for visual testing
    import snap
    graph = MyGraph(name='test', directed=False)
    g = graph.snap
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
    return [graph, pos]


if __name__ == '__main__':
    test_carpet_graph(10, 10)
