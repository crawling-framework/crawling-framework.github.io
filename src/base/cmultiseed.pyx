import logging

from libcpp.vector cimport vector
from libcpp.set cimport set as cset

from cbasic cimport CCrawler, CCrawlerUpdatable
from cgraph cimport CGraph
from base.cbasic import RandomCrawler, NoNextSeedError
from cython.operator cimport dereference as deref

logger = logging.getLogger(__name__)


cdef class MultiCrawler(CCrawler):
    """
    Runs several crawlers in parallel. Each crawler makes a step iteratively in a cycle.
    When the crawler can't get next seed it is discarded from the cycle.
    """
    cdef int next_crawler
    cdef bint keep_node_owners
    cdef dict __dict__  # for pythonic fields, makes it slower
    # cdef cmap[int, int] node_owners  TODO faster?
    # cdef cmap[int, CCrawler] crawlers

    def __init__(self, CGraph graph, crawlers, **kwargs):
        """
        :param crawlers: crawler instances to run in parallel
        """
        cdef CCrawler crawler
        super().__init__(graph,
                         # name='Multi',
                         name='Multi_%sx%s' % (len(crawlers), crawlers[0].name),  # FIXME we suppose all crawlers are the same class
                         **kwargs)  # taking only first 15 into name
        # assert len(crawlers) > 1
        self.crawlers = crawlers  # FIXME can we speed it up?
        self.keep_node_owners = False  # True if any crawler is MOD or POD
        self.node_owner = {}  # node -> index of crawler who owns it. Need for MOD, POD

        # Merge observed graph and crawled set for all crawlers
        cdef int index, x
        cdef cset[int]* a  # TODO ref faster?
        for index, crawler in enumerate(crawlers):
            assert crawler._orig_graph == self._orig_graph

            if isinstance(crawler, CCrawlerUpdatable):
                self.keep_node_owners = True

            # Merge observed graph FIXME seems to slow down if large graphs
            for n in crawler._observed_graph.iter_nodes():  # TODO graph.merge() ?
                if not self._observed_graph.has_node(n):
                    self._observed_graph.add_node(n)
            for i, j in crawler._observed_graph.iter_edges():
                if not self._observed_graph.has_edge(i, j):
                    self._observed_graph.add_edge(i, j)

            # Merge crawled_set and observed_set
            a = crawler._crawled_set
            self._crawled_set.insert(a.begin(), a.end())
            for x in deref(crawler._observed_set):
                # making sure observed_sets are individual FIXME this is for debug, remove it to speedup
                assert self._observed_set.find(x) == self._observed_set.end(), "Crawlers' observed sets are not disjoint!"
            a = crawler._observed_set
            self._observed_set.insert(a.begin(), a.end())
            for n in crawler._observed_set[0]:
                self.node_owner[n] = crawler

        cdef cset[int]* c = self._crawled_set
        for crawler in crawlers:
            crawler.set_observed_graph(self._observed_graph)  # FIXME use ref, otherwise
            crawler.set_crawled_set(c)
        #     # crawler.observed_set is individual

        self.next_crawler = 0  # next crawler index to run

    @property
    def observed_set(self) -> set:
        return self._observed_set[0]

    cdef vector[int] crawl(self, int seed) except *:
        """ Run the next crawler.
        """
        cdef CCrawler c = self.crawlers[self.next_crawler]  # FIXME ref better?
        cdef vector[int] res = c.crawl(seed)
        # cdef vector[int] res = self.crawlers[self.next_crawler].crawl_budget(1)
        logger.debug("res of crawler[%s]: %s" % (self.next_crawler, [n for n in res]))
        cdef vector[int] upd
        cdef int n

        assert self._crawled_set.find(seed) != self._crawled_set.end()  # FIXME do we need it?
        assert self._observed_set.find(seed) != self._observed_set.end()  # FIXME potentially error if node was already removed
        self._observed_set.erase(seed)  # removed crawled node
        for n in res:
            self._observed_set.insert(n)  # add newly observed nodes

        # for c in self.crawlers:
        #     logger.debug("%s.ND_Set: %s" % (c.name, c.observed_skl))

        if self.keep_node_owners:  # TODO can we speed it up?
            # update owners dict
            del self.node_owner[seed]
            for n in res:
                self.node_owner[n] = self.crawlers[self.next_crawler]

            # distribute nodes with changed degree among instances to update their priority structures
            for n in self._observed_graph.neighbors(seed):
                if n in self.node_owner:
                    # print(self.node_owner[n])
                    c = self.node_owner[n]
                    if c != self.crawlers[self.next_crawler] and isinstance(c, CCrawlerUpdatable):
                        c.update([n])

        self.next_crawler = (self.next_crawler+1) % len(self.crawlers)
        # self.seed_sequence_.append(seed)
        return res

    cpdef int next_seed(self) except -1:
        """ The next crawler makes a step. If impossible, it is discarded.
        """
        cdef int n, s
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


# ------------------------------------------------------------

cpdef test_multiseed():
    # cdef cset[int] a
    # print(a.size())
    # cdef cset[int]* p = &a
    # print(p.size())
    # cdef cset[int] b = deref(p)
    # print(b.size())

    from graph_io import GraphCollections
    name = 'dolphins'
    g = GraphCollections.cget(name)
    crawlers = [
        RandomCrawler(g),
        RandomCrawler(g),
        RandomCrawler(g),
    ]
    # for c in crawlers:
    #     c.crawl_budget(1)
    cdef MultiCrawler mc = MultiCrawler(g, crawlers)
    mc.crawl_budget(100)
    cdef int n
    print("c0.o", list(crawlers[0].observed_set))
    print("c1.o", list(crawlers[1].observed_set))
    print("c2.o", list(crawlers[2].observed_set))
    print("mc.o", list(mc.observed_set))
    print("mc.node_owner", mc.node_owner)
    print("mc.keep_node_owners", mc.keep_node_owners)
    # for n in mc._observed_set:
    #     print(n)
