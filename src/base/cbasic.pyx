import logging
import random

from libcpp.set cimport set as cset
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.deque cimport deque
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec, address as addr

from cgraph cimport CGraph, str_to_chars
cimport cbasic
from node_deg_set cimport ND_Set  # FIXME try 'as ND_Set' if error 'ND_Set is not a type identifier'

logger = logging.getLogger(__name__)


cdef class CCrawler:
    def __init__(self, CGraph graph, name=None, **kwargs):
        # original graph
        self._orig_graph = graph  # FIXME conversion here?

        # observed graph
        self._observed_graph = CGraph(directed=self._orig_graph.directed, weighted=self._orig_graph.weighted)

        self._crawled_set = new cset[int]()
        self._observed_set = new cset[int]()

        # self.seed_sequence_ = []  # D: sequence of tries to add nodes to draw history and debug
        name = name if name is not None else type(self).__name__
        self._name = str_to_chars(name)

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set(n for n in self._observed_graph.iter_nodes())

    @property
    def crawled_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return deref(self._crawled_set)

    cdef set_crawled_set(self, cset[int]* new_crawled_set):
        self._crawled_set = new_crawled_set

    cdef set_observed_set(self, cset[int]* new_observed_set):
        self._observed_set = new_observed_set

    cdef set_observed_graph(self, CGraph new_observed_graph):
        self._observed_graph = new_observed_graph

    @property
    def observed_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return deref(self._observed_set)

    @property
    def name(self):
        return bytes.decode(self._name)

    cpdef bint observe(self, int node):
        """ Add the node to observed set and observed graph. """
        cdef bint already = self._observed_graph.has_node(node)
        if not already:
            self._observed_graph.add_node(node)
            self._observed_set.insert(node)
        return already

    # FIXME may be reference to vector?
    cdef vector[int] crawl(self, int seed) except *:
        """ Crawl specified node. The observed graph is updated, also crawled and observed set.

        :param seed: node id to crawl
        :return: pointer to a set of updated nodes
        """
        # seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        cdef vector[int] res
        # cdef cset[int] c = &self._crawled_set
        if self._crawled_set.find(seed) != self._crawled_set.end():  # FIXME simplify!
            logger.error("Already crawled: %s" % seed)
            return res  # if already crawled - do nothing

        # self.seed_sequence_.append(seed)
        self._crawled_set.insert(seed)

        # cdef cset[int]* o = self._observed_set
        # cdef CGraph* g = self._observed_graph
        if self._observed_graph.has_node(seed):  # remove from observed set
            assert self._observed_set.find(seed) != self._observed_set.end()
            self._observed_set.erase(seed)
        else:  # add to observed graph
            self._observed_graph.add_node(seed)

        # iterate over neighbours
        for n in self._orig_graph.neighbors(seed):
            if not self._observed_graph.has_node(n):  # add to observed graph and observed set
                self._observed_graph.add_node(n)
                self._observed_set.insert(n)
                res.push_back(n)
            self._observed_graph.add_edge(seed, n)
        return res

    cpdef int next_seed(self) except -1:
        """
        Core of the crawler - the algorithm to choose the next seed to be crawled.
        Seed must be a node of the original graph.

        :return: node id as int
        """
        # raise CrawlerException("Not implemented") TODO
        raise NotImplementedError()

    cpdef int crawl_budget(self, int budget) except -1:
        """
        Perform `budget` number of crawls according to the algorithm.
        Note that `next_seed()` may be called more times - some returned seeds may not be crawled.

        :param budget: so many nodes will be crawled. If can't crawl any more, raise CrawlerError
        :param args: customizable additional args for subclasses
        :return:
        """
        for i in range(budget):
            self.crawl(self.next_seed())
        return 0  # for exception compatibility


class CrawlerException(Exception):
    pass


class NoNextSeedError(CrawlerException):
    """ Can't get next seed: no more observed nodes."""
    def __init__(self, error_msg=None):
        super().__init__(self)
        self.error_msg = error_msg if error_msg else "Can't get next seed: no more observed nodes."

    def __str__(self):
        return self.error_msg


cdef inline int random_from_iterable(const cset[int]* an_iterable):
    """ Works in O(n) """
    cdef cset[int].iterator it = an_iterable.begin()
    for _ in range(random.randint(0, an_iterable.size()-1)):
        inc(it)
    return deref(it)


cdef class RandomCrawler(CCrawler):
    def __init__(self, CGraph graph, int initial_seed=-1, name=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name=name if name else 'RC_', **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

    cpdef int next_seed(self) except -1:
        if self._observed_set.size() == 0:
            raise NoNextSeedError()
        return random_from_iterable(self._observed_set)
        # return random.choice([n for n in self._observed_set])

    cdef vector[int] crawl(self, int seed):
        return CCrawler.crawl(self, seed)


cdef class RandomWalkCrawler(CCrawler):
    cdef int initial_seed
    cdef int prev_seed

    def __init__(self, CGraph graph, int initial_seed=-1, name=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name='RW_', **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # is not a duplicate code
                initial_seed = self._orig_graph.random_node()
            self.initial_seed = initial_seed
            self.observe(initial_seed)
        else:
            if initial_seed == -1:
                self.initial_seed = random_from_iterable(self._observed_set)

        self.prev_seed = -1

    cpdef int next_seed(self) except -1:
        if self.prev_seed == -1:  # first step
            self.prev_seed = self.initial_seed
            return self.initial_seed

        if self._observed_set.size() == 0:
            raise NoNextSeedError()

        # for walking we need to step on already crawled nodes too
        if self._observed_graph.deg(self.prev_seed) == 0:
            raise NoNextSeedError("No neighbours to go next.")
            # node_neighbours = tuple(self.observed_set)

        # Go to a neighbor until encounter not crawled node
        cdef int seed
        while True:
            seed = self._observed_graph.random_neighbor(self.prev_seed)
            self.prev_seed = seed
            if self._observed_set.find(seed) != self._observed_set.end():
                return seed


cdef class BreadthFirstSearchCrawler(CCrawler):
    cdef queue[int] bfs_queue

    def __init__(self, CGraph graph, int initial_seed=-1, name=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name=name if name else 'BFS', **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        cdef int n
        for n in deref(self._observed_set):
            self.bfs_queue.push(n)

    cpdef int next_seed(self) except -1:
        cdef int seed
        while self.bfs_queue.size() > 0:
            seed = self.bfs_queue.front()
            self.bfs_queue.pop()  # DFS - from back, BFS - from front
            if self._crawled_set.find(seed) == self._crawled_set.end():  # FIXME simplify?
                return seed

        assert self._observed_set.size() == 0
        raise NoNextSeedError()

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = CCrawler.crawl(self, seed)
        for n in res:
            self.bfs_queue.push(n)
        return res


cdef class DepthFirstSearchCrawler(CCrawler):
    cdef deque[int] dfs_queue

    def __init__(self, graph: CGraph, int initial_seed=-1, name=None, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name=name if name else 'DFS', **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        cdef int n
        for n in deref(self._observed_set):
            self.dfs_queue.push_back(n)

    cpdef int next_seed(self) except -1:
        cdef int seed
        while self.dfs_queue.size() > 0:
            seed = self.dfs_queue.back()
            self.dfs_queue.pop_back()  # DFS - from back, BFS - from front
            if self._crawled_set.find(seed) == self._crawled_set.end():  # FIXME simplify?
                return seed

        assert self._observed_set.size() == 0
        raise NoNextSeedError()

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = CCrawler.crawl(self, seed)
        for n in res:
            self.dfs_queue.push_back(n)
        return res


cdef class SnowBallCrawler(CCrawler):
    cdef float p
    cdef deque[int] sbs_queue
    cdef deque[int] sbs_backlog

    def __init__(self, CGraph graph, float p=0.5, int initial_seed=-1, name=None, **kwargs):
        """
        Every step of BFS taking neighbors with probability p.
        http://www.soundarajan.org/papers/CrawlingAnalysis.pdf
        https://arxiv.org/pdf/1004.1729.pdf
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        :param p: probability of taking neighbor into queue
        """
        super().__init__(graph, name=name if name else 'SB_%s' % (int(p * 100) if p != 0.5 else ''), **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        self.p = p
        cdef int n
        for n in deref(self._observed_set):
            self.sbs_queue.push_back(n)

    cpdef int next_seed(self) except -1:
        cdef int seed
        while self.sbs_queue.size() > 0:
            seed = self.sbs_queue.front()
            self.sbs_queue.pop_front()  # like BFS - from front
            if self._crawled_set.find(seed) == self._crawled_set.end():  # FIXME simplify?
                return seed

        while self.sbs_backlog.size() > 0:
            seed = self.sbs_backlog.back()
            self.sbs_backlog.pop_back()  # like DFS - from back
            if self._crawled_set.find(seed) == self._crawled_set.end():  # FIXME simplify?
                return seed

        assert self._observed_set.size() == 0
        raise NoNextSeedError()

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = CCrawler.crawl(self, seed)
        for n in res:
            if random.random() < self.p:
                self.sbs_queue.push_back(n)
            else:
                self.sbs_backlog.push_back(n)
        return res


cdef class CCrawlerUpdatable(CCrawler):
    cpdef void update(self, vector[int] nodes):  # FIXME maybe ref?
        """ Update inner structures, knowing that the specified nodes have changed their degrees.
        """
        raise NotImplementedError()


cdef class MaximumObservedDegreeCrawler(CCrawlerUpdatable):
    cdef int batch
    cdef ND_Set nd_set
    cdef cset[int] mod_set  # FIXME python set would be faster, but how to?

    def __init__(self, CGraph graph, int batch=1, int initial_seed=-1, name=None, **kwargs):
        """
        :param batch: batch size
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. If None is given, a random node of original graph will be used.
        """
        super().__init__(graph, name=name if name else ("MOD-%s" %  batch if batch > 1 else 'MOD'), **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        self.batch = batch
        # self.mod_set = set()  # we need to create, pop and lookup nodes
        cdef int n, d
        self.nd_set = ND_Set()
        for n in deref(self._observed_set):
            self.nd_set.add(n, self._observed_graph.deg(n))

    cpdef void update(self, vector[int] nodes):  # FIXME maybe ref faster?
        """ Update priority structures with specified nodes (suggested their degrees have changed).
        """
        cdef int d, n
        for n in nodes:
            # assert n in self._observed_set and n not in self.mod_queue  # Just for debugging
            # if n in self.mod_set:  # already in batch
            if self.mod_set.find(n) != self.mod_set.end():  # already in batch
                continue
            d = self._observed_graph.deg(n)
            # logger.debug("%s.ND_Set.updating(%s, %s)" % (self.name, n, d))
            self.nd_set.update_1(n, d - 1)

    cpdef vector[int] crawl(self, int seed):
        """ Crawl specified node and update observed ND_Set
        """
        cdef vector[int] res = CCrawler.crawl(self, seed)
        cdef vector[int] upd
        cdef int n
        for n in self._observed_graph.neighbors(seed):
            # if n in self._observed_set:
            if self._observed_set.find(n) != self._observed_set.end():
                upd.push_back(n)
        self.update(upd)
        return res

    cpdef int next_seed(self) except -1:
        """ Next node with highest degree
        """
        cdef int n
        cdef vector[int] vec
        if self.mod_set.size() == 0:  # making array of top-k degrees
            if len(self.nd_set) == 0:
                assert self._observed_set.size() == 0
                raise NoNextSeedError()
            vec = self.nd_set.top(self.batch)
            for n in vec:
                self.mod_set.insert(n)  # TODO could be simplified if nd_set and self.mod_set the same type
            # logger.debug("%s.queue: %s" % (self.name, self.mod_set))
        # return self.mod_set.pop()
        cdef cset[int].iterator it = dec(self.mod_set.end())  # TODO simplify?
        n = deref(it)
        self.mod_set.erase(it)
        return n

from time import time
timer = 0

cdef class PreferentialObservedDegreeCrawler(CCrawlerUpdatable):
    cdef int batch
    cdef ND_Set nd_set
    cdef cset[int] pod_set  # FIXME python set would be faster, but how to?

    def __init__(self, CGraph graph, int batch=1, int initial_seed=-1, name=None, **kwargs):
        super().__init__(graph, name=name if name else ("POD-%s" %  batch if batch > 1 else 'POD'), **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        self.batch = batch
        cdef int n, d
        self.nd_set = ND_Set()
        for n in deref(self._observed_set):
            self.nd_set.add(n, self._observed_graph.deg(n))

    cpdef void update(self, vector[int] nodes):  # FIXME maybe ref faster?
        """ Update priority structures with specified nodes (suggested their degrees have changed).
        """
        cdef int d, n
        for n in nodes:
            # assert n in self._observed_set and n not in self.mod_queue  # Just for debugging
            # if n in self.mod_set:  # already in batch
            if self.pod_set.find(n) != self.pod_set.end():  # already in batch
                continue
            d = self._observed_graph.deg(n)
            # logger.debug("%s.ND_Set.updating(%s, %s)" % (self.name, n, d))
            self.nd_set.update_1(n, d - 1)

    cpdef vector[int] crawl(self, int seed):
        """ Crawl specified node and update observed ND_Set
        """
        cdef vector[int] res = CCrawler.crawl(self, seed)
        cdef vector[int] upd
        cdef int n
        for n in self._observed_graph.neighbors(seed):
            # if n in self._observed_set:
            if self._observed_set.find(n) != self._observed_set.end():
                upd.push_back(n)
        self.update(upd)
        return res

    cpdef int next_seed(self) except -1:
        cdef int n
        if self.pod_set.size() == 0:  # when batch ends, we create another one
            if self._observed_set.size() == 0:
                raise NoNextSeedError()
            for _ in range(min(self.batch, len(self.nd_set))):
                n = self.nd_set.pop_proportional_degree()
                self.pod_set.insert(n)
        cdef cset[int].iterator it = dec(self.pod_set.end())  # TODO simplify?
        n = deref(it)
        self.pod_set.erase(it)
        return n


# --------------------------------------

# cdef extern from "int_iter.cpp":
#     cdef cppclass IntIter:
#         IntIter()
#         int i
#         IntIter operator++(int)


cdef cset[int]* f(int a):
    cdef cset[int] res
    res.insert(10 + a)
    return &res


cpdef cbasic_test():
    # print("cbasic")

    # cgraph = GraphCollections.cget('dolphins')
    # # cgraph = GraphCollections.cget('petster-hamster')
    #
    # crawler = CCrawler(cgraph)
    # for n in cgraph.iter_nodes():
    #     print(n)
    #     crawler.crawl(n)
    #     print("crawled_set:", crawler.crawled_set)
    #     print("observed_set:", crawler.observed_set)
    #     print("nodes_set:", crawler.nodes_set)
    #
    # crawler._observed_set.insert(0)
    # print(crawler._observed_set)

    # from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc
    # cdef IntIter ii = IntIter()
    # print(ii.i)
    # pinc(ii)
    # print(ii.i)

    # import numpy as np
    # cdef cset[int]* r = f(10)
    # cdef queue[int] q
    # a = list(range(10000)) + [1, 2, 3]
    # np.random.shuffle(a)
    #
    # for x in a:
    #     q.push(x)
    #     print("push", x)
    # # print(q)

    cdef cset[int] cy_set
    py_set = set()
    cdef int i, k
    # for i in range(1000000):
    #     cy_set.insert(i)
    #     py_set.add(i)

    # from time import time
    # cdef int a = 0
    #
    # # 2
    # t = time()
    # for _ in range(100000):
    #     py_set.clear()
    #     for k in range(10):
    #         py_set.add(k)
    #         # if k in py_set:
    #         #     a = 0
    # print(a)
    # print("py for %s ms" % ((time()-t)*1000))
    #
    #
    # a = 0
    # # 1
    # t = time()
    # cdef cset[int].iterator it = cy_set.end()
    # for _ in range(100000):
    #     cy_set.clear()
    #     for k in range(10):
    #         cy_set.insert(k)
    #
    # print(a)
    # print("cy for %s ms" % ((time()-t)*1000))

    cdef ND_Set nd = ND_Set()
    print(nd)
    # cdef CGraph g
