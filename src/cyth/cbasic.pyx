import logging

from libcpp.set cimport set as cset

from cgraph cimport CGraph, str_to_chars
from graph_io import GraphCollections


cdef class CCrawler:
    cdef char* _name
    cdef readonly CGraph _orig_graph
    cdef readonly CGraph _observed_graph
    cdef cset[int] _crawled_set
    cdef cset[int] _observed_set


    def __init__(self, graph: CGraph, name=None, **kwargs):
        # original graph
        self._orig_graph = graph  # FIXME conversion here

        # observed graph
        if 'observed_graph' in kwargs:
            self._observed_graph = kwargs['observed_graph']
        else:
            self._observed_graph = CGraph(directed=graph.directed, weighted=graph.weighted)

        # crawled ids set
        if 'crawled_set' in kwargs:
            self._crawled_set = kwargs['crawled_set']
        # else:
        #     self._crawled_set = ()

        # observed ids set excluding crawled ones
        if 'observed_set' in kwargs:
            self._observed_set = kwargs['observed_set']
        # else:
        #     self._observed_set = ()

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
        return self._crawled_set

    @property
    def observed_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return self._observed_set

    cpdef cset[int] crawl(self, int seed):
        """ Crawl specified node. The observed graph is updated, also crawled and observed set.

        :param seed: node id to crawl
        :return: set of updated nodes
        """
        # seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        cdef cset[int] res
        # cdef cset[int] c = &self._crawled_set
        if self._crawled_set.find(seed) != self._crawled_set.end():
            logging.error("Already crawled: %s" % seed)
            return res  # if already crawled - do nothing

        # self.seed_sequence_.append(seed)
        self._crawled_set.insert(seed)

        # cdef cset[int]* o = self._observed_set
        # cdef CGraph* g = self._observed_graph
        if self._observed_graph.has_node(seed):  # remove from observed set
            self._observed_set.erase(seed)
        else:  # add to observed graph
            self._observed_graph.add_node(seed)

        # iterate over neighbours
        for n in self._orig_graph.neighbors(seed):
            if not self._observed_graph.has_node(n):  # add to observed graph and observed set
                self._observed_graph.add_node(n)
                self._observed_set.insert(n)
                res.insert(n)
            self._observed_graph.add_edge(seed, n)
        return res

    cpdef int next_seed(self):
        """
        Core of the crawler - the algorithm to choose the next seed to be crawled.
        Seed must be a node of the original graph.

        :return: node id as int
        """
        # raise CrawlerException("Not implemented") TODO
        raise NotImplementedError()

    cpdef void crawl_budget(self, budget: int):
        """
        Perform `budget` number of crawls according to the algorithm.
        Note that `next_seed()` may be called more times - some returned seeds may not be crawled.

        :param budget: so many nodes will be crawled. If can't crawl any more, raise CrawlerError
        :param args: customizable additional args for subclasses
        :return:
        """
        for _ in range(budget):
            self.crawl(self.next_seed())

cdef extern from "int_iter.cpp":
    cdef cppclass IntIter:
        IntIter()
        int i
        IntIter operator++(int)


def cbasic_test():
    # print("cbasic")

    cgraph = GraphCollections.cget('dolphins')
    # cgraph = GraphCollections.cget('petster-hamster')

    crawler = CCrawler(cgraph)
    for n in cgraph.iter_nodes():
        print(n)
        crawler.crawl(n)
        print("crawled_set:", crawler.crawled_set)
        print("observed_set:", crawler.observed_set)
        print("nodes_set:", crawler.nodes_set)

    #
    # from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc
    # cdef IntIter ii = IntIter()
    # print(ii.i)
    # pinc(ii)
    # print(ii.i)
