from base.cbasic import RandomCrawler, RandomWalkCrawler
from utils import USE_CYTHON_CRAWLERS
from cbasic cimport CCrawler
from cgraph cimport CGraph

from libcpp.set cimport set as cset
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec, address as addr

from graph_io import GraphCollections
from statistics import Stat, get_top_centrality_nodes

cdef class CMetric:
    cdef name
    cdef _callback, _kwargs

    def __init__(self, name, callback, **kwargs):
        self.name = name
        self._callback = callback
        self._kwargs = kwargs

    def __call__(self, CCrawler crawler):
        return self._callback(self, crawler, **self._kwargs)


cdef inline int sets_intersection(cset[int]* a, cset[int]* b):
    cdef int n, res = 0
    if a.size() < b.size():
        for n in deref(a):
            if b.find(n) != b.end():
                res += 1
    else:
        for n in deref(b):
            if a.find(n) != a.end():
                res += 1
    return res


cdef class CoverageTopCentralityMetric(CMetric):
    cdef cset[int]* _target_set
    cdef int _target_set_size
    cdef part

    def __init__(self, CGraph graph, top: float, centrality: Stat, part='crawled', name=None):
        cdef int n
        self._target_set = new cset[int]()
        for n in get_top_centrality_nodes(graph, centrality, count=int(top*graph[Stat.NODES])):
            self._target_set.insert(n)
        self._target_set_size = self._target_set.size()

        assert part in ['crawled', 'observed', 'all']
        if part == 'crawled':
            callback = self._intersect_crawled
        elif part == 'observed':
            callback = self._intersect_observed
        elif part == 'all':
            callback = self._intersect_all
        else:
            raise NotImplementedError()

        super().__init__(name if name else CoverageTopCentralityMetric.to_string(top, centrality, part), callback)

    @staticmethod
    def to_string(top: float, centrality: Stat, part='crawled'):
        # return "%s_%s" % (part, centrality.name)
        return "Coverage_%s_%s_%s" % (part, centrality, top)

    cdef float _intersect_crawled(self, CCrawler crawler):
        return sets_intersection(crawler._crawled_set, self._target_set) / self._target_set_size

    cdef float _intersect_observed(self, CCrawler crawler):
        return sets_intersection(crawler._observed_set, self._target_set) / self._target_set_size

    cdef float _intersect_all(self, CCrawler crawler):
        return (sets_intersection(crawler._crawled_set, self._target_set) +
                sets_intersection(crawler._observed_set, self._target_set)) / self._target_set_size


cpdef test_metrics():
    print("test_metrics")
    # g = GraphCollections.get('Pokec')
    g = GraphCollections.get('dolphins')
    print('1')
    m = CoverageTopCentralityMetric(g, top=0.1, centrality=Stat.DEGREE_DISTR, part='all')
    print('2')
    crawler = RandomWalkCrawler(g)
    # crawler.crawl_budget(50)
    #
    # from time import time
    # t = time()
    # for _ in range(100):
    #     a = m(crawler)
    # # print(a)
    #
    # print(time()-t)

