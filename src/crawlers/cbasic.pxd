from libcpp.vector cimport vector
from base.cgraph cimport MyGraph, str_to_chars
from base.node_deg_set cimport ND_Set  # FIXME try 'as ND_Set' if error 'ND_Set is not a type identifier'


cdef class Crawler:
    cdef readonly MyGraph _orig_graph
    cdef readonly MyGraph _observed_graph
    cdef dict __dict__  # for pythonic fields, makes it slower
    cdef readonly set _crawled_set
    cdef readonly set _observed_set

    cpdef bint observe(self, int node)

    cpdef vector[int] crawl(self, int seed) except *
    cpdef int next_seed(self) except -1
    cpdef int crawl_budget(self, int budget) except -1


# cdef class CrawlerUpdatable(Crawler):
#     cpdef void update(self, vector[int] nodes)
#
cdef class CrawlerWithInitialSeed(Crawler):
    pass
