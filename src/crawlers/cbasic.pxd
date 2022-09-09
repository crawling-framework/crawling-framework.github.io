from libcpp.vector cimport vector
from base.cgraph cimport MyGraph, str_to_chars
from base.node_deg_set cimport ND_Set
from crawlers.declarable cimport Declarable

cdef class Crawler(Declarable):
    cdef readonly MyGraph _orig_graph
    cdef readonly MyGraph _observed_graph
    cdef dict __dict__  # for pythonic fields, makes it slower
    cdef readonly set _crawled_set
    cdef readonly set _observed_set

    cpdef bint observe(self, int node)

    cpdef vector[int] crawl(self, int seed) except *
    cpdef int next_seed(self) except -1
    cpdef int crawl_budget(self, int budget) except -1
