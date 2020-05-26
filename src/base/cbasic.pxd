import logging
import random

from libcpp.set cimport set as cset
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.deque cimport deque
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec

from cgraph cimport CGraph, str_to_chars
from node_deg_set cimport ND_Set  # FIXME try 'as ND_Set' if error 'ND_Set is not a type identifier'


cdef class CCrawler:
    cdef char* _name
    cdef readonly CGraph _orig_graph
    cdef CGraph _observed_graph
    cdef cset[int]* _crawled_set  # FIXME python set is 3x faster than cython
    cdef cset[int]* _observed_set

    cdef set_crawled_set(self, cset[int]* new_crawled_set)
    cdef set_observed_set(self, cset[int]* new_observed_set)
    cdef set_observed_graph(self, CGraph new_observed_graph)

    cpdef bint observe(self, int node)

    cdef vector[int] crawl(self, int seed) except *
    cpdef int next_seed(self) except -1
    cpdef int crawl_budget(self, int budget) except -1


cdef class CCrawlerUpdatable(CCrawler):
    cpdef void update(self, vector[int] nodes)
