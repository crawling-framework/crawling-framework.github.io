import logging
import random

from libcpp.set cimport set as cset
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.deque cimport deque
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec

from base.cgraph cimport CGraph, str_to_chars
from base.node_deg_set cimport ND_Set  # FIXME try 'as ND_Set' if error 'ND_Set is not a type identifier'


cdef class CCrawler:
    cdef char* _name
    cdef readonly CGraph _orig_graph
    cdef readonly CGraph _observed_graph
    cdef dict __dict__  # for pythonic fields, makes it slower
    cdef readonly set _crawled_set
    cdef readonly set _observed_set

    cpdef bint observe(self, int node)

    cpdef vector[int] crawl(self, int seed) except *
    cpdef int next_seed(self) except -1
    cpdef int crawl_budget(self, int budget) except -1


cdef class CCrawlerUpdatable(CCrawler):
    cpdef void update(self, vector[int] nodes)
