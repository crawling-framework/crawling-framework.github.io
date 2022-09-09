from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef extern from "nd_set.cpp":
    cdef cppclass IntPair_Set:
        cppclass iterator:
            pair[int, int]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            pair[int, int]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        IntPair_Set()
        bint add(int node, int deg)
        void remove(int node, int deg)
        # bint update(int node, int deg)
        bint update_1(int node, int deg)
        # int pop()
        pair[int, int] pop()
        pair[int, int] pop_proportional_degree()
        bint empty()
        int size()
        iterator begin()
        iterator end()
        reverse_iterator rbegin()
        void print_me()


cdef class ND_Set:
    cdef IntPair_Set ipset

    cpdef bint add(self, int node, int deg)

    cpdef remove(self, int node, int deg)

    cpdef bint update_1(self, int node, int deg)

    cpdef (int, int) pop(self)

    cpdef int pop_proportional_degree(self)

    cpdef bint empty(self)

    cpdef int size(self)

    cpdef vector[int] pop_top(self, int size)

    cpdef vector[int] top(self, int count)

    cpdef vector[int] bottom(self, int count)
