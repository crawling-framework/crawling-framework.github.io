from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

cimport node_deg_set  # pxd import DON'T DELETE


cdef class ND_Set:

    def __init__(self, iterable=None):
        self.ipset = IntPair_Set()

        if iterable is not None:
            for node, deg in iterable:
                self.ipset.add(node, deg)

    cpdef bint add(self, int node, int deg):
        return self.ipset.add(node, deg)

    # cpdef bint remove(self, int node, int deg):
    #     # print("Remove %d, %d" % (node, deg))
    #     return self.ipset.remove(node, deg)

    cpdef bint update_1(self, int node, int deg):
        """
        Update node degree by 1. Or inserts if didn't exist.
        :param node: node id
        :param deg: old degree
        :return: True if node existed False otherwise
        """
        return self.ipset.update_1(node, deg)

    cpdef remove(self, int node, int deg):
        self.ipset.remove(node, deg)

    cpdef (int, int) pop(self):
        """ Pop (degree, node) with max degree.
        """
        cdef pair[int, int] a = self.ipset.pop()
        return a.first, a.second

    cpdef int pop_proportional_degree(self):
        """ Pop (degree, node) proportional to degree.
        """
        cdef pair[int, int] a = self.ipset.pop_proportional_degree()
        return a.second

    cpdef bint empty(self):
        return self.ipset.empty()

    cpdef int size(self):
        return self.ipset.size()

    def __len__(self):
        return self.ipset.size()

    def __str__(self):
        res = "ND_Set: "
        cdef IntPair_Set.iterator it = self.ipset.begin()
        cdef IntPair_Set.iterator end = self.ipset.end()
        while it != end:
            res += str(deref(it))
            inc(it)
        return res

    def __iter__(self):
        cdef IntPair_Set.iterator it = self.ipset.begin()
        for _ in range(self.ipset.size()):
            yield deref(it)
            inc(it)

    cpdef vector[int] pop_top(self, int size):
        """ Get top-size removing from set"""
        cpdef vector[int] res
        while size > 0 and not self.ipset.empty():
            res.push_back(self.ipset.pop().second)
            size -= 1
        return res

    cpdef vector[int] top(self, int count):
        """ Get top-count not removing from set"""
        cpdef vector[int] res
        cdef IntPair_Set.reverse_iterator it = self.ipset.rbegin()
        count = min(count, self.ipset.size())
        while count > 0:
            res.push_back(deref(it).second)
            inc(it)
            count -= 1
        return res

    cpdef vector[int] bottom(self, int count):
        """ Get bottom-count not removing from set"""
        cpdef vector[int] res
        cdef IntPair_Set.iterator it = self.ipset.begin()
        count = min(count, self.ipset.size())
        while count > 0:
            res.push_back(deref(it).second)
            inc(it)
            count -= 1
        return res

    # def __getitem__(self, item):
    #     # print(self)
    #     cdef IntPair_Set.iterator it
    #     cdef IntPair_Set.reverse_iterator rit
    #     if isinstance(item, slice):
    #         start = item.start
    #         stop = item.stop
    #         if start is None:
    #             if stop < 0:
    #                 stop = self.ipset.size() + start
    #             it = self.ipset.begin()
    #             res = []
    #             for _ in range(max(0, stop)):
    #                 deg, node = deref(it)
    #                 res.append(node)
    #                 inc(it)
    #             # print(res)
    #             return res
    #         if stop is None:
    #             if start < 0:
    #                 start = self.ipset.size() + start
    #             rit = self.ipset.rbegin()
    #             res = []
    #             for _ in range(max(0, self.ipset.size() - start)):
    #                 deg, node = deref(rit)
    #                 print(deg, node)
    #                 res.append(node)
    #                 inc(rit)
    #             # print(res)
    #             return res
    #         raise NotImplementedError()
