from libcpp.set cimport set as cset
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

cimport node_deg_set


cdef class ND_Set:

    def __init__(self, iterable=None):
        self.ipset = IntPair_Set()

        # print(iterable)
        if iterable is not None:
            for node, deg in iterable:
                # deg = key(node)
                self.ipset.add(node, deg)

    cpdef bint add(self, int node, int deg):
        # print("Add %d, %d" % (node, deg))
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

    # cpdef bint update(self, int node, int deg):
    #     """
    #     Update node in set with a new degree. Or inserts if didn't exist.
    #     :param node: node id
    #     :param deg: new degree
    #     :return: True if node existed False otherwise
    #     """
    #     return self.ipset.update(node, deg)
    #
    # cpdef bint remove(self, int node, int deg):
    #     return self.ipset.remove(node, deg)
    #
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

    cpdef vector[int] top(self, int size):
        cpdef vector[int] res
        while size > 0 and not self.ipset.empty():
            # res.append(self.ipset.pop().second)
            res.push_back(self.ipset.pop().second)
            size -= 1
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

def key(n):
    return 20*(n)


cpdef test_ndset():
    # from cyth.cskl import SKL


    elems = [(1, 40),
             (4, 30),
             (6, 20),
             (2, 40),
             (0, 40),
             (3, 40)]
    deg = dict(elems)
    cdef ND_Set skl = ND_Set(elems, key=key)

    # for e in elems:
    #     print("pushing %s" % str(e))
    #     cpq.push(e)
        # skl.add(e[0], e[1])
        # print("pushed %s" % str(e))

    # r = skl.remove(2, 40)
    r = skl.discard(2)
    print(r)

    while not skl.empty():
        a = skl.pop()
        print(a)

