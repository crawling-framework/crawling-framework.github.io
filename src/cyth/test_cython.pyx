import time
# from builtins import function

import numpy as np

from libcpp.map cimport map as cmap


def test():
    n = 100000
    # cdef long[:] keys = np.random.randint(0, 1000, n)
    # cdef set s = set()
    # s = set()

    # cdef long res = 0
    # # res = 0
    # t = time.time()
    # for i in range(n):
    #     # py_dict[keys[i]] = values[i]  # 0.63 ms
    #     # cy_dict[keys[i]] = values[i]  # 2.10 ms
    #     # res += keys[i]
    #     s.add(i)
    #
    # print(1000 * (time.time() - t), "ms")


    cdef long[:] keys = np.random.randint(0, 1000, n)
    cdef long[:] values = np.random.randint(0, 1000, n)
    # values = map(str, keys)

    py_dict = dict()
    cdef int i

    cdef cmap[long, long] cy_dict


    t = time.time()
    for i in range(n):
        # py_dict[i] = i  # 0.45 ms
        cy_dict[i] = i  # 0.78 ms

    print 1000 * (time.time() - t), "ms"



class PyA:
    def __init__(self, a=0):
        self.a = a

    def f(self):
        return self.a ** 2


cdef class CyA:
    cdef int a

    def __init__(self, a=0):
        self.a = a

    def f(self):
        return self.a ** 2


cdef class CyB(CyA):
    # cdef int f(self)

    def __init__(self, a=0):
        super().__init__(a)

    cdef h(self):
        return self.a ** 3


# cdef class CyP(PyA):  # First base of 'CyP' is not an extension type
#     def __init__(self, a=0):
#         super().__init__(a)
#
#     cdef h(self):
#         return self.a ** 3


class PyB(CyA):
    def __init__(self, a=0):
        print('init')
        super().__init__(a)

    def e(self):
        return super().a + 10


cpdef test_class():
    # p = PyC(10)
    # print(p.f())
    # c = CyC(20)
    c = PyB(20)
    # c = CyP(20)
    print(c.f())
    # print(c.h())
    # print(c.e())

# from cyth cimport cskl
# from cyth.cskl import ND_SET
from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair

ctypedef bint (*f_type)(int, int)

# cdef extern from "custom_pq.cpp":
#     cdef cppclass A:
#         A()
#         A(int) # get Cython to accept any arguments and let C++ deal with getting them right
#         int a

# T = int
# cdef extern from "custom_pq.cpp":
#     cdef cppclass CPQ[int]:
#         CPQ()
#         # CPQ(...) # get Cython to accept any arguments and let C++ deal with getting them right
#         CPQ(f_type)
#         bint empty()
#         void pop()
#         void push(int&)
#         size_t size()
#         int& top()
#         # C++11 methods
#         void swap(priority_queue&)
#         # my add-on
#         bint remove(int&)

cdef extern from "nd_set.cpp":
    cdef cppclass IntPair_Set:
        IntPair_Set()
        bint add(int node, int deg)
        bint remove(int node, int deg)
        # int pop()
        pair[int, int] pop()
        # (int, int) pop()
        bint empty()


cdef bint cmp(int a, int b):
    # note - no protection if allocated memory isn't long enough
    # print("comparing %s and %s" % (a, b))
    return a < b

cpdef test_skl():
    # from cyth.cskl import SKL

    elems = [(1, 40),
             (4, 30),
             (6, 20),
             (2, 40),
             (0, 40),
             (3, 40)]
    # cdef A cpq = A(10)
    # print(cpq.a)

    # cdef CPQ cpq = CPQ(cmp)
    # print("cdef CPQ cpq = CPQ(cmp)")

    cdef IntPair_Set skl = IntPair_Set()

    for e in elems:
        print("pushing %s" % str(e))
        # cpq.push(e)
        skl.add(e[0], e[1])
        print("pushed %s" % str(e))

    r = skl.remove(2, 40)
    print(r)

    cdef pair[int, int] a
    while not skl.empty():
        a = skl.pop()
        print(a.first, a.second)


if __name__ == '__main__':
    test()
