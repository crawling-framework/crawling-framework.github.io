import time
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


if __name__ == '__main__':
    test()
