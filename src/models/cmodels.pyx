import random

from base.cgraph cimport MyGraph
from base.cgraph cimport TRnd, TInt, TIntV, PUNGraph, GenConfModel
cimport cmodels

cpdef configuration_model(deg_seq, random_seed: int=-1):
    cdef TIntV DegSeqV = TIntV()
    for deg in deg_seq:
        DegSeqV.Add(TInt(deg))
    cdef TRnd Rnd = TRnd(random_seed if random_seed != -1 else random.randint(0, 1e9), 0)

    cdef PUNGraph g = GenConfModel(DegSeqV, Rnd)
    # for EI in g.Edges():
    #     print("edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
    return MyGraph().new_snap(g, name='Config_model')


cpdef grid2d(int n, int m):
    cdef MyGraph g = MyGraph(name='Grid2d_%sx%s' % (n, m))
    cdef int i, k, node
    for i in range(n * m):
        g.add_node(i)

    for k in range(0, n):
        for i in range(0, m):
            node = i * n + k
            if (node > 0) and (node % n != 0):
                g.add_edge(node, node - 1)
            if node > n - 1:
                g.add_edge(node, node - n)

            # pos[node] = [float(k / n), float(i / m)]
    g.new_snap(g.snap_graph_ptr())  # to update stats and fingerprint
    return g


# cpdef grid3d(int n, int m, int l):
#     cdef MyGraph g = MyGraph(name='Grid2d_%sx%s' % (n, m))
#     cdef int i, k, node
#     for i in range(n * m * l):
#         g.add_node(i)
#
#     for k in range(0, n):
#         for j in range(0, m):
#             for i in range(0, l):
#                 node = k * m * l + j * l + i
#                 if (node > 0) and (node % n != 0):
#                     g.add_edge(node, node - 1)
#                 if node > n - 1:
#                     g.add_edge(node, node - n)
#
#             # pos[node] = [float(k / n), float(i / m)]
#     g.new_snap(g.snap_graph_ptr())  # to update stats and fingerprint
#     return g


cpdef test_models():
    cdef MyGraph g = configuration_model([3, 3, 3, 2, 2, 1, 1, 1, 1, 1])
    print("N=%d, E=%d" % (g.nodes(), g.edges()))
    # print('cmodels')
