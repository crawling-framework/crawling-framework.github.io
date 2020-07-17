import numpy as np
import snap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from graph_io import GraphCollections
from base.cgraph import MyGraph
from cyth.cstatistics import assortativity
from a import truncated_power_law, configuration_model, ba_model

raise Exception("OLD, NEED TO UPDATE")

# def assortativity_of_edge_list(jk_iter):
#     m = 0
#     for j, k in jk_iter:
#         # j, k = e.GetSrcNId(), e.GetDstNId()
#         dj, dk = g.GetNI(j).GetDeg(), g.GetNI(k).GetDeg()
#         mul += dj * dk
#         sum2 += dj + dk
#         sum_sq2 += dj*dj + dk*dk
#
#     # print(mul/m, sum2/2/m, sum_sq2/2/m)
#     r = (mul/m - (sum2/2/m) ** 2) / (sum_sq2/2/m - (sum2/2/m) ** 2)
#     return r


def measure_e_jk(graph: MyGraph):
    g = graph.snap

    # e_jk_dict = {}  # pair -> count
    max_deg = g.GetNI(snap.GetMxDegNId(g)).GetDeg()
    max_deg = min(10000, max_deg)  # fixme
    e_jk_array = np.zeros((max_deg, max_deg), dtype=int)
    for e in g.Edges():
        j, k = e.GetSrcNId(), e.GetDstNId()
        dj, dk = g.GetNI(j).GetDeg() - 1, g.GetNI(k).GetDeg() - 1
        # if e_jk not in e_jk_dict:
        #     e_jk_dict[e_jk] = 0
        # e_jk_dict[e_jk] += 1
        if dj >= max_deg or dk >= max_deg:
            continue
        if dj > dk:
            dj, dk = dk, dj
        e_jk_array[dj][dk] += 1

    print("e_jk_dict", e_jk_array)

    import seaborn as sns

    r = assortativity(graph)
    plt.figure(figsize=(10, 8))
    plt.title("%s N=%d E=%d max_deg=%d r=%.4f" %
              (graph.name, g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg(), r))
    # plt.imshow(np.log(e_jk_array+1), cmap='hot')
    e_jk_array = e_jk_array + 1
    sns.heatmap(e_jk_array, cmap='hot',
                norm=LogNorm(vmin=e_jk_array.min(), vmax=e_jk_array.max()),
                # norm=matplotlib.colors.LogNorm()
                )
    plt.xlim((1, max_deg))
    plt.ylim((1, max_deg))
    # plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    # plt.savefig(PICS_DIR + '/assort__%s.png' % (graph.name))


# def conf_model(deg_seq, directed=False):
#     assert directed == False
#     graph = MyGraph.new_snap(name='conf_model', directed=directed)
#     g = graph.snap
#
#     src_deg_list = []  # each node i encounters deg(i) times. For uniform edge sampling
#     for n, deg in enumerate(deg_seq):
#         src_deg_list.extend([n] * deg)
#     m = len(src_deg_list)
#     print(src_deg_list)
#
#     src_deg_list = np.array(src_deg_list, dtype=int)
#     np.random.shuffle(src_deg_list)
#     dst_deg_list = np.array(src_deg_list, dtype=int)  # edge endpoints
#     np.random.shuffle(dst_deg_list)
#
#     for n in range(len(deg_seq)):
#         g.AddNode(n)
#
#     # generating edges
#     print("expect edges", sum(deg_seq))
#     for i in range(m):
#         src = int(src_deg_list[i])
#         dst = int(dst_deg_list[i])
#         if src != dst and not g.IsEdge(src, dst):
#             print("ok")
#             g.AddEdge(src, dst)
#             continue
#
#         # change the element util satisfy
#         j = i
#         for j in range(i+1, m):
#             print("resampling")
#             dst = int(dst_deg_list[j])
#             if src != dst and not g.IsEdge(src, dst):
#                 print("resampled ok")
#                 # swap elements
#                 tmp = dst_deg_list[i]
#                 dst_deg_list[i], dst_deg_list[j] = dst, tmp
#                 break
#         if j == m-1:
#             print("couldn't resample")
#         g.AddEdge(src, dst)
#
#     return graph


def assort_conf_model_MH(deg_seq, r, directed=False):
    """

    :param deg_seq:
    :param r: target correlation
    :param directed:
    :return:
    """
    assert -1 <= r <= 1
    graph = configuration_model(deg_seq)
    g = graph.snap

    node_deg = dict((n.GetId(), n.GetDeg()) for n in g.Nodes())  # dict {node id -> degree}
    print('actual node deg', node_deg)

    max_deg = g.GetNI(snap.GetMxDegNId(g)).GetDeg()
    # max_deg = min(10000, max_deg)  # fixme
    e_jk_array = np.zeros((max_deg+1, max_deg+1), dtype=int)
    m = g.GetEdges()  # number of edges

    src_deg_list = []  # each node i encounters deg(i) times. For uniform edge sampling
    for n in g.Nodes():
        src_deg_list.extend([n.GetId()] * n.GetDeg())
    print(src_deg_list)

    # Partial sums for correlation coefficient
    mul, sum2, sum_sq2 = 0, 0, 0
    for e in g.Edges():
        j, k = e.GetSrcNId(), e.GetDstNId()
        dj, dk = g.GetNI(j).GetDeg(), g.GetNI(k).GetDeg()
        mul += dj * dk
        sum2 += dj + dk
        sum_sq2 += dj*dj + dk*dk

    mul = mul / m
    sum2 = sum2 / 2/m
    sum_sq2 = sum_sq2 / 2/m
    r1 = (mul - sum2 ** 2) / (sum_sq2 - sum2 ** 2)
    print("Initial r=", r1)

    # edge switches
    for iteration in range(1000000):
        if iteration % 10000 == 0:
            print("r1", r1)  # fixme recompute r1 due to error cumulation
            print("graph N=%d E=%d" % (g.GetNodes(), g.GetEdges()))
            print('iteration %d, assort %s' % (iteration, assortativity(graph)))

        # Pick 2 random edges: j1-j2, k1-k2 such that j1-k2 and k1-j2 don't exist
        j1 = src_deg_list[np.random.randint(m)]
        # j2, k1, k2 = 1, 1, 1
        j2 = list(g.GetNI(j1).GetOutEdges())[np.random.randint(node_deg[j1])]
        while True:
            k1 = src_deg_list[np.random.randint(m)]
            if k1 == j1 or k1 == j2:
                continue

            k2 = list(g.GetNI(k1).GetOutEdges())[np.random.randint(node_deg[k1])]
            if k2 == j1 or k2 == j2:
                continue

            # check that j1-k2 and k1-j2 don't exist
            if g.IsEdge(j1, k2) or g.IsEdge(k1, j2):
                continue

            break  # OK

        dj1 = node_deg[j1] - 1
        dj2 = node_deg[j2] - 1
        dk1 = node_deg[k1] - 1
        dk2 = node_deg[k2] - 1

        # switch probability
        new_mul = mul - (dj1*dj2 + dk1*dk2)/m + (dj1*dk2 + dk1*dj2)/m
        r2 = (new_mul - sum2 ** 2) / (sum_sq2 - sum2 ** 2)
        prob = np.exp((r1-r2)*(r1+r2-2*r)/0.000001)
        # print("prob", prob)

        if np.random.random() < prob:  # make a switch
            g.DelEdge(j1, j2)
            g.DelEdge(k1, k2)
            g.AddEdge(k1, j2)
            g.AddEdge(j1, k2)

            mul = new_mul
            r1 = r2
            if r2 < r1:
                print("iteration r=", r1)

    # create graph
    # graph = MyGraph.new_snap('ass_conf_model', directed=directed)
    # g = graph.snap
    # for n in g0.Nodes():
    #     g.AddNode(n.GetId())
    # for index in range(m):
    #     g.AddEdge(int(src[index]), int(dst[index]))
    print("graph N=%d E=%d" % (g.GetNodes(), g.GetEdges()))
    return graph


# def assort_conf_model_MH(deg_seq, e_jk_func, directed=False):
#     """ """
#     g0 = configuration_model(deg_seq)
#     graph = MyGraph.new_snap('ass_conf_model', directed=directed)
#     graph._snap_graph = g0
#
#     node_deg = dict((n.GetId(), n.GetDeg()) for n in g0.Nodes())  # dict {node id -> degree}
#     print('actual node deg', node_deg)
#
#     max_deg = g0.GetNI(snap.GetMxDegNId(g0)).GetDeg()
#     # max_deg = min(10000, max_deg)  # fixme
#     e_jk_array = np.zeros((max_deg+1, max_deg+1), dtype=int)
#     m = g0.GetEdges()  # number of edges
#
#     src_deg_list = []  # each node i encounters deg(i) times
#     for n in g0.Nodes():
#         src_deg_list.extend([n.GetId()] * n.GetDeg())
#     print(src_deg_list)
#
#     # src, dst = np.ndarray(m, dtype=int), np.ndarray(m, dtype=int)
#     for index, e in enumerate(g0.Edges()):
#         j, k = e.GetSrcNId(), e.GetDstNId()
#         # src[index] = j
#         # dst[index] = k
#         dj, dk = g0.GetNI(j).GetDeg() - 1, g0.GetNI(k).GetDeg() - 1
#         if dj > dk:
#             # fixme this is for undirected only
#             dj, dk = dk, dj
#         e_jk_array[dj][dk] += 1
#
#     g = g0
#     # edge switches
#     for iteration in range(1000000):
#         if iteration % 10000 == 0:
#             # graph = MyGraph.new_snap('ass_conf_model', directed=directed)
#             # g = graph.snap
#             # for n in g0.Nodes():
#             #     g.AddNode(n.GetId())
#             # for index in range(m):
#             #     g.AddEdge(int(src[index]), int(dst[index]))
#             print("graph N=%d E=%d" % (g.GetNodes(), g.GetEdges()))
#             print('iteration %d, assort %s' % (iteration, assortativity(graph)))
#
#         # Pick 2 random edges: j1-j2, k1-k2 such that j1-k2 and k1-j2 don't exist
#         j1 = src_deg_list[np.random.randint(m)]
#         # j2, k1, k2 = 1, 1, 1
#         j2 = list(g.GetNI(j1).GetOutEdges())[np.random.randint(node_deg[j1])]
#         while True:
#             k1 = src_deg_list[np.random.randint(m)]
#             if k1 == j1 or k1 == j2:
#                 continue
#
#             k2 = list(g.GetNI(k1).GetOutEdges())[np.random.randint(node_deg[k1])]
#             if k2 == j1 or k2 == j2:
#                 continue
#
#             # check that j1-k2 and k1-j2 don't exist
#             if g.IsEdge(j1, k2) or g.IsEdge(k1, j2):
#                 continue
#
#             break  # OK
#
#         # j1 = src[j]
#         # j2 = dst[j]
#         # k1 = src[k]
#         # k2 = dst[k]
#
#         dj1 = node_deg[j1] - 1
#         dj2 = node_deg[j2] - 1
#         dk1 = node_deg[k1] - 1
#         dk2 = node_deg[k2] - 1
#
#         # switch probability
#         prob = e_jk_func(dj1, dk2) * e_jk_func(dk1, dj2) / e_jk_func(dj1, dj2) / e_jk_func(dk1, dk2)
#         # print(prob)
#         # prob = 1
#
#         if np.random.random() < prob:  # make a switch
#             g.DelEdge(j1, j2)
#             g.DelEdge(k1, k2)
#             g.AddEdge(k1, j2)
#             g.AddEdge(j1, k2)
#             # dst[j] = k2
#             # dst[k] = j2
#             e_jk_array[min(dj1, dj2)][max(dj1, dj2)] -= 1
#             e_jk_array[min(dk1, dk2)][max(dk1, dk2)] -= 1
#             e_jk_array[min(dj1, dk2)][max(dj1, dk2)] += 1
#             e_jk_array[min(dk1, dj2)][max(dk1, dj2)] += 1
#
#     # create graph
#     # graph = MyGraph.new_snap('ass_conf_model', directed=directed)
#     # g = graph.snap
#     # for n in g0.Nodes():
#     #     g.AddNode(n.GetId())
#     # for index in range(m):
#     #     g.AddEdge(int(src[index]), int(dst[index]))
#     print("graph N=%d E=%d" % (g.GetNodes(), g.GetEdges()))
#     return graph
#
#

def test_conf_model():
    n = 20
    deg_seq = truncated_power_law(n, gamma=1.4, maximal=n)
    # deg_seq = truncated_normal(n, mean=10, variance=1000, min=1, max=20)
    print('deg_seq', deg_seq)

    graph = conf_model(deg_seq)
    print("graph N=%d E=%d" % (graph.nodes(), graph.edges()))
    print('assortativity', assortativity(graph))


def test_assort_config_model():
    n = 23000
    deg_seq = truncated_power_law(n, gamma=2.6, maximal=int(2761))
    # deg_seq = truncated_normal(n, mean=50, variance=1000, min=1, max=20)
    print('deg_seq', deg_seq)

    # kappa = 100
    # ro = 0.1
    # for ro in np.arange(0, 1, 0.001):
    # ea = (8 * ro * (1-ro) - 1) / (2*np.exp(1/kappa) - 1 + 2*(2*ro - 1) ** 2)
    ea = -0.38
    print('expected assortativity', ea)

    # def e_jk_func(dj, dk, kappa=kappa, ro=ro):
    #     bi = binom(dj+dk, dj) * ro ** dj * (1-ro) ** dk + binom(dj+dk, dk) * ro ** dk * (1-ro) ** dj
    #     return np.exp(-(dj+dk)/kappa) * bi

    graph = assort_conf_model_MH(deg_seq, r=ea, directed=False)
    print('assortativity', assortativity(graph))


def test_ba_model():
    graph = ba_model(n=1000, avg_deg=100)

    node_deg = [(n.GetDeg()) for n in graph.snap.Nodes()]
    print(node_deg)

    print('assortativity', assortativity(graph))

    graph = assort_conf_model_MH(node_deg, r=0.6)
    print("graph N=%d E=%d" % (graph.nodes(), graph.edges()))
    print('assortativity', assortativity(graph))


def test_edge_switching(graph, r=None):
    import matplotlib.pyplot as plt
    plt.ion()

    g = graph.snap

    node_deg = dict((n.GetId(), n.GetDeg()) for n in g.Nodes())  # dict {node id -> degree}
    print('actual node deg', node_deg)

    max_deg = g.GetNI(snap.GetMxDegNId(g)).GetDeg()
    # max_deg = min(10000, max_deg)  # fixme
    e_jk_array = np.zeros((max_deg+1, max_deg+1), dtype=int)
    m = g.GetEdges()  # number of edges

    src_deg_list = []  # each node i encounters deg(i) times. For uniform edge sampling
    for n in g.Nodes():
        src_deg_list.extend([n.GetId()] * n.GetDeg())
    print(src_deg_list)

    # Partial sums for correlation coefficient
    mul, sum2, sum_sq2 = 0, 0, 0
    for e in g.Edges():
        j, k = e.GetSrcNId(), e.GetDstNId()
        dj, dk = g.GetNI(j).GetDeg(), g.GetNI(k).GetDeg()
        mul += dj * dk
        sum2 += dj + dk
        sum_sq2 += dj*dj + dk*dk

    mul = mul / m
    sum2 = sum2 / 2/m
    sum_sq2 = sum_sq2 / 2/m
    r1 = (mul - sum2 ** 2) / (sum_sq2 - sum2 ** 2)
    print("Initial r=", r1)

    plt.xlabel('iterations')
    plt.ylabel('assortativity')
    # edge switches
    for iteration in range(1000000):
        if iteration % 10000 == 0:
            print("r1", r1)  # fixme recompute r1 due to error cumulation
            print("graph N=%d E=%d" % (g.GetNodes(), g.GetEdges()))
            print('iteration %d, assort %s' % (iteration, assortativity(graph)))
            plt.plot(iteration, r1, color='r', linestyle='-', marker='o')
            plt.pause(0.005)

        # Pick 2 random edges: j1-j2, k1-k2 such that j1-k2 and k1-j2 don't exist
        j1 = src_deg_list[np.random.randint(m)]
        # j2, k1, k2 = 1, 1, 1
        j2 = list(g.GetNI(j1).GetOutEdges())[np.random.randint(node_deg[j1])]
        while True:
            k1 = src_deg_list[np.random.randint(m)]
            if k1 == j1 or k1 == j2:
                continue

            k2 = list(g.GetNI(k1).GetOutEdges())[np.random.randint(node_deg[k1])]
            if k2 == j1 or k2 == j2:
                continue

            # check that j1-k2 and k1-j2 don't exist
            if g.IsEdge(j1, k2) or g.IsEdge(k1, j2):
                continue

            break  # OK

        dj1 = node_deg[j1] - 1
        dj2 = node_deg[j2] - 1
        dk1 = node_deg[k1] - 1
        dk2 = node_deg[k2] - 1

        # switch probability
        if r:
            new_mul = mul - (dj1*dj2 + dk1*dk2)/m + (dj1*dk2 + dk1*dj2)/m
            r2 = (new_mul - sum2 ** 2) / (sum_sq2 - sum2 ** 2)

            prob = np.exp((r1-r2)*(r1+r2-2*r)/0.000001)
            # print("prob", prob)
        else:
            prob = 1

        if np.random.random() < prob:  # make a switch
            g.DelEdge(j1, j2)
            g.DelEdge(k1, k2)
            g.AddEdge(k1, j2)
            g.AddEdge(j1, k2)

            mul = new_mul
            r1 = r2
            if r2 < r1:
                print("iteration r=", r1)


if __name__ == '__main__':
    # name = 'libimseti'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'petster-friendships-cat'
    # name = 'loc-brightkite_edges'
    # name = 'facebook-wosn-links'
    # name = 'ego-gplus'
    name = 'petster-hamster'
    # for name in ['petster-hamster', 'ego-gplus', 'facebook-wosn-links', 'loc-brightkite_edges', 'petster-friendships-cat', 'digg-friends', 'libimseti', 'soc-pokec-relationships']:
    graph = GraphCollections.get(name, giant_only=True)
    # measure_e_jk(graph)

    # print(assortativity(graph))

    # plt.grid()
    # plt.show()
    # test_assort_config_model()
    # test_conf_model()
    # test_ba_model()
    test_edge_switching(graph, r=-0.8)

    plt.grid()
    plt.show()
