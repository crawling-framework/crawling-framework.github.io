import logging
from operator import itemgetter

import snap
import numpy as np
import matplotlib.pyplot as plt

from crawlers.advanced import AvrachenkovCrawler
from crawlers.basic import Crawler
from graph_io import MyGraph, GraphCollections
from statistics import get_top_centrality_nodes


def get_avg_deg_hubs(graph, count):
    """
    Get average degree of top-count hubs
    :param graph:
    :param count:
    :return:
    """
    node_deg = [(n.GetId(), n.GetDeg()) for n in graph.Nodes()]
    sorted_node_deg = sorted(node_deg, key=itemgetter(1), reverse=True)
    # print(sorted_node_deg)
    return np.mean([d for (n, d) in sorted_node_deg[:count]])


def compute_reachability(graph):
    g = graph.snap
    centrality = 'degree'
    mode = 'top-hubs'
    # mode = 'all'
    # mode = 'top-k'
    p = 0.1
    k = 100
    if mode == 'top-k':
        top_hubs = get_top_centrality_nodes(graph, centrality, count=k)
    if mode in ['all', 'top-hubs']:
        top_hubs = get_top_centrality_nodes(graph, centrality, count=int(p * g.GetNodes()))
    print("top_hubs %d: %s" % (len(top_hubs), top_hubs))

    if mode == 'top-hubs':
        max_k, step = 250, 2
    if mode == 'all':
        max_k, step = 2400, 5
    if mode == 'top-k':
        max_k, step = 200, 2
    reachable_from_random = {}
    graph_nodes = [n.GetId() for n in g.Nodes()]

    random_seeds = [graph_nodes[s] for s in np.random.randint(g.GetNodes(), size=max_k)]
    crawler = Crawler(graph)
    for k in range(1, max_k+1, step):
        seeds = random_seeds[:k]
        print("seeds %d: %s" % (len(seeds), 'seeds'))

        for seed in seeds:
            crawler.crawl(seed)
        neighs = crawler.nodes_set
        print("neighs %d: %s" % (len(neighs), 'neighs'))

        if mode == 'all':
            reachable_from_random[k] = len(neighs) / g.GetNodes()
        if mode in ['top-hubs', 'top-k']:
            covered = 0
            for hub in top_hubs:
                if hub in neighs:
                    covered += 1
            reachable_from_random[k] = covered / len(top_hubs)

    reachable_from_top = {0: 0}
    crawler = Crawler(graph)
    for k in range(1, max_k+1, step):
        top_k = get_top_centrality_nodes(graph, centrality, count=k)
        print("top_k %d: %s" % (len(top_k), 'top_k'))

        for seed in top_k:
            crawler.crawl(seed)
        neighs = crawler.nodes_set
        print("neighs %d: %s" % (len(neighs), 'neighs'))

        if mode == 'all':
            reachable_from_top[k] = len(neighs) / g.GetNodes()
        if mode in ['top-hubs', 'top-k']:
            covered = 0
            for hub in top_hubs:
                if hub in neighs:
                    covered += 1
            reachable_from_top[k] = covered / len(top_hubs)

    plt.figure(figsize=(6.45, 4.5))
    g = graph.snap
    plt.title("Graph N=%d E=%d max_deg=%d" % (g.GetNodes(), g.GetEdges(),
                                              g.GetNI(snap.GetMxDegNId(g)).GetDeg()))
    plt.xlabel("seeds")
    if mode == 'top-hubs':
        plt.ylabel("fraction of reachable top-%d%% hubs" % (100 * p))
    if mode == 'all':
        plt.ylabel("fraction of reachable nodes")
    if mode == 'all':
        plt.ylabel("fraction of reachable top-%d" % k)
    x, y = zip(*reachable_from_top.items())
    plt.plot(x, y, '.', color='b', label='top-k degree')
    x, y = zip(*reachable_from_random.items())
    plt.plot(x, y, '.', color='g', label='random k')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.grid()
    # plt.savefig(PICS_DIR + '/%s_%s.png' % (graph.name, mode))


def test_avrachenkov(graph: MyGraph):
    plt.figure(figsize=(6.45, 4.5))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    # lines = []
    # legends = []

    n_iterations = 5
    k = 100
    n = 1000
    for i, k in enumerate([50, 100, 250]):
        top_hubs = get_top_centrality_nodes(graph, 'degree', count=k)
        print("top_hubs %d: %s" % (len(top_hubs), top_hubs))
        recall_mean, recall_var = {}, {}
        n1_values = list(range(k, n-k+1, 50))
        for n1 in n1_values:
            recalls = []
            for it in range(n_iterations):
                crawler = AvrachenkovCrawler(graph, n=n, n1=n1, k=k)
                print("AvrachenkovCrawler n=%d, n1=%d, k=%d" % (n, n1, k))

                crawler.first_step()
                detected_hubs = crawler.second_step()

                correct = 1 - len(set(top_hubs) - set(detected_hubs)) / len(top_hubs)
                recalls.append(correct)
            recall_mean[n1] = np.mean(recalls)
            recall_var[n1] = np.var(recalls) ** 0.5

            print("recall: %s +- %s" % (recall_mean[n1], recall_var[n1]))

        g = graph.snap
        plt.title("Graph N=%d E=%d max_deg=%d\n n=%d" % (
            g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg(), n))
        plt.xlabel("n1")
        plt.errorbar(n1_values, list(recall_mean.values()), yerr=list(recall_var.values()),
                     fmt='-', color=colors[i%len(colors)][0], label='k=%d' % k)
        # lines.append(plt.plot(x, y, '-o', color=colors[i%len(colors)])[0])
        # legends.append('k=%d' % k)
    plt.legend(loc=0)
    plt.tight_layout()


def test_hubs_search_by_mod(graph: MyGraph):
    """ top-10% hubs detection from top-100 hubs by MOD in their neighbourhood
    """
    from matplotlib_venn import venn3_circles, venn3
    g = graph.snap
    N = g.GetNodes()

    hubs = get_top_centrality_nodes(graph, 'degree')

    # top-10%
    v = int(0.1*N)
    V_target = set(get_top_centrality_nodes(graph, 'degree', count=v))

    ks = range(10, v, int(v/20))
    e2_in_vs = []
    e2s_in_vs = []
    h_in_vs = []
    for k in ks:
        top_hubs = hubs[:k]

        # neighs
        crawler = Crawler(graph)
        [crawler.crawl(s) for s in top_hubs]
        E2 = crawler.observed_set

        o = crawler.observed_graph.snap
        # o = g
        E2_deg = [(n, o.GetNI(n).GetDeg()) for n in E2]

        E2s = set([n for (n, _) in sorted(E2_deg, key=itemgetter(1), reverse=True)[:v-k]])
        Es = E2s.union(top_hubs)

        h = len(top_hubs)/len(V_target)
        e2 = len(V_target.intersection(E2))/len(V_target)
        e = len(V_target.intersection(Es))/len(V_target)
        e2_in_vs.append(e2+h)
        e2s_in_vs.append(e)
        h_in_vs.append(h)
        # print("size of E2", len(E2))
        # print("size of Es", len(Es))
        # print("size of V_target", len(V_target))
        # print("recall", 1-len(V_target-Es)/len(V_target))

        plt.clf()
        plt.plot(ks[:len(e2_in_vs)], e2_in_vs, marker='o', color='g', label=r'seed hubs and $\epsilon_2$')
        plt.plot(ks[:len(e2_in_vs)], e2s_in_vs, marker='o', color='b', label=r'$\epsilon^*$')
        plt.plot(ks[:len(e2_in_vs)], h_in_vs, marker='o', color='r', label=r'seed hubs')
        # venn3([E2, Es, V_target], set_labels=["E2", "E*", "V*"])
        plt.legend()
        plt.xlabel('# of hubs as seeds')
        plt.ylabel(r'$V^*$ coverage')
        plt.title(r"Graph %s:  $N=%d, E=%d, d_{max}=%d$" % (
            graph.name, g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg()))
        plt.grid()
        plt.pause(0.005)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    name = 'petster-hamster'
    # for name in ['libimseti', 'soc-pokec-relationships', 'digg-friends',
    #              'ego-gplus', 'petster-hamster']:
    #     g = read_snap(get_graph_path(name))
    #     print(name, get_avg_deg_hubs(g, 100)/g.GetNodes())
    g = GraphCollections.get(name, giant_only=True)
    
    # compute_reachability(g)
    # test_avrachenkov(g)
    test_hubs_search_by_mod(g)

    plt.show()
