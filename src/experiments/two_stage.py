import logging
import os
from operator import itemgetter

import snap
import numpy as np
import matplotlib.pyplot as plt

from centralities import get_top_centrality_nodes
from crawlers import TwoStageCrawler
from graph_io import MyGraph, GraphCollections
from utils import PICS_DIR


def test_initial_graph(i: str):
    from graph_io import GraphCollections
    if i == "reall":
        # name = 'soc-pokec-relationships'
        # name = 'petster-friendships-cat'
        name = 'petster-hamster'
        # name = 'twitter'
        # name = 'libimseti'
        # name = 'advogato'
        # name = 'facebook-wosn-links'
        # name = 'soc-Epinions1'
        # name = 'douban'
        # name = 'slashdot-zoo'
        # name = 'petster-friendships-cat'  # snap load is long possibly due to unordered ids
        graph = GraphCollections.get(name)
        print("N=%s E=%s" % (graph.snap.GetNodes(), graph.snap.GetEdges()))
    else:
        g = snap.TUNGraph.New()
        g.AddNode(1)
        g.AddNode(2)
        g.AddNode(3)
        g.AddNode(4)
        g.AddNode(5)
        g.AddEdge(1, 2)
        g.AddEdge(2, 3)
        g.AddEdge(4, 2)
        g.AddEdge(4, 3)
        g.AddEdge(5, 4)
        print("N=%s E=%s" % (g.GetNodes(), g.GetEdges()))
        graph = MyGraph.new_snap(name='test', directed=False)
        graph.snap_graph = g
    return graph


def test():

    # GRAPH
    graph = test_initial_graph("reall")

    # Directory for save
    import os
    import glob
    from utils import PICS_DIR
    file_path = PICS_DIR + "/TwoStageCrawler" + "/" + graph.name + "/"
    if os.path.exists(file_path):
        for file in glob.glob(file_path + "*.png"):
            os.remove(file)
    else:
        os.makedirs(file_path)

    # Target array
    p = 0.1
    from centralities import get_top_centrality_nodes
    vs = set(get_top_centrality_nodes(graph, 'degree', int(p*graph.snap.GetNodes())))
    assert abs(len(vs) - p*graph.snap.GetNodes()) <= 1

    # Crawling and drawing
    from matplotlib import pyplot as plt
    for s in range(50, 100, 10):
        print("-----------------%s------------------------" % s)
        history = dict()
        for n in range(s, 242, 50):
            crawler = TwoStageCrawler(graph, n=n, s=s, p=p)
            crawler.first_step()
            hubs_detected = crawler.second_step()
            mu = len(vs.intersection(hubs_detected)) #/ len(vs)
            #mu = len(hubs_detected)
            history[n] = mu
        x, y = zip(*list(history.items()))
        plt.plot(x, y, label="s=%s" % s, marker='o')
        plt.savefig(file_path + str(s) + '_figure.png', dpi=300)
    plt.legend()
    plt.show()


def if_seeds_were_hubs(graph: MyGraph):

    class TwoStageCrawlerSeedsAreHubs(TwoStageCrawler):
        def first_step(self, random_init=None):
            self.hubs = get_top_centrality_nodes(self.orig_graph, 'degree', count=self.s)
            [self.crawl(n) for n in self.hubs]

        def second_step(self):
            super().second_step()
            # Get E2 - second neighbourhood, with their degrees.
            e2 = []
            o = self.observed_graph.snap
            for o_id in self.observed_set:
                deg = o.GetNI(o_id).GetDeg()
                e2.append((o_id, deg))

            # Get v=(pN-n) max degree observed nodes
            self.e2s = [n for n, _ in
                        sorted(e2, key=itemgetter(1), reverse=True)[:self.pN - self.n]]

            # Final answer - E* = hubs + E1* + E2*
            self.es = set(self.hubs + self.e1s + self.e2s)
            return self.es

    g = graph.snap
    N = g.GetNodes()
    p = 0.1
    pN = int(p*N)
    target = set(get_top_centrality_nodes(graph, 'degree', pN))

    seeds = [50, 100, 500, 1000]
    nrows, ncols = 2, 2
    fig = plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
                graph.name, g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg()),
                     figsize=(12, 10))
    for i, s in enumerate(seeds):
        ax = plt.subplot(nrows, ncols, 1+i)

        k_max = min(50*s, 15000)
        ks = range(0, k_max, int(k_max/20))  # nodes to crawl at 2nd step
        o_in_vs = []
        e2s_in_vs = []
        o_in_vs_sh = []
        e2s_in_vs_sh = []
        for k in ks:
            # top_hubs = target[:k]

            try:
                crawler = TwoStageCrawler(graph, s=s, n=k+s, p=p)
                crawler.first_step(random_init=4)
                Es = crawler.second_step()
            except Exception:
                continue

            try:
                crawler_sh = TwoStageCrawlerSeedsAreHubs(graph, s=s, n=k+s, p=p)
                crawler_sh.first_step()
                Es_sh = crawler_sh.second_step()
            except Exception:
                continue

            o = len(target.intersection(crawler.nodes_set))/len(target)
            e = len(target.intersection(Es))/len(target)
            o_in_vs.append(o)
            e2s_in_vs.append(e)

            o_sh = len(target.intersection(crawler_sh.nodes_set))/len(target)
            e_sh = len(target.intersection(Es_sh))/len(target)
            o_in_vs_sh.append(o_sh)
            e2s_in_vs_sh.append(e_sh)

            plt.cla()
            plt.plot(np.array(ks[:len(o_in_vs)])+s, o_in_vs, marker='o', color='g', label=r'by $\epsilon_2^*$')
            plt.plot(np.array(ks[:len(o_in_vs)])+s, e2s_in_vs, marker='o', color='b', label=r'by $\epsilon^*$')

            plt.plot(np.array(ks[:len(o_in_vs)])+s, o_in_vs_sh, linestyle='--', color='g', label=r'by $\epsilon_2^*$, ideal')
            plt.plot(np.array(ks[:len(o_in_vs)])+s, e2s_in_vs_sh, linestyle='--', color='b', label=r'by $\epsilon^*$, ideal')

            plt.legend()
            plt.xlabel('crawled, n')
            plt.ylabel(r'$V^*$ coverage')
            plt.title("s=%s" % s)
            # plt.suptitle(r"Graph %s:  $N=%d, E=%d, d_{max}=%d$" % (
            #     graph.name, g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg()))
            plt.grid()
            plt.tight_layout()
            plt.pause(0.005)

    subdir = os.path.join(PICS_DIR, 'if_seeds_were_hubs')
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    plt.savefig(subdir + '/%s.png' % (graph.name))
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    # test()

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    g = GraphCollections.get(name)

    if_seeds_were_hubs(g)
