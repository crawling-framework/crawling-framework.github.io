import logging

import snap
from matplotlib import pyplot as plt

from centralities import get_top_centrality_nodes
from crawlers.advanced import TwoStageCrawler, TwoStageCrawlerBatches, TwoStageCrawlerBatchesMOD
from crawlers.basic import CrawlerError, Crawler
from experiments.runners import Metric, AnimatedCrawlerRunner
from graph_io import MyGraph, GraphCollections
from statistics import Stat


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
        graph = MyGraph.new_snap(g, name='test', directed=False)
    return graph


# def test():
#
#     # GRAPH
#     graph = test_initial_graph("reall")
#
#     # Directory for save
#     import os
#     import glob
#     from utils import PICS_DIR
#     file_path = PICS_DIR + "/TwoStageCrawler" + "/" + graph.name + "/"
#     if os.path.exists(file_path):
#         for file in glob.glob(file_path + "*.png"):
#             os.remove(file)
#     else:
#         os.makedirs(file_path)
#
#     # Target array
#     p = 0.1
#     from centralities import get_top_centrality_nodes
#     vs = set(get_top_centrality_nodes(graph, 'degree', int(p*graph.snap.GetNodes())))
#     assert abs(len(vs) - p*graph.snap.GetNodes()) <= 1
#
#     # Crawling and drawing
#     from matplotlib import pyplot as plt
#     for s in range(50, 100, 10):
#         print("-----------------%s------------------------" % s)
#         history = dict()
#         for n in range(s, 242, 50):
#             crawler = TwoStageCrawler(graph, n=n, s=s, p=p)
#             crawler.first_step()
#             hubs_detected = crawler.second_step()
#             mu = len(vs.intersection(hubs_detected)) #/ len(vs)
#             #mu = len(hubs_detected)
#             history[n] = mu
#         x, y = zip(*list(history.items()))
#         plt.plot(x, y, label="s=%s" % s, marker='o')
#         plt.savefig(file_path + str(s) + '_figure.png', dpi=300)
#     plt.legend()
#     plt.show()


class TwoStageCrawlerSeedsAreHubs(TwoStageCrawler):
    """
    Artificial version of TwoStageCrawler, where instead of initial random seeds we take hubs
    """
    def _seeds_generator(self):
        # 1) hubs as seeds
        self.hubs = get_top_centrality_nodes(self.orig_graph, 'degree', count=self.s)
        for i in range(self.s):
            yield self.hubs[i]

        # memorize E1
        self.e1 = set(self.observed_set)
        logging.debug("|E1|=", len(self.e1))

        # Check that e1 size is more than (n-s)
        if self.n - self.s > len(self.e1):
            raise CrawlerError("E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." %
                               (len(self.e1), self.n - self.s))

        # 2) detect MOD batch
        self.top_observed_seeds = self._get_mod_nodes(self.observed_set, self.n - self.s)
        self.e1s = set(self.top_observed_seeds)
        logging.debug("|E1*|=", len(self.e1s))

        for node in self.top_observed_seeds:
            yield node

    def _compute_answer(self):  # E* = S + E1* + E2*
        self.e2 = set(self.observed_set)

        # Get v=(pN-n) max degree observed nodes
        self.e2s = set(self._get_mod_nodes(self.e2, self.pN - self.n))

        # Final answer - E* = S + E1* + E2*
        self.answer = set(self.hubs).union(self.e1s.union(self.e2s))


class TargetSetCoverageTester:
    """
    Runs batch crawling and measures target set coverage during crawling.
    """

    def __init__(self, graph: MyGraph, crawler_class, target=None):
        self.graph = graph
        assert isinstance(crawler_class, type(Crawler))
        self.crawler_class = crawler_class

        self.target_set = set(get_top_centrality_nodes(graph, 'degree', target))

    def run(self, crawler_kwargs_list):

        g = self.graph.snap

        nrows = 1
        if len(crawler_kwargs_list) > 1:
            nrows = int(len(crawler_kwargs_list) ** 0.5)
        from math import ceil
        ncols = ceil(len(crawler_kwargs_list) / nrows)
        fig = plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
            self.graph.name, g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg()),
                         figsize=(1 + 5*ncols, 5*nrows))

        for i, crawler_kwargs in enumerate(crawler_kwargs_list):
            ax = plt.subplot(nrows, ncols, 1 + i)

            crawler = self.crawler_class(self.graph, **crawler_kwargs)
            # crawler_sh = TwoStageCrawlerSeedsAreHubs(self.graph, s=crawler_kwargs['s'],
            #                                          n=crawler_kwargs['n'], p=crawler_kwargs['p'])
            # crawler_sh = TwoStageCrawlerBatches(self.graph, **crawler_kwargs, b=10)

            iterations = []
            os = []
            hs = []
            os_sh = []
            hs_sh = []

            try:
                max_budget = crawler_kwargs['n']
            except:
                max_budget = 1000
            # batch = 1
            # s0 = 0
            # crawler.crawl_budget(budget=s0)
            # crawler_sh.crawl_budget(budget=s0)
            # for i in range(s0, max_budget+1, batch):
            batch = max(1, int(max_budget / 30))
            for i in range(batch, max_budget+1, batch):
                iterations.append(i)

                crawler.crawl_budget(budget=batch)
                # crawler_sh.crawl_budget(budget=batch)

                crawler._compute_answer()
                o = len(self.target_set.intersection(crawler.nodes_set)) / len(self.target_set)
                h = len(self.target_set.intersection(crawler.answer)) / len(self.target_set)
                os.append(o)
                hs.append(h)

                # crawler_sh._compute_answer()
                # o_sh = len(self.target_set.intersection(crawler_sh.nodes_set)) / len(self.target_set)
                # h_sh = len(self.target_set.intersection(crawler_sh.answer)) / len(self.target_set)
                # os_sh.append(o_sh)
                # hs_sh.append(h_sh)

                plt.cla()
                plt.axvline(crawler_kwargs['s'], linewidth=2, color='r')
                plt.plot(iterations, os, marker='o', color='g', label=r'all nodes')
                plt.plot(iterations, hs, marker='o', color='b', label=r'candidates')
                # plt.plot(iterations, os_sh, marker='.', linestyle='--', color='g', label=r'HUBS, all nodes')
                # plt.plot(iterations, hs_sh, marker='.', linestyle='--', color='b', label=r'HUBS, candidates')

                plt.legend()
                plt.ylim((0, 1))
                plt.xlabel('crawled, n')
                plt.ylabel(r'$V^*$ coverage')
                g = self.graph.snap
                # plt.title(r"Graph %s:  $N=%d, E=%d, d_{max}=%d$" % (
                #     self.graph.name, g.GetNodes(), g.GetEdges(), g.GetNI(snap.GetMxDegNId(g)).GetDeg())
                #           + "\n%s %s" % (self.crawler_class.__name__, crawler_kwargs))
                plt.title("\n%s %s" % (self.crawler_class.__name__, crawler_kwargs))
                plt.grid()
                plt.tight_layout()
                plt.pause(0.005)

        plt.show()


def test_target_set_coverage():
    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    graph = GraphCollections.get(name)

    p = 0.1
    # # tester = TargetSetCoverageTester(graph, TwoStageCrawler, target=int(p*graph.snap.GetNodes()))
    # # tester.run([
    #     # {'s': 10, 'n': 20, 'p': p},
    #     # {'s': 50, 'n': 200, 'p': p},
    #     # {'s': 100, 'n': 500, 'p': p},
    #     # {'s': 500, 'n': 2500, 'p': p},
    #     # {'s': 1000, 'n': 5000, 'p': p},
    #     # {'s': 2000, 'n': 10000, 'p': p},
    #     # {'s': 5000, 'n': 50000, 'p': p},
    # # ])
    #
    # tester = TargetSetCoverageTester(graph, TwoStageCrawlerBatches, target=int(p * graph.snap.GetNodes()))
    # tester.run([
    #     # {'s': 1, 'n': 50000, 'p': p, 'b': 10},
    #     # {'s': 10, 'n': 2000, 'p': p, 'b': 10},
    #     # {'s': 50, 'n': 2000, 'p': p, 'b': 10},
    #     {'s': 1000, 'n': 20000, 'p': p, 'b': 10},
    # ])

    crawlers = [
        # TwoStageCrawlerBatches(graph, s=1000, n=50000, p=p, b=1),
        TwoStageCrawlerBatchesMOD(graph, s=1000, n=50000, p=p, b=1),
    ]

    target_set = set(get_top_centrality_nodes(graph, 'degree', count=int(p * graph[Stat.NODES])))
    metrics = [
        # Metric(r'$|V_o|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        Metric(r'$|V_o \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=50000, step=100)
    ci.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    # test()
    test_target_set_coverage()