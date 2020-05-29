import logging

import snap
from matplotlib import pyplot as plt

from utils import USE_CYTHON_CRAWLERS

if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
    from base.cadvanced import AvrachenkovCrawler, CrawlerWithAnswer, ThreeStageCrawler, \
        ThreeStageMODCrawler, ThreeStageCrawlerSeedsAreHubs
    from base.cbasic import CrawlerException, MaximumObservedDegreeCrawler
    from base.cmultiseed import MultiCrawler
else:
    from base.graph import MyGraph
    from crawlers.advanced import ThreeStageCrawler, CrawlerWithAnswer, AvrachenkovCrawler, \
        ThreeStageMODCrawler
    from crawlers.basic import CrawlerException, MaximumObservedDegreeCrawler
    from crawlers.multiseed import MultiCrawler

from runners.animated_runner import AnimatedCrawlerRunner, Metric
from graph_io import GraphCollections
from statistics import Stat, get_top_centrality_nodes


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
        graph = GraphCollections.get(name, giant_only=True)
        print("N=%s E=%s" % (graph.nodes(), graph.edges()))
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


if not USE_CYTHON_CRAWLERS:
    class ThreeStageCrawlerSeedsAreHubs(ThreeStageCrawler):
        """
        Artificial version of ThreeStageCrawler, where instead of initial random seeds we take hubs
        """
        def __init__(self, graph: MyGraph, s=500, n=1000, p=0.1):
            super().__init__(graph, s=s, n=n, p=p, name='3-StageHubs_s=%s_n=%s_p=%s' % (s, n, p))
            self.hubs = []

        def _seeds_generator(self):
            # 1) hubs as seeds
            hubs = get_top_centrality_nodes(self.orig_graph, Stat.DEGREE_DISTR, count=self.s)
            for i in range(self.s):
                self.hubs.append(hubs[i])
                yield hubs[i]

            # memorize E1
            self.e1 = set(self._observed_set)
            logging.debug("|E1|=", len(self.e1))

            # Check that e1 size is more than (n-s)
            if self.n - self.s > len(self.e1):
                raise CrawlerException("E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." %
                                       (len(self.e1), self.n - self.s))

            # 2) detect MOD batch
            self.top_observed_seeds = self._get_mod_nodes(self._observed_set, self.n - self.s)
            self.e1s = set(self.top_observed_seeds)
            logging.debug("|E1*|=", len(self.e1s))

            for node in self.top_observed_seeds:
                yield node

        def _compute_answer(self):  # E* = S + E1* + E2*
            self.e2 = set(self._observed_set)

            # Get v=(pN-n+|self.hubs|) max degree observed nodes
            self.e2s = set(self._get_mod_nodes(self.e2, self.pN - self.n + len(self.hubs)))

            # Final answer - E* = S + E1* + E2*, |E*|=pN
            self.answer = set(self.hubs).union(self.e1s.union(self.e2s))


def test_target_set_coverage():
    # name, budget, start_seeds = 'soc-pokec-relationships', 50000, 5000
    # name, budget, start_seeds = 'digg-friends', 5000, 1000
    # name, budget, start_seeds = 'loc-brightkite_edges', 2500, 500
    name, budget, start_seeds = 'petster-hamster', 150, 50

    graph = GraphCollections.get(name, giant_only=True)
    p = 0.1
    target_list = get_top_centrality_nodes(graph, Stat.DEGREE_DISTR, count=int(p * graph[Stat.NODES]))
    thr_degree = graph.deg(target_list[-1])
    target_set = set(target_list)

    crawlers = [
        # MaximumObservedDegreeCrawler(graph, batch=1),
        # ThreeStageCrawler(graph, s=start_seeds, n=budget, p=p),
        # ThreeStageMODCrawler(graph, s=1, n=budget, p=p, b=10),
        # ThreeStageMODCrawler(graph, s=10, n=budget, p=p, b=10),
        # ThreeStageMODCrawler(graph, s=100, n=budget, p=p, b=10),
        # ThreeStageMODCrawler(graph, s=1000, n=budget, p=p, b=10),
        # ThreeStageCrawler(graph, s=start_seeds, n=budget, p=p),
        # ThreeStageMODCrawler(graph, s=start_seeds, n=budget, p=p, b=100),
        # ThreeStageFlexMODCrawler(graph, s=start_seeds, n=budget, p=p, b=1, thr_degree=thr_degree),
        # PreferentialObservedDegreeCrawler(graph, batch=1),
        # BreadthFirstSearchCrawler(graph, initial_seed=None),
        # DepthFirstSearchCrawler(graph, initial_seed=None),
        # RandomCrawler(graph, initial_seed=1),
        # RandomWalkCrawler(graph, initial_seed=None),
        # AvrachenkovCrawler(graph, n=budget, n1=start_seeds, k=int(p * graph.nodes())),
        ThreeStageCrawler(graph, s=start_seeds, n=budget, p=p),
        ThreeStageCrawlerSeedsAreHubs(graph, s=start_seeds, n=budget, p=p),
        # ThreeStageMODCrawler(graph, s=start_seeds, n=budget, p=p, b=10),
        # MultiCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=i+1) for i in range(100)
        # ])
    ]

    def re(result):
        return 0 if len(target_set) == 0 else len(target_set.intersection(result)) / (len(target_set))

    def pr(result):
        return 0 if len(result) == 0 else len(target_set.intersection(result)) / (len(result))

    def f1(result):
        p, r = pr(result), re(result)
        return 0 if p == 0 and r == 0 else 2 * p * r / (p + r)

    def precision(crawler):
        result = crawler.answer if isinstance(crawler, CrawlerWithAnswer) else crawler.nodes_set
        return pr(result)

    def recall(crawler):
        result = crawler.answer if isinstance(crawler, CrawlerWithAnswer) else crawler.nodes_set
        return re(result)

    def f1_measure(crawler):
        result = crawler.answer if isinstance(crawler, CrawlerWithAnswer) else crawler.nodes_set
        return f1(result)

    metrics = [
        # Metric(r'$|V_o|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        # Metric(r'$|V_o \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
        Metric(r'$F_1$', f1_measure),
        # Metric(r'Pr', precision),
        # Metric(r'Re', recall),
        # Metric(r'Re - all nodes', recall_all),
        # Metric(r'Pr - E1*', lambda crawler: pr(crawler.e1s)),
        # Metric(r'Pr - E2*', lambda crawler: pr(crawler.e2s)),
        # Metric(r'Re - E1*', lambda crawler: re(crawler.e1s)),
        # Metric(r'Re - E2*', lambda crawler: re(crawler.e2s)),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=budget, step=int(budget/30))
    ci.run(ylims=(0, 1))


def test_detection_quality():
    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    name = 'petster-hamster'
    graph = GraphCollections.get(name, giant_only=True)

    p = 0.1

    budget = 200  # 2500 5000 50000
    start_seeds = 50  # 500 1000 5000


    crawler = ThreeStageCrawler(graph, s=start_seeds, n=budget, p=p)
    crawler.crawl_budget(budget)

    ograph = crawler.observed_graph
    # nodes = crawler.nodes_set
    # obs = crawler.observed_set
    crawler._compute_answer()
    ans = crawler.answer

    def deg(graph, node):
        return graph.snap.GetNI(node).GetDeg()

    # print("\n\ncrawler.crawled_set")
    # for n in crawler.crawled_set:
    #     o = deg(ograph, n)
    #     r = deg(graph, n)
    #     print("n=%s, real=%s, obs=%s %s" % (n, r, o, o/r))
    #
    # print("\n\ncrawler.observed_set")
    # obs = crawler._get_mod_nodes(crawler.observed_set)
    # for n in obs:
    #     o = deg(ograph, n)
    #     r = deg(graph, n)
    #     print("n=%s, real=%s, obs=%s %s" % (n, r, o, o/r))
    #

    import numpy as np
    plt.figure('', (22, 12))

    target_list = get_top_centrality_nodes(graph, 'degree', count=int(graph[Stat.NODES]))

    xs = np.arange(len(target_list))
    rs = np.array([deg(graph, n) for n in target_list])
    os = np.array([deg(ograph, n) if n in crawler.nodes_set else 0 for n in target_list])
    answs = np.array([0.5 if n in crawler.answer else 0 for n in target_list])
    border = min([deg(ograph, n) for n in ans])
    print('border', border)

    plt.bar(xs, rs, width=1, color='b', label='real degree')
    plt.bar(xs, os, width=1, color='g', label='observed degree')
    plt.bar(xs, answs, width=1, color='r', label='in answer')
    plt.axvline(int(p*graph[Stat.NODES]), color='c')
    plt.axhline(border, color='lightgreen')

    plt.yscale('log')
    plt.xlabel('node')
    plt.ylabel('degree')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # test()
    test_target_set_coverage()
    # test_detection_quality()
