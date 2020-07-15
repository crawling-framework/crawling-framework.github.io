import logging

from matplotlib import pyplot as plt

from running.history_runner import CrawlerHistoryRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric

from base.cgraph import MyGraph
from crawlers.cbasic import Crawler, CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler, \
    BreadthFirstSearchCrawler, RandomCrawler, DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler
from crawlers.advanced import ThreeStageCrawler, CrawlerWithAnswer, AvrachenkovCrawler, \
    ThreeStageMODCrawler, ThreeStageCrawlerSeedsAreHubs
from crawlers.multiseed import MultiInstanceCrawler
from running.animated_runner import AnimatedCrawlerRunner, Metric
from graph_io import GraphCollections
from statistics import Stat, get_top_centrality_nodes


class EmulatorWithAnswerCrawler(CrawlerWithAnswer):
    short = 'EmulatorWA'

    def __init__(self, graph: MyGraph, crawler_def, target_size: int, **kwargs):
        super().__init__(graph, target_size=target_size, crawler_def=crawler_def, **kwargs)
        self.pN = target_size

        _, ckwargs = crawler_def
        ckwargs['observed_graph'] = self._observed_graph
        ckwargs['crawled_set'] = self._crawled_set
        ckwargs['observed_set'] = self._observed_set

        self.crawler = Crawler.from_definition(self._orig_graph, crawler_def)
        # self.next_seed = self.crawler.next_seed

    def crawl(self, seed: int):
        self._actual_answer = False
        return self.crawler.crawl(seed)

    def seeds_generator(self):
        for i in range(self.pN):
            yield self.crawler.next_seed()

    def _compute_answer(self):
        self._answer.clear()
        self._get_mod_nodes(self._crawled_set, self._answer, self.pN)
        return 0


def target_set_coverage_bigruns():

    # for name in ('soc-hamsterster', 'petster-hamster', 'soc-anybeat', 'loc-brightkite_edges', 'soc-themarker',
    #              'soc-slashdot', 'soc-BlogCatalog', 'digg-friends', 'soc-twitter-follows'):
    name = 'digg-friends'
    g = GraphCollections.get(name)
    # g = GraphCollections.get('Pokec')

    p = 0.01
    budget = int(0.005 * g.nodes())
    s = int(budget/2)

    crawler_defs = [
        # (MaximumObservedDegreeCrawler, {'batch': 10}),
        # (MaximumObservedDegreeCrawler, {'batch': 100}),
        # (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 10})}),
        # (ThreeStageCrawler, {'s': s, 'n': budget, 'p': p}),
        # (RandomWalkCrawler, {}),
        # (MaximumObservedDegreeCrawler, {'batch': 1}),
        (EmulatorWithAnswerCrawler, {'crawler_def': (DepthFirstSearchCrawler, {}), 'target_size': int(p*g.nodes())}),
        # DepthFirstSearchCrawler(g).definition,
        # RandomWalkCrawler(g).definition,
        # RandomCrawler(g, initial_seed=1).definition,
        # (AvrachenkovCrawler, {'n': budget, 'n1': s, 'k': int(p * g.nodes())}),
        # ThreeStageCrawler(g, s=s, n=budget, p=p).definition,
        # ThreeStageMODCrawler(g, s=1, n=budget, p=p, b=1).definition,
        # (ThreeStageCrawlerSeedsAreHubs, {'s': int(budget / 2), 'n': budget, 'p': p, 'name': 'hub'}),
        # (ThreeStageMODCrawler, {'s': int(budget / 2), 'n': budget, 'p': p, 'b': 1}),
        # (ThreeStageMODCrawler, {'s': int(budget / 3), 'n': budget, 'p': p, 'b': 1}),
        # (ThreeStageMODCrawler, {'s': int(budget / 10), 'n': budget, 'p': p, 'b': 1}),
        # (ThreeStageMODCrawler, {'s': int(budget / 30), 'n': budget, 'p': p, 'b': 1}),
        # (ThreeStageMODCrawler, {'s': int(budget / 100), 'n': budget, 'p': p, 'b': 1}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 10}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 30}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 100}),
    ]
    metric_defs = [
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.DEGREE_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.PAGERANK_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.BETWEENNESS_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.ECCENTRICITY_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.CLOSENESS_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.K_CORENESS_DISTR.short).definition,
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'crawled'}),
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'answer'}),
    ]
    n_instances = 2
    # Run missing iterations
    chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
    chr.run_parallel(n_instances)

        # Run merger
        # crm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
        # crm.draw_by_crawler(x_lims=(0, budget), x_normalize=False, scale=8, draw_error=False)
        # crm.missing_instances()


# def test_detection_quality():
#     # name = 'libimseti'
#     # name = 'petster-friendships-cat'
#     # name = 'soc-pokec-relationships'
#     # name = 'digg-friends'
#     # name = 'loc-brightkite_edges'
#     # name = 'ego-gplus'
#     name = 'petster-hamster'
#     graph = GraphCollections.get(name, giant_only=True)
#
#     p = 0.1
#
#     budget = 200  # 2500 5000 50000
#     start_seeds = 50  # 500 1000 5000
#
#
#     crawler = ThreeStageCrawler(graph, s=start_seeds, n=budget, p=p)
#     crawler.crawl_budget(budget)
#
#     ograph = crawler.observed_graph
#     # nodes = crawler.nodes_set
#     # obs = crawler.observed_set
#     crawler._compute_answer()
#     ans = crawler.answer
#
#     def deg(graph, node):
#         return graph.snap.GetNI(node).GetDeg()
#
#     # print("\n\ncrawler.crawled_set")
#     # for n in crawler.crawled_set:
#     #     o = deg(ograph, n)
#     #     r = deg(graph, n)
#     #     print("n=%s, real=%s, obs=%s %s" % (n, r, o, o/r))
#     #
#     # print("\n\ncrawler.observed_set")
#     # obs = crawler._get_mod_nodes(crawler.observed_set)
#     # for n in obs:
#     #     o = deg(ograph, n)
#     #     r = deg(graph, n)
#     #     print("n=%s, real=%s, obs=%s %s" % (n, r, o, o/r))
#     #
#
#     import numpy as np
#     plt.figure('', (22, 12))
#
#     target_list = get_top_centrality_nodes(graph, 'degree', count=int(graph[Stat.NODES]))
#
#     xs = np.arange(len(target_list))
#     rs = np.array([deg(graph, n) for n in target_list])
#     os = np.array([deg(ograph, n) if n in crawler.nodes_set else 0 for n in target_list])
#     answs = np.array([0.5 if n in crawler.answer else 0 for n in target_list])
#     border = min([deg(ograph, n) for n in ans])
#     print('border', border)
#
#     plt.bar(xs, rs, width=1, color='b', label='real degree')
#     plt.bar(xs, os, width=1, color='g', label='observed degree')
#     plt.bar(xs, answs, width=1, color='r', label='in answer')
#     plt.axvline(int(p*graph[Stat.NODES]), color='c')
#     plt.axhline(border, color='lightgreen')
#
#     plt.yscale('log')
#     plt.xlabel('node')
#     plt.ylabel('degree')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # test()
    # test_target_set_coverage()
    target_set_coverage_bigruns()
    # test_detection_quality()
