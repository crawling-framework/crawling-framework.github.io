import logging

from matplotlib import pyplot as plt

from running.history_runner import CrawlerHistoryRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric

from base.cgraph import MyGraph
from crawlers.cbasic import CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler, \
    BreadthFirstSearchCrawler, DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler
from crawlers.advanced import ThreeStageCrawler, CrawlerWithAnswer, AvrachenkovCrawler, \
    ThreeStageMODCrawler, ThreeStageCrawlerSeedsAreHubs, EmulatorWithAnswerCrawler
from crawlers.multiseed import MultiInstanceCrawler
from running.animated_runner import AnimatedCrawlerRunner, Metric
from graph_io import GraphCollections
from statistics import Stat, get_top_centrality_nodes


def test_target_set_coverage():
    # name, budget, start_seeds = 'flixster', 50000, 10000
    # name, budget, start_seeds = 'soc-pokec-relationships', 3000, 3000
    name, budget, start_seeds = 'digg-friends', 1000, 500
    # name, budget, start_seeds = 'soc-twitter-follows', 10000, 1000
    # name, budget, start_seeds = 'tech-p2p-gnutella', 3000, 500
    # name, budget, start_seeds = 'loc-brightkite_edges', 2500, 500
    # name, budget, start_seeds = 'petster-hamster', 200, 100
    # name, budget, start_seeds = 'dolphins', 1, 10

    # name, budget, start_seeds = 'ca-CondMat', 1000, 2000
    # name, budget, start_seeds = 'soc-hamsterster', 1000, 2000
    # name, budget, start_seeds = 'tech-WHOIS', 1000, 2000
    # name, budget, start_seeds = 'ca-AstroPh', 1000, 2000
    # name, budget, start_seeds = 'tech-pgp', 1000, 2000
    # name, budget, start_seeds = 'tech-routers-rf', 1000, 2000
    # name, budget, start_seeds = 'web-indochina-2004', 1000, 2000
    # name, budget, start_seeds = 'soc-anybeat', 1000, 2000
    # name, budget, start_seeds = 'web-uk-2005', 1000, 2000

    # name, budget, start_seeds = 'socfb-Bingham82', 1000, 2000
    graph = GraphCollections.get(name, giant_only=True)

    p = 0.01
    # p = 100 / graph.nodes()
    target_list = get_top_centrality_nodes(graph, Stat.DEGREE_DISTR, count=int(p * graph[Stat.NODES]))
    thr_degree = graph.deg(target_list[-1])
    target_set = set(target_list)

    budget = int(0.05 * graph.nodes())
    start_seeds = int(0.7 * budget)

    crawlers = [
        # (DE_Crawler, {'initial_budget': int(0.15*budget)}),
        # BreadthFirstSearchCrawler(graph),
        # RandomWalkCrawler(graph),
        # (MaximumObservedDegreeCrawler, {'batch': 1}),
        # # PreferentialObservedDegreeCrawler(graph, batch=1),
        # MaximumExcessDegreeCrawler(graph),
        (AvrachenkovCrawler, {'n': budget, 'n1': start_seeds, 'k': int(p * graph.nodes())}),
        (ThreeStageCrawler, {'s': start_seeds, 'n': budget, 'p': p}),
        (ThreeStageMODCrawler, {'s': 0, 'n': budget, 'p': p, 'b': 1}),
        (EmulatorWithAnswerCrawler, {'crawler_def': (MaximumObservedDegreeCrawler, {}), 'n': budget, 'target_size': int(p*graph.nodes())}),
        # ThreeStageMODCrawler(graph, s=1, n=budget, p=p, b=10),
        # ThreeStageMODCrawler(graph, s=10, n=budget, p=p, b=10),
        # ThreeStageMODCrawler(graph, s=100, n=budget, p=p, b=10),
        # ThreeStageMODCrawler(graph, s=1000, n=budget, p=p, b=10),
        # (ThreeStageCrawler, {'s': start_seeds, 'n': budget, 'p': p}),
        # (ThreeStageMODCrawler, {'s': start_seeds, 'n': budget, 'p': p, 'b': 2}),
        # ThreeStageFlexMODCrawler(graph, s=start_seeds, n=budget, p=p, b=1, thr_degree=thr_degree),
        # DepthFirstSearchCrawler(graph, initial_seed=1),
        # RandomCrawler(graph, initial_seed=1),
        # ThreeStageCrawler(graph, s=start_seeds, n=budget, p=p),
        # ThreeStageCrawlerSeedsAreHubs(graph, s=start_seeds, n=budget, p=p),
        # ThreeStageMODCrawler(graph, s=1, n=budget, p=p, b=1),
        # ThreeStageMODCrawler(graph, s=100, n=budget, p=p, b=1),
        # MultiInstanceCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=i+1) for i in range(100)
        # ])
    ]

    metrics = [
        # Metric(r'$|V_o|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        # Metric(r'$|V_o \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
        # Metric(r'$F_1$', f1_measure),
        # Metric(r'Pr', precision),
        # Metric(r'Re', recall),
        # Metric(r'Re - all nodes', recall_all),
        # Metric(r'Pr - E1*', lambda crawler: pr(crawler.e1s)),
        # Metric(r'Pr - E2*', lambda crawler: pr(crawler.e2s)),
        # Metric(r'Re - E1*', lambda crawler: re(crawler.e1s)),
        # Metric(r'Re - E2*', lambda crawler: re(crawler.e2s)),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'crawled'}),
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'answer'}),
    ]

    from time import time
    t = time()
    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=budget)
    ci.run()
    print(time()-t)

    # print("time_top", crawlers[0].time_top)
    # print("time_clust", crawlers[0].time_clust)
    # print("time_next", crawlers[0].time_next)


def target_set_coverage_bigruns():
    g = GraphCollections.get('digg-friends')
    # g = GraphCollections.get('Pokec')

    p = 0.1
    budget = int(0.03 * g.nodes())
    s = int(0.5 * budget)

    crawler_defs = [
        # (MaximumObservedDegreeCrawler, {'batch': 1}),
        # (MaximumObservedDegreeCrawler, {'batch': 10}),
        # (MaximumObservedDegreeCrawler, {'batch': 100}),
        # (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 10})}),
        (ThreeStageCrawler, {'s': s, 'n': budget, 'p': p}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p}),
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
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'answer'}),

    ]
    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=budget)
    acr.run()

    # n_instances = 10
    # # Run missing iterations
    # chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
    # chr.run_missing(n_instances)
    #
    # # Run merger
    # crm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
    # crm.draw_by_metric_crawler(x_lims=(0, budget), x_normalize=False, scale=8, swap_coloring_scheme=True, draw_error=False)
    # # crm.missing_instances()


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
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # test()
    test_target_set_coverage()
    # target_set_coverage_bigruns()
    # test_detection_quality()
