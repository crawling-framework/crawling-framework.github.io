from utils import rel_dir

import logging
import matplotlib.pyplot as plt

from crawlers.cbasic import Crawler, RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, \
    DepthFirstSearchCrawler, SnowBallCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler
from crawlers.cadvanced import DE_Crawler
from crawlers.multiseed import MultiInstanceCrawler
from graph_io import GraphCollections
from runners.animated_runner import AnimatedCrawlerRunner, Metric


def test_basic(graph):
    crawler_defs = [
        (RandomCrawler, {'name': 'RC'}),
        (MaximumObservedDegreeCrawler, {'batch': 1}),
        (MaximumObservedDegreeCrawler, {'batch': 1, 'initial_seed': 2}),
        # (MaximumObservedDegreeCrawler, {'batch': 1, 'initial_seed': 2}),
        (MultiInstanceCrawler, {'count': 1, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 1})}),
    ]

    metrics = [
        Metric('observed_set', lambda crawler: len(crawler.observed_set)),
        Metric('nodes_set', lambda crawler: len(crawler.nodes_set)),
    ]
    acr = AnimatedCrawlerRunner(graph, crawler_defs, metrics, budget=1e4, step=1e3)
    acr.run(same_initial_seed=False)

    # iterations = []
    # os = []
    # ns = []
    #
    # n = 50000
    # for i in range(n):
    #     crawler.crawl_budget(1)
    #     if i % 500 == 0:
    #         iterations.append(i)
    #         os.append(len(crawler._observed_set))
    #         ns.append(len(crawler.nodes_set))
    #
    #         plt.cla()
    #         plt.plot(iterations, os, marker='o', color='g', label='observed_set')
    #         plt.plot(iterations, ns, marker='o', color='b', label='nodes_set')
    #         plt.grid()
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.pause(0.005)
    #
    # plt.show()


def test_multi():
    n = 5000
    # name = 'dolphins'
    name = 'digg-friends'
    # name = 'soc-pokec-relationships'
    graph = GraphCollections.get(name)

    crawler = MultiInstanceCrawler(graph, count=100, crawler_def=
        # RandomCrawler(graph) for i in range(10)
        # RandomWalkCrawler(graph) for i in range(10)
        # BreadthFirstSearchCrawler(graph, initial_seed=i+1) for i in range(100)
        (DepthFirstSearchCrawler, {})
        # MaximumObservedDegreeCrawler(graph, name='MOD%s'%i, batch=10, initial_seed=i+1) for i in range(10)
        # PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=i+1) for i in range(10)
        # MaximumObservedDegreeCrawler(graph, name='MOD0', batch=1, initial_seed=1),
        # MaximumObservedDegreeCrawler(graph, name='MOD1', batch=1, initial_seed=2),
    )

    iterations = []
    os = []
    ns = []
    cs = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    n = min(2001, graph['NODES'])
    for i in range(n):
        crawler.crawl_budget(1)
        if i % 100 == 0:
            iterations.append(i)
            os.append(len(crawler.observed_set))
            ns.append(len(crawler.nodes_set))
            cs.append(len(crawler.crawlers))

            ax1.cla()
            ax2.cla()
            ax1.plot(iterations, os, marker='o', color='g', label='observed_set')
            ax1.plot(iterations, ns, marker='o', color='b', label='nodes_set')
            ax2.plot(iterations, cs, marker='.', color='r', label='n_crawlers')
            ax1.legend(loc=0)
            ax2.legend(loc=1)
            ax2.set_ylabel('n_crawlers')
            ax1.grid(axis='both')
            plt.tight_layout()
            plt.pause(0.005)

    plt.show()


def test_snap_times():
    from time import time

    n = 100
    g = GraphCollections.get('ego-gplus').snap
    ids = [n.GetId() for n in g.Nodes()]
    nodes = [n for n in g.Nodes()]

    t = time()
    for i in range(n):
        # a = [n for n in range(g.GetNodes())]  # 1 ms

        # a = [g.GetNI(n).GetDeg() for n in ids]  # 17 ms
        # a = [n for n in g.Nodes()]  # 27 ms
        # a = [n.GetId() for n in g.Nodes()]  # 32 ms

        # a = [n.GetDeg() for n in g.Nodes()]  # 33 ms
        # a = [n.GetDeg() for n in nodes]  # RuntimeError

        a = {n: g.GetNI(n).GetDeg() for n in ids}  # 18 ms
        # a = {n.GetId(): n.GetDeg() for n in g.Nodes()}  # 38 ms

    print("%.3f ms" % ((time()-t)*1000))


def test_crawler_times():
    from time import time

    n = 500
    # name = 'dolphins'
    name = 'digg-friends'
    # name = 'soc-pokec-relationships'
    g = GraphCollections.get(name)

    import numpy as np
    for crawler_def in [
        # (RandomCrawler, {}),
        # (RandomWalkCrawler, {}),
        # (BreadthFirstSearchCrawler, {}),
        # (DepthFirstSearchCrawler, {}),
        # (SnowBallCrawler, {}),
        # (MaximumObservedDegreeCrawler, {'batch': 1}),
        # (PreferentialObservedDegreeCrawler, {'batch': 10}),
        (DE_Crawler, {'initial_budget': 10}),

        # (ThreeStageMODCrawler, {'s': n//2, 'n': n, 'p': 0.01, 'b': 1}),
        # (ThreeStageCrawler, {'s': n//2, 'n': n, 'p': 0.01}),
        # (AvrachenkovCrawler, {'n1': n//2, 'n': n, 'k': 100}),
        #
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (RandomCrawler, {})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (RandomWalkCrawler, {})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (BreadthFirstSearchCrawler, {})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (DepthFirstSearchCrawler, {})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (SnowBallCrawler, {})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 10})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (PreferentialObservedDegreeCrawler, {'batch': 10})}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (DE_Crawler, {})}),
    ]:
        times = []
        it = 1
        for _ in range(it):
            crawler = Crawler.from_definition(g, crawler_def)
            t = time()
            for i in range(n):
                crawler.crawl_budget(1)
            t = (time()-t)*1000
            # print("%s. %.3f ms" % (crawler.name, t))
            times.append(t)
        print("%s %.1f +- %.1f ms" % (crawler.name, np.mean(times), np.var(times)**0.5))


def test_numpy_times():
    from time import time

    n = 1000000
    res = 100
    p = 0.1

    # t = time()
    # for i in range(n):
    #     a = []
    #     for j in range(res):
    #         if np.random.random() < p:
    #             a.append(1)
    #         else:
    #             a.append(0)
    # print("%.3f ms" % ((time()-t)*1000))
    #
    # t = time()
    # for i in range(n):
    #     a = []
    #     neighbors = list(range(res))
    #     binomial_map = np.random.binomial(1, p=p, size=len(neighbors))
    #     [a.append(1) for j in neighbors if (binomial_map[neighbors.index(j)] == 1)]
    #     [a.append(0) for j in neighbors if (binomial_map[neighbors.index(j)] == 0)]
    #
    # print("%.3f ms" % ((time()-t)*1000))

    t = time()
    for i in range(n):
        a = set()
        for k in range(100):
            a.add(k)
    print("%.3f ms" % ((time()-t)*1000))

    t = time()
    for i in range(n):
        a = set()
        a.update(range(100))

    print("%.3f ms" % ((time()-t)*1000))


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # test_snap_times()
    # test_crawler_times()
    # test_numpy_times()

    # name = 'youtube-u-growth'
    # name = 'flixster'
    # name = 'flickr-links'
    # name = 'soc-pokec-relationships'
    name = 'digg-friends'
    # name = 'com-youtube'
    # name = 'Lj'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    # name = 'dolphins'
    # g = test_carpet_graph(8, 8)[0]
    g = GraphCollections.get(name, giant_only=True)

    test_basic(g)
    # test_multi()
    # test_crawler_times()
