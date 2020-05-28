from utils import rel_dir, USE_CYTHON_CRAWLERS

import logging
import matplotlib.pyplot as plt

if USE_CYTHON_CRAWLERS:
    from base.cbasic import CCrawler as Crawler, RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, \
        DepthFirstSearchCrawler, SnowBallCrawler, MaximumObservedDegreeCrawler, \
        PreferentialObservedDegreeCrawler
    from base.cmultiseed import MultiCrawler
else:
    from crawlers.basic import RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, \
        DepthFirstSearchCrawler, SnowBallCrawler, MaximumObservedDegreeCrawler, \
        PreferentialObservedDegreeCrawler
    from crawlers.multiseed import MultiCrawler

from graph_io import GraphCollections
from runners.animated_runner import AnimatedCrawlerRunner, Metric


def test_basic(graph):
    crawlers = [
        MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=1),
        # PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=1),
        # BreadthFirstSearchCrawler(graph, initial_seed=10),
        # RandomCrawler(graph),
    ]

    metrics = [
        Metric('observed_set', lambda crawler: len(crawler.observed_set)),
        Metric('nodes_set', lambda crawler: len(crawler.nodes_set)),
    ]
    acr = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=1e3, step=1e2)
    acr.run()

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

    if USE_CYTHON_CRAWLERS:
        print("After cython crawlers. Graph %s, n=%s steps" % (name, n))
    else:
        print("Before cython crawlers. Graph %s, n=%s steps" % (name, n))

    crawler = MultiCrawler(graph, crawlers=[
        # RandomCrawler(graph) for i in range(10)
        # RandomWalkCrawler(graph) for i in range(10)
        # BreadthFirstSearchCrawler(graph, initial_seed=i+1) for i in range(100)
        DepthFirstSearchCrawler(graph, initial_seed=i+1) for i in range(100)
        # MaximumObservedDegreeCrawler(graph, name='MOD%s'%i, batch=10, initial_seed=i+1) for i in range(10)
        # PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=i+1) for i in range(10)
        # MaximumObservedDegreeCrawler(graph, name='MOD0', batch=1, initial_seed=1),
        # MaximumObservedDegreeCrawler(graph, name='MOD1', batch=1, initial_seed=2),
    ])

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

    n = 50
    # name = 'dolphins'
    name = 'digg-friends'
    # name = 'soc-pokec-relationships'
    g = GraphCollections.get(name)
    if USE_CYTHON_CRAWLERS:
        print("After cython crawlers. Graph %s, n=%s steps" % (name, n))
    else:
        print("Before cython crawlers. Graph %s, n=%s steps" % (name, n))

    import numpy as np
    for crawler_cls in [
        RandomCrawler,
        RandomWalkCrawler,
        BreadthFirstSearchCrawler,
        DepthFirstSearchCrawler,
        SnowBallCrawler,
        MaximumObservedDegreeCrawler,
        PreferentialObservedDegreeCrawler,
    ]:
        times = []
        it = 1
        for _ in range(it):
            # crawler = crawler_cls(g, initial_seed=1, batch=100)
            crawler = MultiCrawler(g, [
                crawler_cls(g, batch=10) for _ in range(10)
            ])
            t = time()
            for i in range(n):
                crawler.crawl_budget(1)
            t = (time()-t)*1000
            print("%s. %.3f ms" % (crawler.name, t))
            times.append(t)
        print("%s %.1f +- %.1f ms" % (crawler.name, np.mean(times), np.var(times)**0.5))

    # print("next_seed times", crawler.timer*1000)


    # SKL vs dict per batches, s. 'digg-friends'. Graph loading time included :(
    # n=5000               n=50000                n=250000
    # batch  SKL   dict  |  batch  SKL   dict  |   batch  SKL   dict
    # 1      10     314  |                     |
    # 10     10     32   |                     |
    # 30     11     12   |                     |
    # 100    9      4.5  |  100    14     53   |   100    17     170
    # 300    9      2.5  |  300    14     19   |   300    16     63
    # 1000   8.4    1.6  |  1000   13     7.5  |   1000   16     20
    # 3000   8.4    1.2  |  3000   13     4.1  |   3000   16     9
    #
    # SKL vs dict per batches, s. 'soc-pokec-relationships'
    # n=5000               n=50000                n=250000
    # batch  SKL   dict  |  batch  SKL   dict  |   batch  SKL   dict
    # 10     18     42   |                     |
    # 30     17     23   |                     |
    # 100    16     17   |  100    51     136  |
    # 300    16     15   |  300    54     53   |   300    234    577
    # 1000   16     15   |  1000   45     30   |   1000   208    192
    # 3000   17     14   |  3000   48     22   |   3000   211    88
    #

    # crawler = PreferentialObservedDegreeCrawler(g, batch=1)
    # t = time()
    # for i in range(n):
    #     crawler.crawl_budget(1)  # initial - 40 s
    #     # crawler.crawl_budget(1)  # self.observed_graph.snap -> g - 38-43 s
    #     # crawler.crawl_budget(1)  # heap -> sort dict, queue - 38-46 s
    #     # crawler.crawl_budget(1)  # batch=10, heap - 5.5 s
    #     # crawler.crawl_budget(1)  # batch=10, sort dict, queue - 4 s
    #
    # print("%.3f ms" % ((time()-t)*1000))


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
    # name = 'digg-friends'
    # name = 'Lj'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    name = 'petster-hamster'
    # name = 'dolphins'
    # g = test_carpet_graph(8, 8)[0]
    g = GraphCollections.get(name, giant_only=True)
    # g = GraphCollections.get('test', 'other', giant_only=True)

    test_basic(g)
    # test_multi()
    # test_crawler_times()

