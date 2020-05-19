import logging

import matplotlib.pyplot as plt

from crawlers.basic import MaximumObservedDegreeCrawler, BreadthFirstSearchCrawler, \
    PreferentialObservedDegreeCrawler
from crawlers.multiseed import MultiCrawler, test_carpet_graph
from graph_io import GraphCollections


def test_basic(graph):
    # crawler = MaximumObservedDegreeCrawler(graph, batch=1, skl_mode=True, initial_seed=1)
    crawler = PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=1)
    # crawler = BreadthFirstSearchCrawler(graph, initial_seed=10)
    # crawler = RandomCrawler(graph)

    iterations = []
    os = []
    ns = []

    n = 50000
    for i in range(n):
        crawler.crawl_budget(1)
        if i % 500 == 0:
            iterations.append(i)
            os.append(len(crawler._observed_set))
            ns.append(len(crawler.nodes_set))

            plt.cla()
            plt.plot(iterations, os, marker='o', color='g', label='observed_set')
            plt.plot(iterations, ns, marker='o', color='b', label='nodes_set')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.pause(0.005)

    plt.show()


def test_multi(graph):
    crawler = MultiCrawler(graph, crawlers=[
        # BreadthFirstSearchCrawler(graph, initial_seed=i+1) for i in range(100)
        MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=i+1) for i in range(10)
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


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    # name = 'dolphins'
    # g = test_carpet_graph(8, 8)[0]
    g = GraphCollections.get(name, giant_only=True)
    # g = GraphCollections.get('test', 'other', giant_only=True)

    test_basic(g)
    # test_multi(g)
