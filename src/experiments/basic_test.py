import logging

import matplotlib.pyplot as plt

from crawlers.basic import MaximumObservedDegreeCrawler, BreadthFirstSearchCrawler
from graph_io import GraphCollections


def test_mod(graph):
    crawler = MaximumObservedDegreeCrawler(graph, batch=10, skl_mode=True, initial_seed=1)
    # crawler = BreadthFirstSearchCrawler(graph, initial_seed=10)
    # crawler = RandomCrawler(graph)

    iterations = []
    os = []
    ns = []

    n = 50
    for i in range(n):
        crawler.crawl_budget(1)
        if i % 1000 == 0:
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


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    g = GraphCollections.get(name, giant_only=True)

    test_mod(g)
